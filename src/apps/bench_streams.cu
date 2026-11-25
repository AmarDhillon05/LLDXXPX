// End-to-end streaming benchmark: a batch of frames goes host -> device ->
// edge pipeline -> host, comparing memory types and stream counts.
//
//   ./bench_streams [--frames 1000] [--size 1080p] [--streams 1,2,4,8]
//                   [--pipeline fused|separate] [--csv results/streams.csv]
//                   [--timeline results/timeline.csv]
//
// Modes
//   pageable_sync     malloc'd host memory, cudaMemcpy + kernel + cudaMemcpy per frame
//   pageable_streams  same buffers, cudaMemcpyAsync on N streams (it cannot overlap:
//                     pageable copies are staged through a driver bounce buffer)
//   pinned_sync       page-locked host memory, still one frame at a time
//   pinned_streams    page-locked + N streams round-robin: H2D of frame i+1,
//                     compute of frame i and D2H of frame i-1 run concurrently
//
// Also reports the per-frame cost of each stage in isolation (H2D, kernel,
// D2H) so the "transfer dominates" story is visible in the numbers.
//
// The 1000 frames are cycled from a ring of distinct host frames (each still
// crosses PCIe), to keep host memory bounded at 8K.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP() ((void)0)
#endif

#include "app_util.hpp"
#include "cpu_reference.hpp"
#include "cuda_utils.cuh"
#include "image.hpp"
#include "kernels.cuh"

namespace {

constexpr int kRing = 8;  // distinct host input frames

struct Config {
    int w, h, frames;
    bool fused;
    Params p;
};

// Device-side working set for one stream.
struct Lane {
    cudaStream_t stream{};
    DeviceBuffer<uchar4> in;
    DeviceBuffer<uint8_t> out;
    DeviceBuffer<float> gray, tmp, blurred;  // only for the separate pipeline
    uint8_t* host_out = nullptr;             // where this lane's results land
    int last_frame = -1;
};

void run_pipeline(const Config& c, Lane& l, cudaStream_t s) {
    if (c.fused) {
        pipeline_fused(l.in.get(), l.out.get(), c.w, c.h, c.p.blur_radius, c.p.sobel_threshold, s);
    } else {
        pipeline_separate(l.in.get(), l.gray.get(), l.tmp.get(), l.blurred.get(), l.out.get(), c.w, c.h,
                          c.p.blur_radius, c.p.sobel_threshold, s);
    }
}

struct Host {
    std::vector<uint8_t*> in;   // kRing frames, RGBA
    std::vector<uint8_t*> out;  // one per lane
    bool pinned;
};

Host make_host(const Config& c, const std::vector<ImageRGBA8>& frames, int lanes, bool pinned) {
    Host h;
    h.pinned = pinned;
    const size_t in_bytes = size_t(c.w) * c.h * 4, out_bytes = size_t(c.w) * c.h;
    for (int i = 0; i < kRing; ++i) {
        uint8_t* p;
        if (pinned) CUDA_CHECK(cudaMallocHost(&p, in_bytes));
        else p = static_cast<uint8_t*>(std::malloc(in_bytes));
        std::memcpy(p, frames[i].data.data(), in_bytes);
        h.in.push_back(p);
    }
    for (int i = 0; i < lanes; ++i) {
        uint8_t* p;
        if (pinned) CUDA_CHECK(cudaMallocHost(&p, out_bytes));
        else p = static_cast<uint8_t*>(std::malloc(out_bytes));
        std::memset(p, 0, out_bytes);
        h.out.push_back(p);
    }
    return h;
}

void free_host(Host& h) {
    for (auto* p : h.in) h.pinned ? (void)cudaFreeHost(p) : std::free(p);
    for (auto* p : h.out) h.pinned ? (void)cudaFreeHost(p) : std::free(p);
}

std::vector<Lane> make_lanes(const Config& c, int n) {
    std::vector<Lane> lanes(n);
    const size_t px = size_t(c.w) * c.h;
    for (auto& l : lanes) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&l.stream, cudaStreamNonBlocking));
        l.in.allocate(px);
        l.out.allocate(px);
        if (!c.fused) {
            l.gray.allocate(px);
            l.tmp.allocate(px);
            l.blurred.allocate(px);
        }
    }
    return lanes;
}

void destroy_lanes(std::vector<Lane>& lanes) {
    for (auto& l : lanes) CUDA_CHECK(cudaStreamDestroy(l.stream));
}

struct TimelineEvent {
    int lane, frame;
    cudaEvent_t e[4];
};

// Process all frames. If `sync`, every frame is fully synchronous (cudaMemcpy).
// If `timeline` is non-null, events are recorded for the first frames.
double process(const Config& c, Host& host, std::vector<Lane>& lanes, bool sync,
               std::vector<TimelineEvent>* timeline = nullptr, cudaEvent_t t0 = nullptr) {
    const size_t in_bytes = size_t(c.w) * c.h * 4, out_bytes = size_t(c.w) * c.h;
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    if (t0) CUDA_CHECK(cudaEventRecord(t0, lanes[0].stream));
    for (int f = 0; f < c.frames; ++f) {
        Lane& l = lanes[f % lanes.size()];
        const uint8_t* src = host.in[f % kRing];
        uint8_t* dst = host.out[f % lanes.size()];
        TimelineEvent* te = nullptr;
        if (timeline && f < int(timeline->capacity())) {
            timeline->push_back({int(f % lanes.size()), f, {}});
            te = &timeline->back();
            for (auto& e : te->e) CUDA_CHECK(cudaEventCreate(&e));
        }
        if (sync) {
            CUDA_CHECK(cudaMemcpy(l.in.get(), src, in_bytes, cudaMemcpyHostToDevice));
            run_pipeline(c, l, 0);
            CUDA_CHECK(cudaMemcpy(dst, l.out.get(), out_bytes, cudaMemcpyDeviceToHost));
        } else {
            if (te) CUDA_CHECK(cudaEventRecord(te->e[0], l.stream));
            CUDA_CHECK(cudaMemcpyAsync(l.in.get(), src, in_bytes, cudaMemcpyHostToDevice, l.stream));
            if (te) CUDA_CHECK(cudaEventRecord(te->e[1], l.stream));
            run_pipeline(c, l, l.stream);
            if (te) CUDA_CHECK(cudaEventRecord(te->e[2], l.stream));
            CUDA_CHECK(cudaMemcpyAsync(dst, l.out.get(), out_bytes, cudaMemcpyDeviceToHost, l.stream));
            if (te) CUDA_CHECK(cudaEventRecord(te->e[3], l.stream));
        }
        l.last_frame = f;
        l.host_out = dst;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Verify each lane's final output against the CPU-computed result for its frame.
bool verify(const Config& c, const std::vector<Lane>& lanes, const std::vector<ImageU8>& ref) {
    const size_t n = size_t(c.w) * c.h;
    for (const auto& l : lanes) {
        if (l.last_frame < 0) continue;
        const ImageU8& r = ref[l.last_frame % kRing];
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += l.host_out[i] != r.data[i];
        if (bad > n / 10000) {
            std::fprintf(stderr, "  verify FAILED: frame %d has %zu mismatching pixels\n", l.last_frame, bad);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv);
    Config c{};
    c.p = args.params();
    const auto sz = parse_sizes(args.str("size", "1080p"));
    if (sz.empty()) return 2;
    c.w = sz[0].w;
    c.h = sz[0].h;
    c.frames = args.i("frames", 1000);
    c.fused = args.str("pipeline", "fused") != "separate";
    std::vector<int> stream_counts;
    {
        const std::string s = args.str("streams", "1,2,4,8");
        size_t pos = 0;
        while (pos < s.size()) {
            size_t e = s.find(',', pos);
            if (e == std::string::npos) e = s.size();
            stream_counts.push_back(std::atoi(s.substr(pos, e - pos).c_str()));
            pos = e + 1;
        }
    }
    const std::string csv = args.str("csv", "results/streams.csv");
    const std::string tl_csv = args.str("timeline", "results/timeline.csv");

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s (async engines: %d)\n", prop.name, prop.asyncEngineCount);
    std::printf("%d frames of %s (%dx%d RGBA8 -> 8-bit edges), %s pipeline\n", c.frames, sz[0].name.c_str(), c.w,
                c.h, c.fused ? "fused" : "separate");

    const auto k1 = cpu::gaussian_kernel_1d(c.p.blur_radius, c.p.blur_sigma);
    gaussian_constant_set_weights(k1.data(), c.p.blur_radius);
    pipeline_fused_set_weights(k1.data(), c.p.blur_radius);

    std::vector<ImageRGBA8> frames;
    std::vector<ImageU8> ref(kRing);
    for (int i = 0; i < kRing; ++i) {
        frames.push_back(make_test_image(c.w, c.h, 100 + i));
        cpu::edge_pipeline(frames.back(), ref[i], c.p.blur_radius, c.p.blur_sigma, c.p.sobel_threshold);
    }

    FILE* f = std::fopen(csv.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", csv.c_str());
        return 1;
    }
    std::fprintf(f, "mode,streams,frames,width,height,pipeline,total_ms,ms_per_frame,fps\n");
    auto emit = [&](const char* mode, int ns, double ms) {
        std::printf("  %-18s streams=%d  total %9.1f ms  %7.3f ms/frame  %8.1f FPS\n", mode, ns, ms, ms / c.frames,
                    1000.0 * c.frames / ms);
        std::fprintf(f, "%s,%d,%d,%d,%d,%s,%.3f,%.4f,%.2f\n", mode, ns, c.frames, c.w, c.h, c.fused ? "fused" : "separate",
                     ms, ms / c.frames, 1000.0 * c.frames / ms);
        std::fflush(stdout);
    };

    bool ok = true;
    // --- per-stage costs in isolation (pinned, 1 stream, event-timed) ---
    {
        auto lanes = make_lanes(c, 1);
        Host host = make_host(c, frames, 1, true);
        const size_t in_bytes = size_t(c.w) * c.h * 4, out_bytes = size_t(c.w) * c.h;
        GpuTimer t;
        auto stage = [&](auto fn) {
            for (int i = 0; i < 3; ++i) fn();
            t.start(lanes[0].stream);
            for (int i = 0; i < 50; ++i) fn();
            t.stop(lanes[0].stream);
            return t.elapsed_ms() / 50.0;
        };
        cudaStream_t s = lanes[0].stream;
        const double h2d = stage([&] {
            CUDA_CHECK(cudaMemcpyAsync(lanes[0].in.get(), host.in[0], in_bytes, cudaMemcpyHostToDevice, s));
        });
        const double ker = stage([&] { run_pipeline(c, lanes[0], s); });
        const double d2h = stage([&] {
            CUDA_CHECK(cudaMemcpyAsync(host.out[0], lanes[0].out.get(), out_bytes, cudaMemcpyDeviceToHost, s));
        });
        std::printf("Per-frame stage cost: H2D %.3f ms  kernel %.3f ms  D2H %.3f ms  (sum %.3f, max %.3f)\n", h2d, ker,
                    d2h, h2d + ker + d2h, std::max({h2d, ker, d2h}));
        std::fprintf(f, "stage_h2d,1,1,%d,%d,%s,%.4f,%.4f,0\n", c.w, c.h, c.fused ? "fused" : "separate", h2d, h2d);
        std::fprintf(f, "stage_kernel,1,1,%d,%d,%s,%.4f,%.4f,0\n", c.w, c.h, c.fused ? "fused" : "separate", ker, ker);
        std::fprintf(f, "stage_d2h,1,1,%d,%d,%s,%.4f,%.4f,0\n", c.w, c.h, c.fused ? "fused" : "separate", d2h, d2h);
        free_host(host);
        destroy_lanes(lanes);
    }

    // --- pageable ---
    {
        auto lanes = make_lanes(c, 1);
        Host host = make_host(c, frames, 1, false);
        NVTX_PUSH("pageable_sync");
        emit("pageable_sync", 1, process(c, host, lanes, true));
        NVTX_POP();
        ok &= verify(c, lanes, ref);
        free_host(host);
        destroy_lanes(lanes);
    }
    {
        const int ns = *std::max_element(stream_counts.begin(), stream_counts.end());
        auto lanes = make_lanes(c, ns);
        Host host = make_host(c, frames, ns, false);
        NVTX_PUSH("pageable_streams");
        emit("pageable_streams", ns, process(c, host, lanes, false));
        NVTX_POP();
        ok &= verify(c, lanes, ref);
        free_host(host);
        destroy_lanes(lanes);
    }
    // --- pinned ---
    {
        auto lanes = make_lanes(c, 1);
        Host host = make_host(c, frames, 1, true);
        NVTX_PUSH("pinned_sync");
        emit("pinned_sync", 1, process(c, host, lanes, true));
        NVTX_POP();
        ok &= verify(c, lanes, ref);
        free_host(host);
        destroy_lanes(lanes);
    }
    for (int ns : stream_counts) {
        auto lanes = make_lanes(c, ns);
        Host host = make_host(c, frames, ns, true);
        NVTX_PUSH(("pinned_streams_" + std::to_string(ns)).c_str());
        emit("pinned_streams", ns, process(c, host, lanes, false));
        NVTX_POP();
        ok &= verify(c, lanes, ref);
        free_host(host);
        destroy_lanes(lanes);
    }
    std::fclose(f);

    // --- event timeline for the copy/compute overlap plot ---
    {
        const int ns = std::min(4, *std::max_element(stream_counts.begin(), stream_counts.end()));
        Config tc = c;
        tc.frames = 16;
        auto lanes = make_lanes(tc, ns);
        Host host = make_host(tc, frames, ns, true);
        process(tc, host, lanes, false);  // warm-up
        std::vector<TimelineEvent> tl;
        tl.reserve(tc.frames);
        cudaEvent_t t0;
        CUDA_CHECK(cudaEventCreate(&t0));
        process(tc, host, lanes, false, &tl, t0);
        FILE* g = std::fopen(tl_csv.c_str(), "w");
        if (g) {
            std::fprintf(g, "stream,frame,op,start_ms,end_ms\n");
            const char* names[3] = {"H2D", "kernel", "D2H"};
            for (const auto& e : tl) {
                for (int k = 0; k < 3; ++k) {
                    float a, b;
                    CUDA_CHECK(cudaEventElapsedTime(&a, t0, e.e[k]));
                    CUDA_CHECK(cudaEventElapsedTime(&b, t0, e.e[k + 1]));
                    std::fprintf(g, "%d,%d,%s,%.4f,%.4f\n", e.lane, e.frame, names[k], a, b);
                }
            }
            std::fclose(g);
            std::printf("wrote %s (event-based; use Nsight Systems for the exact timeline)\n", tl_csv.c_str());
        }
        for (auto& e : tl)
            for (auto& ev : e.e) cudaEventDestroy(ev);
        cudaEventDestroy(t0);
        free_host(host);
        destroy_lanes(lanes);
    }

    std::printf("wrote %s\noutputs %s\n", csv.c_str(), ok ? "verified against CPU reference" : "MISMATCH");
    return ok ? 0 : 1;
}
