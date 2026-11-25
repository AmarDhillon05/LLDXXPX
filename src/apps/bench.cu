// Kernel benchmark: every version of every operation at several resolutions.
//
//   ./bench [--sizes 1080p,4k,8k] [--reps 30] [--cpu-reps 3] [--skip-cpu]
//           [--csv results/bench.csv] [--device-json results/device.json]
//           [--cpu-only]          (no CUDA calls at all; CPU baselines only)
//
// GPU times are cudaEvent-timed, kernel-only (data already resident), median
// of --reps runs after 3 warm-ups. CPU times are wall-clock medians.
// Bandwidth columns use each implementation's *own* minimal DRAM traffic
// (e.g. a two-pass separable blur moves 16 B/px, a fused one 8 B/px), so
// "% of peak" says how close that design is to its own roofline.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "app_util.hpp"
#include "cpu_reference.hpp"
#include "cuda_utils.cuh"
#include "image.hpp"
#include "kernels.cuh"
#ifdef HAVE_OPENCV
#include <opencv2/core/utility.hpp>

#include "opencv_ref.hpp"
#endif
#ifdef HAVE_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif
#ifdef HAVE_NPP
#include "npp_baseline.hpp"
#endif

namespace {

struct Row {
    std::string size;
    int w, h;
    std::string op, impl, device;
    double ms;
    double bytes_per_px;  // 0 = not meaningful (compute bound / irregular)
};

std::vector<Row> g_rows;
double g_peak_gbps = 0;  // measured device-to-device copy bandwidth

void add(const Resolution& r, const std::string& op, const std::string& impl, const std::string& dev,
         double ms, double bpp) {
    g_rows.push_back({r.name, r.w, r.h, op, impl, dev, ms, bpp});
    const double px = double(r.w) * r.h;
    std::printf("  %-10s %-22s %-4s %10.3f ms %9.1f Mpx/s", op.c_str(), impl.c_str(), dev.c_str(), ms,
                px / (ms * 1e3));
    if (bpp > 0 && dev == "gpu") {
        const double gbps = px * bpp / (ms * 1e6);
        std::printf("  %7.1f GB/s", gbps);
        if (g_peak_gbps > 0) std::printf(" (%4.1f%% of copy peak)", 100.0 * gbps / g_peak_gbps);
    }
    std::printf("\n");
    std::fflush(stdout);
}

template <typename F>
double time_gpu(F&& fn, int reps) {
    GpuTimer t;
    for (int i = 0; i < 3; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) {
        t.start();
        fn();
        t.stop();
        v.push_back(t.elapsed_ms());
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Device-to-device copy bandwidth (read + write), the practical roofline.
double measure_copy_bandwidth() {
    const size_t bytes = size_t(512) << 20;
    DeviceBuffer<char> a(bytes), b(bytes);
    CUDA_CHECK(cudaMemset(a.get(), 1, bytes));
    const double ms = time_gpu([&] { CUDA_CHECK(cudaMemcpy(b.get(), a.get(), bytes, cudaMemcpyDeviceToDevice)); }, 10);
    return 2.0 * bytes / (ms * 1e6);
}

double measure_pcie(bool pinned, cudaMemcpyKind kind) {
    const size_t bytes = size_t(256) << 20;
    DeviceBuffer<char> d(bytes);
    char* h = nullptr;
    std::vector<char> pageable;
    if (pinned) CUDA_CHECK(cudaMallocHost(&h, bytes));
    else {
        pageable.resize(bytes, 1);
        h = pageable.data();
    }
    std::fill(h, h + bytes, 1);
    const double ms = time_gpu([&] {
        if (kind == cudaMemcpyHostToDevice) CUDA_CHECK(cudaMemcpy(d.get(), h, bytes, kind));
        else CUDA_CHECK(cudaMemcpy(h, d.get(), bytes, kind));
    }, 5);
    if (pinned) CUDA_CHECK(cudaFreeHost(h));
    return bytes / (ms * 1e6);
}

void write_device_json(const std::string& path, bool cpu_only) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"cpu_threads\": %u,\n", std::thread::hardware_concurrency());
#ifdef HAVE_OPENCV
    std::fprintf(f, "  \"opencv_version\": \"%s\",\n  \"opencv_threads\": %d,\n", CV_VERSION, cv::getNumThreads());
#endif
    if (!cpu_only) {
        int dev = 0, mem_clock_khz = 0, bus_bits = 0, rt = 0, drv = 0;
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
        cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
        cudaRuntimeGetVersion(&rt);
        cudaDriverGetVersion(&drv);
        const double theo = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
        std::fprintf(f, "  \"gpu\": \"%s\",\n  \"sm\": \"%d.%d\",\n  \"sms\": %d,\n", p.name, p.major, p.minor,
                     p.multiProcessorCount);
        std::fprintf(f, "  \"cuda_runtime\": %d,\n  \"cuda_driver\": %d,\n", rt, drv);
        std::fprintf(f, "  \"theoretical_bandwidth_gbps\": %.1f,\n", theo);
        std::fprintf(f, "  \"copy_bandwidth_gbps\": %.1f,\n", g_peak_gbps);
        std::fprintf(f, "  \"pcie_h2d_pinned_gbps\": %.2f,\n", measure_pcie(true, cudaMemcpyHostToDevice));
        std::fprintf(f, "  \"pcie_d2h_pinned_gbps\": %.2f,\n", measure_pcie(true, cudaMemcpyDeviceToHost));
        std::fprintf(f, "  \"pcie_h2d_pageable_gbps\": %.2f,\n", measure_pcie(false, cudaMemcpyHostToDevice));
        std::fprintf(f, "  \"pcie_d2h_pageable_gbps\": %.2f,\n", measure_pcie(false, cudaMemcpyDeviceToHost));
    }
    std::fprintf(f, "  \"note\": \"copy_bandwidth = device-to-device cudaMemcpy (read+write); used as 100%% in plots\"\n}\n");
    std::fclose(f);
}

void bench_cpu(const Resolution& r, const Params& p, int reps) {
    const ImageRGBA8 rgba = make_test_image(r.w, r.h, 3);
    const ImageF32 gray = rgba_to_gray_f32(rgba);
    const ImageU8 gray8 = f32_to_u8(gray);
    ImageF32 out, mag;
    ImageU8 out8;
    const int big = r.w * r.h > 4000 * 2500;  // 8K: one rep of the slow ones
    const int slow_reps = big ? 1 : reps;

    add(r, "gaussian", "cpu_2d_1t", "cpu",
        time_cpu_ms([&] { cpu::gaussian_blur_2d(gray, out, p.blur_radius, p.blur_sigma); }, slow_reps), 0);
    add(r, "gaussian", "cpu_separable_1t", "cpu",
        time_cpu_ms([&] { cpu::gaussian_blur_separable(gray, out, p.blur_radius, p.blur_sigma); }, reps), 0);
    cpu::gaussian_blur_separable(gray, out, p.blur_radius, p.blur_sigma);
    add(r, "sobel", "cpu_1t", "cpu",
        time_cpu_ms([&] { cpu::sobel(out, nullptr, &out8, p.sobel_threshold); }, reps), 0);
    add(r, "bilateral", "cpu_1t", "cpu",
        time_cpu_ms([&] { cpu::bilateral(gray, mag, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space); },
                    slow_reps), 0);
    add(r, "histogram", "cpu_1t", "cpu", time_cpu_ms([&] { volatile auto h = cpu::histogram(gray8)[0]; (void)h; }, reps), 0);
    add(r, "equalize", "cpu_1t", "cpu", time_cpu_ms([&] { cpu::equalize(gray8, out8); }, reps), 0);
    add(r, "pipeline", "cpu_1t", "cpu",
        time_cpu_ms([&] { cpu::edge_pipeline(rgba, out8, p.blur_radius, p.blur_sigma, p.sobel_threshold); }, reps), 0);
    add(r, "canny", "cpu_1t", "cpu",
        time_cpu_ms([&] { cpu::canny(gray, out8, p.blur_radius, p.blur_sigma, p.canny_low, p.canny_high); }, reps), 0);

#ifdef HAVE_OPENCV
    // OpenCV: SIMD + multithreaded, i.e. a strong CPU baseline.
    const std::string cvname = "opencv_" + std::to_string(cv::getNumThreads()) + "t";
    cv::Mat g = cvref::wrap(gray), g8 = cvref::wrap(gray8), dst, dx, dy, m;
    cv::Mat rgbaf;
    const int k = 2 * p.blur_radius + 1;
    add(r, "gaussian", cvname, "cpu", time_cpu_ms([&] {
            cv::GaussianBlur(g, dst, cv::Size(k, k), p.blur_sigma, p.blur_sigma, cv::BORDER_REPLICATE);
        }, reps), 0);
    cv::Mat blurred = dst.clone();
    add(r, "sobel", cvname, "cpu", time_cpu_ms([&] {
            cv::Sobel(blurred, dx, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::Sobel(blurred, dy, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::magnitude(dx, dy, m);
            cv::threshold(m, dst, p.sobel_threshold, 255, cv::THRESH_BINARY);
        }, reps), 0);
    add(r, "bilateral", cvname, "cpu", time_cpu_ms([&] {
            cv::bilateralFilter(g, dst, 2 * p.bil_radius + 1, p.bil_sigma_color, p.bil_sigma_space,
                                cv::BORDER_REPLICATE);
        }, reps), 0);
    add(r, "histogram", cvname, "cpu", time_cpu_ms([&] { volatile auto h = cvref::histogram(gray8)[0]; (void)h; }, reps), 0);
    add(r, "equalize", cvname, "cpu", time_cpu_ms([&] { cv::equalizeHist(g8, dst); }, reps), 0);
    add(r, "pipeline", cvname, "cpu", time_cpu_ms([&] {
            cv::Mat f, gg, b;
            cvref::wrap(rgba).convertTo(f, CV_32FC4);
            cv::cvtColor(f, gg, cv::COLOR_RGBA2GRAY);
            cv::GaussianBlur(gg, b, cv::Size(k, k), p.blur_sigma, p.blur_sigma, cv::BORDER_REPLICATE);
            cv::Sobel(b, dx, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::Sobel(b, dy, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
            cv::magnitude(dx, dy, m);
            cv::threshold(m, dst, p.sobel_threshold, 255, cv::THRESH_BINARY);
        }, reps), 0);
    // cv::Canny wants 8-bit input and does its own (unblurred) Sobel, so it is
    // a timing reference only, not an identical algorithm.
    add(r, "canny", cvname + "_8u", "cpu", time_cpu_ms([&] {
            cv::Mat b8;
            cv::GaussianBlur(g8, b8, cv::Size(k, k), p.blur_sigma, p.blur_sigma, cv::BORDER_REPLICATE);
            cv::Canny(b8, dst, p.canny_low, p.canny_high, 3, true);
        }, reps), 0);
#endif
}

void bench_gpu(const Resolution& r, const Params& p, int reps) {
    const int w = r.w, h = r.h;
    const size_t n = size_t(w) * h;
    const ImageRGBA8 rgba = make_test_image(w, h, 3);
    const ImageF32 gray = rgba_to_gray_f32(rgba);
    const ImageU8 gray8 = f32_to_u8(gray);

    DeviceBuffer<float> d_gray(n), d_tmp(n), d_out(n), d_mag(n), d_gy(n);
    DeviceBuffer<uint8_t> d_edges(n), d_gray8(n), d_u8(n), d_lut(256), d_dir(n), d_state(n);
    DeviceBuffer<uint32_t> d_hist(256);
    DeviceBuffer<uchar4> d_rgba(n);
    DeviceBuffer<int> d_flag(1);
    d_gray.upload(gray.data.data(), n);
    d_gray8.upload(gray8.data.data(), n);
    d_rgba.upload(reinterpret_cast<const uchar4*>(rgba.data.data()), n);

    const auto k1 = cpu::gaussian_kernel_1d(p.blur_radius, p.blur_sigma);
    const int nk = 2 * p.blur_radius + 1;
    std::vector<float> k2(nk * nk);
    for (int j = 0; j < nk; ++j)
        for (int i = 0; i < nk; ++i) k2[j * nk + i] = k1[j] * k1[i];
    DeviceBuffer<float> d_k1(nk), d_k2(nk * nk);
    d_k1.upload(k1.data(), nk);
    d_k2.upload(k2.data(), nk * nk);
    gaussian_constant_set_weights(k1.data(), p.blur_radius);
    gaussian_texture_set_weights(k1.data(), p.blur_radius);
    pipeline_fused_set_weights(k1.data(), p.blur_radius);

    const float* gin = d_gray.get();
    // ---- Gaussian ----
    add(r, "gaussian", "v1_naive_2d", "gpu",
        time_gpu([&] { gaussian_naive_2d(gin, d_out.get(), w, h, d_k2.get(), p.blur_radius); }, reps), 8);
    add(r, "gaussian", "v2_separable", "gpu",
        time_gpu([&] { gaussian_separable(gin, d_tmp.get(), d_out.get(), w, h, d_k1.get(), p.blur_radius); }, reps), 16);
    add(r, "gaussian", "v3_tiled_smem", "gpu",
        time_gpu([&] { gaussian_tiled(gin, d_tmp.get(), d_out.get(), w, h, d_k1.get(), p.blur_radius); }, reps), 16);
    add(r, "gaussian", "v4_tiled_constant", "gpu",
        time_gpu([&] { gaussian_constant(gin, d_tmp.get(), d_out.get(), w, h, p.blur_radius); }, reps), 16);
    {
        float *pin, *ptmp, *pout;
        size_t a, b, c;
        CUDA_CHECK(cudaMallocPitch(&pin, &a, w * sizeof(float), h));
        CUDA_CHECK(cudaMallocPitch(&ptmp, &b, w * sizeof(float), h));
        CUDA_CHECK(cudaMallocPitch(&pout, &c, w * sizeof(float), h));
        CUDA_CHECK(cudaMemcpy2D(pin, a, gray.data.data(), w * sizeof(float), w * sizeof(float), h,
                                cudaMemcpyHostToDevice));
        add(r, "gaussian", "v5_texture", "gpu",
            time_gpu([&] { gaussian_texture(pin, a, ptmp, b, pout, c, w, h, p.blur_radius); }, reps), 16);
        cudaFree(pin);
        cudaFree(ptmp);
        cudaFree(pout);
    }
#ifdef HAVE_NPP
    add(r, "gaussian", "npp_gauss_advanced", "gpu",
        time_gpu([&] { nppb::gauss(gin, d_out.get(), w, h, d_k1.get(), nk); }, reps), 8);
#endif

    // ---- Sobel (on the blurred image) ----
    gaussian_constant(gin, d_tmp.get(), d_out.get(), w, h, p.blur_radius);
    const float* blurred = d_out.get();
    add(r, "sobel", "naive_fused_mag_thr", "gpu",
        time_gpu([&] { sobel_naive(blurred, nullptr, d_edges.get(), w, h, p.sobel_threshold); }, reps), 5);
    add(r, "sobel", "tiled_fused_mag_thr", "gpu",
        time_gpu([&] { sobel_tiled(blurred, nullptr, d_edges.get(), w, h, p.sobel_threshold); }, reps), 5);
#ifdef HAVE_NPP
    // NPP: two passes producing gx and gy only (no magnitude/threshold).
    add(r, "sobel", "npp_gx_gy_only", "gpu",
        time_gpu([&] { nppb::sobel_xy(blurred, d_mag.get(), d_gy.get(), w, h); }, reps), 16);
#endif

    // ---- Bilateral ----
    add(r, "bilateral", "naive_expf", "gpu", time_gpu([&] {
            bilateral_naive(gin, d_tmp.get(), w, h, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space);
        }, reps), 8);
    add(r, "bilateral", "tiled_fast_expf", "gpu", time_gpu([&] {
            bilateral_tiled(gin, d_tmp.get(), w, h, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space);
        }, reps), 8);
#ifdef HAVE_NPP
    add(r, "bilateral", "npp_square_window", "gpu", time_gpu([&] {
            nppb::bilateral(gin, d_tmp.get(), w, h, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space);
        }, reps), 8);
#endif

    // ---- Histogram / equalize ----
    const uint8_t* g8 = d_gray8.get();
    add(r, "histogram", "global_atomics", "gpu", time_gpu([&] { histogram_global_atomics(g8, n, d_hist.get()); }, reps), 1);
    add(r, "histogram", "shared_privatized", "gpu", time_gpu([&] { histogram_shared(g8, n, d_hist.get()); }, reps), 1);
    add(r, "histogram", "warp_privatized", "gpu", time_gpu([&] { histogram_warp_private(g8, n, d_hist.get()); }, reps), 1);
#ifdef HAVE_NPP
    {
        DeviceBuffer<int32_t> d_h32(256);
        add(r, "histogram", "npp_histogram_even", "gpu",
            time_gpu([&] { nppb::histogram(g8, w, h, d_h32.get()); }, reps), 1);
    }
#endif
    add(r, "equalize", "hist_scan_lut", "gpu",
        time_gpu([&] { equalize(g8, d_u8.get(), n, d_hist.get(), d_lut.get()); }, reps), 3);
    add(r, "equalize_lut_only", "scan_256", "gpu",
        time_gpu([&] { equalize_build_lut(d_hist.get(), n, d_lut.get()); }, reps), 0);

    // ---- Pipeline ----
    add(r, "pipeline", "separate_kernels", "gpu", time_gpu([&] {
            pipeline_separate(d_rgba.get(), d_gray.get(), d_tmp.get(), d_mag.get(), d_edges.get(), w, h,
                              p.blur_radius, p.sobel_threshold);
        }, reps), 29);
    add(r, "pipeline", "fused_kernel", "gpu", time_gpu([&] {
            pipeline_fused(d_rgba.get(), d_edges.get(), w, h, p.blur_radius, p.sobel_threshold);
        }, reps), 5);

    // ---- Canny ----
    d_gray.upload(gray.data.data(), n);
    {
        CannyBuffers cb{d_out.get(), d_tmp.get(), d_mag.get(), d_dir.get(), d_state.get(), d_flag.get()};
        int iters = 0;
        const double ms = time_gpu([&] {
            iters = canny(d_gray.get(), d_edges.get(), w, h, p.blur_radius, p.canny_low, p.canny_high, cb);
        }, reps);
        add(r, "canny", "gpu", "gpu", ms, 0);
        std::printf("             (canny: %d hysteresis launches)\n", iters);
    }

#ifdef HAVE_OPENCV_CUDA
    {
        cv::cuda::GpuMat gg(h, w, CV_32F, d_gray.get()), go(h, w, CV_32F, d_out.get());
        cv::cuda::GpuMat g8m(h, w, CV_8U, d_gray8.get()), o8(h, w, CV_8U, d_u8.get());
        auto gf = cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(nk, nk), p.blur_sigma, p.blur_sigma,
                                                 cv::BORDER_REPLICATE, cv::BORDER_REPLICATE);
        add(r, "gaussian", "opencv_cuda", "gpu", time_gpu([&] { gf->apply(gg, go); }, reps), 8);
        add(r, "bilateral", "opencv_cuda", "gpu", time_gpu([&] {
                cv::cuda::bilateralFilter(gg, go, 2 * p.bil_radius + 1, p.bil_sigma_color, p.bil_sigma_space,
                                          cv::BORDER_REPLICATE);
            }, reps), 8);
        add(r, "equalize", "opencv_cuda", "gpu", time_gpu([&] { cv::cuda::equalizeHist(g8m, o8); }, reps), 3);
    }
#endif
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv);
    const Params p = args.params();
    const bool cpu_only = args.has("cpu-only");
    const bool skip_cpu = args.has("skip-cpu");
    const int reps = args.i("reps", 30);
    const int cpu_reps = args.i("cpu-reps", 3);
    const auto sizes = parse_sizes(args.str("sizes", "1080p,4k,8k"));
    const std::string csv = args.str("csv", "results/bench.csv");

    if (!cpu_only) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        g_peak_gbps = measure_copy_bandwidth();
        std::printf("Device: %s, D2D copy bandwidth %.1f GB/s\n", prop.name, g_peak_gbps);
    }
    write_device_json(args.str("device-json", "results/device.json"), cpu_only);

    for (const auto& r : sizes) {
        std::printf("\n=== %s (%d x %d) ===\n", r.name.c_str(), r.w, r.h);
        if (!cpu_only) bench_gpu(r, p, reps);
        if (!skip_cpu) bench_cpu(r, p, cpu_reps);
    }

    FILE* f = std::fopen(csv.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", csv.c_str());
        return 1;
    }
    std::fprintf(f, "size,width,height,op,impl,device,ms,mpix_per_s,bytes_per_px,gbps,pct_copy_peak\n");
    for (const auto& r : g_rows) {
        const double px = double(r.w) * r.h;
        const double gbps = r.bytes_per_px > 0 ? px * r.bytes_per_px / (r.ms * 1e6) : 0;
        std::fprintf(f, "%s,%d,%d,%s,%s,%s,%.4f,%.2f,%.0f,%.2f,%.2f\n", r.size.c_str(), r.w, r.h, r.op.c_str(),
                     r.impl.c_str(), r.device.c_str(), r.ms, px / (r.ms * 1e3), r.bytes_per_px, gbps,
                     g_peak_gbps > 0 && gbps > 0 ? 100 * gbps / g_peak_gbps : 0.0);
    }
    std::fclose(f);
    std::printf("\nwrote %s\n", csv.c_str());
    return 0;
}
