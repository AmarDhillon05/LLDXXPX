// Video demo: decode a video with OpenCV, run the fused edge pipeline on the
// GPU, encode the edge video, and report FPS.
//
//   ./demo_video input.mp4 output.mp4 [--threshold 40] [--radius 7]
//
// Two pinned host slots + two streams: while the GPU processes frame i, the
// CPU decodes frame i+1 and encodes frame i-1.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>

#include "app_util.hpp"
#include "cpu_reference.hpp"
#include "cuda_utils.cuh"
#include "kernels.cuh"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s input_video output_video [--threshold T] [--radius R]\n", argv[0]);
        return 2;
    }
    Args args(argc, argv);
    const Params p = args.params();
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    const int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH)), h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps_in = cap.get(cv::CAP_PROP_FPS);
    if (fps_in <= 0) fps_in = 30;
    cv::VideoWriter writer(argv[2], cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_in, cv::Size(w, h), false);
    if (!writer.isOpened()) {
        std::fprintf(stderr, "cannot open writer for %s\n", argv[2]);
        return 1;
    }
    std::printf("%s: %dx%d @ %.1f fps\n", argv[1], w, h, fps_in);

    const auto k1 = cpu::gaussian_kernel_1d(p.blur_radius, p.blur_sigma);
    pipeline_fused_set_weights(k1.data(), p.blur_radius);

    const size_t px = size_t(w) * h;
    struct Slot {
        cudaStream_t s;
        uint8_t *h_in, *h_out;
        DeviceBuffer<uchar4> d_in;
        DeviceBuffer<uint8_t> d_out;
        bool busy = false;
    } slots[2];
    for (auto& s : slots) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.s, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMallocHost(&s.h_in, px * 4));
        CUDA_CHECK(cudaMallocHost(&s.h_out, px));
        s.d_in.allocate(px);
        s.d_out.allocate(px);
    }

    int n = 0;
    cv::Mat bgr;
    auto flush = [&](Slot& s) {
        CUDA_CHECK(cudaStreamSynchronize(s.s));
        writer.write(cv::Mat(h, w, CV_8U, s.h_out));
        s.busy = false;
    };
    const auto t0 = std::chrono::steady_clock::now();
    for (;; ++n) {
        Slot& s = slots[n & 1];
        if (s.busy) flush(s);  // frame n-2 used this slot
        if (!cap.read(bgr)) break;
        cv::cvtColor(bgr, cv::Mat(h, w, CV_8UC4, s.h_in), cv::COLOR_BGR2RGBA);
        CUDA_CHECK(cudaMemcpyAsync(s.d_in.get(), s.h_in, px * 4, cudaMemcpyHostToDevice, s.s));
        pipeline_fused(s.d_in.get(), s.d_out.get(), w, h, p.blur_radius, p.sobel_threshold, s.s);
        CUDA_CHECK(cudaMemcpyAsync(s.h_out, s.d_out.get(), px, cudaMemcpyDeviceToHost, s.s));
        s.busy = true;
    }
    for (auto& s : slots)
        if (s.busy) flush(s);
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d frames in %.1f ms: %.1f FPS end-to-end (decode + upload + GPU + download + encode)\n", n, wall,
                1000.0 * n / wall);
    for (auto& s : slots) {
        cudaFreeHost(s.h_in);
        cudaFreeHost(s.h_out);
        cudaStreamDestroy(s.s);
    }
    return 0;
}
