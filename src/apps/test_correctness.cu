// Correctness test: every GPU kernel version vs the CPU reference, and (when
// built with OpenCV) vs OpenCV itself.
//
//   ./test_correctness [--sizes 1920x1080,1013x701] [--image photo.png]
//
// Pass criteria
//   float outputs : PSNR > 60 dB (peak 255) and small max |diff|
//   binary outputs: < 0.01% of pixels differ (threshold ties from FMA rounding)
//   histogram/LUT : exact
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "app_util.hpp"
#include "cpu_reference.hpp"
#include "cuda_utils.cuh"
#include "image.hpp"
#include "kernels.cuh"
#ifdef HAVE_OPENCV
#include "opencv_ref.hpp"
#endif
#ifdef HAVE_NPP
#include "npp_baseline.hpp"
#endif

namespace {

int g_failures = 0, g_checks = 0;

void report(const std::string& name, bool pass, const std::string& detail) {
    ++g_checks;
    if (!pass) ++g_failures;
    std::printf("  [%s] %-44s %s\n", pass ? "PASS" : "FAIL", name.c_str(), detail.c_str());
}

void info(const std::string& name, const std::string& detail) {
    std::printf("  [INFO] %-44s %s\n", name.c_str(), detail.c_str());
}

std::string fmt_float(const DiffStats& d) {
    char b[128];
    std::snprintf(b, sizeof b, "PSNR %7.2f dB  max|d| %.3g", d.psnr_db, d.max_abs);
    return b;
}
std::string fmt_binary(const DiffStats& d, size_t n) {
    char b[128];
    std::snprintf(b, sizeof b, "%zu / %zu px differ (%.4f%%)", d.mismatches, n, 100.0 * d.mismatches / n);
    return b;
}

void check_float(const std::string& name, const ImageF32& got, const ImageF32& ref, double min_psnr = 60.0,
                 double max_abs = 0.5) {
    const DiffStats d = compare(got, ref);
    report(name, d.psnr_db > min_psnr && d.max_abs <= max_abs, fmt_float(d));
}
void check_binary(const std::string& name, const ImageU8& got, const ImageU8& ref, double max_frac = 1e-4) {
    const DiffStats d = compare(got, ref, 0.0);
    report(name, double(d.mismatches) <= max_frac * got.pixels(), fmt_binary(d, got.pixels()));
}
void check_exact(const std::string& name, const ImageU8& got, const ImageU8& ref) {
    const DiffStats d = compare(got, ref, 0.0);
    report(name, d.mismatches == 0, fmt_binary(d, got.pixels()));
}
template <size_t N, typename A, typename B>
void check_array(const std::string& name, const std::array<A, N>& got, const std::array<B, N>& ref) {
    size_t bad = 0;
    for (size_t i = 0; i < N; ++i) bad += uint64_t(got[i]) != uint64_t(ref[i]);
    report(name, bad == 0, std::to_string(bad) + " / " + std::to_string(N) + " entries differ");
}

ImageF32 download_f32(const DeviceBuffer<float>& d, int w, int h) {
    ImageF32 out(w, h);
    d.download(out.data.data(), out.pixels());
    return out;
}
ImageU8 download_u8(const DeviceBuffer<uint8_t>& d, int w, int h) {
    ImageU8 out(w, h);
    d.download(out.data.data(), out.pixels());
    return out;
}

void run_size(const ImageRGBA8& rgba, const Params& p) {
    const int w = rgba.width, h = rgba.height;
    const size_t n = rgba.pixels();
    std::printf("\n=== %d x %d ===\n", w, h);

    // ---------------- inputs + CPU reference ----------------
    const ImageF32 gray = rgba_to_gray_f32(rgba);
    const ImageU8 gray8 = f32_to_u8(gray);
    ImageF32 ref_blur, ref_blur2d, ref_mag, ref_bil;
    ImageU8 ref_edges, ref_eq, ref_pipe, ref_canny;
    cpu::gaussian_blur_separable(gray, ref_blur, p.blur_radius, p.blur_sigma);
    cpu::gaussian_blur_2d(gray, ref_blur2d, p.blur_radius, p.blur_sigma);
    cpu::sobel(ref_blur, &ref_mag, &ref_edges, p.sobel_threshold);
    cpu::bilateral(gray, ref_bil, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space);
    const auto ref_hist = cpu::histogram(gray8);
    const auto ref_lut = cpu::equalize_lut(ref_hist, n);
    cpu::equalize(gray8, ref_eq);
    cpu::edge_pipeline(rgba, ref_pipe, p.blur_radius, p.blur_sigma, p.sobel_threshold);
    cpu::canny(gray, ref_canny, p.blur_radius, p.blur_sigma, p.canny_low, p.canny_high);

    std::printf("CPU reference self-consistency\n");
    check_float("cpu 2D blur == cpu separable blur", ref_blur2d, ref_blur, 80.0, 1e-3);

    // ---------------- device buffers ----------------
    DeviceBuffer<float> d_gray(n), d_tmp(n), d_out(n), d_mag(n);
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

    // ---------------- Gaussian ----------------
    std::printf("Gaussian blur (%dx%d, sigma %.2f) vs CPU\n", nk, nk, p.blur_sigma);
    gaussian_naive_2d(d_gray.get(), d_out.get(), w, h, d_k2.get(), p.blur_radius);
    check_float("v1 naive 2D", download_f32(d_out, w, h), ref_blur);
    gaussian_separable(d_gray.get(), d_tmp.get(), d_out.get(), w, h, d_k1.get(), p.blur_radius);
    check_float("v2 separable", download_f32(d_out, w, h), ref_blur);
    gaussian_tiled(d_gray.get(), d_tmp.get(), d_out.get(), w, h, d_k1.get(), p.blur_radius);
    check_float("v3 separable + shared tiles", download_f32(d_out, w, h), ref_blur);
    gaussian_constant(d_gray.get(), d_tmp.get(), d_out.get(), w, h, p.blur_radius);
    const ImageF32 gpu_blur = download_f32(d_out, w, h);
    check_float("v4 tiled + constant weights", gpu_blur, ref_blur);
    {
        float *pin, *ptmp, *pout;
        size_t pitch_in, pitch_tmp, pitch_out;
        CUDA_CHECK(cudaMallocPitch(&pin, &pitch_in, w * sizeof(float), h));
        CUDA_CHECK(cudaMallocPitch(&ptmp, &pitch_tmp, w * sizeof(float), h));
        CUDA_CHECK(cudaMallocPitch(&pout, &pitch_out, w * sizeof(float), h));
        CUDA_CHECK(cudaMemcpy2D(pin, pitch_in, gray.data.data(), w * sizeof(float), w * sizeof(float), h,
                                cudaMemcpyHostToDevice));
        gaussian_texture(pin, pitch_in, ptmp, pitch_tmp, pout, pitch_out, w, h, p.blur_radius);
        ImageF32 t(w, h);
        CUDA_CHECK(cudaMemcpy2D(t.data.data(), w * sizeof(float), pout, pitch_out, w * sizeof(float), h,
                                cudaMemcpyDeviceToHost));
        check_float("v5 texture path", t, ref_blur);
        cudaFree(pin);
        cudaFree(ptmp);
        cudaFree(pout);
    }

    // ---------------- Sobel ----------------
    std::printf("Sobel + magnitude + threshold (%.0f) vs CPU\n", p.sobel_threshold);
    DeviceBuffer<float> d_ref_blur(n);
    d_ref_blur.upload(ref_blur.data.data(), n);  // isolate Sobel from blur rounding
    sobel_naive(d_ref_blur.get(), d_mag.get(), d_edges.get(), w, h, p.sobel_threshold);
    check_float("sobel naive: magnitude", download_f32(d_mag, w, h), ref_mag);
    check_binary("sobel naive: edges", download_u8(d_edges, w, h), ref_edges);
    sobel_tiled(d_ref_blur.get(), d_mag.get(), d_edges.get(), w, h, p.sobel_threshold);
    check_float("sobel tiled: magnitude", download_f32(d_mag, w, h), ref_mag);
    check_binary("sobel tiled: edges", download_u8(d_edges, w, h), ref_edges);

    // ---------------- Bilateral ----------------
    std::printf("Bilateral (r=%d, sc=%.0f, ss=%.1f) vs CPU\n", p.bil_radius, p.bil_sigma_color,
                p.bil_sigma_space);
    bilateral_naive(d_gray.get(), d_out.get(), w, h, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space);
    check_float("bilateral naive (expf)", download_f32(d_out, w, h), ref_bil);
    bilateral_tiled(d_gray.get(), d_out.get(), w, h, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space);
    const ImageF32 gpu_bil = download_f32(d_out, w, h);
    check_float("bilateral tiled (__expf)", gpu_bil, ref_bil);

    // ---------------- Histogram / equalization ----------------
    std::printf("Histogram + equalization vs CPU\n");
    auto get_hist = [&]() {
        std::array<uint32_t, 256> hh{};
        d_hist.download(hh.data(), 256);
        return hh;
    };
    histogram_global_atomics(d_gray8.get(), n, d_hist.get());
    check_array("histogram: global atomics", get_hist(), ref_hist);
    histogram_shared(d_gray8.get(), n, d_hist.get());
    check_array("histogram: shared privatized", get_hist(), ref_hist);
    histogram_warp_private(d_gray8.get(), n, d_hist.get());
    check_array("histogram: per-warp privatized", get_hist(), ref_hist);
    equalize_build_lut(d_hist.get(), n, d_lut.get());
    std::array<uint8_t, 256> lut{};
    d_lut.download(lut.data(), 256);
    check_array("equalize: scan -> LUT", lut, ref_lut);
    equalize(d_gray8.get(), d_u8.get(), n, d_hist.get(), d_lut.get());
    const ImageU8 gpu_eq = download_u8(d_u8, w, h);
    check_exact("equalize: full", gpu_eq, ref_eq);
    {
        // Edge cases: constant image and odd length tail.
        ImageU8 flat(37, 3);
        std::fill(flat.data.begin(), flat.data.end(), uint8_t(77));
        DeviceBuffer<uint8_t> df(flat.pixels()), dfo(flat.pixels());
        df.upload(flat.data.data(), flat.pixels());
        equalize(df.get(), dfo.get(), flat.pixels(), d_hist.get(), d_lut.get());
        ImageU8 got(37, 3), ref;
        dfo.download(got.data.data(), got.pixels());
        cpu::equalize(flat, ref);
        check_exact("equalize: constant 37x3 image", got, ref);
    }

    // ---------------- Pipeline ----------------
    std::printf("Edge pipeline (gray -> blur -> sobel -> threshold) vs CPU\n");
    pipeline_separate(d_rgba.get(), d_gray.get(), d_tmp.get(), d_out.get(), d_edges.get(), w, h,
                      p.blur_radius, p.sobel_threshold);
    const ImageU8 gpu_pipe_sep = download_u8(d_edges, w, h);
    check_binary("pipeline: separate kernels", gpu_pipe_sep, ref_pipe);
    pipeline_fused(d_rgba.get(), d_edges.get(), w, h, p.blur_radius, p.sobel_threshold);
    const ImageU8 gpu_pipe_fused = download_u8(d_edges, w, h);
    check_binary("pipeline: fused kernel", gpu_pipe_fused, ref_pipe);
    check_binary("pipeline: fused vs separate", gpu_pipe_fused, gpu_pipe_sep);

    // ---------------- Canny ----------------
    std::printf("Canny (low %.0f, high %.0f) vs CPU\n", p.canny_low, p.canny_high);
    {
        CannyBuffers cb{d_out.get(), d_tmp.get(), d_mag.get(), d_dir.get(), d_state.get(), d_flag.get()};
        d_gray.upload(gray.data.data(), n);
        const int iters = canny(d_gray.get(), d_edges.get(), w, h, p.blur_radius, p.canny_low,
                                p.canny_high, cb);
        check_binary("canny (" + std::to_string(iters) + " hysteresis launches)",
                     download_u8(d_edges, w, h), ref_canny, 1e-3);
    }

#ifdef HAVE_OPENCV
    std::printf("vs OpenCV %s\n", CV_VERSION);
    check_float("gray: cpu vs cv::cvtColor", gray, cvref::gray(rgba), 60.0, 1e-3);
    const ImageF32 cv_blur = cvref::gaussian(gray, p.blur_radius, p.blur_sigma);
    check_float("blur: gpu v4 vs cv::GaussianBlur", gpu_blur, cv_blur);
    check_float("sobel mag: gpu vs cv::Sobel+magnitude", ref_mag, cvref::sobel_mag(ref_blur));
    check_float("bilateral: gpu tiled vs cv::bilateralFilter", gpu_bil,
                cvref::bilateral(gray, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space));
    check_array("histogram: gpu vs cv::calcHist", ref_hist, cvref::histogram(gray8));
    check_exact("equalize: gpu vs cv::equalizeHist", gpu_eq, cvref::equalize(gray8));
    check_binary("pipeline: gpu fused vs OpenCV chain", gpu_pipe_fused,
                 cvref::edge_pipeline(rgba, p.blur_radius, p.blur_sigma, p.sobel_threshold), 1e-3);
#endif

#ifdef HAVE_NPP
    std::printf("vs NPP\n");
    nppb::gauss(d_gray.get(), d_out.get(), w, h, d_k1.get(), nk);
    check_float("blur: NPP GaussAdvanced vs CPU", download_f32(d_out, w, h), ref_blur);
    {
        DeviceBuffer<float> d_gy(n);
        nppb::sobel_xy(d_ref_blur.get(), d_mag.get(), d_gy.get(), w, h);
        ImageF32 gx = download_f32(d_mag, w, h), gy = download_f32(d_gy, w, h), m(w, h);
        for (size_t i = 0; i < n; ++i) m.data[i] = std::sqrt(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
        check_float("sobel mag: NPP vs CPU", m, ref_mag);
    }
    {
        DeviceBuffer<int32_t> d_h32(256);
        nppb::histogram(d_gray8.get(), w, h, d_h32.get());
        std::array<int32_t, 256> hh{};
        d_h32.download(hh.data(), 256);
        check_array("histogram: NPP HistogramEven vs CPU", hh, ref_hist);
    }
    nppb::bilateral(d_gray.get(), d_out.get(), w, h, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space);
    info("bilateral: NPP (square window) vs CPU", fmt_float(compare(download_f32(d_out, w, h), ref_bil)));
#endif
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv);
    const Params p = args.params();
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (args.has("image")) {
        ImageRGBA8 img;
        if (!load_rgba(args.str("image", ""), img)) {
            std::fprintf(stderr, "cannot load %s\n", args.str("image", "").c_str());
            return 2;
        }
        run_size(img, p);
    } else {
        // An odd size exercises partial tiles and the uchar4 tail paths.
        for (const auto& r : parse_sizes(args.str("sizes", "1920x1080,1013x701")))
            run_size(make_test_image(r.w, r.h, 3), p);
    }
    std::printf("\n%d / %d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures ? 1 : 0;
}
