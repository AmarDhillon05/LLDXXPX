// CPU-only check: the single-threaded reference implementations (which the GPU
// tests use as ground truth) against OpenCV. Needs no GPU.
//
//   ./check_cpu_vs_opencv [--sizes 1920x1080,1013x701] [--image photo.png]
#include <cstdio>

#include "app_util.hpp"
#include "cpu_reference.hpp"
#include "image.hpp"
#include "opencv_ref.hpp"

static int g_fail = 0, g_n = 0;

static void report(const char* name, bool ok, const DiffStats& d, bool binary, size_t n) {
    ++g_n;
    g_fail += !ok;
    if (binary)
        std::printf("  [%s] %-40s %zu / %zu px differ\n", ok ? "PASS" : "FAIL", name, d.mismatches, n);
    else
        std::printf("  [%s] %-40s PSNR %7.2f dB  max|d| %.3g\n", ok ? "PASS" : "FAIL", name, d.psnr_db, d.max_abs);
}

static void run(const ImageRGBA8& rgba, const Params& p) {
    std::printf("\n=== %d x %d (OpenCV %s) ===\n", rgba.width, rgba.height, CV_VERSION);
    const size_t n = rgba.pixels();
    const ImageF32 gray = rgba_to_gray_f32(rgba);
    const ImageU8 gray8 = f32_to_u8(gray);

    DiffStats d = compare(gray, cvref::gray(rgba));
    report("grayscale", d.max_abs < 1e-3, d, false, n);

    ImageF32 blur;
    cpu::gaussian_blur_separable(gray, blur, p.blur_radius, p.blur_sigma);
    d = compare(blur, cvref::gaussian(gray, p.blur_radius, p.blur_sigma));
    report("gaussian (separable)", d.psnr_db > 60, d, false, n);

    ImageF32 mag;
    ImageU8 edges;
    cpu::sobel(blur, &mag, &edges, p.sobel_threshold);
    d = compare(mag, cvref::sobel_mag(blur));
    report("sobel magnitude", d.psnr_db > 60, d, false, n);
    d = compare(edges, cvref::threshold(mag, p.sobel_threshold));
    report("sobel threshold", d.mismatches == 0, d, true, n);

    ImageF32 bil;
    cpu::bilateral(gray, bil, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space);
    d = compare(bil, cvref::bilateral(gray, p.bil_radius, p.bil_sigma_color, p.bil_sigma_space));
    report("bilateral", d.psnr_db > 60, d, false, n);

    const auto h = cpu::histogram(gray8), hcv = cvref::histogram(gray8);
    size_t bad = 0;
    for (int i = 0; i < 256; ++i) bad += h[i] != hcv[i];
    d = DiffStats{};
    d.mismatches = bad;
    report("histogram", bad == 0, d, true, 256);

    ImageU8 eq;
    cpu::equalize(gray8, eq);
    d = compare(eq, cvref::equalize(gray8));
    report("equalizeHist", d.mismatches == 0, d, true, n);

    ImageU8 pipe;
    cpu::edge_pipeline(rgba, pipe, p.blur_radius, p.blur_sigma, p.sobel_threshold);
    d = compare(pipe, cvref::edge_pipeline(rgba, p.blur_radius, p.blur_sigma, p.sobel_threshold));
    report("edge pipeline", d.mismatches <= n / 10000, d, true, n);
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    const Params p = args.params();
    if (args.has("image")) {
        ImageRGBA8 img;
        if (!load_rgba(args.str("image", ""), img)) return 2;
        run(img, p);
    } else {
        for (const auto& r : parse_sizes(args.str("sizes", "1920x1080,1013x701")))
            run(make_test_image(r.w, r.h, 3), p);
    }
    std::printf("\n%d / %d checks passed\n", g_n - g_fail, g_n);
    return g_fail ? 1 : 0;
}
