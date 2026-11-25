// Host-side image containers, I/O (stb_image) and comparison metrics.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Interleaved RGBA8 image.
struct ImageRGBA8 {
    int width = 0, height = 0;
    std::vector<uint8_t> data;  // width*height*4
    size_t pixels() const { return size_t(width) * height; }
};

// Single-channel float image (values in [0,255] by convention).
struct ImageF32 {
    int width = 0, height = 0;
    std::vector<float> data;
    ImageF32() = default;
    ImageF32(int w, int h) : width(w), height(h), data(size_t(w) * h, 0.f) {}
    size_t pixels() const { return size_t(width) * height; }
    float& at(int x, int y) { return data[size_t(y) * width + x]; }
    float at(int x, int y) const { return data[size_t(y) * width + x]; }
};

// Single-channel 8-bit image.
struct ImageU8 {
    int width = 0, height = 0;
    std::vector<uint8_t> data;
    ImageU8() = default;
    ImageU8(int w, int h) : width(w), height(h), data(size_t(w) * h, 0) {}
    size_t pixels() const { return size_t(width) * height; }
};

// ---- I/O ------------------------------------------------------------------
bool load_rgba(const std::string& path, ImageRGBA8& out);
bool save_png(const std::string& path, const ImageU8& img);
bool save_png(const std::string& path, const ImageF32& img);  // clamps to [0,255]
bool save_png(const std::string& path, const ImageRGBA8& img);

// Raw float dump (header: int32 w, int32 h, then w*h float32). Used to hand
// outputs to scripts/opencv_check.py without any precision loss.
bool save_raw(const std::string& path, const ImageF32& img);
bool save_raw(const std::string& path, const ImageU8& img);

// ---- Synthetic inputs -------------------------------------------------------
// Deterministic test card: gradients, rings, rectangles, text-like stripes and
// noise. Has edges at every orientation and a non-trivial histogram, so it
// exercises blur, Sobel, bilateral and equalization. `seed` varies the frame
// (used for the multi-frame streaming benchmark).
ImageRGBA8 make_test_image(int width, int height, uint32_t seed = 1);

ImageF32 rgba_to_gray_f32(const ImageRGBA8& img);  // 0.299R + 0.587G + 0.114B
ImageU8 f32_to_u8(const ImageF32& img);            // round + saturate

// ---- Metrics ----------------------------------------------------------------
struct DiffStats {
    double max_abs = 0.0;
    double mse = 0.0;
    double psnr_db = 0.0;     // vs peak 255; +inf reported as 999
    size_t mismatches = 0;    // count of |a-b| > tolerance
};
DiffStats compare(const ImageF32& a, const ImageF32& b, double tol = 0.0);
DiffStats compare(const ImageU8& a, const ImageU8& b, double tol = 0.0);
