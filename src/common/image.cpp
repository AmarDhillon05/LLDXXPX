#include "image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool load_rgba(const std::string& path, ImageRGBA8& out) {
    int w, h, c;
    unsigned char* p = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!p) return false;
    out.width = w;
    out.height = h;
    out.data.assign(p, p + size_t(w) * h * 4);
    stbi_image_free(p);
    return true;
}

bool save_png(const std::string& path, const ImageU8& img) {
    return stbi_write_png(path.c_str(), img.width, img.height, 1, img.data.data(), img.width) != 0;
}

bool save_png(const std::string& path, const ImageF32& img) { return save_png(path, f32_to_u8(img)); }

bool save_png(const std::string& path, const ImageRGBA8& img) {
    return stbi_write_png(path.c_str(), img.width, img.height, 4, img.data.data(), img.width * 4) != 0;
}

template <typename T>
static bool save_raw_impl(const std::string& path, int w, int h, const T* data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    int32_t hdr[2] = {w, h};
    bool ok = std::fwrite(hdr, sizeof(hdr), 1, f) == 1 &&
              std::fwrite(data, sizeof(T), size_t(w) * h, f) == size_t(w) * h;
    std::fclose(f);
    return ok;
}
bool save_raw(const std::string& path, const ImageF32& img) {
    return save_raw_impl(path, img.width, img.height, img.data.data());
}
bool save_raw(const std::string& path, const ImageU8& img) {
    return save_raw_impl(path, img.width, img.height, img.data.data());
}

// Small, fast, deterministic PRNG (xorshift32).
static inline uint32_t xorshift(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

ImageRGBA8 make_test_image(int width, int height, uint32_t seed) {
    ImageRGBA8 img;
    img.width = width;
    img.height = height;
    img.data.resize(size_t(width) * height * 4);
    const float cx = width * (0.45f + 0.02f * (seed % 7));
    const float cy = height * 0.5f;
    const float scale = 1080.f / height;  // keep features similar across resolutions
    uint32_t rng = 0x9E3779B9u ^ (seed * 2654435761u);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float fx = x * scale, fy = y * scale;
            // Background: diagonal gradient (exercises equalization).
            float r = 40.f + 120.f * x / width;
            float g = 30.f + 100.f * y / height;
            float b = 60.f + 60.f * (x + y) / (width + height);
            // Concentric rings (edges at every orientation).
            float dx = (x - cx) * scale, dy = (y - cy) * scale;
            float d = std::sqrt(dx * dx + dy * dy);
            if (d < 400.f && (int(d / 24.f) % 2 == 0)) { r += 70.f; g += 50.f; b += 20.f; }
            // Rectangles.
            int bx = int(fx) / 160, by = int(fy) / 120;
            if (((bx * 7 + by * 3 + int(seed)) % 11) == 0) { r = 230.f; g = 220.f; b = 200.f; }
            // Thin vertical stripes ("text").
            if (fy > 900.f && fy < 1000.f && (int(fx) % 9) < 2) { r *= 0.3f; g *= 0.3f; b *= 0.3f; }
            // Noise (gives the bilateral filter something to do).
            float n = float(int(xorshift(rng) & 31u) - 16);
            auto sat = [](float v) { return (uint8_t)std::min(255.f, std::max(0.f, v)); };
            uint8_t* p = &img.data[(size_t(y) * width + x) * 4];
            p[0] = sat(r + n);
            p[1] = sat(g + n);
            p[2] = sat(b + n);
            p[3] = 255;
        }
    }
    return img;
}

ImageF32 rgba_to_gray_f32(const ImageRGBA8& img) {
    ImageF32 out(img.width, img.height);
    for (size_t i = 0; i < img.pixels(); ++i) {
        const uint8_t* p = &img.data[i * 4];
        out.data[i] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
    return out;
}

ImageU8 f32_to_u8(const ImageF32& img) {
    ImageU8 out(img.width, img.height);
    for (size_t i = 0; i < img.pixels(); ++i) {
        float v = std::nearbyint(img.data[i]);
        out.data[i] = (uint8_t)std::min(255.f, std::max(0.f, v));
    }
    return out;
}

template <typename A>
static DiffStats compare_impl(const A& a, const A& b, double tol) {
    DiffStats s;
    if (a.width != b.width || a.height != b.height) {
        s.max_abs = 1e30;
        s.psnr_db = 0;
        s.mismatches = std::max(a.pixels(), b.pixels());
        return s;
    }
    double sum = 0;
    for (size_t i = 0; i < a.pixels(); ++i) {
        double d = std::fabs(double(a.data[i]) - double(b.data[i]));
        s.max_abs = std::max(s.max_abs, d);
        sum += d * d;
        if (d > tol) ++s.mismatches;
    }
    s.mse = sum / double(a.pixels());
    s.psnr_db = s.mse == 0 ? 999.0 : 10.0 * std::log10(255.0 * 255.0 / s.mse);
    return s;
}
DiffStats compare(const ImageF32& a, const ImageF32& b, double tol) { return compare_impl(a, b, tol); }
DiffStats compare(const ImageU8& a, const ImageU8& b, double tol) { return compare_impl(a, b, tol); }
