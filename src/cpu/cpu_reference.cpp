#include "cpu_reference.hpp"

#include <algorithm>
#include <cmath>

namespace cpu {

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

std::vector<float> gaussian_kernel_1d(int radius, float sigma) {
    // Mirrors cv::getGaussianKernel for CV_32F: exp in double, sum in float.
    const int n = 2 * radius + 1;
    std::vector<float> k(n);
    const double scale2x = -0.5 / (double(sigma) * sigma);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        double x = i - radius;
        k[i] = float(std::exp(scale2x * x * x));
        sum += k[i];
    }
    const double inv = 1.0 / sum;
    for (float& v : k) v = float(v * inv);
    return k;
}

void gaussian_blur_2d(const ImageF32& in, ImageF32& out, int radius, float sigma) {
    const auto k = gaussian_kernel_1d(radius, sigma);
    const int n = 2 * radius + 1, w = in.width, h = in.height;
    std::vector<float> k2(n * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) k2[j * n + i] = k[j] * k[i];
    out = ImageF32(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int j = -radius; j <= radius; ++j) {
                const float* row = &in.data[size_t(clampi(y + j, 0, h - 1)) * w];
                const float* kr = &k2[(j + radius) * n + radius];
                for (int i = -radius; i <= radius; ++i) acc += kr[i] * row[clampi(x + i, 0, w - 1)];
            }
            out.at(x, y) = acc;
        }
    }
}

void gaussian_blur_separable(const ImageF32& in, ImageF32& out, int radius, float sigma) {
    const auto k = gaussian_kernel_1d(radius, sigma);
    const int w = in.width, h = in.height;
    ImageF32 tmp(w, h);
    for (int y = 0; y < h; ++y) {
        const float* row = &in.data[size_t(y) * w];
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int i = -radius; i <= radius; ++i) acc += k[i + radius] * row[clampi(x + i, 0, w - 1)];
            tmp.at(x, y) = acc;
        }
    }
    out = ImageF32(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int j = -radius; j <= radius; ++j) acc += k[j + radius] * tmp.at(x, clampi(y + j, 0, h - 1));
            out.at(x, y) = acc;
        }
    }
}

// gx/gy for a 3x3 Sobel at (x,y) with replicate borders.
static inline void sobel_at(const ImageF32& in, int x, int y, float& gx, float& gy) {
    const int w = in.width, h = in.height;
    const int xm = clampi(x - 1, 0, w - 1), xp = clampi(x + 1, 0, w - 1);
    const int ym = clampi(y - 1, 0, h - 1), yp = clampi(y + 1, 0, h - 1);
    const float a = in.at(xm, ym), b = in.at(x, ym), c = in.at(xp, ym);
    const float d = in.at(xm, y), f = in.at(xp, y);
    const float g = in.at(xm, yp), hh = in.at(x, yp), i = in.at(xp, yp);
    gx = (c - a) + 2.f * (f - d) + (i - g);
    gy = (g - a) + 2.f * (hh - b) + (i - c);
}

void sobel(const ImageF32& in, ImageF32* mag, ImageU8* edges, float threshold) {
    const int w = in.width, h = in.height;
    if (mag) *mag = ImageF32(w, h);
    if (edges) *edges = ImageU8(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float gx, gy;
            sobel_at(in, x, y, gx, gy);
            const float m = std::sqrt(gx * gx + gy * gy);
            if (mag) mag->at(x, y) = m;
            if (edges) edges->data[size_t(y) * w + x] = m > threshold ? 255 : 0;
        }
    }
}

void bilateral(const ImageF32& in, ImageF32& out, int radius, float sigma_color, float sigma_space) {
    const int w = in.width, h = in.height;
    const float color_coeff = -0.5f / (sigma_color * sigma_color);
    const float space_coeff = -0.5f / (sigma_space * sigma_space);
    // Precompute the circular window (offsets + spatial weights), like OpenCV.
    std::vector<int> dx, dy;
    std::vector<float> ws;
    for (int j = -radius; j <= radius; ++j)
        for (int i = -radius; i <= radius; ++i) {
            const float r = std::sqrt(float(i * i + j * j));
            if (r > radius) continue;
            dx.push_back(i);
            dy.push_back(j);
            ws.push_back(std::exp(r * r * space_coeff));
        }
    out = ImageF32(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float c = in.at(x, y);
            float sum = 0.f, wsum = 0.f;
            for (size_t k = 0; k < ws.size(); ++k) {
                const float v = in.at(clampi(x + dx[k], 0, w - 1), clampi(y + dy[k], 0, h - 1));
                const float d = v - c;
                const float wgt = ws[k] * std::exp(d * d * color_coeff);
                sum += v * wgt;
                wsum += wgt;
            }
            out.at(x, y) = sum / wsum;
        }
    }
}

std::array<uint32_t, 256> histogram(const ImageU8& in) {
    std::array<uint32_t, 256> hist{};
    for (uint8_t v : in.data) ++hist[v];
    return hist;
}

std::array<uint8_t, 256> equalize_lut(const std::array<uint32_t, 256>& hist, size_t total) {
    // Exactly cv::equalizeHist: skip leading empty bins, scale the CDF so the
    // first occupied bin maps to 0 and the last to 255, round-half-even.
    std::array<uint8_t, 256> lut{};
    int i = 0;
    while (i < 256 && hist[i] == 0) ++i;
    if (i == 256) return lut;
    if (hist[i] == total) {
        lut.fill(uint8_t(i));
        return lut;
    }
    const float scale = 255.f / float(total - hist[i]);
    int sum = 0;
    for (lut[i] = 0; ++i < 256;) {
        sum += int(hist[i]);
        const float v = std::nearbyint(float(sum) * scale);
        lut[i] = uint8_t(std::min(255.f, std::max(0.f, v)));
    }
    return lut;
}

void equalize(const ImageU8& in, ImageU8& out) {
    const auto lut = equalize_lut(histogram(in), in.pixels());
    out = ImageU8(in.width, in.height);
    for (size_t i = 0; i < in.pixels(); ++i) out.data[i] = lut[in.data[i]];
}

void edge_pipeline(const ImageRGBA8& in, ImageU8& edges, int radius, float sigma, float threshold) {
    ImageF32 gray = rgba_to_gray_f32(in), blurred;
    gaussian_blur_separable(gray, blurred, radius, sigma);
    sobel(blurred, nullptr, &edges, threshold);
}

// Direction codes shared with the GPU Canny kernels:
// 0 = horizontal gradient (compare W/E), 1 = 45deg (NW/SE),
// 2 = vertical (N/S),                    3 = 135deg (NE/SW).
static inline int quantize_dir(float gx, float gy) {
    const float ax = std::fabs(gx), ay = std::fabs(gy);
    if (ay <= 0.41421356f * ax) return 0;
    if (ay >= 2.41421356f * ax) return 2;
    return ((gx > 0) == (gy > 0)) ? 1 : 3;
}

void canny(const ImageF32& gray, ImageU8& edges, int radius, float sigma, float low, float high) {
    const int w = gray.width, h = gray.height;
    ImageF32 blurred;
    gaussian_blur_separable(gray, blurred, radius, sigma);
    ImageF32 mag(w, h);
    std::vector<uint8_t> dir(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float gx, gy;
            sobel_at(blurred, x, y, gx, gy);
            mag.at(x, y) = std::sqrt(gx * gx + gy * gy);
            dir[size_t(y) * w + x] = uint8_t(quantize_dir(gx, gy));
        }
    // NMS + double threshold. state: 0 none, 1 weak, 2 strong.
    static const int kOff[4][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}};
    std::vector<uint8_t> state(size_t(w) * h, 0);
    std::vector<int> stack;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float m = mag.at(x, y);
            if (m <= low) continue;
            const int d = dir[size_t(y) * w + x];
            const float ma = mag.at(clampi(x + kOff[d][0], 0, w - 1), clampi(y + kOff[d][1], 0, h - 1));
            const float mb = mag.at(clampi(x - kOff[d][0], 0, w - 1), clampi(y - kOff[d][1], 0, h - 1));
            if (!(m > ma && m >= mb)) continue;
            const size_t idx = size_t(y) * w + x;
            if (m > high) {
                state[idx] = 2;
                stack.push_back(int(idx));
            } else {
                state[idx] = 1;
            }
        }
    // Hysteresis: flood fill from strong pixels through weak ones.
    while (!stack.empty()) {
        const int idx = stack.back();
        stack.pop_back();
        const int x = idx % w, y = idx / w;
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i) {
                const int nx = x + i, ny = y + j;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                const size_t n = size_t(ny) * w + nx;
                if (state[n] == 1) {
                    state[n] = 2;
                    stack.push_back(int(n));
                }
            }
    }
    edges = ImageU8(w, h);
    for (size_t i = 0; i < state.size(); ++i) edges.data[i] = state[i] == 2 ? 255 : 0;
}

}  // namespace cpu
