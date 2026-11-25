// Single-threaded CPU reference implementations.
//
// These serve two purposes:
//   1. Ground truth for the GPU kernels (tests/test_correctness).
//   2. The "single-thread CPU" baseline in the benchmarks.
//
// Every function uses replicate (clamp-to-edge) borders, which matches
// OpenCV's BORDER_REPLICATE. The test suite checks these against OpenCV.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "image.hpp"

namespace cpu {

// 1D Gaussian taps, length 2*radius+1, normalized to sum 1. Same formula as
// cv::getGaussianKernel(2r+1, sigma, CV_32F).
std::vector<float> gaussian_kernel_1d(int radius, float sigma);

// Naive 2D convolution with the outer-product kernel: (2r+1)^2 taps/pixel.
void gaussian_blur_2d(const ImageF32& in, ImageF32& out, int radius, float sigma);
// Two 1D passes (horizontal then vertical): 2*(2r+1) taps/pixel.
void gaussian_blur_separable(const ImageF32& in, ImageF32& out, int radius, float sigma);

// 3x3 Sobel. Writes gradient magnitude sqrt(gx^2+gy^2) (if mag != nullptr) and
// a binary edge map (255 where mag > threshold, else 0).
void sobel(const ImageF32& in, ImageF32* mag, ImageU8* edges, float threshold);

// Bilateral filter with a circular window of the given radius, matching the
// semantics of cv::bilateralFilter(src, dst, 2r+1, sigma_color, sigma_space).
void bilateral(const ImageF32& in, ImageF32& out, int radius, float sigma_color, float sigma_space);

// 256-bin histogram and histogram equalization (cv::equalizeHist semantics).
std::array<uint32_t, 256> histogram(const ImageU8& in);
// LUT used by equalization; exposed so the GPU LUT can be checked directly.
std::array<uint8_t, 256> equalize_lut(const std::array<uint32_t, 256>& hist, size_t total);
void equalize(const ImageU8& in, ImageU8& out);

// Full edge pipeline: RGBA -> gray -> Gaussian -> Sobel magnitude -> threshold.
void edge_pipeline(const ImageRGBA8& in, ImageU8& edges, int radius, float sigma, float threshold);

// Canny: Gaussian -> Sobel (L2 magnitude, 4-way quantized direction) ->
// non-max suppression -> double threshold -> hysteresis (8-connectivity).
void canny(const ImageF32& gray, ImageU8& edges, int radius, float sigma, float low, float high);

}  // namespace cpu
