// NVIDIA NPP vendor baselines (only built when NPP is found: HAVE_NPP).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace nppb {

// Separable Gaussian with caller-provided taps (device pointer), replicate border.
void gauss(const float* in, float* out, int w, int h, const float* d_taps, int ntaps,
           cudaStream_t stream = 0);
// Horizontal + vertical Sobel into two float images (NPP has no fused magnitude).
void sobel_xy(const float* in, float* gx, float* gy, int w, int h, cudaStream_t stream = 0);
// Bilateral with a square (2r+1)^2 window.
void bilateral(const float* in, float* out, int w, int h, int radius, float sigma_color,
               float sigma_space, cudaStream_t stream = 0);
// 256-bin histogram; scratch buffer is allocated on first use and cached.
void histogram(const uint8_t* in, int w, int h, int32_t* d_hist, cudaStream_t stream = 0);

}  // namespace nppb
