// Host-side launchers for every GPU kernel version. Each group lives in its own
// file under src/kernels/. All pointers are device pointers, all images are
// tightly packed (row stride == width), and all borders are replicate/clamp.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

constexpr int kMaxGaussRadius = 16;     // 33 taps
constexpr int kMaxBilateralRadius = 10;  // 21x21 window

// ---------------------------------------------------------------------------
// Gaussian blur (float32, single channel)
// ---------------------------------------------------------------------------
// v1 gaussian_naive.cu: one thread per pixel, (2r+1)^2 taps, 2D weights in global memory.
void gaussian_naive_2d(const float* in, float* out, int w, int h, const float* d_weights2d,
                       int radius, cudaStream_t stream = 0);

// v2 gaussian_separable.cu: horizontal + vertical 1D passes straight from global memory.
void gaussian_separable(const float* in, float* tmp, float* out, int w, int h,
                        const float* d_weights1d, int radius, cudaStream_t stream = 0);

// v3 gaussian_tiled.cu: separable, each pass stages a tile + halo in shared memory.
// Weights still read from global memory.
void gaussian_tiled(const float* in, float* tmp, float* out, int w, int h,
                    const float* d_weights1d, int radius, cudaStream_t stream = 0);

// v4 gaussian_constant.cu: v3 with the weights in __constant__ memory.
// Call gaussian_constant_set_weights() once per kernel change.
void gaussian_constant_set_weights(const float* h_weights1d, int radius);
void gaussian_constant(const float* in, float* tmp, float* out, int w, int h, int radius,
                       cudaStream_t stream = 0);

// v5 gaussian_texture.cu (stretch): separable passes reading through the texture
// unit (hardware clamp addressing, no explicit border math). Inputs must be
// pitched allocations (cudaMallocPitch); pitch is in bytes.
void gaussian_texture_set_weights(const float* h_weights1d, int radius);
void gaussian_texture(const float* in, size_t in_pitch, float* tmp, size_t tmp_pitch, float* out,
                      size_t out_pitch, int w, int h, int radius, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Sobel (sobel.cu). Magnitude + threshold fused in one kernel. `mag` may be null.
// ---------------------------------------------------------------------------
void sobel_naive(const float* in, float* mag, uint8_t* edges, int w, int h, float threshold,
                 cudaStream_t stream = 0);
void sobel_tiled(const float* in, float* mag, uint8_t* edges, int w, int h, float threshold,
                 cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Bilateral filter (bilateral.cu). Circular window, OpenCV semantics.
// ---------------------------------------------------------------------------
void bilateral_naive(const float* in, float* out, int w, int h, int radius, float sigma_color,
                     float sigma_space, cudaStream_t stream = 0);
// Shared-memory tile + constant-memory window table + __expf.
void bilateral_tiled(const float* in, float* out, int w, int h, int radius, float sigma_color,
                     float sigma_space, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Histogram + equalization (histogram.cu). d_hist is 256 uint32 and is zeroed
// by the launcher. n = number of pixels.
// ---------------------------------------------------------------------------
void histogram_global_atomics(const uint8_t* in, size_t n, uint32_t* d_hist, cudaStream_t stream = 0);
void histogram_shared(const uint8_t* in, size_t n, uint32_t* d_hist, cudaStream_t stream = 0);
void histogram_warp_private(const uint8_t* in, size_t n, uint32_t* d_hist, cudaStream_t stream = 0);
// Single-block scan of the 256-bin histogram into the cv::equalizeHist LUT.
void equalize_build_lut(const uint32_t* d_hist, size_t n, uint8_t* d_lut, cudaStream_t stream = 0);
void equalize_apply_lut(const uint8_t* in, uint8_t* out, size_t n, const uint8_t* d_lut,
                        cudaStream_t stream = 0);
// Convenience: histogram_shared + build_lut + apply_lut.
void equalize(const uint8_t* in, uint8_t* out, size_t n, uint32_t* d_hist, uint8_t* d_lut,
              cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Edge pipeline: RGBA8 -> gray -> Gaussian -> Sobel -> threshold -> uint8.
// ---------------------------------------------------------------------------
// pipeline.cu
void rgba_to_gray(const uchar4* in, float* gray, int w, int h, cudaStream_t stream = 0);
// Separate kernels: gray, blur H, blur V, Sobel. Uses gaussian_constant weights,
// so call gaussian_constant_set_weights() first. tmp buffers are w*h floats.
void pipeline_separate(const uchar4* in, float* gray, float* tmp, float* blurred, uint8_t* edges,
                       int w, int h, int radius, float threshold, cudaStream_t stream = 0);
// One fused kernel: everything stays in shared memory; reads RGBA once, writes
// the uint8 edge map once.
void pipeline_fused_set_weights(const float* h_weights1d, int radius);
void pipeline_fused(const uchar4* in, uint8_t* edges, int w, int h, int radius, float threshold,
                    cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Canny (canny.cu, stretch). Uses gaussian_constant weights for the blur.
// scratch buffers: blurred/mag are w*h floats, dir/state are w*h bytes.
// ---------------------------------------------------------------------------
struct CannyBuffers {
    float* blurred;
    float* tmp;
    float* mag;
    uint8_t* dir;
    uint8_t* state;
    int* d_changed;  // single int
};
// Returns the number of hysteresis iterations (host loop) that were needed.
int canny(const float* gray, uint8_t* edges, int w, int h, int radius, float low, float high,
          const CannyBuffers& buf, cudaStream_t stream = 0);
