// Sobel edge detection with fused magnitude + threshold.
//
// Both versions compute gx, gy, |g| = sqrt(gx^2 + gy^2) and the binary edge
// map in a single kernel, so the gradient images never touch global memory.
//
//  sobel_naive: 8 clamped global loads per pixel (L1 catches most of them).
//  sobel_tiled: the block stages a (32+2) x (8+2) tile in shared memory once,
//               then every thread reads its 3x3 neighbourhood from there.
#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace {

__device__ __forceinline__ void sobel_3x3(float a, float b, float c, float d, float f, float g,
                                          float hh, float i, float& gx, float& gy) {
    // a b c
    // d . f
    // g h i
    gx = (c - a) + 2.f * (f - d) + (i - g);
    gy = (g - a) + 2.f * (hh - b) + (i - c);
}

__global__ void sobel_naive_kernel(const float* __restrict__ in, float* __restrict__ mag,
                                   uint8_t* __restrict__ edges, int w, int h, float thr) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int xm = max(x - 1, 0), xp = min(x + 1, w - 1);
    const float* rm = in + size_t(max(y - 1, 0)) * w;
    const float* r0 = in + size_t(y) * w;
    const float* rp = in + size_t(min(y + 1, h - 1)) * w;
    float gx, gy;
    sobel_3x3(rm[xm], rm[x], rm[xp], r0[xm], r0[xp], rp[xm], rp[x], rp[xp], gx, gy);
    const float m = sqrtf(gx * gx + gy * gy);
    const size_t o = size_t(y) * w + x;
    if (mag) mag[o] = m;
    edges[o] = m > thr ? 255 : 0;
}

constexpr int TX = 32, TY = 8;

__global__ void __launch_bounds__(TX* TY)
    sobel_tiled_kernel(const float* __restrict__ in, float* __restrict__ mag,
                       uint8_t* __restrict__ edges, int w, int h, float thr) {
    __shared__ float s[TY + 2][TX + 2];
    const int x0 = blockIdx.x * TX - 1, y0 = blockIdx.y * TY - 1;
    const int tid = threadIdx.y * TX + threadIdx.x;
    for (int i = tid; i < (TX + 2) * (TY + 2); i += TX * TY) {
        const int ly = i / (TX + 2), lx = i % (TX + 2);
        s[ly][lx] = in[size_t(clampi(y0 + ly, 0, h - 1)) * w + clampi(x0 + lx, 0, w - 1)];
    }
    __syncthreads();
    const int x = blockIdx.x * TX + threadIdx.x;
    const int y = blockIdx.y * TY + threadIdx.y;
    if (x >= w || y >= h) return;
    const int lx = threadIdx.x + 1, ly = threadIdx.y + 1;
    float gx, gy;
    sobel_3x3(s[ly - 1][lx - 1], s[ly - 1][lx], s[ly - 1][lx + 1], s[ly][lx - 1], s[ly][lx + 1],
              s[ly + 1][lx - 1], s[ly + 1][lx], s[ly + 1][lx + 1], gx, gy);
    const float m = sqrtf(gx * gx + gy * gy);
    const size_t o = size_t(y) * w + x;
    if (mag) mag[o] = m;
    edges[o] = m > thr ? 255 : 0;
}

}  // namespace

void sobel_naive(const float* in, float* mag, uint8_t* edges, int w, int h, float threshold,
                 cudaStream_t stream) {
    const dim3 block(32, 8), grid(div_up(w, 32), div_up(h, 8));
    sobel_naive_kernel<<<grid, block, 0, stream>>>(in, mag, edges, w, h, threshold);
    CUDA_CHECK_LAUNCH();
}

void sobel_tiled(const float* in, float* mag, uint8_t* edges, int w, int h, float threshold,
                 cudaStream_t stream) {
    const dim3 block(TX, TY), grid(div_up(w, TX), div_up(h, TY));
    sobel_tiled_kernel<<<grid, block, 0, stream>>>(in, mag, edges, w, h, threshold);
    CUDA_CHECK_LAUNCH();
}
