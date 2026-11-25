// Gaussian v1: naive 2D convolution.
//
// One thread per output pixel, (2r+1)^2 multiply-adds, every tap is a global
// load (served mostly by L1/L2 because neighbouring threads share inputs), and
// the 2D weight table lives in global memory. This is the baseline every other
// version is measured against.
#include "cuda_utils.cuh"
#include "kernels.cuh"

__global__ void gaussian_naive_2d_kernel(const float* __restrict__ in, float* __restrict__ out,
                                         int w, int h, const float* __restrict__ weights,
                                         int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int n = 2 * radius + 1;
    float acc = 0.f;
    for (int j = -radius; j <= radius; ++j) {
        const float* row = in + size_t(clampi(y + j, 0, h - 1)) * w;
        const float* wrow = weights + (j + radius) * n + radius;
        for (int i = -radius; i <= radius; ++i) acc += wrow[i] * row[clampi(x + i, 0, w - 1)];
    }
    out[size_t(y) * w + x] = acc;
}

void gaussian_naive_2d(const float* in, float* out, int w, int h, const float* d_weights2d,
                       int radius, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(div_up(w, block.x), div_up(h, block.y));
    gaussian_naive_2d_kernel<<<grid, block, 0, stream>>>(in, out, w, h, d_weights2d, radius);
    CUDA_CHECK_LAUNCH();
}
