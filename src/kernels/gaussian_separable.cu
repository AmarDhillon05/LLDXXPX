// Gaussian v2: separable, straight from global memory.
//
// A Gaussian is the outer product of two 1D Gaussians, so a horizontal pass
// followed by a vertical pass gives the same result with 2(2r+1) taps instead
// of (2r+1)^2: 30 vs 225 for a 15x15 kernel. The price is one extra round trip
// of the intermediate image through global memory.
#include "cuda_utils.cuh"
#include "kernels.cuh"

__global__ void gaussian_sep_row_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int w, int h, const float* __restrict__ k, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* row = in + size_t(y) * w;
    float acc = 0.f;
    for (int i = -radius; i <= radius; ++i) acc += k[i + radius] * row[clampi(x + i, 0, w - 1)];
    out[size_t(y) * w + x] = acc;
}

__global__ void gaussian_sep_col_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int w, int h, const float* __restrict__ k, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float acc = 0.f;
    for (int j = -radius; j <= radius; ++j)
        acc += k[j + radius] * in[size_t(clampi(y + j, 0, h - 1)) * w + x];
    out[size_t(y) * w + x] = acc;
}

void gaussian_separable(const float* in, float* tmp, float* out, int w, int h,
                        const float* d_weights1d, int radius, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(div_up(w, block.x), div_up(h, block.y));
    gaussian_sep_row_kernel<<<grid, block, 0, stream>>>(in, tmp, w, h, d_weights1d, radius);
    CUDA_CHECK_LAUNCH();
    gaussian_sep_col_kernel<<<grid, block, 0, stream>>>(tmp, out, w, h, d_weights1d, radius);
    CUDA_CHECK_LAUNCH();
}
