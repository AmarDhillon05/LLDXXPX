// Shared implementation of the tiled separable Gaussian (v3 and v4).
//
// The two versions differ only in where the 1D weights come from, so the
// kernels are templated on a weight-accessor policy:
//   v3 gaussian_tiled.cu    -> GlobalWeights   (plain global loads)
//   v4 gaussian_constant.cu -> ConstantWeights (__constant__ bank, broadcast)
//
// Both are also templated on the radius so the tap loop fully unrolls.
//
// Row pass: a 32x8 block produces a 128x8 output tile. It first stages
// 8 rows of (128 + 2R) inputs (tile + halo) in shared memory with coalesced
// loads, then each thread computes 4 outputs spaced 32 apart, so a warp reads
// 32 consecutive shared-memory words per tap (no bank conflicts) and writes
// 32 consecutive floats (fully coalesced).
//
// Column pass: a 32x8 block produces a 32x32 tile from a (32 + 2R) x 32
// staged column strip. Each thread computes 4 outputs spaced 8 rows apart.
#pragma once

#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace gauss_tiled {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kRowSteps = 4;  // outputs per thread in the row pass
constexpr int kColSteps = 4;  // outputs per thread in the column pass
constexpr int kRowTileW = kBlockX * kRowSteps;  // 128
constexpr int kColTileH = kBlockY * kColSteps;  // 32

template <int R, class Weights>
__global__ void __launch_bounds__(kBlockX* kBlockY)
    row_kernel(const float* __restrict__ in, float* __restrict__ out, int w, int h, Weights wts) {
    __shared__ float s[kBlockY][kRowTileW + 2 * R];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x0 = blockIdx.x * kRowTileW;
    const int y = blockIdx.y * kBlockY + ty;
    const int yc = min(y, h - 1);  // rows past the bottom still load (and are discarded)
    const float* row = in + size_t(yc) * w;

#pragma unroll
    for (int i = tx; i < kRowTileW + 2 * R; i += kBlockX) s[ty][i] = row[clampi(x0 - R + i, 0, w - 1)];
    __syncthreads();
    if (y >= h) return;

#pragma unroll
    for (int step = 0; step < kRowSteps; ++step) {
        const int lx = tx + step * kBlockX;
        const int x = x0 + lx;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k <= 2 * R; ++k) acc += wts(k) * s[ty][lx + k];
        if (x < w) out[size_t(y) * w + x] = acc;
    }
}

template <int R, class Weights>
__global__ void __launch_bounds__(kBlockX* kBlockY)
    col_kernel(const float* __restrict__ in, float* __restrict__ out, int w, int h, Weights wts) {
    __shared__ float s[kColTileH + 2 * R][kBlockX];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = blockIdx.x * kBlockX + tx;
    const int y0 = blockIdx.y * kColTileH;
    const int xc = min(x, w - 1);

#pragma unroll
    for (int r = ty; r < kColTileH + 2 * R; r += kBlockY)
        s[r][tx] = in[size_t(clampi(y0 - R + r, 0, h - 1)) * w + xc];
    __syncthreads();
    if (x >= w) return;

#pragma unroll
    for (int step = 0; step < kColSteps; ++step) {
        const int ly = ty + step * kBlockY;
        const int y = y0 + ly;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k <= 2 * R; ++k) acc += wts(k) * s[ly + k][tx];
        if (y < h) out[size_t(y) * w + x] = acc;
    }
}

template <int R, class Weights>
void launch(const float* in, float* tmp, float* out, int w, int h, Weights wts, cudaStream_t st) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid_r(div_up(w, kRowTileW), div_up(h, kBlockY));
    row_kernel<R, Weights><<<grid_r, block, 0, st>>>(in, tmp, w, h, wts);
    CUDA_CHECK_LAUNCH();
    const dim3 grid_c(div_up(w, kBlockX), div_up(h, kColTileH));
    col_kernel<R, Weights><<<grid_c, block, 0, st>>>(tmp, out, w, h, wts);
    CUDA_CHECK_LAUNCH();
}

// Runtime radius -> compile-time template dispatch.
template <class Weights>
void dispatch(int radius, const float* in, float* tmp, float* out, int w, int h, Weights wts,
              cudaStream_t st) {
    switch (radius) {
#define GT_CASE(R) \
    case R: launch<R>(in, tmp, out, w, h, wts, st); break;
        GT_CASE(1) GT_CASE(2) GT_CASE(3) GT_CASE(4) GT_CASE(5) GT_CASE(6) GT_CASE(7) GT_CASE(8)
        GT_CASE(9) GT_CASE(10) GT_CASE(11) GT_CASE(12) GT_CASE(13) GT_CASE(14) GT_CASE(15) GT_CASE(16)
#undef GT_CASE
        default:
            std::fprintf(stderr, "gaussian: radius %d unsupported (1..%d)\n", radius, kMaxGaussRadius);
            std::exit(EXIT_FAILURE);
    }
}

}  // namespace gauss_tiled
