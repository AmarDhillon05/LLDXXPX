// Edge pipeline: RGBA8 -> grayscale -> Gaussian blur -> Sobel -> threshold.
//
// pipeline_separate: four kernels (gray, blur-row, blur-col, Sobel). Global
//   traffic per pixel: 4B RGBA in, then 3 float images written and read back
//   (gray, row-blurred, blurred) = 4 + 3*(4+4) + 1 = 29 bytes/pixel minimum.
//
// pipeline_fused: one kernel. A 32x32 output tile loads its RGBA footprint
//   (tile + R+1 halo) once, and gray, horizontal blur, vertical blur and the
//   Sobel stencil all live in shared memory. Global traffic is 4B in + 1B out
//   = 5 bytes/pixel (+ halo re-reads, mostly L2 hits). The price is redundant
//   halo computation (~1.6x row-blur work for R=7) and lower occupancy from
//   ~20 KB shared memory per block.
//
// Border semantics match the separate path exactly: every stage replicates
// at the image border, so blurred values in the Sobel halo are computed at the
// *clamped* coordinate rather than from clamped gray values around an
// out-of-image centre.
#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace {

// Grayscale with explicit round-to-nearest ops so nvcc cannot contract to FMA:
// the result is bit-identical to the CPU reference.
__device__ __forceinline__ float gray_of(uchar4 p) {
    return __fadd_rn(__fadd_rn(__fmul_rn(0.299f, float(p.x)), __fmul_rn(0.587f, float(p.y))),
                     __fmul_rn(0.114f, float(p.z)));
}

__global__ void rgba_to_gray_kernel(const uchar4* __restrict__ in, float* __restrict__ gray, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x)
        gray[i] = gray_of(in[i]);
}

__constant__ float c_fused_w[2 * kMaxGaussRadius + 1];

constexpr int FT = 32;          // output tile is FT x FT
constexpr int FBX = 32, FBY = 8;  // 256 threads

template <int R>
__global__ void __launch_bounds__(FBX* FBY)
    fused_kernel(const uchar4* __restrict__ in, uint8_t* __restrict__ edges, int w, int h, float thr) {
    constexpr int BW = FT + 2;       // blurred region (tile + Sobel halo)
    constexpr int GW = BW + 2 * R;   // gray region (+ blur halo)
    __shared__ float sG[GW][GW];
    __shared__ float sH[GW][BW];
    __shared__ float sB[BW][BW + 1];  // +1: Sobel reads are column-strided across rows

    const int tid = threadIdx.y * FBX + threadIdx.x;
    const int ox = blockIdx.x * FT, oy = blockIdx.y * FT;
    const int bx0 = ox - 1, by0 = oy - 1;          // global coord of sB[0][0]
    const int gx0 = bx0 - R, gy0 = by0 - R;        // global coord of sG[0][0]

    // 1. RGBA -> gray for the whole footprint (clamped loads).
    for (int i = tid; i < GW * GW; i += FBX * FBY) {
        const int ly = i / GW, lx = i - ly * GW;
        const int gx = clampi(gx0 + lx, 0, w - 1), gy = clampi(gy0 + ly, 0, h - 1);
        sG[ly][lx] = gray_of(in[size_t(gy) * w + gx]);
    }
    __syncthreads();

    // 2. Horizontal blur for every gray row, at the blurred-region columns.
    //    Column c is evaluated at the clamped coordinate (replicate border).
    for (int i = tid; i < GW * BW; i += FBX * FBY) {
        const int r = i / BW, c = i - r * BW;
        const int lc = clampi(bx0 + c, 0, w - 1) - gx0;  // centre column in sG
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k <= 2 * R; ++k) acc += c_fused_w[k] * sG[r][lc - R + k];
        sH[r][c] = acc;
    }
    __syncthreads();

    // 3. Vertical blur, again centred on the clamped row.
    for (int i = tid; i < BW * BW; i += FBX * FBY) {
        const int r = i / BW, c = i - r * BW;
        const int lr = clampi(by0 + r, 0, h - 1) - gy0;  // centre row in sH
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k <= 2 * R; ++k) acc += c_fused_w[k] * sH[lr - R + k][c];
        sB[r][c] = acc;
    }
    __syncthreads();

    // 4. Sobel + magnitude + threshold; each thread writes FT/FBY outputs.
    const int tx = threadIdx.x;
    const int x = ox + tx;
#pragma unroll
    for (int s = 0; s < FT / FBY; ++s) {
        const int ty = threadIdx.y + s * FBY;
        const int y = oy + ty;
        if (x >= w || y >= h) continue;
        const int lx = tx + 1, ly = ty + 1;
        const float a = sB[ly - 1][lx - 1], b = sB[ly - 1][lx], c = sB[ly - 1][lx + 1];
        const float d = sB[ly][lx - 1], f = sB[ly][lx + 1];
        const float g = sB[ly + 1][lx - 1], hh = sB[ly + 1][lx], ii = sB[ly + 1][lx + 1];
        const float gxv = (c - a) + 2.f * (f - d) + (ii - g);
        const float gyv = (g - a) + 2.f * (hh - b) + (ii - c);
        edges[size_t(y) * w + x] = sqrtf(gxv * gxv + gyv * gyv) > thr ? 255 : 0;
    }
}

}  // namespace

void rgba_to_gray(const uchar4* in, float* gray, int w, int h, cudaStream_t stream) {
    const size_t n = size_t(w) * h;
    const int grid = int(std::min<size_t>((n + 255) / 256, 65535));
    rgba_to_gray_kernel<<<grid, 256, 0, stream>>>(in, gray, n);
    CUDA_CHECK_LAUNCH();
}

void pipeline_separate(const uchar4* in, float* gray, float* tmp, float* blurred, uint8_t* edges,
                       int w, int h, int radius, float threshold, cudaStream_t stream) {
    rgba_to_gray(in, gray, w, h, stream);
    gaussian_constant(gray, tmp, blurred, w, h, radius, stream);
    sobel_tiled(blurred, nullptr, edges, w, h, threshold, stream);
}

void pipeline_fused_set_weights(const float* h_weights1d, int radius) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_fused_w, h_weights1d, sizeof(float) * (2 * radius + 1)));
}

void pipeline_fused(const uchar4* in, uint8_t* edges, int w, int h, int radius, float threshold,
                    cudaStream_t stream) {
    const dim3 block(FBX, FBY), grid(div_up(w, FT), div_up(h, FT));
    switch (radius) {
#define PF_CASE(R)                                                                 \
    case R:                                                                        \
        fused_kernel<R><<<grid, block, 0, stream>>>(in, edges, w, h, threshold);   \
        break;
        PF_CASE(1) PF_CASE(2) PF_CASE(3) PF_CASE(4) PF_CASE(5) PF_CASE(6) PF_CASE(7) PF_CASE(8)
#undef PF_CASE
        default:
            std::fprintf(stderr, "pipeline_fused: radius %d unsupported (1..8)\n", radius);
            std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK_LAUNCH();
}
