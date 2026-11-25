// Bilateral filter: the compute-heavy kernel.
//
// out(p) = sum_q  v(q) * Ws(|p-q|) * Wc(|v(q)-v(p)|)  /  sum_q Ws * Wc
// with a circular window |p-q| <= r (cv::bilateralFilter semantics).
//
// bilateral_naive: per tap, clamped global load + expf() for BOTH weights.
// bilateral_tiled:
//   * (16+2r)^2 input tile staged in shared memory (no clamping in the loop),
//   * the circular window is precomputed on the host as a list of
//     (shared-memory offset, spatial weight) pairs in __constant__ memory:
//     every thread walks the same list in lock-step -> constant-cache
//     broadcast, no divergence, no wasted taps outside the circle,
//   * the per-tap range weight uses the __expf intrinsic (SFU ex2 + 1 FMUL)
//     instead of the accurate, ~20-instruction expf().
#include <cmath>
#include <vector>

#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace {

constexpr int kMaxTaps = (2 * kMaxBilateralRadius + 1) * (2 * kMaxBilateralRadius + 1);
constexpr int BT = 16;  // tile edge

__constant__ int c_bl_off[kMaxTaps];
__constant__ float c_bl_ws[kMaxTaps];

__global__ void bilateral_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int w,
                                       int h, int r, float color_coeff, float space_coeff) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float c = in[size_t(y) * w + x];
    float sum = 0.f, wsum = 0.f;
    for (int j = -r; j <= r; ++j) {
        const float* row = in + size_t(clampi(y + j, 0, h - 1)) * w;
        for (int i = -r; i <= r; ++i) {
            const float d2 = float(i * i + j * j);
            if (sqrtf(d2) > r) continue;
            const float v = row[clampi(x + i, 0, w - 1)];
            const float dv = v - c;
            const float wgt = expf(d2 * space_coeff) * expf(dv * dv * color_coeff);
            sum += v * wgt;
            wsum += wgt;
        }
    }
    out[size_t(y) * w + x] = sum / wsum;
}

__global__ void __launch_bounds__(BT* BT)
    bilateral_tiled_kernel(const float* __restrict__ in, float* __restrict__ out, int w, int h,
                           int r, int ntaps, float color_coeff) {
    extern __shared__ float s[];
    const int S = BT + 2 * r;  // shared tile stride
    const int x0 = blockIdx.x * BT - r, y0 = blockIdx.y * BT - r;
    const int tid = threadIdx.y * BT + threadIdx.x;
    for (int i = tid; i < S * S; i += BT * BT) {
        const int ly = i / S, lx = i - ly * S;
        s[i] = in[size_t(clampi(y0 + ly, 0, h - 1)) * w + clampi(x0 + lx, 0, w - 1)];
    }
    __syncthreads();
    const int x = blockIdx.x * BT + threadIdx.x;
    const int y = blockIdx.y * BT + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* center = s + (threadIdx.y + r) * S + (threadIdx.x + r);
    const float c = *center;
    float sum = 0.f, wsum = 0.f;
#pragma unroll 4
    for (int k = 0; k < ntaps; ++k) {
        const float v = center[c_bl_off[k]];
        const float dv = v - c;
        const float wgt = c_bl_ws[k] * __expf(dv * dv * color_coeff);
        sum = fmaf(v, wgt, sum);
        wsum += wgt;
    }
    out[size_t(y) * w + x] = __fdividef(sum, wsum);
}

void check_radius(int r) {
    if (r < 1 || r > kMaxBilateralRadius) {
        std::fprintf(stderr, "bilateral: radius %d unsupported (1..%d)\n", r, kMaxBilateralRadius);
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

void bilateral_naive(const float* in, float* out, int w, int h, int radius, float sigma_color,
                     float sigma_space, cudaStream_t stream) {
    check_radius(radius);
    const dim3 block(32, 8), grid(div_up(w, 32), div_up(h, 8));
    bilateral_naive_kernel<<<grid, block, 0, stream>>>(in, out, w, h, radius,
                                                       -0.5f / (sigma_color * sigma_color),
                                                       -0.5f / (sigma_space * sigma_space));
    CUDA_CHECK_LAUNCH();
}

void bilateral_tiled(const float* in, float* out, int w, int h, int radius, float sigma_color,
                     float sigma_space, cudaStream_t stream) {
    check_radius(radius);
    // (Re)build the window table only when the parameters change.
    static int cached_r = -1, cached_taps = 0;
    static float cached_ss = -1.f;
    if (radius != cached_r || sigma_space != cached_ss) {
        const int S = BT + 2 * radius;
        const float space_coeff = -0.5f / (sigma_space * sigma_space);
        std::vector<int> off;
        std::vector<float> ws;
        for (int j = -radius; j <= radius; ++j)
            for (int i = -radius; i <= radius; ++i) {
                const float rr = std::sqrt(float(i * i + j * j));
                if (rr > radius) continue;
                off.push_back(j * S + i);
                ws.push_back(std::exp(rr * rr * space_coeff));
            }
        CUDA_CHECK(cudaMemcpyToSymbol(c_bl_off, off.data(), off.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_bl_ws, ws.data(), ws.size() * sizeof(float)));
        cached_r = radius;
        cached_ss = sigma_space;
        cached_taps = int(off.size());
    }
    const int S = BT + 2 * radius;
    const size_t smem = size_t(S) * S * sizeof(float);
    const dim3 block(BT, BT), grid(div_up(w, BT), div_up(h, BT));
    bilateral_tiled_kernel<<<grid, block, smem, stream>>>(in, out, w, h, radius, cached_taps,
                                                          -0.5f / (sigma_color * sigma_color));
    CUDA_CHECK_LAUNCH();
}
