// Histogram + histogram equalization.
//
// Three histogram versions, in increasing order of privatization:
//
//  histogram_global_atomics: one global atomicAdd per pixel. Natural images
//      concentrate in few bins, so thousands of threads serialize on the same
//      L2 atomic units.
//  histogram_shared: each block keeps a private 256-bin histogram in shared
//      memory (shared atomics are resolved in the SM), reads 4 pixels per
//      32-bit load, then merges with at most 256 global atomics per block.
//  histogram_warp_private: one sub-histogram per warp inside the block, which
//      removes inter-warp contention on hot bins; sub-histograms are summed
//      before the global merge.
//
// Equalization then needs the CDF: a single 256-thread block does a warp-
// shuffle inclusive scan and writes the cv::equalizeHist LUT, and a final
// bandwidth-bound kernel applies the LUT (staged in shared memory).
#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace {

constexpr int kBins = 256;
constexpr int kHistThreads = 256;

int hist_grid(size_t n) {
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    }
    // ~8 resident blocks per SM is plenty to saturate bandwidth; more blocks
    // only means more global merges.
    const size_t needed = (n / 4 + kHistThreads - 1) / kHistThreads;
    return int(std::max<size_t>(1, std::min<size_t>(needed, size_t(sms) * 8)));
}

__global__ void hist_global_kernel(const uint8_t* __restrict__ in, size_t n, uint32_t* hist) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x)
        atomicAdd(&hist[in[i]], 1u);
}

__global__ void hist_shared_kernel(const uint8_t* __restrict__ in, size_t n, uint32_t* hist) {
    __shared__ uint32_t s[kBins];
    for (int i = threadIdx.x; i < kBins; i += blockDim.x) s[i] = 0;
    __syncthreads();

    const size_t n4 = n / 4;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4; i += stride) {
        const uchar4 v = in4[i];
        atomicAdd(&s[v.x], 1u);
        atomicAdd(&s[v.y], 1u);
        atomicAdd(&s[v.z], 1u);
        atomicAdd(&s[v.w], 1u);
    }
    if (blockIdx.x == 0)  // tail (n not a multiple of 4)
        for (size_t i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) atomicAdd(&s[in[i]], 1u);
    __syncthreads();

    for (int i = threadIdx.x; i < kBins; i += blockDim.x)
        if (s[i]) atomicAdd(&hist[i], s[i]);
}

__global__ void hist_warp_private_kernel(const uint8_t* __restrict__ in, size_t n, uint32_t* hist) {
    constexpr int kWarps = kHistThreads / 32;
    __shared__ uint32_t s[kWarps][kBins];
    for (int i = threadIdx.x; i < kWarps * kBins; i += blockDim.x) (&s[0][0])[i] = 0;
    __syncthreads();

    uint32_t* mine = s[threadIdx.x / 32];
    const size_t n4 = n / 4;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4; i += stride) {
        const uchar4 v = in4[i];
        atomicAdd(&mine[v.x], 1u);
        atomicAdd(&mine[v.y], 1u);
        atomicAdd(&mine[v.z], 1u);
        atomicAdd(&mine[v.w], 1u);
    }
    if (blockIdx.x == 0)
        for (size_t i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) atomicAdd(&mine[in[i]], 1u);
    __syncthreads();

    for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
        uint32_t total = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) total += s[w][b];
        if (total) atomicAdd(&hist[b], total);
    }
}

// One block, 256 threads: thread t owns bin t.
__global__ void build_lut_kernel(const uint32_t* __restrict__ hist, unsigned long long total,
                                 uint8_t* __restrict__ lut) {
    __shared__ uint32_t warp_sums[kBins / 32];
    __shared__ uint32_t s_hist[kBins];
    __shared__ int first;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const uint32_t v = hist[t];
    s_hist[t] = v;
    if (t == 0) first = kBins;
    __syncthreads();
    if (v) atomicMin(&first, t);

    // Inclusive scan: within each warp via shuffles, then across warp totals.
    uint32_t x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint32_t ws = lane < kBins / 32 ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, ws, o);
            if (lane >= o) ws += y;
        }
        if (lane < kBins / 32) warp_sums[lane] = ws;
    }
    __syncthreads();
    const uint32_t cdf = x + (warp ? warp_sums[warp - 1] : 0);

    const int i0 = first;
    if (i0 == kBins) {  // empty image
        lut[t] = 0;
        return;
    }
    const uint32_t h0 = s_hist[i0];
    if (h0 == total) {  // constant image: cv::equalizeHist maps everything to i0
        lut[t] = uint8_t(i0);
        return;
    }
    if (t <= i0) {
        lut[t] = 0;
        return;
    }
    const float scale = 255.f / float(total - h0);
    const int val = __float2int_rn(float(int(cdf - h0)) * scale);  // round-half-even
    lut[t] = uint8_t(min(255, max(0, val)));
}

__global__ void apply_lut_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, size_t n,
                                 const uint8_t* __restrict__ lut) {
    __shared__ uint8_t s[kBins];
    for (int i = threadIdx.x; i < kBins; i += blockDim.x) s[i] = lut[i];
    __syncthreads();
    const size_t n4 = n / 4;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);
    uchar4* out4 = reinterpret_cast<uchar4*>(out);
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4; i += stride) {
        const uchar4 v = in4[i];
        out4[i] = make_uchar4(s[v.x], s[v.y], s[v.z], s[v.w]);
    }
    if (blockIdx.x == 0)
        for (size_t i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) out[i] = s[in[i]];
}

}  // namespace

void histogram_global_atomics(const uint8_t* in, size_t n, uint32_t* d_hist, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, kBins * sizeof(uint32_t), stream));
    const int grid = int(std::min<size_t>((n + kHistThreads - 1) / kHistThreads, 65535 * 4));
    hist_global_kernel<<<grid, kHistThreads, 0, stream>>>(in, n, d_hist);
    CUDA_CHECK_LAUNCH();
}

void histogram_shared(const uint8_t* in, size_t n, uint32_t* d_hist, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, kBins * sizeof(uint32_t), stream));
    hist_shared_kernel<<<hist_grid(n), kHistThreads, 0, stream>>>(in, n, d_hist);
    CUDA_CHECK_LAUNCH();
}

void histogram_warp_private(const uint8_t* in, size_t n, uint32_t* d_hist, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, kBins * sizeof(uint32_t), stream));
    hist_warp_private_kernel<<<hist_grid(n), kHistThreads, 0, stream>>>(in, n, d_hist);
    CUDA_CHECK_LAUNCH();
}

void equalize_build_lut(const uint32_t* d_hist, size_t n, uint8_t* d_lut, cudaStream_t stream) {
    build_lut_kernel<<<1, kBins, 0, stream>>>(d_hist, (unsigned long long)n, d_lut);
    CUDA_CHECK_LAUNCH();
}

void equalize_apply_lut(const uint8_t* in, uint8_t* out, size_t n, const uint8_t* d_lut,
                        cudaStream_t stream) {
    apply_lut_kernel<<<hist_grid(n), 256, 0, stream>>>(in, out, n, d_lut);
    CUDA_CHECK_LAUNCH();
}

void equalize(const uint8_t* in, uint8_t* out, size_t n, uint32_t* d_hist, uint8_t* d_lut,
              cudaStream_t stream) {
    histogram_shared(in, n, d_hist, stream);
    equalize_build_lut(d_hist, n, d_lut, stream);
    equalize_apply_lut(in, out, n, d_lut, stream);
}
