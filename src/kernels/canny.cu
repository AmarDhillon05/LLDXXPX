// Canny edge detector (stretch goal).
//
//   1. Gaussian blur                     (reuses gaussian_constant, v4)
//   2. Sobel gx, gy -> L2 magnitude + direction quantized to 4 sectors
//   3. Non-max suppression + double threshold -> state {0 none, 1 weak, 2 strong}
//   4. Hysteresis: weak pixels 8-connected to a strong pixel become strong.
//
// Hysteresis is the interesting part on a GPU: it is a connected-components
// problem with no fixed amount of work. Each launch propagates "strong"
// through a 32x8 tile *inside shared memory* until the tile converges
// (__syncthreads_or), so one launch moves information across a whole tile
// instead of one pixel. The host relaunches until no block changed anything.
// Updates are monotonic (weak -> strong only), so reading a neighbour tile's
// halo while that tile is being updated is benign: a stale value only delays
// convergence by one launch, it cannot produce a wrong fixed point.
#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace {

// Same direction codes as cpu_reference.cpp.
__device__ __forceinline__ int quantize_dir(float gx, float gy) {
    const float ax = fabsf(gx), ay = fabsf(gy);
    if (ay <= 0.41421356f * ax) return 0;
    if (ay >= 2.41421356f * ax) return 2;
    return ((gx > 0) == (gy > 0)) ? 1 : 3;
}
__constant__ int c_off[4][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}};

__global__ void grad_kernel(const float* __restrict__ in, float* __restrict__ mag,
                            uint8_t* __restrict__ dir, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int xm = max(x - 1, 0), xp = min(x + 1, w - 1);
    const float* rm = in + size_t(max(y - 1, 0)) * w;
    const float* r0 = in + size_t(y) * w;
    const float* rp = in + size_t(min(y + 1, h - 1)) * w;
    const float gx = (rm[xp] - rm[xm]) + 2.f * (r0[xp] - r0[xm]) + (rp[xp] - rp[xm]);
    const float gy = (rp[xm] - rm[xm]) + 2.f * (rp[x] - rm[x]) + (rp[xp] - rm[xp]);
    const size_t o = size_t(y) * w + x;
    mag[o] = sqrtf(gx * gx + gy * gy);
    dir[o] = uint8_t(quantize_dir(gx, gy));
}

__global__ void nms_kernel(const float* __restrict__ mag, const uint8_t* __restrict__ dir,
                           uint8_t* __restrict__ state, int w, int h, float low, float high) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = size_t(y) * w + x;
    const float m = mag[o];
    uint8_t s = 0;
    if (m > low) {
        const int d = dir[o];
        const int dx = c_off[d][0], dy = c_off[d][1];
        const float ma = mag[size_t(clampi(y + dy, 0, h - 1)) * w + clampi(x + dx, 0, w - 1)];
        const float mb = mag[size_t(clampi(y - dy, 0, h - 1)) * w + clampi(x - dx, 0, w - 1)];
        if (m > ma && m >= mb) s = m > high ? 2 : 1;
    }
    state[o] = s;
}

constexpr int HX = 32, HY = 8;

__global__ void __launch_bounds__(HX* HY)
    hysteresis_kernel(uint8_t* __restrict__ state, int w, int h, int* changed) {
    __shared__ uint8_t s[HY + 2][HX + 2];
    const int x0 = blockIdx.x * HX - 1, y0 = blockIdx.y * HY - 1;
    const int tid = threadIdx.y * HX + threadIdx.x;
    for (int i = tid; i < (HX + 2) * (HY + 2); i += HX * HY) {
        const int ly = i / (HX + 2), lx = i % (HX + 2);
        const int gx = x0 + lx, gy = y0 + ly;
        s[ly][lx] = (gx >= 0 && gy >= 0 && gx < w && gy < h) ? state[size_t(gy) * w + gx] : 0;
    }
    __syncthreads();

    const int lx = threadIdx.x + 1, ly = threadIdx.y + 1;
    const int x = blockIdx.x * HX + threadIdx.x, y = blockIdx.y * HY + threadIdx.y;
    const bool inside = x < w && y < h;
    bool promoted = false;
    for (;;) {
        bool local = false;
        if (inside && s[ly][lx] == 1) {
            const bool strong_nb = s[ly - 1][lx - 1] == 2 || s[ly - 1][lx] == 2 || s[ly - 1][lx + 1] == 2 ||
                                   s[ly][lx - 1] == 2 || s[ly][lx + 1] == 2 || s[ly + 1][lx - 1] == 2 ||
                                   s[ly + 1][lx] == 2 || s[ly + 1][lx + 1] == 2;
            if (strong_nb) {
                s[ly][lx] = 2;
                local = promoted = true;
            }
        }
        if (!__syncthreads_or(local)) break;
    }
    if (promoted) {
        state[size_t(y) * w + x] = 2;
        *changed = 1;
    }
}

__global__ void finalize_kernel(const uint8_t* __restrict__ state, uint8_t* __restrict__ edges, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x)
        edges[i] = state[i] == 2 ? 255 : 0;
}

}  // namespace

int canny(const float* gray, uint8_t* edges, int w, int h, int radius, float low, float high,
          const CannyBuffers& b, cudaStream_t stream) {
    gaussian_constant(gray, b.tmp, b.blurred, w, h, radius, stream);
    const dim3 blk(32, 8), grd(div_up(w, 32), div_up(h, 8));
    grad_kernel<<<grd, blk, 0, stream>>>(b.blurred, b.mag, b.dir, w, h);
    CUDA_CHECK_LAUNCH();
    nms_kernel<<<grd, blk, 0, stream>>>(b.mag, b.dir, b.state, w, h, low, high);
    CUDA_CHECK_LAUNCH();

    const dim3 hgrd(div_up(w, HX), div_up(h, HY));
    int iters = 0, changed = 1;
    while (changed) {
        CUDA_CHECK(cudaMemsetAsync(b.d_changed, 0, sizeof(int), stream));
        hysteresis_kernel<<<hgrd, dim3(HX, HY), 0, stream>>>(b.state, w, h, b.d_changed);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpyAsync(&changed, b.d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++iters;
    }
    const size_t n = size_t(w) * h;
    finalize_kernel<<<int(std::min<size_t>((n + 255) / 256, 65535)), 256, 0, stream>>>(b.state, edges, n);
    CUDA_CHECK_LAUNCH();
    return iters;
}
