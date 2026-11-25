// Gaussian v3: separable + shared-memory tiles with halo.
//
// Each input pixel is fetched from DRAM/L2 once per pass into shared memory
// (plus a 2R halo per tile row/column) instead of 2R+1 times through the
// cache hierarchy. Weights are still read from global memory. See
// gaussian_tiled_impl.cuh for the tiling scheme.
#include "gaussian_tiled_impl.cuh"

namespace {
struct GlobalWeights {
    const float* __restrict__ k;
    __device__ __forceinline__ float operator()(int i) const { return k[i]; }
};
}  // namespace

void gaussian_tiled(const float* in, float* tmp, float* out, int w, int h,
                    const float* d_weights1d, int radius, cudaStream_t stream) {
    gauss_tiled::dispatch(radius, in, tmp, out, w, h, GlobalWeights{d_weights1d}, stream);
}
