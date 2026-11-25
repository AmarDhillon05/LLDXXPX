// Gaussian v4: v3 with the weights in __constant__ memory.
//
// Every thread in a warp reads the same tap weight at the same time, which is
// exactly the access pattern the constant cache is built for: one broadcast
// per warp, no load/store unit traffic, and the value can be used directly as
// an FFMA operand (c[bank][offset]) instead of occupying a register.
#include "gaussian_tiled_impl.cuh"

namespace {
__constant__ float c_gauss[2 * kMaxGaussRadius + 1];

struct ConstantWeights {
    __device__ __forceinline__ float operator()(int i) const { return c_gauss[i]; }
};
}  // namespace

void gaussian_constant_set_weights(const float* h_weights1d, int radius) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, h_weights1d, sizeof(float) * (2 * radius + 1)));
}

void gaussian_constant(const float* in, float* tmp, float* out, int w, int h, int radius,
                       cudaStream_t stream) {
    gauss_tiled::dispatch(radius, in, tmp, out, w, h, ConstantWeights{}, stream);
}
