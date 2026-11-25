// Gaussian v5 (stretch): separable passes through the texture path.
//
// No shared memory at all: the texture unit handles the 2D-local caching and
// the clamp-to-edge addressing in hardware, so the kernel has no border
// arithmetic. On modern GPUs the texture cache and L1 are the same SRAM, so
// expect this to land between v2 (global, no tiling) and v4 (tiled).
//
// Inputs must be pitched allocations (cudaMallocPitch) so they can be bound
// as 2D textures.
#include <map>
#include <tuple>

#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace {
__constant__ float c_gauss_tex[2 * kMaxGaussRadius + 1];

// Texture objects are cached per (pointer, pitch, size) so the benchmark does
// not time cudaCreateTextureObject on every call.
cudaTextureObject_t get_texture(const float* ptr, size_t pitch, int w, int h) {
    static std::map<std::tuple<const float*, size_t, int, int>, cudaTextureObject_t> cache;
    auto key = std::make_tuple(ptr, pitch, w, h);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    cudaResourceDesc res{};
    res.resType = cudaResourceTypePitch2D;
    res.res.pitch2D.devPtr = const_cast<float*>(ptr);
    res.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    res.res.pitch2D.width = w;
    res.res.pitch2D.height = h;
    res.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc td{};
    td.addressMode[0] = cudaAddressModeClamp;  // == BORDER_REPLICATE
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 0;
    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res, &td, nullptr));
    cache.emplace(key, tex);
    return tex;
}

__global__ void tex_row_kernel(cudaTextureObject_t tex, float* __restrict__ out, size_t out_pitch,
                               int w, int h, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float fy = y + 0.5f;
    float acc = 0.f;
    for (int i = -radius; i <= radius; ++i)
        acc += c_gauss_tex[i + radius] * tex2D<float>(tex, x + i + 0.5f, fy);
    reinterpret_cast<float*>(reinterpret_cast<char*>(out) + y * out_pitch)[x] = acc;
}

__global__ void tex_col_kernel(cudaTextureObject_t tex, float* __restrict__ out, size_t out_pitch,
                               int w, int h, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float fx = x + 0.5f;
    float acc = 0.f;
    for (int j = -radius; j <= radius; ++j)
        acc += c_gauss_tex[j + radius] * tex2D<float>(tex, fx, y + j + 0.5f);
    reinterpret_cast<float*>(reinterpret_cast<char*>(out) + y * out_pitch)[x] = acc;
}
}  // namespace

void gaussian_texture_set_weights(const float* h_weights1d, int radius) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss_tex, h_weights1d, sizeof(float) * (2 * radius + 1)));
}

void gaussian_texture(const float* in, size_t in_pitch, float* tmp, size_t tmp_pitch, float* out,
                      size_t out_pitch, int w, int h, int radius, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(div_up(w, block.x), div_up(h, block.y));
    tex_row_kernel<<<grid, block, 0, stream>>>(get_texture(in, in_pitch, w, h), tmp, tmp_pitch, w, h,
                                               radius);
    CUDA_CHECK_LAUNCH();
    tex_col_kernel<<<grid, block, 0, stream>>>(get_texture(tmp, tmp_pitch, w, h), out, out_pitch, w,
                                               h, radius);
    CUDA_CHECK_LAUNCH();
}
