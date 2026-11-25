#include <nppi.h>

#include "cuda_utils.cuh"
#include "npp_baseline.hpp"

#define NPP_CHECK(expr)                                                                \
    do {                                                                               \
        NppStatus st__ = (expr);                                                       \
        if (st__ < 0) {                                                                \
            std::fprintf(stderr, "NPP error %d at %s:%d\n", int(st__), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

namespace nppb {

static NppStreamContext ctx_for(cudaStream_t stream) {
    static bool init = false;
    static NppStreamContext base{};
    if (!init) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
        base.nCudaDeviceId = dev;
        base.nMultiProcessorCount = p.multiProcessorCount;
        base.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
        base.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
        base.nSharedMemPerBlock = p.sharedMemPerBlock;
        base.nCudaDevAttrComputeCapabilityMajor = p.major;
        base.nCudaDevAttrComputeCapabilityMinor = p.minor;
        init = true;
    }
    NppStreamContext c = base;
    c.hStream = stream;
    CUDA_CHECK(cudaStreamGetFlags(stream, &c.nStreamFlags));
    return c;
}

void gauss(const float* in, float* out, int w, int h, const float* d_taps, int ntaps,
           cudaStream_t stream) {
    const NppiSize sz{w, h};
    const int step = w * int(sizeof(float));
    NPP_CHECK(nppiFilterGaussAdvancedBorder_32f_C1R_Ctx(in, step, sz, NppiPoint{0, 0}, out, step, sz,
                                                        ntaps, d_taps, NPP_BORDER_REPLICATE,
                                                        ctx_for(stream)));
}

void sobel_xy(const float* in, float* gx, float* gy, int w, int h, cudaStream_t stream) {
    const NppiSize sz{w, h};
    const int step = w * int(sizeof(float));
    const NppStreamContext c = ctx_for(stream);
    NPP_CHECK(nppiFilterSobelVertBorder_32f_C1R_Ctx(in, step, sz, NppiPoint{0, 0}, gx, step, sz,
                                                    NPP_BORDER_REPLICATE, c));
    NPP_CHECK(nppiFilterSobelHorizBorder_32f_C1R_Ctx(in, step, sz, NppiPoint{0, 0}, gy, step, sz,
                                                     NPP_BORDER_REPLICATE, c));
}

void bilateral(const float* in, float* out, int w, int h, int radius, float sigma_color,
               float sigma_space, cudaStream_t stream) {
    const NppiSize sz{w, h};
    const int step = w * int(sizeof(float));
    NPP_CHECK(nppiFilterBilateralGaussBorder_32f_C1R_Ctx(in, step, sz, NppiPoint{0, 0}, out, step, sz,
                                                         radius, 1, sigma_color * sigma_color,
                                                         sigma_space * sigma_space,
                                                         NPP_BORDER_REPLICATE, ctx_for(stream)));
}

void histogram(const uint8_t* in, int w, int h, int32_t* d_hist, cudaStream_t stream) {
    static Npp8u* buf = nullptr;
    static size_t buf_size = 0;
    const NppiSize sz{w, h};
    const NppStreamContext c = ctx_for(stream);
    size_t need = 0;
    NPP_CHECK(nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(sz, 257, &need, c));
    if (need > buf_size) {
        if (buf) CUDA_CHECK(cudaFree(buf));
        CUDA_CHECK(cudaMalloc(&buf, need));
        buf_size = need;
    }
    NPP_CHECK(nppiHistogramEven_8u_C1R_Ctx(in, w, sz, d_hist, 257, 0, 256, buf, c));
}

}  // namespace nppb
