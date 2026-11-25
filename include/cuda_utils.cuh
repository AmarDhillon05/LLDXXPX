// Small CUDA helpers shared by every kernel file and app.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <utility>

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t err__ = (expr);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                 \
                         cudaGetErrorName(err__), __FILE__, __LINE__,             \
                         cudaGetErrorString(err__));                              \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// Check for launch errors right after a <<<>>> launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

inline int div_up(int a, int b) { return (a + b - 1) / b; }

// RAII device buffer. Move-only.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept { *this = std::move(o); }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            count_ = o.count_;
            o.ptr_ = nullptr;
            o.count_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        release();
        if (count) CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        count_ = count;
    }
    void release() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }
    void upload(const T* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// cudaEvent-based timer. Times work submitted to `stream` between start() and stop().
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
    void stop(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(stop_, s)); }
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// Replicate-border clamp, matching OpenCV BORDER_REPLICATE.
__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
