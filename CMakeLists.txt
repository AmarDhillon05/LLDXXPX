cmake_minimum_required(VERSION 3.24)
project(gpu_image_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Default: build for the GPU in this machine. Override with
#   -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(GIP_WITH_OPENCV "Use OpenCV for baselines, tests and the video demo (if found)" ON)
option(GIP_WITH_NPP "Use NPP as a vendor baseline (if found)" ON)

find_package(CUDAToolkit REQUIRED)

# -lineinfo lets Nsight Compute attribute stalls to source lines.
set(GIP_CUDA_FLAGS -lineinfo --expt-relaxed-constexpr -Xptxas=-warn-spills)

# ---------------------------------------------------------------------------
# Core library: CPU reference + every GPU kernel version.
# ---------------------------------------------------------------------------
add_library(gip_cpu STATIC src/common/image.cpp src/cpu/cpu_reference.cpp)
target_include_directories(gip_cpu PUBLIC include third_party/stb)
# The CPU reference doubles as ground truth: forbid FMA contraction so its
# results don't depend on the host compiler / -march.
target_compile_options(gip_cpu PRIVATE -O3 -ffp-contract=off)

add_library(gip_kernels STATIC
  src/kernels/gaussian_naive.cu
  src/kernels/gaussian_separable.cu
  src/kernels/gaussian_tiled.cu
  src/kernels/gaussian_constant.cu
  src/kernels/gaussian_texture.cu
  src/kernels/sobel.cu
  src/kernels/bilateral.cu
  src/kernels/histogram.cu
  src/kernels/pipeline.cu
  src/kernels/canny.cu)
target_include_directories(gip_kernels PUBLIC include)
target_compile_options(gip_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GIP_CUDA_FLAGS}>)
target_link_libraries(gip_kernels PUBLIC CUDA::cudart)

# ---------------------------------------------------------------------------
# Optional baselines
# ---------------------------------------------------------------------------
set(GIP_EXTRA_DEFS "")
set(GIP_EXTRA_LIBS "")

if(GIP_WITH_NPP AND TARGET CUDA::nppif AND TARGET CUDA::nppist AND TARGET CUDA::nppc)
  add_library(gip_npp STATIC src/common/npp_baseline.cu)
  target_include_directories(gip_npp PUBLIC include)
  target_link_libraries(gip_npp PUBLIC CUDA::nppif CUDA::nppist CUDA::nppc CUDA::cudart)
  list(APPEND GIP_EXTRA_DEFS HAVE_NPP=1)
  list(APPEND GIP_EXTRA_LIBS gip_npp)
  message(STATUS "NPP baseline: enabled")
else()
  message(STATUS "NPP baseline: disabled")
endif()

set(GIP_HAVE_OPENCV OFF)
if(GIP_WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core imgproc)
  if(OpenCV_FOUND)
    set(GIP_HAVE_OPENCV ON)
    list(APPEND GIP_EXTRA_DEFS HAVE_OPENCV=1)
    list(APPEND GIP_EXTRA_LIBS ${OpenCV_LIBS})
    message(STATUS "OpenCV ${OpenCV_VERSION}: enabled")
    # cv::cuda baselines if OpenCV was built with CUDA (opencv_contrib).
    find_package(OpenCV QUIET COMPONENTS cudafilters cudaimgproc)
    if(OpenCV_cudafilters_FOUND OR TARGET opencv_cudafilters)
      list(APPEND GIP_EXTRA_DEFS HAVE_OPENCV_CUDA=1)
      list(APPEND GIP_EXTRA_LIBS opencv_cudafilters opencv_cudaimgproc)
      message(STATUS "OpenCV CUDA modules: enabled")
    endif()
  else()
    message(STATUS "OpenCV: not found (tests compare against the CPU reference only)")
  endif()
endif()

function(gip_app name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE gip_kernels gip_cpu ${GIP_EXTRA_LIBS})
  target_compile_definitions(${name} PRIVATE ${GIP_EXTRA_DEFS})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GIP_CUDA_FLAGS}>)
  if(GIP_HAVE_OPENCV)
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()
endfunction()

gip_app(test_correctness src/apps/test_correctness.cu)
gip_app(bench src/apps/bench.cu)
gip_app(bench_streams src/apps/bench_streams.cu)

# CPU-only: checks the CPU reference against OpenCV. Runs without a GPU.
if(GIP_HAVE_OPENCV)
  add_executable(check_cpu_vs_opencv src/apps/check_cpu_vs_opencv.cpp)
  target_link_libraries(check_cpu_vs_opencv PRIVATE gip_cpu ${OpenCV_LIBS})
  target_include_directories(check_cpu_vs_opencv PRIVATE ${OpenCV_INCLUDE_DIRS})

  find_package(OpenCV QUIET COMPONENTS videoio)
  if(OpenCV_videoio_FOUND OR TARGET opencv_videoio)
    gip_app(demo_video src/apps/demo_video.cu)
    target_link_libraries(demo_video PRIVATE opencv_videoio)
  endif()
endif()

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
if(TARGET check_cpu_vs_opencv)
  add_test(NAME cpu_vs_opencv COMMAND check_cpu_vs_opencv)
endif()
