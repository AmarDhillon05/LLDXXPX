# GPU image-processing pipeline + performance study

CUDA C++ implementations of classic image filters, each written in several
versions so every optimization can be measured on its own. Includes a CPU
reference (checked against OpenCV), NPP and OpenCV baselines, a streaming
benchmark (pageable vs pinned vs multi-stream), plotting, and profiler scripts.

| Operation | Versions (one file per kernel group) |
|---|---|
| Gaussian blur | `gaussian_naive.cu` (2D, (2r+1)² taps) → `gaussian_separable.cu` (two 1D passes) → `gaussian_tiled.cu` (shared-memory tiles + halo) → `gaussian_constant.cu` (+ `__constant__` weights) → `gaussian_texture.cu` (texture path, stretch) |
| Sobel | `sobel.cu`: naive and shared-memory tiled; gx, gy, magnitude and threshold fused in one kernel |
| Bilateral | `bilateral.cu`: naive (`expf`, clamped global loads) vs tiled (shared tile, constant-memory window table, `__expf`) |
| Histogram + equalization | `histogram.cu`: global atomics → per-block shared privatization → per-warp privatization; warp-shuffle scan for the CDF/LUT; LUT apply |
| Fused pipeline | `pipeline.cu`: RGBA → gray → blur → Sobel → threshold as 4 separate kernels vs 1 fused kernel |
| Canny (stretch) | `canny.cu`: blur, gradient + direction, NMS + double threshold, tile-local iterative hysteresis |
| Streams | `apps/bench_streams.cu`: 1000-frame batch, H2D/compute/D2H overlap with pinned memory |

Everything is single-channel `float32` (values in 0–255) except the RGBA8 pipeline
input, the 8-bit histogram, and the 8-bit edge maps. All borders use replicate
(clamp-to-edge), the same as OpenCV's `BORDER_REPLICATE`.

## Status: what has been verified

* **Builds**: the whole project (kernels, NPP baseline, OpenCV paths, video demo)
  compiles with CUDA 13.5 for sm_80/86/89 and GCC 13.
* **CPU reference vs OpenCV 4.6**: all checks pass (`check_cpu_vs_opencv`):
  Gaussian, Sobel and bilateral at about 142–145 dB PSNR, with histogram,
  `equalizeHist`, Sobel threshold and the full edge pipeline matching
  bit-for-bit, at 1920×1080 and 1013×701.
* **GPU kernels: not run yet.** The development container had no GPU, so the
  GPU tests and benchmarks are written and compiled but have not been run.
  The repo contains **no GPU numbers** until you run `scripts/run_all.sh` on a
  machine with an NVIDIA GPU. The plots and `results/summary.md` come only
  from that run.

## Build and run

Requirements: CUDA toolkit ≥ 12, CMake ≥ 3.24, a C++17 compiler. Optional:
OpenCV (baselines, tests against OpenCV, video demo), NPP (ships with the
toolkit), Python 3 + matplotlib for plots.

```bash
scripts/run_all.sh                 # build, test, benchmark 1080p/4K/8K, 1000 frames, plots
QUICK=1 scripts/run_all.sh         # ~1 minute version
scripts/profile.sh                 # Nsight Compute metrics + Nsight Systems timeline
```

Or by hand:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build -j
./build/test_correctness                       # every GPU version vs CPU (and OpenCV, NPP)
./build/bench --sizes 1080p,4k,8k              # -> results/bench.csv, results/device.json
./build/bench_streams --frames 1000 --streams 1,2,4,8   # -> results/streams.csv, timeline.csv
python3 scripts/plot.py                        # -> results/plots/*.png, results/summary.md
./build/demo_video in.mp4 edges.mp4            # video in, edge video out, prints FPS
```

Useful flags: `--radius 7 --sigma 2.5` (15×15 Gaussian), `--bil-radius 5`,
`--threshold 40`, `--image photo.png` (test on a real photo instead of the
synthetic test card), `--skip-cpu` (the 8K single-thread CPU baselines take a
few minutes), and `--cpu-only` (CPU baselines only, no GPU needed).

## Layout

```
include/            cuda_utils.cuh (CUDA_CHECK, DeviceBuffer, GpuTimer), kernels.cuh (all launchers),
                    cpu_reference.hpp, opencv_ref.hpp, npp_baseline.hpp, image.hpp, app_util.hpp
src/kernels/        one file per kernel version / group
src/cpu/            single-threaded reference = ground truth + "1 CPU thread" baseline
src/common/         image I/O (stb_image), synthetic test card, metrics, NPP wrappers
src/apps/           test_correctness, check_cpu_vs_opencv, bench, bench_streams, demo_video
scripts/            run_all.sh, profile.sh, plot.py
docs/WRITEUP.md     the performance study: what each optimization does and how to prove it
results/            benchmark output (generated)
```

## How correctness is tested

`test_correctness` runs every GPU version on a 1920×1080 test card and an odd
1013×701 one (partial tiles, uchar4 tails) and compares against the CPU
reference. When built with OpenCV, it also compares against OpenCV directly.

* float outputs: PSNR > 60 dB and max |diff| ≤ 0.5
* binary edge maps: ≤ 0.01% of pixels differ (FMA contraction can flip pixels
  that sit exactly on the threshold); Canny ≤ 0.1%
* histograms, equalization LUT, equalized image: exact match

The CPU reference is built with `-ffp-contract=off`, so it doesn't change with
compiler flags. The grayscale conversion on the GPU uses `__fmul_rn`/`__fadd_rn`
so it is bit-identical to the CPU.
