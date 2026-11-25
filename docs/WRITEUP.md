# Performance study: a GPU image-processing pipeline

> **Status.** This document explains what each optimization does, the reasoning
> about bytes and FLOPs behind it, the static evidence collected at build time,
> and the profiler metric that should confirm it. The **measured** numbers come
> from `scripts/run_all.sh` and `scripts/profile.sh`, which write
> `results/summary.md`, `results/plots/*.png` and `results/profile/*`. The
> machine this was written on had no GPU. Every number below marked
> *expected* is a prediction, not a measurement. Replace those with your results.

Default parameters: 15×15 Gaussian (r = 7, σ = 2.5); Sobel threshold 40;
bilateral r = 5 (circular window, 81 taps), σ_color = 25, σ_space = 3;
float32 single channel unless stated otherwise.

## 0. Method

* **Timing.** cudaEvent pairs around each launch with data already on the
  device: 3 warm-up runs, then the median of 30. CPU baselines are wall-clock
  medians. End-to-end numbers (section 6) use wall clock around the whole batch.
* **Roofline yardstick.** `bench` measures a 512 MB device-to-device
  `cudaMemcpy` (read + write) and uses it as "100% bandwidth". This is the
  bandwidth a real kernel can reach, about 85–92% of the datasheet figure. The
  theoretical figure is also recorded in `device.json`.
* **Bytes per pixel** for the bandwidth plots is each design's *own* minimum
  DRAM traffic: 16 B/px for a two-pass separable blur, 8 B/px for a
  single-pass one, 5 B/px for the fused pipeline. "% of peak" therefore shows
  how close the kernel gets to its own roofline. It doesn't rank the designs;
  wall time does that.
* **Baselines.** `cpu_*_1t` is the single-threaded reference, plain C++ with
  `-O3` (scalar clamps prevent most auto-vectorization). `opencv_Nt` is OpenCV
  with SIMD and all cores, a strong CPU baseline. NPP is the vendor GPU
  baseline. `cv::cuda` is benchmarked too if OpenCV was built with CUDA.

## 1. Gaussian blur: four steps, four separate effects

| Version | Taps/px | DRAM bytes/px (ideal) | What changes |
|---|---|---|---|
| v1 naive 2D | 225 | 8 | baseline: every tap is an L1/L2 load + FFMA, weights in global |
| v2 separable | 2 × 15 | 16 | 7.5× fewer FLOPs and loads, one extra round trip for the intermediate image |
| v3 tiled | 2 × 15 | 16 | each input is loaded into shared memory once per tile (plus a 2r halo) |
| v4 constant | 2 × 15 | 16 | weights are read from the constant bank (broadcast) |
| v5 texture | 2 × 15 | 16 | no shared memory; the texture unit does the 2D caching and clamping |

**v1 → v2.** At r = 7, v1 issues 225 loads and 225 FFMAs per pixel. Even at
100% L1 hit rate it is bound by load/store-unit throughput. Its `dram__bytes`
stays close to 8 B/px, but `l1tex__t_sector_hit_rate` is very high and
`sm__throughput` is the limiter. Going separable divides the instruction count
by 7.5 and doubles DRAM traffic. *Expected: 3–6× faster at 4K.* Check in
`ncu_metrics_4k.csv`: `smsp__inst_executed_op_global_ld.sum` falls by about 7×.

**v2 → v3.** v2's horizontal pass is already cache-friendly. Its vertical
pass reads `in[(y+j)*w + x]`, which is coalesced across x, but each value is
re-fetched 15 times through L1. v3 stages a 128×8 row tile (or 32×(32+2r)
column strip) in shared memory with coalesced loads, then reads 15 neighbours
from shared memory without bank conflicts: consecutive threads read
consecutive words, and each thread produces 4 outputs spaced 32 apart. The
global-load count per output drops from 15 to about (1 + 2r/tile). Check:
`smsp__inst_executed_op_global_ld.sum` drops sharply, and
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` should be close to
0. *Expected: 1.3–2× over v2, landing at 70–90% of copy bandwidth.*

**v3 → v4 (constant weights).** Static evidence from `ptxas -v` (sm_86, r = 7):

| Kernel | Registers | Static smem |
|---|---|---|
| `row_kernel<7, GlobalWeights>` (v3) | 50 | 4544 B |
| `row_kernel<7, ConstantWeights>` (v4) | 39 | 4544 B |
| `col_kernel<7, GlobalWeights>` (v3) | 48 (4 B spill at r = 15) | 5888 B |
| `col_kernel<7, ConstantWeights>` (v4) | 40 | 5888 B |

Since the tap loop is fully unrolled, the v3 compiler hoists all 15 weights
into registers. That costs about 10 extra registers, plus a small spill at
r = 15. v4 can feed the weights to FFMA directly as constant-bank operands
(`c[0x3][...]`), so it doesn't need those registers. Confirm this in the SASS
with `cuobjdump -sass`. Both versions are memory-bound at this point, so **expect a
small difference** (0–10%), mostly from occupancy. That is a real finding:
constant memory matters most when weights are indexed in a non-unrolled loop
or don't fit in registers. To see it, compare
`sm__warps_active.avg.pct_of_peak_sustained_active` between the two kernels.

**v5 texture.** No border math and no shared memory. On Volta and later, the
texture cache and L1 share the same SRAM, so this should land between v2 and
v4. Expected: close to v2's vertical pass, because tap reuse across threads
still goes through the cache rather than explicit shared memory.

**Versus the CPU.** *Expected (from the project brief):* v1 should be 50–150×
faster than `cpu_2d_1t` at 4K. The more honest comparison is v4 against
OpenCV on all cores, which is the realistic alternative. See
`results/plots/speedup_4k.png`.

**Versus NPP.** `nppiFilterGaussAdvancedBorder_32f_C1R` with the same taps
and border. It is a single-pass 2D filter, so it reads 8 B/px but does
(2r+1)² work per pixel. Expect v4 to be competitive with it or faster at
r = 7.

## 2. Sobel: fusing the epilogue

Both versions compute gx, gy, `sqrtf(gx²+gy²)` and the threshold in one
kernel and write a single byte per pixel: 4 B in + 1 B out = **5 B/px**. The
unfused design (NPP's `FilterSobelHoriz/Vert` + magnitude + threshold) writes
and re-reads two float images, about 21 B/px. That is the `npp_gx_gy_only`
row in the bench, and it already costs 16 B/px *before* computing the
magnitude.

Tiled vs naive: the 3×3 stencil reuses data so heavily that L1 already
absorbs naive's 8 neighbour loads. Expect tiled to be only marginally faster,
or no faster. Both should sit near the bandwidth limit. A small or zero
difference here is a useful result in its own right: shared-memory tiling pays
off when the reuse distance is larger than what L1 holds comfortably, as in
the blur's 15-row column pass.

## 3. Bilateral: compute-bound, so optimize instructions

81 taps per pixel at r = 5, each needing a `exp()` of the range difference.
The spatial weight depends only on the offset. Bytes per pixel are the same as
a blur, but the arithmetic is about 10× heavier, so this kernel is
**compute-bound** and the levers are different:

1. **Spatial weights and window shape precomputed.** The host builds a list of
   (shared-memory offset, spatial weight) pairs for the circular window in
   `__constant__` memory. Every thread walks the same list in lock-step, so
   the constant cache broadcasts each entry. There is no divergence and no
   wasted work on the 40 of 121 square-window taps outside the circle. The
   naive kernel computes `sqrtf` and a second `expf` per tap and branches.
2. **`__expf`.** One `FMUL` plus `MUFU.EX2` on the SFU, instead of `expf`'s
   accurate range reduction (about 10–20 instructions). Check:
   `smsp__inst_executed_pipe_xu.sum` (the SFU pipe) is about 1 per tap in the
   tiled kernel. Total instructions drop by about 3–5×. Accuracy stays well
   inside the 60 dB bar, because `__expf`'s relative error (about 2⁻²¹ scaled
   by the argument) is far below the 8-bit output quantization.
3. **Shared-memory tile.** A (16 + 2r)² tile removes the clamps and global
   loads from the inner loop.

*Expected (brief):* about 1–3 ms at 4K for the tiled kernel, versus hundreds
of ms for one CPU thread (measured on the dev container: 965 ms at 1080p on
one core, so about 4 s at 4K). The interesting ncu number is
`sm__throughput` (should be high) next to `dram__throughput` (should be low).
This confirms the kernel is compute-bound, and that more memory tricks would
not help.

NPP's bilateral uses a square window, so it does about 50% more taps. The test
reports its PSNR for information only.

## 4. Histogram and equalization: atomics contention

A histogram of a natural image has a few very hot bins. With
`global_atomics`, every pixel performs an L2 atomic and those serialize per
address. The test card has large flat regions, which makes this worse, and
that is on purpose.

* **Shared privatization:** atomics resolve in the SM's shared memory. Each
  block then performs at most 256 global atomics. Pixels are read 4 per
  `uchar4` load, and the grid is capped at about 8 blocks per SM, so merge
  traffic is negligible. Check: `smsp__sass_inst_executed_op_global_atom.sum`
  falls from *n* to ≤ 256 × blocks.
* **Per-warp privatization (8 KB smem/block):** removes inter-warp contention
  on the same shared-memory address. It helps most on low-entropy images. On
  high-entropy ones, expect a small or zero difference compared with shared.
* **Scan → LUT:** only 256 elements, so a single block does a warp-shuffle
  inclusive scan (`__shfl_up_sync`) and a cross-warp fix-up, and produces the
  `cv::equalizeHist` LUT exactly: first occupied bin maps to 0,
  `round_half_even(255·(cdf−cdf_min)/(N−cdf_min))`. This takes microseconds
  and is launch-latency bound (`equalize_lut_only` in the bench).
* **Apply:** pure bandwidth, 2 B/px, LUT in shared memory, `uchar4` I/O.

## 5. Fused pipeline: the cost of global round trips

Four separate kernels (gray, blur-row, blur-col, Sobel) move about
**29 B/px** through DRAM: 4 (RGBA) + 4+4 (gray) + 4+4 (row) + 4+4 (blurred)
+ 1 (edges). The fused kernel moves **5 B/px** plus halo re-reads, which
mostly hit L2.

What fusion costs (static, from `ptxas -v` on sm_86):

* `fused_kernel<7>`: 36 registers, **20,504 B** static shared memory per
  256-thread block. At about 100 KB of shared memory per SM on sm_86, that
  allows at most 4 blocks, or 1024 of 1536 threads (67% theoretical
  occupancy). The separate kernels use ≤ 5.9 KB.
* Redundant halo work: a 32×32 output tile computes the row blur on
  48×34 = 1632 positions (1.6× the useful work) and the column blur on
  34×34 = 1156 (1.13×).
* Border handling is exact: blurred values in the Sobel halo are evaluated at
  the clamped coordinate. That matches the separate path's
  "replicate-at-every-stage" semantics, and the test checks fused vs separate
  directly.

*Expected:* 2–3× faster than separate kernels. If you see less, check
`dram__bytes_read.sum` (should be about 5–6 B/px) against
`sm__throughput`. If the fused kernel is SM-bound, the redundant halo work is
the price, and a larger tile (64×32) or a register-blocked column pass are the
next steps.

## 6. Streams: when PCIe is the real bottleneck

For a 1080p RGBA frame, the H2D copy is 8.3 MB and the D2H copy is 2.1 MB.
Over PCIe 4.0 ×16 (about 25 GB/s pinned in practice), that is about 0.33 ms +
0.08 ms. The fused kernel needs about 10 MB of DRAM traffic, roughly 0.02 ms
on a 500 GB/s GPU. **The transfer is about 10–20× longer than the
computation.** That is the key point of this section, and
`bench_streams` prints it directly ("Per-frame stage cost" and
`results/streams.png`, right-hand panel).

Modes measured (1000 frames, cycled from 8 distinct host frames, outputs
verified against the CPU reference):

| Mode | Expected behaviour |
|---|---|
| `pageable_sync` | driver stages through a bounce buffer: lower bandwidth, nothing overlaps |
| `pageable_streams` | **no better**: pageable `cudaMemcpyAsync` is effectively synchronous |
| `pinned_sync` | full PCIe bandwidth, but H2D, kernel and D2H still run back to back |
| `pinned_streams` ×N | H2D(i+1) ‖ kernel(i) ‖ D2H(i−1); bound by the slowest stage (H2D) |

With 2 copy engines (`asyncEngineCount` is printed), H2D and D2H also
overlap each other. The ideal streamed time per frame is therefore the
maximum of (H2D, kernel, D2H), not their sum. `summary.md` reports
**overlap efficiency** = max-stage / measured, where 1.0 means the transfer
fully hides the computation. *Expected (brief):* 1.5–2.5× end to end from
pinned memory + streams over pageable memory. Gains flatten after 2–4 streams
because there is only one H2D engine.

**Timeline.** `results/plots/timeline.png` is built from cudaEvent
timestamps recorded in each stream for the first 16 frames. It is an
approximation: an event can't tell queueing apart from execution. The exact
picture comes from `scripts/profile.sh`, which records an Nsight Systems
trace with NVTX ranges per mode. Open `results/profile/streams_timeline.nsys-rep`,
zoom into `pinned_streams_4`, and screenshot it into `docs/img/`.

## 7. Canny (stretch)

Blur → gradient + 4-way direction → NMS + double threshold → hysteresis.
Hysteresis is a connected-components problem. Each launch propagates "strong"
through a 32×8 tile in shared memory until that tile converges
(`__syncthreads_or`), so one launch crosses a whole tile rather than one
pixel. The host relaunches until a global flag stays clear. Updates are
monotonic (weak → strong), so stale halo reads only delay convergence and
can't change the fixed point. The bench prints the number of launches. Long
diagonal weak chains are the worst case. The usual next step would be a
union-find or a global worklist.

`cv::Canny` is included as a timing reference only. It works on 8-bit input,
uses its own Sobel and a different NMS tie rule, so its output isn't
bit-comparable.

## 8. Checklist: profiler evidence per claim

| Claim | Metric (ncu) | Expected direction |
|---|---|---|
| separable cuts work | `smsp__inst_executed_op_global_ld.sum`, `sm__throughput` | ↓ about 7× from v1 to v2 |
| tiling cuts global loads | `smsp__inst_executed_op_global_ld.sum`, `..._shared_ld.sum` | global ↓, shared ↑ from v2 to v3 |
| no bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | ≈ 0 for v3/v4/fused |
| blur is memory-bound | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 70–90% |
| bilateral is compute-bound | `sm__throughput` high, `dram__throughput` low; `smsp__inst_executed_pipe_xu` | SFU ≈ 1 per tap |
| privatization removes global atomics | `smsp__sass_inst_executed_op_global_atom.sum` | *n* → ≤ 256 × blocks |
| fusion removes round trips | `dram__bytes_read.sum + dram__bytes_write.sum` per pixel | about 29 → about 5–6 B/px |
| copies dominate end to end | Nsight Systems timeline, `stage_*` rows | kernel ≪ H2D |

## Appendix: CPU reference vs OpenCV (measured)

`./build/check_cpu_vs_opencv` (OpenCV 4.6.0, x86-64 container), 1920×1080:

```
[PASS] grayscale                PSNR 154.78 dB  max|d| 3.05e-05
[PASS] gaussian (separable)     PSNR 144.98 dB  max|d| 9.16e-05
[PASS] sobel magnitude          PSNR 145.31 dB  max|d| 0.000107
[PASS] sobel threshold          0 / 2073600 px differ
[PASS] bilateral                PSNR 142.30 dB  max|d| 0.000214
[PASS] histogram                0 / 256 bins differ
[PASS] equalizeHist             0 / 2073600 px differ
[PASS] edge pipeline            0 / 2073600 px differ
```

The same checks pass at 1013×701.
