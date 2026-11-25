#!/usr/bin/env bash
# Collect the profiler evidence referenced in docs/WRITEUP.md.
#   scripts/profile.sh            # needs ncu + nsys on PATH, and usually sudo/
#                                 # NVreg_RestrictProfilingToAdminUsers=0 for ncu counters
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/profile

# --- Nsight Compute: one launch of every kernel at 4K, key metrics as CSV ---
METRICS=gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_global_ld.sum,\
smsp__inst_executed_pipe_xu.sum,\
smsp__sass_inst_executed_op_global_atom.sum,\
smsp__sass_inst_executed_op_shared_atom.sum,\
launch__registers_per_thread,launch__shared_mem_per_block_static

ncu --metrics "$METRICS" --csv --page raw \
    --launch-skip 0 --launch-count 200 \
    ./build/bench --sizes 4k --reps 1 --skip-cpu --csv results/profile/bench_ncu.csv \
    > results/profile/ncu_metrics_4k.csv
echo "wrote results/profile/ncu_metrics_4k.csv"

# Full report for the GUI (roofline, source-level stalls thanks to -lineinfo).
ncu --set full -k regex:'row_kernel|col_kernel|naive_2d|fused_kernel|bilateral_tiled|hist_' \
    --launch-count 12 -o results/profile/ncu_full_4k -f \
    ./build/bench --sizes 4k --reps 1 --skip-cpu --csv /dev/null

# --- Nsight Systems: copy/compute overlap timeline ---
nsys profile -t cuda,nvtx -o results/profile/streams_timeline -f true \
    ./build/bench_streams --frames 200 --size 1080p --streams 1,4 \
    --csv results/profile/streams_nsys.csv --timeline /dev/null
echo "open results/profile/streams_timeline.nsys-rep in nsys-ui and screenshot the"
echo "pinned_streams_4 NVTX range -> docs/img/nsys_timeline.png"
