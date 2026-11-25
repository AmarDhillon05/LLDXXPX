#!/usr/bin/env bash
# Build, test, benchmark, plot. Run from the repo root on a machine with an NVIDIA GPU.
#   scripts/run_all.sh            # full run (1080p, 4K, 8K + 1000-frame stream batch)
#   QUICK=1 scripts/run_all.sh    # fewer reps, no 8K, 200 frames
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH=${CUDA_ARCH:-native}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build build -j

mkdir -p results
echo "== correctness =="
./build/test_correctness | tee results/test_correctness.txt
[ -x build/check_cpu_vs_opencv ] && ./build/check_cpu_vs_opencv | tee results/check_cpu_vs_opencv.txt

if [ "${QUICK:-0}" = 1 ]; then
  SIZES=1080p,4k; REPS=10; CPU_REPS=1; FRAMES=200
else
  SIZES=1080p,4k,8k; REPS=30; CPU_REPS=3; FRAMES=1000
fi

echo "== kernel benchmarks =="
./build/bench --sizes "$SIZES" --reps "$REPS" --cpu-reps "$CPU_REPS" \
  --csv results/bench.csv --device-json results/device.json | tee results/bench.txt

echo "== streaming benchmark =="
./build/bench_streams --frames "$FRAMES" --size 1080p --streams 1,2,4,8 \
  --csv results/streams.csv --timeline results/timeline.csv | tee results/streams.txt
./build/bench_streams --frames "$FRAMES" --size 4k --streams 4 \
  --csv results/streams_4k.csv --timeline results/timeline_4k.csv | tee results/streams_4k.txt

echo "== plots =="
python3 scripts/plot.py --results results
