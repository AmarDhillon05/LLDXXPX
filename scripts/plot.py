#!/usr/bin/env python3
"""Turn the benchmark CSVs into the plots used by the write-up.

    python3 scripts/plot.py [--results results] [--out results/plots]

Reads (whichever exist):
    results/bench.csv     from ./build/bench
    results/streams.csv   from ./build/bench_streams
    results/timeline.csv  from ./build/bench_streams
    results/device.json   from ./build/bench
Writes PNGs into --out.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

# Validated categorical palette (fixed order) + neutral inks.
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
SIZE_ORDER = ["1080p", "4k", "8k"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
    "ytick.color": TEXT2, "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def sizes_in(rows):
    seen = {r["size"] for r in rows}
    return [s for s in SIZE_ORDER if s in seen] + sorted(seen - set(SIZE_ORDER))


def label_bars(ax, bars, fmt):
    for b in bars:
        w = b.get_width()
        ax.text(w, b.get_y() + b.get_height() / 2, " " + fmt(w), va="center", ha="left", fontsize=8, color=TEXT)


def plot_op_times(rows, out):
    """One figure per op: ms per implementation, one panel per resolution."""
    ops = sorted({r["op"] for r in rows if r["op"] != "equalize_lut_only"})
    sizes = sizes_in(rows)
    for op in ops:
        sub = [r for r in rows if r["op"] == op]
        n_impl = max(len([r for r in sub if r["size"] == s]) for s in sizes)
        fig, axes = plt.subplots(1, len(sizes), figsize=(max(7.5, 4.4 * len(sizes)), 0.38 * n_impl + 1.8),
                                 squeeze=False)
        for ax, size in zip(axes[0], sizes):
            rs = sorted([r for r in sub if r["size"] == size], key=lambda r: -float(r["ms"]))
            if not rs:
                ax.set_visible(False)
                continue
            names = [r["impl"] for r in rs]
            ms = [float(r["ms"]) for r in rs]
            colors = [C[0] if r["device"] == "gpu" else C[1] for r in rs]
            bars = ax.barh(names, ms, color=colors, height=0.6)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(PLAIN)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{size}", loc="left")
            ax.set_xlabel("ms (log scale, lower is better)")
            ax.grid(axis="y", visible=False)
            label_bars(ax, bars, lambda v: f"{v:.3g}")
            ax.set_xlim(min(ms) / 2, max(ms) * 6)
        handles = [plt.Rectangle((0, 0), 1, 1, color=C[0]), plt.Rectangle((0, 0), 1, 1, color=C[1])]
        fig.suptitle(f"{op}: time per implementation", x=0.01, ha="left", fontweight="bold", color=TEXT)
        fig.legend(handles, ["GPU", "CPU"], loc="lower right", ncol=2, bbox_to_anchor=(0.99, 0.0))
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        fig.savefig(os.path.join(out, f"ms_{op}.png"))
        plt.close(fig)


def best(rows, op, size, pred):
    rs = [float(r["ms"]) for r in rows if r["op"] == op and r["size"] == size and pred(r)]
    return min(rs) if rs else None


def plot_speedup(rows, out, size):
    ops = ["gaussian", "sobel", "bilateral", "histogram", "equalize", "pipeline", "canny"]
    labels, vs_1t, vs_cv = [], [], []
    for op in ops:
        g = best(rows, op, size, lambda r: r["device"] == "gpu" and not r["impl"].startswith(("npp", "opencv")))
        c1 = best(rows, op, size, lambda r: r["device"] == "cpu" and r["impl"].endswith("_1t")
                  and r["impl"] != "cpu_separable_1t")
        cv = best(rows, op, size, lambda r: r["device"] == "cpu" and r["impl"].startswith("opencv"))
        if g is None or c1 is None:
            continue
        labels.append(op)
        vs_1t.append(c1 / g)
        vs_cv.append(cv / g if cv else 0)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(labels))
    wdt = 0.38
    b1 = ax.bar([i - wdt / 2 - 0.01 for i in x], vs_1t, wdt, color=C[0], label="vs single-thread CPU (naive)")
    b2 = ax.bar([i + wdt / 2 + 0.01 for i in x], vs_cv, wdt, color=C[1], label="vs OpenCV CPU (SIMD, all cores)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("speedup (x, log scale)")
    ax.set_title(f"Best GPU kernel vs CPU, {size} (kernel time only, data resident)", loc="left")
    ax.grid(axis="x", visible=False)
    for bars in (b1, b2):
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.0f}x",
                        ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"speedup_{size}.png"))
    plt.close(fig)


def plot_bandwidth(rows, out):
    impls = [("gaussian", "v1_naive_2d"), ("gaussian", "v2_separable"), ("gaussian", "v3_tiled_smem"),
             ("gaussian", "v4_tiled_constant"), ("gaussian", "v5_texture"), ("sobel", "tiled_fused_mag_thr"),
             ("histogram", "warp_privatized"), ("pipeline", "separate_kernels"), ("pipeline", "fused_kernel")]
    sizes = sizes_in([r for r in rows if r["device"] == "gpu"])
    if not sizes:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    n = len(sizes)
    wdt = 0.8 / n
    names = [f"{o}\n{i}" for o, i in impls]
    for k, size in enumerate(sizes):
        vals = []
        for op, impl in impls:
            r = [r for r in rows if r["op"] == op and r["impl"] == impl and r["size"] == size]
            vals.append(float(r[0]["pct_copy_peak"]) if r else 0)
        ax.bar([i + (k - (n - 1) / 2) * wdt for i in range(len(impls))], vals, wdt * 0.92, color=C[k], label=size)
    ax.axhline(100, color=TEXT2, linewidth=1, linestyle="--")
    ax.text(len(impls) - 0.5, 101, "device-to-device copy", ha="right", va="bottom", fontsize=8, color=TEXT2)
    ax.set_xticks(range(len(impls)), names, fontsize=7.5)
    ax.set_ylabel("achieved DRAM bandwidth, % of copy peak")
    ax.set_title("How close each kernel gets to its own memory roofline", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(title=None, ncol=n, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bandwidth.png"))
    plt.close(fig)


def plot_streams(rows, out):
    if not rows:
        return
    stages = {r["mode"]: float(r["ms_per_frame"]) for r in rows if r["mode"].startswith("stage_")}
    runs = [r for r in rows if not r["mode"].startswith("stage_")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    names = [f"{r['mode']}\n({r['streams']} stream{'s' if r['streams'] != '1' else ''})" for r in runs]
    ms = [float(r["ms_per_frame"]) for r in runs]
    colors = [C[1] if r["mode"].startswith("pageable") else C[0] for r in runs]
    bars = a1.barh(names, ms, color=colors, height=0.6)
    a1.invert_yaxis()
    for b, r in zip(bars, runs):
        a1.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {b.get_width():.2f} ms  ({float(r['fps']):.0f} FPS)",
                va="center", fontsize=8, color=TEXT)
    a1.set_xlim(0, max(ms) * 1.45)
    a1.set_xlabel("ms per frame, end to end (lower is better)")
    r0 = runs[0]
    a1.set_title(f"{r0['frames']} frames, {r0['width']}x{r0['height']}, {r0['pipeline']} pipeline", loc="left")
    a1.grid(axis="y", visible=False)
    if stages:
        keys = ["stage_h2d", "stage_kernel", "stage_d2h"]
        lab = ["H2D copy", "kernel", "D2H copy"]
        vals = [stages.get(k, 0) for k in keys]
        b2 = a2.bar(lab, vals, color=[C[0], C[1], C[2]], width=0.6)
        for b in b2:
            a2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}", ha="center",
                    va="bottom", fontsize=8, color=TEXT)
        a2.set_ylabel("ms per frame")
        a2.set_title("Stage cost in isolation", loc="left")
        a2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "streams.png"))
    plt.close(fig)


def plot_timeline(rows, out):
    if not rows:
        return
    colors = {"H2D": C[0], "kernel": C[1], "D2H": C[2]}
    streams = sorted({int(r["stream"]) for r in rows})
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(streams) + 1.6))
    for r in rows:
        s, a, b = int(r["stream"]), float(r["start_ms"]), float(r["end_ms"])
        ax.barh(s, max(b - a, 1e-3), left=a, height=0.6, color=colors[r["op"]], edgecolor="white", linewidth=1)
        if r["op"] == "kernel" and b - a > 0.15:
            ax.text((a + b) / 2, s, r["frame"], ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks(streams, [f"stream {s}" for s in streams])
    ax.invert_yaxis()
    ax.set_xlabel("time (ms)")
    ax.set_title("Copy/compute overlap across streams (cudaEvent timestamps; numbers = frame)", loc="left")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
    ax.legend(handles, list(colors), ncol=3, loc="upper right", bbox_to_anchor=(1, 1.18))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timeline.png"))
    plt.close(fig)


def write_summary(bench, streams, dev, path):
    """Headline ratios as Markdown, so the write-up can quote measured numbers."""
    L = ["# Measured summary", ""]
    if dev:
        L += [f"- GPU: **{dev.get('gpu', '?')}** (sm {dev.get('sm', '?')}), "
              f"D2D copy {dev.get('copy_bandwidth_gbps', 0):.0f} GB/s "
              f"(theoretical {dev.get('theoretical_bandwidth_gbps', 0):.0f} GB/s)",
              f"- PCIe pinned H2D {dev.get('pcie_h2d_pinned_gbps', 0):.1f} GB/s, "
              f"pageable H2D {dev.get('pcie_h2d_pageable_gbps', 0):.1f} GB/s",
              f"- CPU threads: {dev.get('cpu_threads', '?')}, OpenCV {dev.get('opencv_version', 'n/a')}", ""]

    def ms(op, impl, size):
        r = [x for x in bench if x["op"] == op and x["impl"] == impl and x["size"] == size]
        return float(r[0]["ms"]) if r else None

    def pct(op, impl, size):
        r = [x for x in bench if x["op"] == op and x["impl"] == impl and x["size"] == size]
        return float(r[0]["pct_copy_peak"]) if r else None

    def ratio(a, b):
        return f"{a / b:.1f}x" if a and b else "n/a"

    if bench:
        L += ["| metric | " + " | ".join(sizes_in(bench)) + " |", "|---|" + "---|" * len(sizes_in(bench))]
        rows = [
            ("Gaussian v1 naive GPU vs 1-thread CPU 2D", lambda s: ratio(ms("gaussian", "cpu_2d_1t", s), ms("gaussian", "v1_naive_2d", s))),
            ("Gaussian v2 separable vs v1 naive", lambda s: ratio(ms("gaussian", "v1_naive_2d", s), ms("gaussian", "v2_separable", s))),
            ("Gaussian v4 tiled+constant vs v1 naive", lambda s: ratio(ms("gaussian", "v1_naive_2d", s), ms("gaussian", "v4_tiled_constant", s))),
            ("Gaussian v4 vs v3 (constant vs global weights)", lambda s: ratio(ms("gaussian", "v3_tiled_smem", s), ms("gaussian", "v4_tiled_constant", s))),
            ("Gaussian v4 % of copy bandwidth (16 B/px)", lambda s: f"{pct('gaussian', 'v4_tiled_constant', s):.0f}%" if pct('gaussian', 'v4_tiled_constant', s) else "n/a"),
            ("Gaussian v4 vs NPP", lambda s: ratio(ms("gaussian", "npp_gauss_advanced", s), ms("gaussian", "v4_tiled_constant", s))),
            ("Bilateral tiled+__expf vs naive", lambda s: ratio(ms("bilateral", "naive_expf", s), ms("bilateral", "tiled_fast_expf", s))),
            ("Bilateral tiled ms", lambda s: f"{ms('bilateral', 'tiled_fast_expf', s):.2f} ms" if ms('bilateral', 'tiled_fast_expf', s) else "n/a"),
            ("Bilateral CPU 1-thread ms", lambda s: f"{ms('bilateral', 'cpu_1t', s):.0f} ms" if ms('bilateral', 'cpu_1t', s) else "n/a"),
            ("Histogram shared vs global atomics", lambda s: ratio(ms("histogram", "global_atomics", s), ms("histogram", "shared_privatized", s))),
            ("Histogram warp-private vs shared", lambda s: ratio(ms("histogram", "shared_privatized", s), ms("histogram", "warp_privatized", s))),
            ("Pipeline fused vs separate kernels", lambda s: ratio(ms("pipeline", "separate_kernels", s), ms("pipeline", "fused_kernel", s))),
            ("Pipeline fused % of copy bandwidth (5 B/px)", lambda s: f"{pct('pipeline', 'fused_kernel', s):.0f}%" if pct('pipeline', 'fused_kernel', s) else "n/a"),
        ]
        for name, fn in rows:
            L.append(f"| {name} | " + " | ".join(fn(s) for s in sizes_in(bench)) + " |")
        L.append("")
    if streams:
        st = {r["mode"]: float(r["ms_per_frame"]) for r in streams if r["mode"].startswith("stage_")}
        runs = [r for r in streams if not r["mode"].startswith("stage_")]
        page = [float(r["ms_per_frame"]) for r in runs if r["mode"] == "pageable_sync"]
        pin = [float(r["ms_per_frame"]) for r in runs if r["mode"] == "pinned_streams"]
        if st:
            tot = sum(st.values())
            L += [f"- Stage cost per frame: H2D {st.get('stage_h2d', 0):.3f} ms, kernel {st.get('stage_kernel', 0):.3f} ms, "
                  f"D2H {st.get('stage_d2h', 0):.3f} ms -> kernel is **{100 * st.get('stage_kernel', 0) / tot:.0f}%** of the serial total"]
        if page and pin:
            L += [f"- End to end: pageable sync {page[0]:.3f} ms/frame, best pinned+streams {min(pin):.3f} ms/frame "
                  f"-> **{page[0] / min(pin):.2f}x**"]
            if st:
                L += [f"- Overlap efficiency: best streamed {min(pin):.3f} ms/frame vs bottleneck stage "
                      f"{max(st.values()):.3f} ms (1.00 = perfect overlap): {max(st.values()) / min(pin):.2f}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "plots")
    os.makedirs(out, exist_ok=True)
    bench = read_csv(os.path.join(a.results, "bench.csv"))
    if bench:
        plot_op_times(bench, out)
        for s in sizes_in(bench):
            plot_speedup(bench, out, s)
        plot_bandwidth(bench, out)
    plot_streams(read_csv(os.path.join(a.results, "streams.csv")), out)
    plot_timeline(read_csv(os.path.join(a.results, "timeline.csv")), out)
    dev_path = os.path.join(a.results, "device.json")
    dev = json.load(open(dev_path)) if os.path.exists(dev_path) else {}
    write_summary(bench, read_csv(os.path.join(a.results, "streams.csv")), dev,
                  os.path.join(a.results, "summary.md"))
    print("plots written to", out)


if __name__ == "__main__":
    main()
