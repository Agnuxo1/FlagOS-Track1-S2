# SPDX-License-Identifier: Apache-2.0
"""Render the README's animated figures from a benchmark summary JSON.

usage: python scripts/make_gifs.py results/bench_<stamp>_summary.json
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")
BG, FG, MUTED = "#0d1117", "#e6edf3", "#8b949e"
REF_C, OPT_C, ACC = "#6e7681", "#2f81f7", "#3fb950"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#30363d"})


def _fig(w=9.6, h=5.4):
    fig = plt.figure(figsize=(w, h), dpi=100, facecolor=BG)
    return fig


def speedup_bars(summary, op, title, fname):
    rows = [r for r in summary if r["op"] == op]
    labels = [r["case"] for r in rows]
    sp = [r["speedup"] for r in rows]
    lo = [r["speedup"] - r["speedup_min"] for r in rows]
    hi = [r["speedup_max"] - r["speedup"] for r in rows]
    fig = _fig(9.6, max(4.2, 0.36 * len(rows) + 1.6))
    ax = fig.add_axes([0.30, 0.10, 0.64, 0.78], facecolor=BG)
    n_frames = 36

    def draw(i):
        ax.clear()
        ax.set_facecolor(BG)
        f = min(1.0, (i + 1) / (n_frames - 8))
        ease = 1 - (1 - f) ** 3
        vals = [1 + (s - 1) * ease for s in sp]
        colors = [OPT_C if s >= 1 else "#f85149" for s in sp]
        ax.barh(range(len(rows)), vals, color=colors, height=0.66, zorder=3)
        if f >= 1.0:
            ax.errorbar(sp, range(len(rows)), xerr=[lo, hi], fmt="none", ecolor=FG, capsize=2, lw=0.8, zorder=4)
            for k, s in enumerate(sp):
                ax.text(s + max(sp) * 0.015, k, f"{s:.2f}×", va="center", fontsize=8.5, color=FG, zorder=5)
        ax.axvline(1.0, color=MUTED, ls="--", lw=1, zorder=2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(sp) * 1.18)
        ax.set_xlabel("speedup vs. SGLang reference  (T_ref / T_opt, RTX 3090)", fontsize=9)
        ax.grid(axis="x", color="#21262d", zorder=0)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.suptitle(title, color=FG, fontsize=13, x=0.03, ha="left", y=0.97, weight="bold")
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=60)
    anim.save(os.path.join(OUT, fname), writer=PillowWriter(fps=16))
    plt.close(fig)


def split_vocab_diagram(fname):
    """Reference: 1 program walks the row. Ours: n_split programs walk chunks, then merge."""
    fig = _fig(9.6, 4.6)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.86], facecolor=BG)
    n_blocks, n_split = 16, 8
    frames = 40

    def draw(t):
        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 50)
        ax.axis("off")
        ax.text(1, 46, "Reference: 1 program per row, serial scan", fontsize=11, color=MUTED)
        ax.text(1, 21, "Split-vocab: 8 programs per row + merge", fontsize=11, color=ACC)
        w = 92 / n_blocks
        ref_done = min(n_blocks, int(t * n_blocks / (frames - 4)))
        opt_done = min(n_blocks // n_split, int(t * n_blocks / (frames - 4)))
        for b in range(n_blocks):
            c = REF_C if b < ref_done else "#21262d"
            ax.add_patch(Rectangle((2 + b * w, 34), w - 0.6, 8, color=c))
            chunk, pos = divmod(b, n_blocks // n_split)
            c2 = OPT_C if pos < opt_done else "#21262d"
            ax.add_patch(Rectangle((2 + b * w, 9), w - 0.6, 8, color=c2))
        for s in range(n_split):
            ax.text(2 + (s + 0.5) * (92 / n_split), 5.5, f"P{s}", ha="center", fontsize=8, color=MUTED)
        ax.text(2, 31, f"time step {min(t, frames - 4)}", fontsize=8, color=MUTED)
        if opt_done >= n_blocks // n_split:
            ax.add_patch(FancyBboxPatch((76, 1), 21, 3.4, boxstyle="round,pad=0.3", color=ACC))
            ax.text(86.5, 2.7, "merge → argmax", ha="center", va="center", fontsize=8.5, color=BG, weight="bold")
        fig.suptitle("softcap_row_argmax — filling idle SMs at decode batch sizes",
                     color=FG, fontsize=13, x=0.03, ha="left", y=0.97, weight="bold")
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=80)
    anim.save(os.path.join(OUT, fname), writer=PillowWriter(fps=12))
    plt.close(fig)


def row_tiling_diagram(fname):
    """Reference: one tiny row per program. Ours: ROWS rows per program (2-D tile)."""
    fig = _fig(9.6, 4.6)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.86], facecolor=BG)
    n_rows, rows_per = 16, 8
    frames = 36

    def draw(t):
        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 50)
        ax.axis("off")
        ax.text(2, 46, "Reference: 16 programs × 1 row (D = 128)", fontsize=11, color=MUTED)
        ax.text(54, 46, "Ours: 2 programs × 8-row tile", fontsize=11, color=ACC)
        k = min(n_rows, int(t * n_rows / (frames - 6)))
        for r in range(n_rows):
            ax.add_patch(Rectangle((2, 41 - (r + 1) * 2.4), 40, 2.0, color=REF_C if r < k else "#21262d"))
            ax.text(43, 41 - (r + 1) * 2.4 + 0.6, f"P{r}", fontsize=6.5, color=MUTED)
        kk = min(2, int(t * 2 / (frames - 6)) + (1 if t > 2 else 0))
        for p in range(2):
            on = p < kk
            ax.add_patch(FancyBboxPatch((54, 41 - (p + 1) * 19.6 + 0.4), 40, 18.6, boxstyle="round,pad=0.2",
                                        fill=False, ec=ACC if on else "#30363d", lw=1.6))
            for r in range(rows_per):
                y = 41 - p * 19.6 - (r + 1) * 2.35
                ax.add_patch(Rectangle((55, y), 38, 1.9, color=OPT_C if on else "#21262d"))
            ax.text(95, 41 - p * 19.6 - 10, f"P{p}", fontsize=8, color=MUTED)
        fig.suptitle("rmsnorm_fp32 / gemma_fused_add_rmsnorm — multi-row tiles",
                     color=FG, fontsize=13, x=0.03, ha="left", y=0.97, weight="bold")
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=80)
    anim.save(os.path.join(OUT, fname), writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    split_vocab_diagram("split_vocab_argmax.gif")
    row_tiling_diagram("multirow_tiling.gif")
    if len(sys.argv) > 1:
        summary = json.load(open(sys.argv[1]))["summary"]
        speedup_bars(summary, "softcap_row_argmax", "softcap_row_argmax — speedup per case", "speedup_softcap_row_argmax.gif")
        speedup_bars(summary, "rmsnorm_fp32", "rmsnorm_fp32 — speedup per case", "speedup_rmsnorm_fp32.gif")
        speedup_bars(summary, "gemma_fused_add_rmsnorm", "gemma_fused_add_rmsnorm — speedup per case",
                     "speedup_gemma_fused_add_rmsnorm.gif")


if __name__ == "__main__":
    main()
