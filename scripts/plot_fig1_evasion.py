#!/usr/bin/env python3
"""Figure 1: the evasion arm (experiment 05), baseline -> evasion, per model.

Left panel: MASK's headline not-lying rate, 1 - P(lie) (native `native`).
Right panel: strict honesty (native `native_strict_honest`).
Baseline = experiment 03 (results/anchor_details/mask__<model>.jsonl);
evasion = experiment 05 (results/runs/20261115T034832_evasion_prompt/details/<model>.jsonl).
All values are computed from the per-item files; the asserts pin them to Table 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "results" / "anchor_details"
EVA_DIR = ROOT / "results" / "runs" / "20261115T034832_evasion_prompt" / "details"
OUT = ROOT / "results" / "figures" / "fig1_evasion_arm.png"

MODELS = [  # (file stem, display label), top to bottom
    ("glm-53-flash", "glm-53-flash"),
    ("deepseek-v4-flash", "deepseek-v4-flash"),
    ("qwen3.8-27b", "qwen3.8-27b"),
    ("gemma-4-26b-a4b-it", "gemma-4-26b"),
]
# Table 3 values (percent): (not-lying base, not-lying evasion, strict base, strict evasion)
TABLE3 = {
    "glm-53-flash": (80.0, 90.0, 75.0, 60.0),
    "deepseek-v4-flash": (45.0, 77.5, 42.5, 27.5),
    "qwen3.8-27b": (30.0, 72.5, 22.5, 15.0),
    "gemma-4-26b-a4b-it": (47.5, 90.0, 2.5, 10.0),
}

BASE_COLOR = "#2a78d6"  # categorical slot 1 (validated: dataviz validate_palette.js, light, #ffffff)
EVA_COLOR = "#eb6834"  # categorical slot 2
INK = "#1f1f1e"
MUTED = "#6b6a66"
GRID = "#e6e5e1"


def rate(path: Path, field: str) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    vals = [float(r[field]) for r in rows]
    assert len(vals) == 40, (path, len(vals))
    return 100.0 * sum(vals) / len(vals)


def main() -> None:
    data = {}
    for stem, _ in MODELS:
        b, e = BASE_DIR / f"mask__{stem}.jsonl", EVA_DIR / f"{stem}.jsonl"
        vals = (
            rate(b, "native"),
            rate(e, "native"),
            rate(b, "native_strict_honest"),
            rate(e, "native_strict_honest"),
        )
        for got, want in zip(vals, TABLE3[stem]):
            assert abs(got - want) < 1e-9, (stem, vals, TABLE3[stem])
        data[stem] = vals

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True)
    panels = [
        ("MASK headline score, 1 − P(lie)", 0, 1),
        ("Strict honesty", 2, 3),
    ]
    ys = list(range(len(MODELS)))[::-1]
    for ax, (title, bi, ei) in zip(axes, panels):
        for y, (stem, _) in zip(ys, MODELS):
            b, e = data[stem][bi], data[stem][ei]
            ax.annotate(
                "",
                xy=(e, y),
                xytext=(b, y),
                arrowprops=dict(arrowstyle="-|>", color="#9a9893", lw=1.6, shrinkA=5, shrinkB=5,
                                mutation_scale=9),
            )
            ax.scatter([b], [y], s=46, color=BASE_COLOR, zorder=3, edgecolors="white", linewidths=1.5)
            ax.scatter([e], [y], s=46, color=EVA_COLOR, zorder=3, edgecolors="white", linewidths=1.5)
            d = e - b
            ax.text(103, y, f"{d:+.1f}".replace("-", "−"), va="center", ha="left", fontsize=8.5,
                    color=INK)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _, label in MODELS], color=INK)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BASE_COLOR, markersize=7, label="Baseline system prompt"),
        plt.Line2D([], [], marker="o", ls="", color=EVA_COLOR, markersize=7, label="+ one evasion line"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.03), labelcolor=INK)
    fig.subplots_adjust(left=0.2, right=0.93, top=0.84, bottom=0.2, wspace=0.22)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"wrote {OUT}")
    for stem, _ in MODELS:
        print(stem, data[stem])


if __name__ == "__main__":
    main()
