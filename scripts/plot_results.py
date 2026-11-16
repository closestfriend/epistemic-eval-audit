#!/usr/bin/env python3
"""Generate publication-grade figures in results/figures/ from experimental JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 180,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_redteam() -> None:
    base_p = RESULTS / "baseline_redteam_rates.json"
    hard_p = RESULTS / "hardened_redteam_rates.json"
    if not (base_p.exists() and hard_p.exists()):
        return

    base = json.loads(base_p.read_text())
    hard = json.loads(hard_p.read_text())
    attacks = sorted(base["mask"]["by_attack"].keys())
    y = np.arange(len(attacks))

    fig, axes = plt.subplots(1, 3, figsize=(15, 7.5), sharey=True)
    for ax, bname, title in zip(
        axes,
        ["mask", "syceval", "darkbench"],
        ["MASK (n=560)", "SycEval (n=560)", "DarkBench (n=480)"],
    ):
        nat_vals = [base[bname]["by_attack"][a]["false_pass_rate"] * 100 for a in attacks]
        hrd_vals = [hard[bname]["by_attack"][a]["false_pass_rate"] * 100 for a in attacks]
        ax.barh(y - 0.18, nat_vals, height=0.35, label="Native Scorer", color="#d95f02")
        ax.barh(y + 0.18, hrd_vals, height=0.35, label="Hardened Scorer", color="#1b9e77")
        ax.set_title(title)
        ax.set_xlim(0, 108)
        ax.set_xlabel("False-Pass Rate (%)")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(attacks)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    fig.suptitle("Scorer Red-Teaming: 16 Degenerate & Adversarial Attacks (Native vs. Hardened)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_redteam_exploitability.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_redteam_exploitability.png")


def plot_judge_meta() -> None:
    base_p = RESULTS / "judge_meta_baseline.json"
    hard_p = RESULTS / "judge_meta_hardened.json"
    if not (base_p.exists() and hard_p.exists()):
        return

    base = json.loads(base_p.read_text())
    hard = json.loads(hard_p.read_text())

    metrics = [
        ("Krippendorff's α", base["krippendorff_alpha"], hard["krippendorff_alpha"]),
        ("Test–Retest Consistency", base.get("test_retest_consistency", 0.0), hard.get("test_retest_consistency", 0.0)),
        ("Position Consistency (1 - Flip)", 1.0 - base["position_flip_rate_mask"], 1.0 - hard["position_flip_rate_mask"]),
    ]
    labels = [m[0] for m in metrics]
    nat_v = [m[1] for m in metrics]
    hrd_v = [m[2] for m in metrics]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - 0.18, nat_v, width=0.35, label="Native Scorers", color="#d95f02")
    ax.bar(x + 0.18, hrd_v, width=0.35, label="Hardened Scorers", color="#1b9e77")
    ax.axhline(0.85, color="#555555", linestyle="--", linewidth=1, label="Target Threshold (0.85)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Reliability / Consistency")
    ax.set_title(f"Multi-Judge Meta-Evaluation (n={base['n_items']}, 3 Model Families)")
    ax.legend(loc="lower right")
    for i, (nv, hv) in enumerate(zip(nat_v, hrd_v)):
        ax.text(i - 0.18, nv + 0.02, f"{nv:.2f}", ha="center", fontsize=9)
        ax.text(i + 0.18, hv + 0.02, f"{hv:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_judge_meta_reliability.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_judge_meta_reliability.png")


def plot_anchors() -> None:
    anc_p = RESULTS / "anchor_calibration.json"
    if not anc_p.exists():
        return
    data = json.loads(anc_p.read_text())
    benchmarks = list(data.keys())
    configs = list(data[benchmarks[0]].keys())

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(15, 5), sharey=True)
    if len(benchmarks) == 1:
        axes = [axes]
    x = np.arange(len(configs))
    for ax, bname in zip(axes, benchmarks):
        nat = [data[bname][c]["native_pass_rate"] * 100 for c in configs]
        hrd = [data[bname][c]["hardened_pass_rate"] * 100 for c in configs]
        ax.bar(x - 0.18, nat, width=0.35, label="Native", color="#d95f02")
        ax.bar(x + 0.18, hrd, width=0.35, label="Hardened", color="#1b9e77")
        ax.set_title(bname.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Pass Rate (%)")
    axes[0].legend(loc="upper left")
    fig.suptitle("4-Anchor Calibration: Null, Adversarial, Oracle & 4 Adaption Target Models", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_anchor_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_anchor_calibration.png")


def plot_implicit() -> None:
    imp_p = RESULTS / "implicit_evasion_rates.json"
    if not imp_p.exists():
        return
    data = json.loads(imp_p.read_text())
    suites = list(data["suites"].keys())
    chosen = [data["suites"][s]["chosen_pass_rate"] * 100 for s in suites]
    rejected = [data["suites"][s]["rejected_false_pass_rate"] * 100 for s in suites]
    x = np.arange(len(suites))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - 0.18, chosen, width=0.35, label="Chosen (Honest Non-Completion)", color="#1b9e77")
    ax.bar(x + 0.18, rejected, width=0.35, label="Rejected (Plausible Hallucination)", color="#d95f02")
    ax.set_xticks(x)
    ax.set_xticklabels(suites)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title(f"Implicit Non-Completion Audit (n={data['n_pairs']} pairs)")
    ax.legend()
    for i, (cv, rv) in enumerate(zip(chosen, rejected)):
        ax.text(i - 0.18, cv + 2, f"{cv:.1f}%", ha="center", fontsize=8.5)
        ax.text(i + 0.18, rv + 2, f"{rv:.1f}%", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_implicit_evasion.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_implicit_evasion.png")


def main() -> None:
    plot_redteam()
    plot_judge_meta()
    plot_anchors()
    plot_implicit()


if __name__ == "__main__":
    main()

