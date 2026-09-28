"""
experiment_scalability.py — Scalability Study
==============================================
Varies team size M in {2, 3, 4, 6} for baseline and v2.
Studies whether coordination benefit scales with team size.

Usage:
  python experiment_scalability.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from core import run_multiple, T_FINAL, N_RUNS

OUTDIR = "./outputs"
os.makedirs(OUTDIR, exist_ok=True)

TEAM_SIZES   = [2, 3, 4, 6]
ALPHA        = 0.4   # Fig. 7 / outputs/fig_scalability.png were generated with α=0.4.
                     # Main results (Table I, Figs. 1–4) use α=0.2 — see simulation_main.py.
N_RUNS_SCALE = 3

COLORS = {"baseline": "#4C72B0", "v2": "#55A868"}
LABELS = {"baseline": "Baseline (α=0)", "v2": f"V2 (α={ALPHA})"}


def run_scalability():
    results = {"baseline": {}, "v2": {}}
    for M in TEAM_SIZES:
        for mode in ["baseline", "v2"]:
            alpha_val = ALPHA if mode == "v2" else 0.0
            print(f"  M={M}, mode={mode}...")
            results[mode][M] = run_multiple(
                mode=mode, alpha=alpha_val,
                n_runs=N_RUNS_SCALE, n_uavs=M
            )
    return results


def plot_scalability(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── Final regret vs M ──
    ax = axes[0]
    for mode in ["baseline", "v2"]:
        means = [results[mode][M]["regret"][0][-1] for M in TEAM_SIZES]
        stds  = [results[mode][M]["regret"][1][-1] for M in TEAM_SIZES]
        ax.errorbar(TEAM_SIZES, means, yerr=stds, marker='o', lw=2,
                    capsize=4, color=COLORS[mode], label=LABELS[mode])
    ax.set_xlabel("Team size M", fontsize=12)
    ax.set_ylabel("Final Time-Averaged Regret", fontsize=12)
    ax.set_title("Regret vs. Team Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_xticks(TEAM_SIZES)

    # ── Last ROI discovery time vs M ──
    ax = axes[1]
    for mode in ["baseline", "v2"]:
        last_times = [max(results[mode][M]["roi_times"][k][0]
                          for k in results[mode][M]["roi_times"])
                      for M in TEAM_SIZES]
        ax.plot(TEAM_SIZES, last_times, marker='o', lw=2,
                color=COLORS[mode], label=LABELS[mode])
    ax.set_xlabel("Team size M", fontsize=12)
    ax.set_ylabel("Time to Discover Last ROI", fontsize=12)
    ax.set_title("Last ROI Discovery vs. Team Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_xticks(TEAM_SIZES)

    # ── Mean spatial overlap vs M ──
    ax = axes[2]
    for mode in ["baseline", "v2"]:
        overlaps = [np.mean(results[mode][M]["overlap"][0][-100:]) for M in TEAM_SIZES]
        ax.plot(TEAM_SIZES, overlaps, marker='o', lw=2,
                color=COLORS[mode], label=LABELS[mode])
    ax.set_xlabel("Team size M", fontsize=12)
    ax.set_ylabel("Mean Spatial Overlap (last 100 steps)", fontsize=12)
    ax.set_title("Spatial Overlap vs. Team Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_xticks(TEAM_SIZES)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_scalability.png", dpi=150)
    plt.close()
    print("Saved: fig_scalability.png")


def print_summary(results):
    print("\n── Scalability Summary ──")
    print(f"{'Condition':<20} {'M':>3} {'Final Regret':>14} {'Last ROI':>10} {'Overlap':>10}")
    print("-" * 62)
    for M in TEAM_SIZES:
        for mode in ["baseline", "v2"]:
            mu_r = results[mode][M]["regret"][0][-1]
            last = max(results[mode][M]["roi_times"][k][0]
                       for k in results[mode][M]["roi_times"])
            ov   = np.mean(results[mode][M]["overlap"][0][-100:])
            print(f"{LABELS[mode]:20} {M:3}  {mu_r:12.4f}  {last:8.1f}  {ov:8.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Scalability experiment")
    print("=" * 60)
    results = run_scalability()
    plot_scalability(results)
    print_summary(results)
    print(f"\nAll figures saved to {OUTDIR}/")
