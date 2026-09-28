"""
experiment_tau_sweep.py — Belief/Policy Update Rate Experiment
==============================================================
Varies tau_GP and tau_P across coordination conditions.
Studies how update frequency interacts with coordination strength.

Mendoza's suggestion: study tau before dynamic environments.

Usage:
  python experiment_tau_sweep.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from core import run_multiple, T_FINAL, N_RUNS

OUTDIR = "./outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Sweep parameters
TAU_VALUES  = [10, 20, 50, 100]
ALPHA_VALUES = [0.0, 0.2, 0.4]   # 0.0 = baseline
N_RUNS_SWEEP = 3                  # fewer runs for speed

COLORS_TAU = {10: "#2196F3", 20: "#4CAF50", 50: "#FF9800", 100: "#E91E63"}
COLORS_ALPHA = {0.0: "#4C72B0", 0.2: "#DD8452", 0.4: "#55A868"}


def run_tau_sweep():
    """
    For each (alpha, tau) combination, run N_RUNS_SWEEP seeds.
    Returns nested dict: results[mode][alpha][tau] = run_multiple output
    """
    results = {"v2": {}, "baseline": {}}

    # Baseline across tau values (alpha=0, mode=baseline)
    print("Running BASELINE across tau values...")
    results["baseline"] = {}
    for tau in TAU_VALUES:
        print(f"  tau={tau}...")
        results["baseline"][tau] = run_multiple(
            mode="baseline", alpha=0.0, n_runs=N_RUNS_SWEEP,
            tau_gp=tau, tau_p=tau
        )

    # V2 across alpha and tau
    print("\nRunning V2 across alpha and tau values...")
    for alpha in ALPHA_VALUES:
        if alpha == 0.0:
            continue   # covered by baseline
        results["v2"][alpha] = {}
        for tau in TAU_VALUES:
            print(f"  alpha={alpha}, tau={tau}...")
            results["v2"][alpha][tau] = run_multiple(
                mode="v2", alpha=alpha, n_runs=N_RUNS_SWEEP,
                tau_gp=tau, tau_p=tau
            )
    return results


def plot_tau_regret(results):
    """
    For each alpha value, plot final regret vs. tau.
    Shows whether coordination benefit degrades at high tau.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: final regret vs tau for each alpha
    ax = axes[0]
    # Baseline
    base_means = [results["baseline"][tau]["regret"][0][-1] for tau in TAU_VALUES]
    base_stds  = [results["baseline"][tau]["regret"][1][-1] for tau in TAU_VALUES]
    ax.errorbar(TAU_VALUES, base_means, yerr=base_stds, marker='o', lw=2,
                capsize=4, color="#4C72B0", label="Baseline (α=0)", linestyle='--')

    for alpha in [0.2, 0.4]:
        means = [results["v2"][alpha][tau]["regret"][0][-1] for tau in TAU_VALUES]
        stds  = [results["v2"][alpha][tau]["regret"][1][-1] for tau in TAU_VALUES]
        ax.errorbar(TAU_VALUES, means, yerr=stds, marker='s', lw=2,
                    capsize=4, color=COLORS_ALPHA[alpha], label=f"V2 (α={alpha})")

    ax.set_xlabel("Update period τ (τGP = τP)", fontsize=12)
    ax.set_ylabel("Final Time-Averaged Regret", fontsize=12)
    ax.set_title("Regret vs. Update Period", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)

    # Right: last ROI discovery time vs tau
    ax = axes[1]
    base_last = [max(results["baseline"][tau]["roi_times"][k][0]
                     for k in results["baseline"][tau]["roi_times"])
                 for tau in TAU_VALUES]
    ax.plot(TAU_VALUES, base_last, marker='o', lw=2, linestyle='--',
            color="#4C72B0", label="Baseline (α=0)")

    for alpha in [0.2, 0.4]:
        last_times = [max(results["v2"][alpha][tau]["roi_times"][k][0]
                          for k in results["v2"][alpha][tau]["roi_times"])
                      for tau in TAU_VALUES]
        ax.plot(TAU_VALUES, last_times, marker='s', lw=2,
                color=COLORS_ALPHA[alpha], label=f"V2 (α={alpha})")

    ax.set_xlabel("Update period τ (τGP = τP)", fontsize=12)
    ax.set_ylabel("Time to Discover Last ROI", fontsize=12)
    ax.set_title("Last ROI Discovery vs. Update Period", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_tau_sweep.png", dpi=150)
    plt.close()
    print("Saved: fig_tau_sweep.png")


def plot_tau_regret_curves(results):
    """
    For a fixed alpha=0.4, show regret curves for each tau value.
    Helps visualize how update frequency affects convergence speed.
    """
    alpha = 0.4   # outputs/fig_tau_curves.png was generated with α=0.4 (not a paper figure).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    timesteps = np.arange(1, T_FINAL + 1)

    for ax, mode, title in zip(axes,
                                ["baseline", "v2"],
                                ["Baseline", f"V2 (α={alpha})"]):
        for tau in TAU_VALUES:
            if mode == "v2":
                data = results["v2"][alpha][tau]
            else:
                data = results["baseline"][tau]
            mu, std = data["regret"]
            ax.plot(timesteps, mu, color=COLORS_TAU[tau], lw=1.5, label=f"τ={tau}")
            ax.fill_between(timesteps, mu-std, mu+std, alpha=0.1, color=COLORS_TAU[tau])

        ax.set_xlabel("Timestep k", fontsize=12)
        ax.set_ylabel("Time-Averaged Regret", fontsize=12)
        ax.set_title(f"Regret Curves: {title}", fontsize=12)
        ax.legend(fontsize=10, title="Update period τ")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_tau_curves.png", dpi=150)
    plt.close()
    print("Saved: fig_tau_curves.png")


def print_summary(results):
    print("\n── Tau Sweep Summary ──")
    print(f"{'Condition':<25} {'tau':>5} {'Final Regret':>14} {'Last ROI':>10}")
    print("-" * 58)

    for tau in TAU_VALUES:
        mu = results["baseline"][tau]["regret"][0][-1]
        last = max(results["baseline"][tau]["roi_times"][k][0]
                   for k in results["baseline"][tau]["roi_times"])
        print(f"{'Baseline':25} {tau:5}  {mu:12.4f}  {last:8.1f}")

    for alpha in [0.2, 0.4]:
        for tau in TAU_VALUES:
            mu = results["v2"][alpha][tau]["regret"][0][-1]
            last = max(results["v2"][alpha][tau]["roi_times"][k][0]
                       for k in results["v2"][alpha][tau]["roi_times"])
            label = f"V2 α={alpha}"
            print(f"{label:25} {tau:5}  {mu:12.4f}  {last:8.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Tau sweep experiment")
    print("=" * 60)
    results = run_tau_sweep()
    plot_tau_regret(results)
    plot_tau_regret_curves(results)
    print_summary(results)
    print(f"\nAll figures saved to {OUTDIR}/")
