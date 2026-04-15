"""
r1_label_noise_figure.py  —  Publication figure for Phase 3A
=============================================================

Generates Figure 3 (or new panel): D₂(p) for both architectures
as a function of label-noise fraction.

Input:  data/main/revision1/label_noise_sweep.json
Output: paper/figures/revision1/label_noise_d2.png
        paper/figures/revision1/label_noise_d2.pdf
"""

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    data_path = os.path.join(
        repo_root, "data", "main", "revision1", "label_noise_sweep.json")
    fig_dir = os.path.join(repo_root, "paper", "figures", "revision1")
    os.makedirs(fig_dir, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    # ── Extract ──────────────────────────────────────────────
    cnn = data["conditions"]["cnn_cifar"]
    mlp = data["conditions"]["mlp_cifar_w85"]

    noise_levels = cnn["noise_levels"]

    cnn_d2_mean = [cnn["results"][str(p)]["d2_mean"] for p in noise_levels]
    cnn_d2_std  = [cnn["results"][str(p)]["d2_std"]  for p in noise_levels]
    mlp_d2_mean = [mlp["results"][str(p)]["d2_mean"] for p in noise_levels]
    mlp_d2_std  = [mlp["results"][str(p)]["d2_std"]  for p in noise_levels]

    # ── Figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.4, 2.8))  # PRL single-column width

    ax.errorbar(noise_levels, cnn_d2_mean, yerr=cnn_d2_std,
                fmt='o-', color='#D64545', markersize=5, capsize=3,
                linewidth=1.4, label='CNN/CIFAR (30% EoS)')
    ax.errorbar(noise_levels, mlp_d2_mean, yerr=mlp_d2_std,
                fmt='s-', color='#2E6EB5', markersize=5, capsize=3,
                linewidth=1.4, label='MLP 269K/CIFAR (90% EoS)')

    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Label-noise fraction $p$', fontsize=9)
    ax.set_ylabel('Correlation dimension $D_2$', fontsize=9)
    ax.legend(fontsize=7, loc='center left')
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 5.5)
    ax.tick_params(labelsize=8)

    # Light grid
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()

    png_path = os.path.join(fig_dir, "label_noise_d2.png")
    pdf_path = os.path.join(fig_dir, "label_noise_d2.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # ── Print summary for manuscript ─────────────────────────
    print("\nCNN/CIFAR at 30% EoS:")
    for p, m, s in zip(noise_levels, cnn_d2_mean, cnn_d2_std):
        print(f"  p={p:.2f}: D₂ = {m:.2f} ± {s:.2f}")

    print("\nMLP 269K at 90% EoS:")
    for p, m, s in zip(noise_levels, mlp_d2_mean, mlp_d2_std):
        print(f"  p={p:.2f}: D₂ = {m:.2f} ± {s:.2f}")

    # Monotonicity check
    cnn_diffs = np.diff(cnn_d2_mean)
    mlp_diffs = np.diff(mlp_d2_mean)
    print(f"\nCNN D₂ monotonically decreasing: {all(d <= 0 for d in cnn_diffs)}")
    print(f"MLP D₂ monotonically increasing: {all(d >= 0 for d in mlp_diffs)}")
    print(f"CNN total drop: {cnn_d2_mean[0]:.2f} → {cnn_d2_mean[-1]:.2f} "
          f"(Δ = {cnn_d2_mean[-1] - cnn_d2_mean[0]:.2f})")
    print(f"MLP total rise: {mlp_d2_mean[0]:.2f} → {mlp_d2_mean[-1]:.2f} "
          f"(Δ = {mlp_d2_mean[-1] - mlp_d2_mean[0]:.2f})")


if __name__ == "__main__":
    main()
