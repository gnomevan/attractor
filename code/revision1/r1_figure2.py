#!/usr/bin/env python3
"""
Generate Figure 2: D₂ vs EoS fraction for all five conditions.
Uses N=10 merged data for CNN/CIFAR, MLP/CIFAR (both widths), CNN/synthetic.
MLP/synthetic uses depth_scaling.json baseline (N=5 seeds per LR, 20 total).

Referee issue #1, #2: error bars now reflect N=10 seed variability.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_MAIN = ROOT / "data" / "main"
DATA_SUPP = ROOT / "data" / "supplemental"
OUT = ROOT / "paper" / "figures" / "figure2_cross_experiments.png"


def load_merged(path):
    """Load a merged JSON and return (fractions, means, stds)."""
    with open(path) as f:
        d = json.load(f)
    fracs = d["lr_fractions"]
    means, stds = [], []
    for i in range(len(fracs)):
        vals = d[f"lr_{i}"]["corr_dim"]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
    return np.array(fracs) * 100, np.array(means), np.array(stds)


def load_cnn_cifar(path):
    """Load CNN/CIFAR-10 data (different key: 'seeds' not 'seeds_run')."""
    with open(path) as f:
        d = json.load(f)
    fracs = d["lr_fractions"]
    means, stds = [], []
    for i in range(len(fracs)):
        vals = d[f"lr_{i}"]["corr_dim"]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
    return np.array(fracs) * 100, np.array(means), np.array(stds)


def load_mlp_synth(path):
    """Load MLP/synthetic baseline from depth_scaling.json.
    This uses absolute LRs, not EoS fractions. We plot it as a horizontal band."""
    with open(path) as f:
        d = json.load(f)
    baseline = d["depth=2 (baseline)"]
    all_vals = []
    for lr_key, vals in baseline["corr_dims"].items():
        all_vals.extend(vals)
    return np.mean(all_vals), np.std(all_vals, ddof=1)


def main():
    # Load all conditions
    x_cnn, y_cnn, e_cnn = load_cnn_cifar(DATA_MAIN / "cifar10_eos_10seeds.json")
    x_w85, y_w85, e_w85 = load_merged(DATA_MAIN / "revision1" / "cross_small_mlp_cifar_w85_seeds_merged.json")
    x_w50, y_w50, e_w50 = load_merged(DATA_MAIN / "revision1" / "cross_small_mlp_cifar_w50_seeds_merged.json")
    x_syn, y_syn, e_syn = load_merged(DATA_MAIN / "revision1" / "cross_cnn_synthetic_seeds_merged.json")
    mlp_synth_mean, mlp_synth_std = load_mlp_synth(DATA_SUPP / "depth_scaling.json")

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.8))  # PRL single-column width

    # CNN/CIFAR-10
    ax.errorbar(x_cnn, y_cnn, yerr=e_cnn, fmt='o-', color='#d62728', markersize=4,
                linewidth=1.2, capsize=2, capthick=0.8, label='CNN + CIFAR-10 (269K)', zorder=5)

    # MLP/CIFAR-10 w85 (269K)
    ax.errorbar(x_w85, y_w85, yerr=e_w85, fmt='s-', color='#1f77b4', markersize=3.5,
                linewidth=1.0, capsize=2, capthick=0.8, label='MLP + CIFAR-10 (269K)', zorder=4)

    # MLP/CIFAR-10 w50 (156K)
    ax.errorbar(x_w50, y_w50, yerr=e_w50, fmt='^-', color='#2ca02c', markersize=3.5,
                linewidth=1.0, capsize=2, capthick=0.8, label='MLP + CIFAR-10 (156K)', zorder=4)

    # CNN/synthetic (open markers)
    ax.errorbar(x_syn, y_syn, yerr=e_syn, fmt='o--', color='#d62728', markersize=3.5,
                linewidth=0.8, capsize=2, capthick=0.8, markerfacecolor='white',
                markeredgewidth=0.8, label='CNN + synthetic (269K)', zorder=3)

    # MLP/synthetic as horizontal band
    ax.axhspan(mlp_synth_mean - mlp_synth_std, mlp_synth_mean + mlp_synth_std,
               color='#7f7f7f', alpha=0.15, zorder=1)
    ax.axhline(mlp_synth_mean, color='#7f7f7f', linewidth=0.8, linestyle=':', zorder=2,
               label='MLP + synthetic (14K)')

    # Formatting
    ax.set_xlabel('Fraction of EoS threshold (%)', fontsize=8)
    ax.set_ylabel('Correlation dimension $D_2$', fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 5.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, loc='upper left', framealpha=0.9, handlelength=2.0)

    # Reference line at D₂ = 1
    ax.axhline(1.0, color='black', linewidth=0.4, linestyle='--', alpha=0.3, zorder=1)

    plt.tight_layout()
    fig.savefig(str(OUT), dpi=300, bbox_inches='tight')
    print(f"Saved: {OUT}")
    print(f"  CNN/CIFAR peak: {y_cnn.max():.2f} ± {e_cnn[np.argmax(y_cnn)]:.2f} at {x_cnn[np.argmax(y_cnn)]:.0f}% EoS")
    print(f"  MLP w50 peak:   {y_w50.max():.2f} ± {e_w50[np.argmax(y_w50)]:.2f} at {x_w50[np.argmax(y_w50)]:.0f}% EoS")
    print(f"  MLP w85 peak:   {y_w85.max():.2f} ± {e_w85[np.argmax(y_w85)]:.2f} at {x_w85[np.argmax(y_w85)]:.0f}% EoS")
    print(f"  CNN/synth peak: {y_syn.max():.2f} ± {e_syn[np.argmax(y_syn)]:.2f} at {x_syn[np.argmax(y_syn)]:.0f}% EoS")
    print(f"  MLP/synth band: {mlp_synth_mean:.2f} ± {mlp_synth_std:.2f}")


if __name__ == "__main__":
    main()
