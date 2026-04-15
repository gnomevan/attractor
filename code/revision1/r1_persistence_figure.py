#!/usr/bin/env python3
"""
Generate persistence diagram figure for supplement (referee issue #11).

Three panels:
  (a) CNN/CIFAR-10 at 5% EoS — trivial (zero features)
  (b) CNN/CIFAR-10 at 30% EoS — strange attractor
  (c) MLP/CIFAR-10 (269K) at 90% EoS — strange attractor (different arch)

Points are (birth, death); distance from diagonal = lifetime.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CNN_PATH = ROOT / "data" / "supplemental" / "revision1" / "tda_cnn_cifar_diagrams.json"
MLP85_PATH = ROOT / "data" / "supplemental" / "revision1" / "tda_mlp_cifar_w85.json"
OUT = ROOT / "paper" / "figures" / "revision1" / "persistence_diagrams.png"


def plot_diagram(ax, diagrams, title, max_val=None):
    """Plot a single persistence diagram with H1 and H2."""
    h1 = np.array(diagrams.get('H1', []))
    h2 = np.array(diagrams.get('H2', []))

    # Find axis limits
    all_pts = []
    if len(h1) > 0:
        all_pts.append(h1)
    if len(h2) > 0:
        all_pts.append(h2)

    if len(all_pts) > 0:
        all_pts = np.vstack(all_pts)
        if max_val is None:
            max_val = all_pts.max() * 1.1
    else:
        max_val = 1.0

    # Diagonal
    ax.plot([0, max_val], [0, max_val], 'k-', lw=0.5, alpha=0.3)

    # Plot H2 first (behind)
    if len(h2) > 0:
        ax.scatter(h2[:, 0], h2[:, 1], s=8, alpha=0.4, c='#2ca02c',
                   edgecolors='none', label=f'$H_2$ ({len(h2)})', zorder=3)

    # Plot H1 on top
    if len(h1) > 0:
        ax.scatter(h1[:, 0], h1[:, 1], s=8, alpha=0.5, c='#1f77b4',
                   edgecolors='none', label=f'$H_1$ ({len(h1)})', zorder=4)

    if len(h1) == 0 and len(h2) == 0:
        ax.text(0.5, 0.5, 'No features', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='gray')

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel('Birth', fontsize=7)
    ax.set_ylabel('Death', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, loc='lower right', framealpha=0.9)


def main():
    # Load CNN diagrams
    with open(CNN_PATH) as f:
        cnn = json.load(f)

    # Load MLP w85 data
    with open(MLP85_PATH) as f:
        w85 = json.load(f)

    # CNN 5% EoS (lr_0 in CNN diagrams, frac=0.05)
    cnn_5 = cnn['lr_0']['diagrams']

    # CNN 30% EoS (lr_1 in CNN diagrams, frac=0.30)
    cnn_30 = cnn['lr_1']['diagrams']

    # MLP w85 90% EoS (lr_11 in w85, frac=0.90)
    # Find the index for 90% in w85
    w85_fracs = w85['lr_fractions']
    idx_90 = w85_fracs.index(0.9)
    # Use seed 0 diagrams (first in the list)
    mlp_90 = w85[f'lr_{idx_90}']['diagrams'][0]

    # Find common axis scale from the two non-trivial diagrams
    all_vals = []
    for dgm in [cnn_30, mlp_90]:
        for key in ['H1', 'H2']:
            pts = np.array(dgm.get(key, []))
            if len(pts) > 0:
                all_vals.extend(pts.flatten())
    max_val = max(all_vals) * 1.05 if all_vals else 1.0

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4))

    plot_diagram(axes[0], cnn_5,
                 '(a) CNN/CIFAR, 5% EoS\n$D_2 = 0.99$', max_val=max_val)
    plot_diagram(axes[1], cnn_30,
                 '(b) CNN/CIFAR, 30% EoS\n$D_2 = 3.67$', max_val=max_val)
    plot_diagram(axes[2], mlp_90,
                 '(c) MLP (269K)/CIFAR, 90% EoS\n$D_2 = 4.37$', max_val=max_val)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=300, bbox_inches='tight')
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
