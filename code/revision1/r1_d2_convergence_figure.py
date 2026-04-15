#!/usr/bin/env python3
"""
Generate D₂(N) convergence figure for supplement.
Shows neural network conditions alongside Lorenz and MG τ=30 references.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NEURAL = ROOT / "data" / "supplemental" / "revision1" / "d2_vs_n_neural.json"
REF = ROOT / "data" / "supplemental" / "revision1" / "d2_vs_n_reference.json"
OUT = ROOT / "paper" / "figures" / "revision1" / "d2_convergence.png"


def main():
    with open(NEURAL) as f:
        neural = json.load(f)
    with open(REF) as f:
        ref = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.8))

    # Reference systems (dashed, gray tones)
    for sys_name, data in ref.items():
        ns = [r['n_points'] for r in data['results']]
        d2s = [r['d2_fixed'] for r in data['results']]
        expected = data['expected']
        label = f'{sys_name} (true $D_2$={expected})'
        color = '#888888' if sys_name == 'Lorenz' else '#bbbbbb'
        ax.plot(ns, d2s, 'x--', color=color, markersize=5, linewidth=0.8,
                label=label, zorder=2)

    # Neural conditions
    styles = {
        'cnn_cifar':     ('o-', '#d62728', 'CNN/CIFAR, 30% EoS'),
        'mlp_cifar_w85': ('s-', '#1f77b4', 'MLP 269K/CIFAR, 90% EoS'),
        'mlp_cifar_w50': ('^-', '#2ca02c', 'MLP 156K/CIFAR, 90% EoS'),
    }

    for cond_name, data in neural['conditions'].items():
        fmt, color, label = styles[cond_name]
        ns = [r['n'] for r in data['d2_vs_n'] if r['d2'] is not None]
        d2s = [r['d2'] for r in data['d2_vs_n'] if r['d2'] is not None]
        ax.plot(ns, d2s, fmt, color=color, markersize=4, linewidth=1.0,
                label=label, zorder=4)

    ax.set_xscale('log')
    ax.set_xlabel('Trajectory length $N$', fontsize=8)
    ax.set_ylabel('Correlation dimension $D_2$', fontsize=8)
    ax.set_xlim(80, 5000)
    ax.set_ylim(0, 6.5)
    ax.tick_params(labelsize=7)

    # Mark production N≈400
    ax.axvline(400, color='black', linewidth=0.4, linestyle=':', alpha=0.4)
    ax.text(420, 0.3, '$N{\\approx}400$\n(production)', fontsize=5.5,
            color='gray', va='bottom')

    ax.legend(fontsize=5, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=300, bbox_inches='tight')
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
