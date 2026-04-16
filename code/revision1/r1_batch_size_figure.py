"""
r1_batch_size_figure.py  —  Publication figure for batch-size sweep
====================================================================

Generates a two-panel figure with both CNN and MLP conditions:
  (a) D₂ vs batch size — shows attractor dimension retention under SGD
  (b) λ vs batch size — shows Lyapunov exponent amplification under SGD

Input:  data/main/revision1/batch_size_sweep_merged.json
        (falls back to batch_size_sweep.json if merged not found)
Output: paper/figures/revision1/batch_size_d2.png
        paper/figures/revision1/batch_size_d2.pdf

Addresses: PRL referee concern about full-batch GD limitation.
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_condition(cond_data):
    """Pull D₂ and λ arrays from a condition's results dict."""
    batch_sizes = cond_data["batch_sizes"]
    d2_mean, d2_std, lam_mean, lam_std = [], [], [], []
    for B in batch_sizes:
        r = cond_data["results"][str(B)]
        d2_mean.append(np.mean(r["corr_dim"]))
        d2_std.append(np.std(r["corr_dim"]))
        lam_mean.append(np.mean(r["lyapunov"]))
        lam_std.append(np.std(r["lyapunov"]))
    return (batch_sizes,
            np.array(d2_mean), np.array(d2_std),
            np.array(lam_mean), np.array(lam_std))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    data_dir = os.path.join(repo_root, "data", "main", "revision1")
    # Prefer merged file, fall back to single-condition file
    merged_path = os.path.join(data_dir, "batch_size_sweep_merged.json")
    single_path = os.path.join(data_dir, "batch_size_sweep.json")
    data_path = merged_path if os.path.exists(merged_path) else single_path

    fig_dir = os.path.join(repo_root, "paper", "figures", "revision1")
    os.makedirs(fig_dir, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    has_mlp = "mlp_cifar_w85" in data["conditions"]
    print(f"Data source: {os.path.basename(data_path)}")
    print(f"Conditions: {list(data['conditions'].keys())}")

    # ── Extract data ────────────────────────────────────────
    cnn_bs, cnn_d2, cnn_d2s, cnn_lam, cnn_lams = extract_condition(
        data["conditions"]["cnn_cifar"])

    if has_mlp:
        mlp_bs, mlp_d2, mlp_d2s, mlp_lam, mlp_lams = extract_condition(
            data["conditions"]["mlp_cifar_w85"])

    # ── Two-panel figure ────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    color_cnn = '#D64545'
    color_mlp = '#2E6EB5'

    # Panel (a): D₂ vs batch size
    ax1.errorbar(cnn_bs, cnn_d2, yerr=cnn_d2s,
                 fmt='o-', color=color_cnn, markersize=5, capsize=3,
                 linewidth=1.4, markeredgecolor='white', markeredgewidth=0.5,
                 label='CNN (30% EoS)', zorder=3)

    if has_mlp:
        ax1.errorbar(mlp_bs, mlp_d2, yerr=mlp_d2s,
                     fmt='s-', color=color_mlp, markersize=5, capsize=3,
                     linewidth=1.4, markeredgecolor='white',
                     markeredgewidth=0.5,
                     label='MLP 269K (90% EoS)', zorder=3)

    ax1.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # Retention annotations for CNN
    cnn_full = cnn_d2[0]
    for i, B in enumerate(cnn_bs):
        if B == 2000:
            continue
        pct = cnn_d2[i] / cnn_full * 100
        ax1.annotate(f'{pct:.0f}%',
                     xy=(B, cnn_d2[i]),
                     xytext=(0, -13),
                     textcoords='offset points',
                     fontsize=6, color=color_cnn, alpha=0.7,
                     ha='center', va='top')

    # Retention annotations for MLP
    if has_mlp:
        mlp_full = mlp_d2[0]
        for i, B in enumerate(mlp_bs):
            if B == 2000:
                continue
            pct = mlp_d2[i] / mlp_full * 100
            ax1.annotate(f'{pct:.0f}%',
                         xy=(B, mlp_d2[i]),
                         xytext=(0, 9),
                         textcoords='offset points',
                         fontsize=6, color=color_mlp, alpha=0.7,
                         ha='center', va='bottom')

    ax1.set_xlabel('Batch size $B$', fontsize=9)
    ax1.set_ylabel('Correlation dimension $D_2$', fontsize=9)
    ax1.set_xscale('log')
    ax1.set_xticks(cnn_bs)
    ax1.set_xticklabels([str(B) for B in cnn_bs])
    ax1.set_xlim(70, 3000)
    ax1.set_ylim(0, 5.5)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.legend(fontsize=7, loc='lower right')
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold', va='top')

    # Panel (b): λ vs batch size
    ax2.errorbar(cnn_bs, cnn_lam * 1e4, yerr=cnn_lams * 1e4,
                 fmt='o-', color=color_cnn, markersize=5, capsize=3,
                 linewidth=1.4, markeredgecolor='white', markeredgewidth=0.5,
                 label='CNN (30% EoS)', zorder=3)

    if has_mlp:
        ax2.errorbar(mlp_bs, mlp_lam * 1e4, yerr=mlp_lams * 1e4,
                     fmt='s-', color=color_mlp, markersize=5, capsize=3,
                     linewidth=1.4, markeredgecolor='white',
                     markeredgewidth=0.5,
                     label='MLP 269K (90% EoS)', zorder=3)

    ax2.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    ax2.set_xlabel('Batch size $B$', fontsize=9)
    ax2.set_ylabel(r'Lyapunov exponent $\lambda$ ($\times 10^{-4}$/step)',
                   fontsize=9)
    ax2.set_xscale('log')
    ax2.set_xticks(cnn_bs)
    ax2.set_xticklabels([str(B) for B in cnn_bs])
    ax2.set_xlim(70, 3000)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold', va='top')

    plt.tight_layout(w_pad=2.0)

    png_path = os.path.join(fig_dir, "batch_size_d2.png")
    pdf_path = os.path.join(fig_dir, "batch_size_d2.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")

    # ── Print summary tables ────────────────────────────────
    for label, bs, d2, d2s, lam, lams in [
        ("CNN/CIFAR-10 at 30% EoS", cnn_bs, cnn_d2, cnn_d2s,
         cnn_lam, cnn_lams),
    ] + ([
        ("MLP 269K at 90% EoS", mlp_bs, mlp_d2, mlp_d2s,
         mlp_lam, mlp_lams),
    ] if has_mlp else []):
        full = d2[0]
        print(f"\n{label}:")
        print(f"{'B':>6}  {'D₂':>12}  {'λ (×10⁻⁴)':>14}  {'retention':>10}")
        print("-" * 50)
        for i, B in enumerate(bs):
            pct = d2[i] / full * 100
            print(f"{B:>6d}  {d2[i]:.2f} ± {d2s[i]:.2f}"
                  f"  {lam[i]*1e4:>6.2f} ± {lams[i]*1e4:.2f}"
                  f"  {pct:>8.1f}%")


if __name__ == "__main__":
    main()
