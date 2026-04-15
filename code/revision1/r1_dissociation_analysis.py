"""
r1_dissociation_analysis.py  —  Phase 3B: λ–D₂ dissociation universality
=========================================================================

Addresses referee issue #9: the dissociation between peak λ (chaos)
and peak D₂ (geometric complexity) was shown only for CNN/CIFAR in v1.
Phase 1 already produced N=10 data for all CIFAR conditions. This
script extracts the dissociation quantitatively and generates a
supplemental figure.

The analysis is purely post-hoc on existing data — no new training.

Input files (all from Phase 1 / existing data):
  - data/main/cifar10_eos_10seeds.json         (CNN/CIFAR, different schema)
  - data/main/revision1/cross_small_mlp_cifar_w50_seeds_merged.json
  - data/main/revision1/cross_small_mlp_cifar_w85_seeds_merged.json

Output:
  - data/supplemental/revision1/dissociation_analysis.json
  - paper/figures/revision1/dissociation_figure.png  (supplemental figure)
  - paper/figures/revision1/dissociation_figure.pdf

Usage:
   python r1_dissociation_analysis.py
   python r1_dissociation_analysis.py --no-figure   # skip figure generation

Addresses referee issue: #9 (dissociation universality)
"""

import argparse
import json
import os
import sys

import numpy as np

# ============================================================
# DATA LOADING
# ============================================================

def load_merged_data(filepath):
    """Load a merged JSON from r1_cross_experiments.py / r1_merge.py."""
    with open(filepath) as f:
        data = json.load(f)
    lr_fractions = data["lr_fractions"]
    n_lrs = len(lr_fractions)
    lam_all = []
    d2_all = []
    for li in range(n_lrs):
        key = f"lr_{li}"
        bucket = data[key]
        lam_all.append(bucket["lyapunov"])
        d2_all.append(bucket["corr_dim"])
    return {
        "lr_fractions": lr_fractions,
        "lam_per_lr": lam_all,       # list of lists (per seed)
        "d2_per_lr": d2_all,
        "lam_max": data.get("lam_max"),
        "lr_eos": data.get("lr_eos"),
        "n_params": data.get("n_params"),
        "seeds_run": data.get("seeds_run"),
    }


def load_cnn_cifar_data(filepath):
    """
    Load the CNN/CIFAR 10-seed data. Different schema: uses 'seeds' key
    and nested structure. Adapted from the original analysis scripts.
    """
    with open(filepath) as f:
        data = json.load(f)

    # CNN/CIFAR data schema has lr_fractions and per-lr buckets
    # but uses slightly different key naming
    lr_fractions = data.get("lr_fractions",
                            data.get("fractions", []))
    n_lrs = len(lr_fractions)

    lam_all = []
    d2_all = []
    for li in range(n_lrs):
        key = f"lr_{li}"
        if key not in data:
            # Try alternative key format
            key = f"frac_{li}"
        if key not in data:
            continue
        bucket = data[key]
        lam_all.append(bucket.get("lyapunov", bucket.get("lyap", [])))
        d2_all.append(bucket.get("corr_dim", bucket.get("d2", [])))

    return {
        "lr_fractions": lr_fractions,
        "lam_per_lr": lam_all,
        "d2_per_lr": d2_all,
        "lam_max": data.get("lam_max"),
        "lr_eos": data.get("lr_eos"),
        "n_params": data.get("n_params"),
        "seeds_run": data.get("seeds", data.get("seeds_run")),
    }


# ============================================================
# ANALYSIS
# ============================================================

def analyze_dissociation(name, cond_data):
    """
    For a given condition, find:
      - argmax(mean λ) across LR fractions
      - argmax(mean D₂) across LR fractions
      - whether they dissociate (different LR fraction)
    """
    fracs = cond_data["lr_fractions"]
    lam_means = [np.mean(vals) for vals in cond_data["lam_per_lr"]]
    lam_stds = [np.std(vals) for vals in cond_data["lam_per_lr"]]
    d2_means = [np.mean(vals) for vals in cond_data["d2_per_lr"]]
    d2_stds = [np.std(vals) for vals in cond_data["d2_per_lr"]]

    i_lam_peak = int(np.argmax(lam_means))
    i_d2_peak = int(np.argmax(d2_means))

    dissociates = i_lam_peak != i_d2_peak

    return {
        "name": name,
        "lr_fractions": fracs,
        "lam_means": [float(x) for x in lam_means],
        "lam_stds": [float(x) for x in lam_stds],
        "d2_means": [float(x) for x in d2_means],
        "d2_stds": [float(x) for x in d2_stds],
        "peak_lam_frac": fracs[i_lam_peak],
        "peak_lam_value": float(lam_means[i_lam_peak]),
        "peak_lam_std": float(lam_stds[i_lam_peak]),
        "peak_d2_frac": fracs[i_d2_peak],
        "peak_d2_value": float(d2_means[i_d2_peak]),
        "peak_d2_std": float(d2_stds[i_d2_peak]),
        "dissociates": dissociates,
        "frac_gap": abs(fracs[i_d2_peak] - fracs[i_lam_peak]),
        "n_seeds": len(cond_data["lam_per_lr"][0]),
        "n_params": cond_data["n_params"],
    }


# ============================================================
# FIGURE
# ============================================================

def make_dissociation_figure(analyses, output_dir):
    """
    Supplemental figure: stacked λ(LR) and D₂(LR) for all CIFAR conditions.
    Three rows (one per condition), two columns (λ, D₂).
    Peak locations marked with vertical dashed lines.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping figure generation")
        return None

    fig, axes = plt.subplots(3, 2, figsize=(8, 9), sharex=True)

    conditions_order = [
        ("CNN/CIFAR-10 (269K)", "cnn_cifar"),
        ("MLP/CIFAR-10 (156K, w=50)", "mlp_cifar_w50"),
        ("MLP/CIFAR-10 (269K, w=85)", "mlp_cifar_w85"),
    ]

    colors = {"lam": "#D64545", "d2": "#2E6EB5"}

    for row, (label, key) in enumerate(conditions_order):
        if key not in analyses:
            continue
        a = analyses[key]
        fracs = a["lr_fractions"]
        fracs_pct = [f * 100 for f in fracs]

        # Left column: Lyapunov
        ax_lam = axes[row, 0]
        ax_lam.errorbar(fracs_pct, a["lam_means"], yerr=a["lam_stds"],
                        fmt='o-', color=colors["lam"], markersize=4,
                        capsize=3, linewidth=1.2)
        ax_lam.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax_lam.axvline(a["peak_lam_frac"] * 100, color=colors["lam"],
                       linestyle='--', alpha=0.5, linewidth=1.0)
        ax_lam.set_ylabel(r"$\lambda$ (step$^{-1}$)", fontsize=9)
        ax_lam.set_title(label if row == 0 else "", fontsize=10)
        ax_lam.text(0.02, 0.95, label, transform=ax_lam.transAxes,
                    fontsize=8, verticalalignment='top',
                    fontweight='bold')
        ax_lam.text(0.98, 0.95,
                    f"peak: {a['peak_lam_frac']*100:.0f}% EoS",
                    transform=ax_lam.transAxes, fontsize=7,
                    verticalalignment='top', horizontalalignment='right',
                    color=colors["lam"])

        # Right column: D₂
        ax_d2 = axes[row, 1]
        ax_d2.errorbar(fracs_pct, a["d2_means"], yerr=a["d2_stds"],
                       fmt='s-', color=colors["d2"], markersize=4,
                       capsize=3, linewidth=1.2)
        ax_d2.axhline(1, color='gray', linestyle=':', linewidth=0.8)
        ax_d2.axvline(a["peak_d2_frac"] * 100, color=colors["d2"],
                      linestyle='--', alpha=0.5, linewidth=1.0)
        ax_d2.set_ylabel(r"$D_2$", fontsize=9)
        ax_d2.text(0.98, 0.95,
                   f"peak: {a['peak_d2_frac']*100:.0f}% EoS",
                   transform=ax_d2.transAxes, fontsize=7,
                   verticalalignment='top', horizontalalignment='right',
                   color=colors["d2"])

    # Bottom axis labels
    for ax in axes[2, :]:
        ax.set_xlabel("Fraction of EoS threshold (%)", fontsize=9)

    # Overall title
    fig.suptitle(
        r"$\lambda$–$D_2$ dissociation across architectures",
        fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "dissociation_figure.png")
    pdf_path = os.path.join(output_dir, "dissociation_figure.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"  Figure saved: {png_path}")
    print(f"  Figure saved: {pdf_path}")
    return png_path


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3B: λ–D₂ dissociation analysis (referee #9)")
    parser.add_argument("--no-figure", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default=None)
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = args.data_dir or os.path.join(repo_root, "data")
    output_dir = args.output_dir or os.path.join(
        data_dir, "supplemental", "revision1")
    figure_dir = args.figure_dir or os.path.join(
        repo_root, "paper", "figures", "revision1")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("=" * 64)
    print("Phase 3B: λ–D₂ dissociation universality analysis")
    print("=" * 64)

    # Load all CIFAR conditions
    conditions = {}

    # CNN/CIFAR (different schema)
    cnn_path = os.path.join(data_dir, "main", "cifar10_eos_10seeds.json")
    if os.path.exists(cnn_path):
        print(f"\nLoading CNN/CIFAR: {cnn_path}")
        conditions["cnn_cifar"] = load_cnn_cifar_data(cnn_path)
    else:
        print(f"WARNING: {cnn_path} not found")

    # MLP w50
    w50_path = os.path.join(
        data_dir, "main", "revision1",
        "cross_small_mlp_cifar_w50_seeds_merged.json")
    if os.path.exists(w50_path):
        print(f"Loading MLP w50: {w50_path}")
        conditions["mlp_cifar_w50"] = load_merged_data(w50_path)
    else:
        print(f"WARNING: {w50_path} not found")

    # MLP w85
    w85_path = os.path.join(
        data_dir, "main", "revision1",
        "cross_small_mlp_cifar_w85_seeds_merged.json")
    if os.path.exists(w85_path):
        print(f"Loading MLP w85: {w85_path}")
        conditions["mlp_cifar_w85"] = load_merged_data(w85_path)
    else:
        print(f"WARNING: {w85_path} not found")

    if not conditions:
        print("ERROR: No data files found. Cannot proceed.")
        sys.exit(1)

    # Analyze each condition
    analyses = {}
    print("\n" + "-" * 64)
    print("Dissociation analysis:")
    print("-" * 64)
    print(f"{'Condition':<25} {'Peak λ':>10} {'Peak D₂':>10} "
          f"{'Gap':>8} {'Dissoc?':>8}")
    print("-" * 64)

    for key, cond_data in conditions.items():
        a = analyze_dissociation(key, cond_data)
        analyses[key] = a
        dissoc_str = "YES" if a["dissociates"] else "no"
        print(f"{key:<25} {a['peak_lam_frac']*100:>6.0f}% EoS "
              f"{a['peak_d2_frac']*100:>6.0f}% EoS "
              f"{a['frac_gap']*100:>5.0f}pp   {dissoc_str:>8}")

    # Summary
    all_dissociate = all(a["dissociates"] for a in analyses.values())
    print(f"\nAll conditions dissociate: {all_dissociate}")

    if all_dissociate:
        print("→ Dissociation is UNIVERSAL across all CIFAR architectures.")
        print("  Strengthens KAM framing in Discussion.")
    else:
        non_dissoc = [k for k, a in analyses.items() if not a["dissociates"]]
        print(f"→ Non-dissociating conditions: {non_dissoc}")
        print("  Discussion framing should be softened to architecture-dependent.")

    # Save analysis
    output_path = os.path.join(output_dir, "dissociation_analysis.json")
    output_data = {
        "experiment": "dissociation_analysis",
        "all_dissociate": all_dissociate,
        "conditions": analyses,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAnalysis saved: {output_path}")

    # Generate figure
    if not args.no_figure:
        print("\nGenerating dissociation figure...")
        make_dissociation_figure(analyses, figure_dir)

    print("\nPhase 3B complete.")


if __name__ == "__main__":
    main()
