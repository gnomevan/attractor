#!/usr/bin/env python3
"""
r1_label_noise_merge_and_plot.py — Merge 3-seed + 4-seed label-noise data,
regenerate figure, and print updated manuscript values.

Input:
  data/main/revision1/label_noise_sweep.json           (seeds 0,1,2)
  data/main/revision1/label_noise_sweep_extra_seeds.json (seeds 3,4,5,6)

Output:
  data/main/revision1/label_noise_sweep_merged.json    (seeds 0–6, 7 total)
  paper/figures/revision1/label_noise_d2.png            (updated figure)
  paper/figures/revision1/label_noise_d2.pdf            (updated figure)

Usage:
  python -u code/revision1/r1_label_noise_merge_and_plot.py
"""

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    orig_path = os.path.join(
        repo_root, "data", "main", "revision1", "label_noise_sweep.json")
    extra_path = os.path.join(
        repo_root, "data", "main", "revision1",
        "label_noise_sweep_extra_seeds.json")
    merged_path = os.path.join(
        repo_root, "data", "main", "revision1",
        "label_noise_sweep_merged.json")
    fig_dir = os.path.join(repo_root, "paper", "figures", "revision1")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load ────────────────────────────────────────────────
    with open(orig_path) as f:
        orig = json.load(f)
    with open(extra_path) as f:
        extra = json.load(f)

    print("Original seeds:", orig["conditions"]["cnn_cifar"]["seeds"])
    print("Extra seeds:   ", extra["conditions"]["cnn_cifar"]["seeds"])

    # ── Merge ───────────────────────────────────────────────
    import copy
    merged = copy.deepcopy(orig)
    merged["revision1_metadata"]["merge_note"] = (
        "Merged original seeds [0,1,2] with extra seeds [3,4,5,6]. "
        "7 seeds total per condition per noise level."
    )

    for cname in ["cnn_cifar", "mlp_cifar_w85"]:
        if cname not in extra["conditions"]:
            print(f"  WARNING: {cname} not in extra seeds file, skipping")
            continue

        orig_cond = merged["conditions"][cname]
        extra_cond = extra["conditions"][cname]

        orig_seeds = orig_cond.get("seeds", [0, 1, 2])
        extra_seeds = extra_cond.get("seeds", [3, 4, 5, 6])
        all_seeds = orig_seeds + extra_seeds
        orig_cond["seeds"] = all_seeds

        noise_levels = orig_cond["noise_levels"]

        for p in noise_levels:
            p_key = str(p)
            if p_key not in extra_cond["results"]:
                print(f"  WARNING: {cname} p={p} not in extra, skipping")
                continue

            ob = orig_cond["results"][p_key]
            eb = extra_cond["results"][p_key]

            # Merge per-seed arrays
            for key in ["lyapunov", "corr_dim", "pc1", "pc2"]:
                ob[key] = ob[key] + eb[key]

            for key in ["sharpness_series", "grad_norm_series", "loss_series"]:
                if key in ob and key in eb:
                    ob[key] = ob[key] + eb[key]

            # Recompute summary stats
            ob["d2_mean"] = float(np.mean(ob["corr_dim"]))
            ob["d2_std"] = float(np.std(ob["corr_dim"]))
            ob["lam_mean"] = float(np.mean(ob["lyapunov"]))
            ob["lam_std"] = float(np.std(ob["lyapunov"]))

    # ── Save merged ─────────────────────────────────────────
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged data saved to {merged_path}")

    # ── Comparison table ────────────────────────────────────
    print("\n" + "=" * 72)
    print("COMPARISON: 3 seeds → 7 seeds")
    print("=" * 72)
    print(f"{'Condition':<18} {'p':>5}  {'D₂(3)':>8} {'σ(3)':>7}  "
          f"{'D₂(7)':>8} {'σ(7)':>7}  {'Δμ':>7} {'Δσ':>7}")
    print("-" * 72)

    for cname in ["cnn_cifar", "mlp_cifar_w85"]:
        cond = merged["conditions"][cname]
        noise_levels = cond["noise_levels"]
        for p in noise_levels:
            p_key = str(p)
            bucket = cond["results"][p_key]
            all_d2 = bucket["corr_dim"]

            d2_3_mean = float(np.mean(all_d2[:3]))
            d2_3_std = float(np.std(all_d2[:3]))
            d2_7_mean = float(np.mean(all_d2))
            d2_7_std = float(np.std(all_d2))

            delta_mu = d2_7_mean - d2_3_mean
            delta_sig = d2_7_std - d2_3_std

            marker = " ←" if abs(delta_mu) > 0.3 or abs(delta_sig) > 0.2 else ""
            print(f"{cname:<18} {p:>5.2f}  {d2_3_mean:>8.2f} {d2_3_std:>7.2f}  "
                  f"{d2_7_mean:>8.2f} {d2_7_std:>7.2f}  "
                  f"{delta_mu:>+7.2f} {delta_sig:>+7.2f}{marker}")
        print()

    # ── Generate figure ─────────────────────────────────────
    print("Generating updated figure...")

    cnn = merged["conditions"]["cnn_cifar"]
    mlp = merged["conditions"]["mlp_cifar_w85"]
    noise_levels = cnn["noise_levels"]

    cnn_d2_mean = [cnn["results"][str(p)]["d2_mean"] for p in noise_levels]
    cnn_d2_std  = [cnn["results"][str(p)]["d2_std"]  for p in noise_levels]
    mlp_d2_mean = [mlp["results"][str(p)]["d2_mean"] for p in noise_levels]
    mlp_d2_std  = [mlp["results"][str(p)]["d2_std"]  for p in noise_levels]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

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
    ax.set_ylim(0, 5.8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()

    png_path = os.path.join(fig_dir, "label_noise_d2.png")
    pdf_path = os.path.join(fig_dir, "label_noise_d2.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # ── Manuscript values ───────────────────────────────────
    print("\n" + "=" * 72)
    print("UPDATED VALUES FOR MANUSCRIPT (7 seeds)")
    print("=" * 72)

    print("\nFor main text (label-noise paragraph):")
    print(f"  CNN at 30% EoS: D₂ falls from "
          f"{cnn_d2_mean[0]:.2f} ± {cnn_d2_std[0]:.2f} to "
          f"{cnn_d2_mean[-1]:.2f} ± {cnn_d2_std[-1]:.2f}")
    print(f"  MLP at 90% EoS: D₂ rises from "
          f"{mlp_d2_mean[0]:.2f} ± {mlp_d2_std[0]:.2f} to "
          f"{mlp_d2_mean[-1]:.2f} ± {mlp_d2_std[-1]:.2f}")

    print("\nFor Figure 3 caption:")
    print(f"  Seven seeds per condition; error bars show standard deviation.")

    print("\nFor supplemental Table (label-noise sweep):")
    print(f"\n  {'':>4}  {'CNN/CIFAR (30% EoS)':>24}  {'MLP 269K (90% EoS)':>24}")
    print(f"  {'p':>4}  {'D₂':>10} {'σ':>10}  {'D₂':>10} {'σ':>10}")
    print("  " + "-" * 50)
    for p in noise_levels:
        p_key = str(p)
        cm = cnn["results"][p_key]["d2_mean"]
        cs = cnn["results"][p_key]["d2_std"]
        mm = mlp["results"][p_key]["d2_mean"]
        ms = mlp["results"][p_key]["d2_std"]
        print(f"  {p:>4.2f}  {cm:>10.2f} {cs:>10.2f}  {mm:>10.2f} {ms:>10.2f}")

    # Monotonicity check
    cnn_diffs = np.diff(cnn_d2_mean)
    mlp_diffs = np.diff(mlp_d2_mean)
    print(f"\n  CNN D₂ monotonically decreasing: "
          f"{all(d <= 0 for d in cnn_diffs)}")
    print(f"  MLP D₂ monotonically increasing: "
          f"{all(d >= 0 for d in mlp_diffs)}")
    print(f"  CNN total drop: {cnn_d2_mean[0]:.2f} → {cnn_d2_mean[-1]:.2f} "
          f"(Δ = {cnn_d2_mean[-1] - cnn_d2_mean[0]:.2f})")
    print(f"  MLP total rise: {mlp_d2_mean[0]:.2f} → {mlp_d2_mean[-1]:.2f} "
          f"(Δ = {mlp_d2_mean[-1] - mlp_d2_mean[0]:.2f})")


if __name__ == "__main__":
    main()
