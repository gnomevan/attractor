#!/usr/bin/env python3
"""
r1_tda_mlp_cifar.py — Phase 2C: Persistent homology for MLP/CIFAR-10
======================================================================

Referee issue #3: TDA currently supports only the CNN claim. The MLP/CIFAR
result — which the "data complexity is sufficient" argument hinges on —
has no topological cross-check.

Referee issue #11: Persistence diagrams (not just feature counts) should
be shown.

Approach:
  For both MLP widths (50 and 85), at each of the 12 LR fractions:
    - Train 3 seeds (matching TDA seed count for CNN)
    - Record function-space trajectory
    - PCA-reduce to 10 dimensions
    - Compute persistent homology via ripser (H₁ and H₂)
    - Extract: feature counts, gap ratios, mean lifetimes, full diagrams

Also compute for CNN/CIFAR at 4 representative LR fractions for the
publication persistence-diagram figure:
    - 5% EoS (trivial convergence)
    - 30% EoS (peak D₂ — strange attractor)
    - 90% EoS (late plateau)

OUTPUT:
  data/supplemental/revision1/tda_mlp_cifar_w50.json
  data/supplemental/revision1/tda_mlp_cifar_w85.json
  data/supplemental/revision1/tda_cnn_cifar_diagrams.json  (for figure)

USAGE:
  python -u code/revision1/r1_tda_mlp_cifar.py                   # full
  python -u code/revision1/r1_tda_mlp_cifar.py --quick            # 3 LR points
  python -u code/revision1/r1_tda_mlp_cifar.py --dry-run
  python -u code/revision1/r1_tda_mlp_cifar.py --condition w50    # one width only
  python -u code/revision1/r1_tda_mlp_cifar.py --cnn-diagrams-only

REQUIREMENTS: torch, torchvision, numpy, scipy, ripser
HARDWARE: GPU recommended; ripser is CPU-only but benefits from fast
          trajectory generation on GPU.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code", "revision1"))

from r1_cross_experiments import (
    MLPCifar, SmallCNN, Condition, CONDITIONS,
    build_model, load_condition_data,
    find_eos_threshold, compute_sharpness,
    _collect_metadata, _serialize,
    LR_FRACTIONS_FULL, LR_FRACTIONS_QUICK,
)

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("WARNING: ripser not installed. Install with: pip install ripser")
    print("         TDA computation will fail without it.")

OUT_DIR = os.path.join(ROOT, "data", "supplemental", "revision1")

N_STEPS = 5000
TDA_MAX_POINTS = 400
TDA_PCA_DIM = 10
TDA_MAX_DIM = 2  # H₀, H₁, H₂
TDA_SEEDS = [0, 1, 2]  # 3 seeds for TDA (matching CNN TDA protocol)

CNN_DIAGRAM_FRACS = [0.05, 0.30, 0.90]


# ============================================================
# TRAJECTORY GENERATION (no perturbation needed for TDA)
# ============================================================

def generate_trajectory(cond, seed, lr, X, y, X_eval, device, n_steps=5000):
    """
    Train for n_steps and record function-space trajectory.
    Returns trajectory array of shape (n_post_transient, n_eval * n_classes).
    """
    criterion = nn.MSELoss()
    model = build_model(cond, seed).to(device)
    outputs_rec = []

    for t in range(n_steps):
        if t % 10 == 0:
            with torch.no_grad():
                outputs_rec.append(model(X_eval).cpu().numpy())

        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

    outputs = np.array(outputs_rec)
    # Discard transient (first 20%)
    n_transient = len(outputs) // 5
    traj = outputs[n_transient:].reshape(len(outputs) - n_transient, -1)
    return traj


def pca_reduce(traj, n_components=10):
    """PCA-reduce trajectory to n_components dimensions."""
    centered = traj - traj.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n = min(n_components, Vt.shape[0])
    return centered @ Vt[:n].T


# ============================================================
# PERSISTENT HOMOLOGY
# ============================================================

def compute_persistence(traj_pca, max_dim=2, max_points=400, seed=42):
    """
    Compute persistent homology on PCA-reduced trajectory.
    Returns diagrams dict and raw ripser result.
    """
    if not HAS_RIPSER:
        raise RuntimeError("ripser is required for TDA. pip install ripser")

    # Subsample if needed
    if len(traj_pca) > max_points:
        idx = np.random.RandomState(seed).choice(
            len(traj_pca), max_points, replace=False)
        traj_pca = traj_pca[idx]

    result = ripser(traj_pca, maxdim=max_dim)

    diagrams = {}
    for dim in range(max_dim + 1):
        dgm = result['dgms'][dim]
        # Filter infinite death times
        finite_mask = np.isfinite(dgm[:, 1])
        diagrams[f'H{dim}'] = dgm[finite_mask].tolist()

    return diagrams


def persistence_summary(diagrams):
    """Extract summary statistics from persistence diagrams."""
    summary = {}
    for dim_key, pts in diagrams.items():
        pts = np.array(pts)
        if len(pts) == 0:
            summary[dim_key] = {
                'n_features': 0,
                'max_lifetime': 0.0,
                'mean_lifetime': 0.0,
                'total_persistence': 0.0,
                'gap_ratio': None,
                'top_3_lifetimes': [],
            }
            continue

        lifetimes = pts[:, 1] - pts[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        sorted_lt = np.sort(lifetimes)[::-1]

        summary[dim_key] = {
            'n_features': len(lifetimes),
            'max_lifetime': float(sorted_lt[0]) if len(sorted_lt) > 0 else 0.0,
            'mean_lifetime': float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0,
            'total_persistence': float(np.sum(lifetimes)),
            'gap_ratio': float(sorted_lt[0] / sorted_lt[1]) if len(sorted_lt) > 1 else float('inf'),
            'top_3_lifetimes': sorted_lt[:3].tolist(),
        }
    return summary


# ============================================================
# MAIN RUNNERS
# ============================================================

def run_mlp_tda(cond_key, seeds, lr_fractions, device, data_root):
    """Run TDA for one MLP condition across all LR fractions and seeds."""
    cond = CONDITIONS[cond_key]
    print(f"\n{'='*60}")
    print(f"TDA: {cond.name}")
    print(f"  seeds: {seeds}")
    print(f"  LR fractions: {lr_fractions}")
    print(f"{'='*60}")

    X, y = load_condition_data(cond, n_samples=2000, data_root=data_root)
    X = X.to(device)
    y = y.to(device)

    torch.manual_seed(0)
    n_eval = min(100, X.shape[0])
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    lam_max, lr_eos = find_eos_threshold(cond, X, y, device)
    test_lrs = [frac * lr_eos for frac in lr_fractions]

    results = {
        "experiment": cond.name,
        "lam_max": lam_max,
        "lr_eos": lr_eos,
        "lr_fractions": lr_fractions,
        "test_lrs": test_lrs,
        "n_params": build_model(cond, 0).count_params(),
        "seeds": seeds,
        "metadata": _collect_metadata(),
    }

    for li, (frac, lr) in enumerate(zip(lr_fractions, test_lrs)):
        lr_key = f"lr_{li}"
        print(f"\n  --- {frac*100:.0f}% EoS (lr={lr:.6f}) ---")

        bucket = {
            "H1_features": [], "H2_features": [],
            "H1_gap_ratio": [], "H2_gap_ratio": [],
            "H1_mean_lifetime": [], "H2_mean_lifetime": [],
            "H1_total_persistence": [], "H2_total_persistence": [],
            "diagrams": [],  # full diagrams for representative figure
        }

        for seed in seeds:
            t0 = time.time()

            # Generate trajectory
            traj = generate_trajectory(
                cond, seed, lr, X, y, X_eval, device, n_steps=N_STEPS)

            # PCA reduce
            traj_pca = pca_reduce(traj, n_components=TDA_PCA_DIM)

            # Persistent homology
            diagrams = compute_persistence(
                traj_pca, max_dim=TDA_MAX_DIM,
                max_points=TDA_MAX_POINTS,
                seed=li * 1000 + seed)
            summary = persistence_summary(diagrams)

            # Collect stats
            h1 = summary.get('H1', {'n_features': 0, 'gap_ratio': None,
                                     'mean_lifetime': 0, 'total_persistence': 0})
            h2 = summary.get('H2', {'n_features': 0, 'gap_ratio': None,
                                     'mean_lifetime': 0, 'total_persistence': 0})

            bucket["H1_features"].append(h1['n_features'])
            bucket["H2_features"].append(h2['n_features'])
            bucket["H1_gap_ratio"].append(h1['gap_ratio'])
            bucket["H2_gap_ratio"].append(h2['gap_ratio'])
            bucket["H1_mean_lifetime"].append(h1['mean_lifetime'])
            bucket["H2_mean_lifetime"].append(h2['mean_lifetime'])
            bucket["H1_total_persistence"].append(h1['total_persistence'])
            bucket["H2_total_persistence"].append(h2['total_persistence'])
            bucket["diagrams"].append(diagrams)

            elapsed = time.time() - t0
            print(f"    seed {seed}: H1={h1['n_features']:4d}  "
                  f"H2={h2['n_features']:4d}  "
                  f"gap={h1['gap_ratio']:.2f}  ({elapsed:.1f}s)"
                  if h1['gap_ratio'] is not None else
                  f"    seed {seed}: H1={h1['n_features']:4d}  "
                  f"H2={h2['n_features']:4d}  gap=---  ({elapsed:.1f}s)")

        results[lr_key] = _serialize(bucket)

    return results


def run_cnn_diagrams(device, data_root):
    """
    Generate full persistence diagrams for CNN/CIFAR at representative
    LR fractions (for the publication figure).
    """
    from r1_cross_experiments import SmallCNN

    cond = Condition(
        name="cnn_cifar", arch="cnn", hidden_dim=None,
        data="cifar_image", output_stem="cnn_cifar")

    print(f"\n{'='*60}")
    print(f"CNN/CIFAR persistence diagrams for figure")
    print(f"  LR fractions: {CNN_DIAGRAM_FRACS}")
    print(f"{'='*60}")

    # Load CIFAR as images
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), 2000, replace=False)
    images, labels = [], []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)
    X = torch.stack(images).to(device)
    y_onehot = torch.zeros(2000, 10)
    for i, label in enumerate(labels):
        y_onehot[i, label] = 1.0
    y = y_onehot.to(device)

    torch.manual_seed(0)
    eval_idx = torch.randperm(X.shape[0])[:100]
    X_eval = X[eval_idx]

    lam_max, lr_eos = find_eos_threshold(cond, X, y, device)

    results = {
        "experiment": "cnn_cifar_diagrams",
        "lam_max": lam_max,
        "lr_eos": lr_eos,
        "lr_fractions": CNN_DIAGRAM_FRACS,
        "metadata": _collect_metadata(),
    }

    for li, frac in enumerate(CNN_DIAGRAM_FRACS):
        lr = frac * lr_eos
        lr_key = f"lr_{li}"
        print(f"\n  --- {frac*100:.0f}% EoS (lr={lr:.6f}) ---")

        traj = generate_trajectory(
            cond, seed=0, lr=lr, X=X, y=y, X_eval=X_eval,
            device=device, n_steps=N_STEPS)
        traj_pca = pca_reduce(traj, n_components=TDA_PCA_DIM)
        diagrams = compute_persistence(
            traj_pca, max_dim=TDA_MAX_DIM,
            max_points=TDA_MAX_POINTS, seed=li * 1000)
        summary = persistence_summary(diagrams)

        h1 = summary.get('H1', {'n_features': 0})
        h2 = summary.get('H2', {'n_features': 0})
        print(f"    H1={h1['n_features']}  H2={h2['n_features']}")

        results[lr_key] = _serialize({
            "frac": frac,
            "diagrams": diagrams,
            "summary": summary,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2C: Persistent homology for MLP/CIFAR-10")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="3 LR fractions instead of 12")
    parser.add_argument("--condition", choices=["w50", "w85", "both"],
                        default="both", help="Which MLP width to run")
    parser.add_argument("--cnn-diagrams-only", action="store_true",
                        help="Only generate CNN diagrams for the figure")
    parser.add_argument("--seeds", type=int, nargs="+", default=TDA_SEEDS)
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ripser available: {HAS_RIPSER}")

    lr_fracs = LR_FRACTIONS_QUICK if args.quick else LR_FRACTIONS_FULL

    if args.dry_run:
        print("\n=== DRY RUN ===")
        if not args.cnn_diagrams_only:
            if args.condition in ("w50", "both"):
                print(f"  MLP w50: {len(lr_fracs)} LR fracs × {len(args.seeds)} seeds")
            if args.condition in ("w85", "both"):
                print(f"  MLP w85: {len(lr_fracs)} LR fracs × {len(args.seeds)} seeds")
        print(f"  CNN diagrams: {len(CNN_DIAGRAM_FRACS)} LR fracs × 1 seed")
        print(f"  Output → {OUT_DIR}/tda_mlp_cifar_w*.json")
        return

    if not HAS_RIPSER:
        print("ERROR: ripser required. Install with: pip install ripser")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.cnn_diagrams_only:
        if args.condition in ("w50", "both"):
            r50 = run_mlp_tda("mlp_cifar_w50", args.seeds, lr_fracs,
                              device, args.data_root)
            path = os.path.join(OUT_DIR, "tda_mlp_cifar_w50.json")
            with open(path, "w") as f:
                json.dump(_serialize(r50), f, indent=2)
            print(f"\nSaved → {path}")

        if args.condition in ("w85", "both"):
            r85 = run_mlp_tda("mlp_cifar_w85", args.seeds, lr_fracs,
                              device, args.data_root)
            path = os.path.join(OUT_DIR, "tda_mlp_cifar_w85.json")
            with open(path, "w") as f:
                json.dump(_serialize(r85), f, indent=2)
            print(f"\nSaved → {path}")

    # CNN diagrams for the figure
    cnn_results = run_cnn_diagrams(device, args.data_root)
    path = os.path.join(OUT_DIR, "tda_cnn_cifar_diagrams.json")
    with open(path, "w") as f:
        json.dump(_serialize(cnn_results), f, indent=2)
    print(f"\nSaved → {path}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
