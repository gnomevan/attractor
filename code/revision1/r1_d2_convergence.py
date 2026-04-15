#!/usr/bin/env python3
"""
r1_d2_convergence.py — Phase 2B: D₂ vs trajectory length convergence
=====================================================================

Referee issue #4: MLP/CIFAR D₂ values sit near the Eckmann-Ruelle ceiling
for n≈400 points. We must show they plateau (not grow) when N increases.

APPROACH (revised):
  The original plan extended training to 32k steps, but this changes the
  dynamical regime — the attractor collapses as the network converges
  further (CNN loss dropped 800× between step 5k and 25k).

  Instead: record outputs EVERY STEP (not every 10) within the standard
  5000-step protocol. This yields ~4000 post-transient points WITHOUT
  changing the dynamics. We then compute D₂ at subsampled lengths
  N ∈ {100, 200, 400, 800, 1600, 3200} from the same trajectory.

  For ONE seed each of:
    - CNN/CIFAR-10 at 30% EoS
    - MLP/CIFAR-10 269K (w=85) at 90% EoS
    - MLP/CIFAR-10 156K (w=50) at 90% EoS

  The reference D₂(N) curves for Lorenz and MG τ=30 are produced by
  r1_calibration_n400.py (already run).

OUTPUT:
  data/supplemental/revision1/d2_vs_n_neural.json

USAGE:
  python -u code/revision1/r1_d2_convergence.py              # full run
  python -u code/revision1/r1_d2_convergence.py --dry-run    # print plan
  python -u code/revision1/r1_d2_convergence.py --quick      # only N={200,400,800}

REQUIREMENTS: torch, torchvision, numpy, scipy
HARDWARE: GPU recommended but 3 runs of 5000 steps — fast (~5 min total)
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
    find_eos_threshold, correlation_dimension,
    _collect_metadata, _serialize,
)

OUT_DIR = os.path.join(ROOT, "data", "supplemental", "revision1")

# ── Conditions to test ──
CONVERGENCE_CONDITIONS = [
    {"condition": "cnn_cifar",     "lr_frac": 0.30, "seed": 0},
    {"condition": "mlp_cifar_w85", "lr_frac": 0.90, "seed": 0},
    {"condition": "mlp_cifar_w50", "lr_frac": 0.90, "seed": 0},
]

CNN_CIFAR_CONDITION = Condition(
    name="cnn_cifar", arch="cnn", hidden_dim=None,
    data="cifar_image", output_stem="cnn_cifar")

N_STEPS = 5000
TRANSIENT_FRAC = 0.20
# Record EVERY step (not every 10) for dense trajectory
RECORD_EVERY = 1

SAMPLE_SIZES = [100, 200, 400, 800, 1600, 3200]
SAMPLE_SIZES_QUICK = [200, 400, 800]


def load_cifar10_image(n_samples=2000, seed=42, data_root="./data"):
    """Load CIFAR-10 as images (3x32x32) for the CNN condition."""
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), n_samples, replace=False)
    images, labels = [], []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)
    X = torch.stack(images)
    y = torch.zeros(n_samples, 10)
    for i, label in enumerate(labels):
        y[i, label] = 1.0
    return X, y


def run_dense_trajectory(cond, seed, lr, X, y, X_eval, device, n_steps=5000):
    """
    Train for n_steps, recording outputs EVERY step for dense trajectory.
    Same protocol as r1_cross_experiments.py except for recording frequency.
    """
    criterion = nn.MSELoss()
    model = build_model(cond, seed).to(device)
    outputs_rec = []

    for t in range(n_steps):
        # Record every step
        if t % RECORD_EVERY == 0:
            with torch.no_grad():
                outputs_rec.append(model(X_eval).cpu().numpy())

        # Train
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        if t % 1000 == 0:
            print(f"    step {t}/{n_steps}: loss={loss.item():.6f}")

    return np.array(outputs_rec)


def d2_at_subsample_sizes(full_traj, sample_sizes, seed=42):
    """
    Compute D₂ at various trajectory lengths by taking the LAST n points
    of the full post-transient trajectory (most dynamically settled).
    """
    results = []
    n_full = len(full_traj)
    for n in sample_sizes:
        if n > n_full:
            results.append({"n": n, "d2": None, "status": "too_few_points"})
            print(f"      n={n:5d}: SKIP (need {n}, have {n_full})")
            continue
        traj_slice = full_traj[-n:]
        d2 = correlation_dimension(traj_slice, seed)
        results.append({"n": n, "d2": float(d2)})
        print(f"      n={n:5d}: D₂ = {d2:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2B: D₂(N) convergence for neural networks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer sample sizes")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sample_sizes = SAMPLE_SIZES_QUICK if args.quick else SAMPLE_SIZES

    # Post-transient points: 5000 * (1 - 0.20) = 4000
    n_post_transient = int(N_STEPS * (1 - TRANSIENT_FRAC))
    print(f"Recording every step → {N_STEPS} total, "
          f"{n_post_transient} post-transient")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for cc in CONVERGENCE_CONDITIONS:
            print(f"  {cc['condition']} at {cc['lr_frac']*100:.0f}% EoS, "
                  f"seed {cc['seed']}, {N_STEPS} steps (every-step recording)")
        print(f"  D₂ at N = {sample_sizes}")
        print(f"  Output → {OUT_DIR}/d2_vs_n_neural.json")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"metadata": _collect_metadata(), "conditions": {}}

    for cc in CONVERGENCE_CONDITIONS:
        cond_name = cc["condition"]
        lr_frac = cc["lr_frac"]
        seed = cc["seed"]

        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} at {lr_frac*100:.0f}% EoS, seed {seed}")
        print(f"{'='*60}")

        t0 = time.time()

        # Load data and find EoS threshold
        if cond_name == "cnn_cifar":
            cond = CNN_CIFAR_CONDITION
            X, y = load_cifar10_image(2000, seed=42, data_root=args.data_root)
        else:
            cond = CONDITIONS[cond_name]
            X, y = load_condition_data(cond, n_samples=2000,
                                       data_root=args.data_root)

        X = X.to(device)
        y = y.to(device)

        # Eval set (same protocol)
        torch.manual_seed(0)
        n_eval = min(100, X.shape[0])
        eval_idx = torch.randperm(X.shape[0])[:n_eval]
        X_eval = X[eval_idx]

        lam_max, lr_eos = find_eos_threshold(cond, X, y, device)
        lr = lr_frac * lr_eos
        print(f"  lr = {lr_frac} × {lr_eos:.6f} = {lr:.6f}")

        # Dense trajectory recording
        print(f"  Training for {N_STEPS} steps (recording every step)...")
        outputs = run_dense_trajectory(
            cond, seed, lr, X, y, X_eval, device, n_steps=N_STEPS)

        # Post-transient trajectory
        n_total = len(outputs)
        n_transient = int(n_total * TRANSIENT_FRAC)
        traj = outputs[n_transient:].reshape(n_total - n_transient, -1)
        print(f"  Post-transient trajectory: {len(traj)} points")

        # D₂ at various N
        print(f"  Computing D₂(N):")
        d2_results = d2_at_subsample_sizes(traj, sample_sizes, seed=seed)

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")

        results["conditions"][cond_name] = {
            "lr_frac": lr_frac,
            "seed": seed,
            "lam_max": float(lam_max),
            "lr_eos": float(lr_eos),
            "lr_actual": float(lr),
            "n_steps": N_STEPS,
            "record_every": RECORD_EVERY,
            "n_trajectory_points": len(traj),
            "approach": "dense_recording_every_step",
            "d2_vs_n": d2_results,
        }

    # Save
    out_path = os.path.join(OUT_DIR, "d2_vs_n_neural.json")
    with open(out_path, "w") as f:
        json.dump(_serialize(results), f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary
    print("\n=== D₂(N) CONVERGENCE SUMMARY ===")
    for cond_name, data in results["conditions"].items():
        print(f"\n  {cond_name} at {data['lr_frac']*100:.0f}% EoS "
              f"({data['n_trajectory_points']} pts):")
        for r in data["d2_vs_n"]:
            if r["d2"] is not None:
                print(f"    N={r['n']:5d}  D₂={r['d2']:.3f}")
            else:
                print(f"    N={r['n']:5d}  {r['status']}")


if __name__ == "__main__":
    main()
