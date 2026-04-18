#!/usr/bin/env python3
"""
r1_windowed_d2.py — Windowed D₂ stationarity test
===================================================

Addresses referee issue #9: demonstrate that D₂ is stable across
different temporal windows of the trajectory. If the system hasn't
reached its attractor, D₂ computed over early vs late windows will
differ systematically.

APPROACH:
  Record outputs EVERY STEP within the standard 5000-step protocol
  (same as r1_d2_convergence.py). Then compute D₂ over disjoint
  temporal windows:

    Window 1: steps 1000–2000  (points 1000–2000)
    Window 2: steps 2000–3000  (points 2000–3000)
    Window 3: steps 3000–4000  (points 3000–4000)
    Window 4: steps 4000–5000  (points 4000–5000)

  Also compute D₂ over sliding windows of width 1500 steps, stride 500:
    [1000–2500], [1500–3000], [2000–3500], [2500–4000], [3000–4500], [3500–5000]

  If D₂ is stationary, all windows should give similar values.

  We test at the three main conditions:
    - CNN/CIFAR-10 at 30% EoS  (peak D₂ for CNN)
    - MLP/CIFAR-10 269K at 90% EoS  (peak D₂ for MLP)
    - CNN/CIFAR-10 at 95% EoS  (where sharpness may not have equilibrated)

  3 seeds each to get error bars.

OUTPUT:
  data/supplemental/revision1/windowed_d2_stationarity.json

USAGE:
  python -u code/revision1/r1_windowed_d2.py
  python -u code/revision1/r1_windowed_d2.py --dry-run
  python -u code/revision1/r1_windowed_d2.py --quick   # 1 seed only

REQUIREMENTS: torch, torchvision, numpy, scipy
"""

import argparse
import copy
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
    correlation_dimension,
    _collect_metadata, _serialize,
)


def compute_sharpness(model, X, y, criterion, n_iter=15):
    """Top Hessian eigenvalue via power iteration (MPS-safe)."""
    on_mps = next(model.parameters()).device.type == "mps"
    if on_mps:
        model_h = copy.deepcopy(model).cpu()
        X_h, y_h = X.cpu(), y.cpu()
    else:
        model_h = model
        X_h, y_h = X, y

    v = [torch.randn_like(p) for p in model_h.parameters()]
    v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
    v = [vi / v_norm for vi in v]
    eigenvalue = 0.0
    for _ in range(n_iter):
        model_h.zero_grad()
        loss = criterion(model_h(X_h), y_h)
        grads = torch.autograd.grad(loss, model_h.parameters(),
                                    create_graph=True)
        Hv_terms = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(Hv_terms, model_h.parameters())
        eigenvalue = sum((hv * vi).sum().item()
                         for hv, vi in zip(Hv, v))
        hv_norm = sum((hv ** 2).sum() for hv in Hv).sqrt().item()
        if hv_norm < 1e-12:
            break
        v = [hv.detach() / hv_norm for hv in Hv]
    return abs(eigenvalue)


def find_eos_threshold(cond, X, y, device,
                       warmup_lr=0.01, warmup_steps=1000):
    """Train a fresh seed-0 copy to measure lambda_max (MPS-safe)."""
    model = build_model(cond, seed=0).to(device)
    criterion = nn.MSELoss()
    print(f"  warmup: {warmup_steps} steps at lr={warmup_lr}")
    for t in range(warmup_steps):
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= warmup_lr * p.grad
        if t % 200 == 0:
            print(f"    step {t}: loss = {loss.item():.4f}")
    lam_max = compute_sharpness(model, X, y, criterion, n_iter=15)
    lr_eos = 2.0 / lam_max
    print(f"  lambda_max = {lam_max:.4f}, EoS = {lr_eos:.6f}")
    return float(lam_max), float(lr_eos)

OUT_DIR = os.path.join(ROOT, "data", "supplemental", "revision1")

# ── Conditions to test ──
STATIONARITY_CONDITIONS = [
    {"name": "cnn_cifar_30pct",     "condition": "cnn_cifar",     "lr_frac": 0.30},
    {"name": "mlp_cifar_w85_90pct", "condition": "mlp_cifar_w85", "lr_frac": 0.90},
    {"name": "cnn_cifar_95pct",     "condition": "cnn_cifar",     "lr_frac": 0.95},
]

CNN_CIFAR_CONDITION = Condition(
    name="cnn_cifar", arch="cnn", hidden_dim=None,
    data="cifar_image", output_stem="cnn_cifar")

N_STEPS = 5000
SEEDS_FULL = [0, 1, 2]
SEEDS_QUICK = [0]

# Disjoint windows (each 1000 steps)
DISJOINT_WINDOWS = [
    {"label": "steps_1000_2000", "start": 1000, "end": 2000},
    {"label": "steps_2000_3000", "start": 2000, "end": 3000},
    {"label": "steps_3000_4000", "start": 3000, "end": 4000},
    {"label": "steps_4000_5000", "start": 4000, "end": 5000},
]

# Sliding windows (1500 steps wide, stride 500)
SLIDING_WINDOWS = [
    {"label": "steps_1000_2500", "start": 1000, "end": 2500},
    {"label": "steps_1500_3000", "start": 1500, "end": 3000},
    {"label": "steps_2000_3500", "start": 2000, "end": 3500},
    {"label": "steps_2500_4000", "start": 2500, "end": 4000},
    {"label": "steps_3000_4500", "start": 3000, "end": 4500},
    {"label": "steps_3500_5000", "start": 3500, "end": 5000},
]


def load_cifar10_image(n_samples=2000, seed=42, data_root="./data"):
    """Load CIFAR-10 as images (3x32x32) for the CNN condition."""
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    try:
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform)
    except Exception:
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=False, transform=transform)
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
    """Train for n_steps, recording outputs EVERY step."""
    criterion = nn.MSELoss()
    model = build_model(cond, seed).to(device)
    outputs_rec = []

    for t in range(n_steps):
        with torch.no_grad():
            outputs_rec.append(model(X_eval).cpu().numpy())

        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        if t % 1000 == 0:
            print(f"      step {t}/{n_steps}: loss={loss.item():.6f}")

    return np.array(outputs_rec)


def compute_windowed_d2(outputs, windows, seed):
    """
    Compute D₂ over each specified temporal window.

    Parameters
    ----------
    outputs : ndarray of shape (n_steps, n_eval, n_classes)
        Raw model outputs at every step.
    windows : list of dict with 'label', 'start', 'end'
    seed : int for reproducible subsampling

    Returns
    -------
    list of dict with 'label', 'start', 'end', 'n_points', 'd2'
    """
    results = []
    for w in windows:
        s, e = w["start"], w["end"]
        if e > len(outputs):
            results.append({
                "label": w["label"], "start": s, "end": e,
                "n_points": 0, "d2": None, "status": "out_of_range"
            })
            continue

        window_outputs = outputs[s:e]
        traj = window_outputs.reshape(len(window_outputs), -1)
        d2 = correlation_dimension(traj, seed)
        results.append({
            "label": w["label"],
            "start": s,
            "end": e,
            "n_points": len(traj),
            "d2": float(d2),
        })
        print(f"        {w['label']}: D₂ = {d2:.3f} ({len(traj)} pts)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Windowed D₂ stationarity test (referee issue #9)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="1 seed instead of 3")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for sc in STATIONARITY_CONDITIONS:
            print(f"  {sc['name']} ({sc['condition']} at {sc['lr_frac']*100:.0f}% EoS)")
        print(f"  Seeds: {seeds}")
        print(f"  Disjoint windows: {len(DISJOINT_WINDOWS)}")
        print(f"  Sliding windows: {len(SLIDING_WINDOWS)}")
        print(f"  Output → {OUT_DIR}/windowed_d2_stationarity.json")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"metadata": _collect_metadata(), "conditions": {}}

    # Resolve data root — try multiple known locations
    data_root = args.data_root
    if data_root is None:
        for candidate in [
            os.path.join(ROOT, "data"),
            os.path.join(ROOT, "Torus Theory", "Pytorch", "data"),
            "./data",
        ]:
            if os.path.isdir(os.path.join(candidate, "cifar-10-batches-py")):
                data_root = candidate
                break
        if data_root is None:
            data_root = os.path.join(ROOT, "Torus Theory", "Pytorch", "data")
    print(f"Data root: {data_root}")

    # Pre-load data
    print("Loading CIFAR-10...")
    X_img, y_img = load_cifar10_image(2000, seed=42, data_root=data_root)
    X_flat = X_img.view(2000, -1)

    t0_global = time.time()

    for sc in STATIONARITY_CONDITIONS:
        cond_name = sc["condition"]
        lr_frac = sc["lr_frac"]
        run_name = sc["name"]

        print(f"\n{'='*60}")
        print(f"Condition: {run_name} ({cond_name} at {lr_frac*100:.0f}% EoS)")
        print(f"{'='*60}")

        # Load appropriate data
        if "cnn" in cond_name:
            cond = CNN_CIFAR_CONDITION
            X = X_img.to(device)
            y = y_img.to(device)
        else:
            cond = CONDITIONS[cond_name]
            X, y = load_condition_data(cond, n_samples=2000,
                                       data_root=data_root)
            X = X.to(device)
            y = y.to(device)

        # Eval set
        torch.manual_seed(0)
        n_eval = min(100, X.shape[0])
        eval_idx = torch.randperm(X.shape[0])[:n_eval]
        X_eval = X[eval_idx]

        # EoS threshold
        lam_max, lr_eos = find_eos_threshold(cond, X, y, device)
        lr = lr_frac * lr_eos
        print(f"  lr = {lr_frac} × {lr_eos:.6f} = {lr:.6f}")

        cond_results = {
            "lr_frac": lr_frac,
            "lam_max": float(lam_max),
            "lr_eos": float(lr_eos),
            "lr_actual": float(lr),
            "seeds": seeds,
            "seed_results": {},
        }

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            t0 = time.time()

            # Dense trajectory
            print(f"    Training {N_STEPS} steps (every-step recording)...")
            outputs = run_dense_trajectory(
                cond, seed, lr, X, y, X_eval, device, n_steps=N_STEPS)

            print(f"    Trajectory: {len(outputs)} points")

            # Full post-transient D₂ (for reference)
            n_transient = len(outputs) // 5  # first 20%
            full_traj = outputs[n_transient:].reshape(
                len(outputs) - n_transient, -1)
            d2_full = correlation_dimension(full_traj, seed)
            print(f"    Full post-transient D₂ = {d2_full:.3f} "
                  f"({len(full_traj)} pts)")

            # Disjoint windows
            print(f"    Disjoint windows:")
            disjoint_results = compute_windowed_d2(
                outputs, DISJOINT_WINDOWS, seed)

            # Sliding windows
            print(f"    Sliding windows:")
            sliding_results = compute_windowed_d2(
                outputs, SLIDING_WINDOWS, seed)

            elapsed = time.time() - t0
            print(f"    ({elapsed:.1f}s)")

            cond_results["seed_results"][str(seed)] = {
                "d2_full_post_transient": float(d2_full),
                "n_full_post_transient": len(full_traj),
                "disjoint_windows": disjoint_results,
                "sliding_windows": sliding_results,
            }

        # Compute summary statistics across seeds
        summary = {"disjoint": {}, "sliding": {}}
        for window_type, window_list, key in [
            ("disjoint", DISJOINT_WINDOWS, "disjoint_windows"),
            ("sliding", SLIDING_WINDOWS, "sliding_windows"),
        ]:
            for i, w in enumerate(window_list):
                d2_vals = []
                for seed in seeds:
                    sr = cond_results["seed_results"][str(seed)]
                    d2 = sr[key][i]["d2"]
                    if d2 is not None:
                        d2_vals.append(d2)
                if d2_vals:
                    summary[window_type][w["label"]] = {
                        "d2_mean": float(np.mean(d2_vals)),
                        "d2_std": float(np.std(d2_vals)),
                        "n_seeds": len(d2_vals),
                    }

        # Full post-transient summary
        full_d2s = [cond_results["seed_results"][str(s)]["d2_full_post_transient"]
                    for s in seeds]
        summary["full_post_transient"] = {
            "d2_mean": float(np.mean(full_d2s)),
            "d2_std": float(np.std(full_d2s)),
        }

        cond_results["summary"] = summary
        results["conditions"][run_name] = cond_results

    # Save
    out_path = os.path.join(OUT_DIR, "windowed_d2_stationarity.json")
    with open(out_path, "w") as f:
        json.dump(_serialize(results), f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print summary table
    elapsed_total = time.time() - t0_global
    print(f"\n{'='*70}")
    print(f"STATIONARITY SUMMARY ({elapsed_total/60:.1f} min total)")
    print(f"{'='*70}")

    for sc in STATIONARITY_CONDITIONS:
        run_name = sc["name"]
        cond_data = results["conditions"][run_name]
        s = cond_data["summary"]

        print(f"\n  {run_name}:")
        print(f"    Full post-transient: D₂ = {s['full_post_transient']['d2_mean']:.2f} "
              f"± {s['full_post_transient']['d2_std']:.2f}")
        print(f"    Disjoint windows:")
        for label, vals in s["disjoint"].items():
            print(f"      {label}: D₂ = {vals['d2_mean']:.2f} ± {vals['d2_std']:.2f}")
        print(f"    Sliding windows:")
        for label, vals in s["sliding"].items():
            print(f"      {label}: D₂ = {vals['d2_mean']:.2f} ± {vals['d2_std']:.2f}")

    # Stationarity verdict
    print(f"\n  STATIONARITY DIAGNOSTICS:")
    for sc in STATIONARITY_CONDITIONS:
        run_name = sc["name"]
        s = results["conditions"][run_name]["summary"]
        disjoint_means = [v["d2_mean"] for v in s["disjoint"].values()]
        if disjoint_means:
            d2_range = max(disjoint_means) - min(disjoint_means)
            d2_mean = np.mean(disjoint_means)
            cv = d2_range / d2_mean * 100 if d2_mean > 0 else float("inf")
            print(f"    {run_name}: D₂ range across windows = {d2_range:.2f} "
                  f"(CV = {cv:.1f}% of mean)")
            if cv < 15:
                print(f"      → STATIONARY (variation < 15%)")
            elif cv < 30:
                print(f"      → MARGINAL (variation 15-30%)")
            else:
                print(f"      → NON-STATIONARY (variation > 30%)")


if __name__ == "__main__":
    main()
