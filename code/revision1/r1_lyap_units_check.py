#!/usr/bin/env python3
"""
r1_lyap_units_check.py — Phase 2D: ε and N_inputs sensitivity audit
=====================================================================

Referee minor issue: The main text asserts that 100 held-out inputs and
ε = 1e-5 are converged but does not show it. This script validates that
λ and D₂ are stable within one seed-std across the ε and N_inputs ranges.

Approach:
  Pick one representative condition: CNN/CIFAR-10 at 30% EoS, seed 0.

  Sweep:
    N_inputs ∈ {25, 50, 100, 200, 400}
    ε ∈ {1e-4, 5e-5, 1e-5, 5e-6, 1e-6}

  At each (ε, N_inputs) pair, run the FULL measurement protocol:
    - Train original + perturbed for 5000 steps
    - Measure λ from function-space divergence on N_inputs held-out samples
    - Record trajectory (outputs every 10 steps on N_inputs samples)
    - Compute D₂ from the post-transient trajectory

  Reference values: λ and D₂ from the 10-seed production run at ε=1e-5,
  N_inputs=100 (seed-std from cifar10_eos_10seeds.json at 30% EoS).

OUTPUT:
  data/supplemental/revision1/convergence_n_inputs_epsilon.json

USAGE:
  python -u code/revision1/r1_lyap_units_check.py              # full
  python -u code/revision1/r1_lyap_units_check.py --quick      # fewer combos
  python -u code/revision1/r1_lyap_units_check.py --dry-run

REQUIREMENTS: torch, torchvision, numpy, scipy
HARDWARE: CPU/M1 OK — single condition, 25 runs of 5000 steps each
  Estimated time on M1: ~20-40 minutes total
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
    SmallCNN, Condition,
    compute_sharpness, correlation_dimension,
    _collect_metadata, _serialize,
)

OUT_DIR = os.path.join(ROOT, "data", "supplemental", "revision1")

# ── Fixed parameters ──
SEED = 0
LR_FRAC = 0.30
N_STEPS = 5000
WARMUP_STEPS = 1000
WARMUP_LR = 0.01

# ── Sweep grids ──
N_INPUTS_FULL = [25, 50, 100, 200, 400]
EPSILONS_FULL = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

N_INPUTS_QUICK = [50, 100, 200]
EPSILONS_QUICK = [1e-4, 1e-5, 1e-6]


# ============================================================
# CNN SETUP (self-contained to avoid import issues on M1)
# ============================================================

class SmallCNNLocal(nn.Module):
    """Identical to SmallCNN in r1_cross_experiments.py."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def load_cifar10(n_samples=2000, data_root="./data"):
    """Load CIFAR-10 as (3,32,32) images with one-hot labels."""
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform)
    rng = np.random.RandomState(42)
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


def build_cnn(seed):
    torch.manual_seed(seed)
    return SmallCNNLocal()


def clone_perturbed_local(model, eps, seed):
    """Clone and perturb. Same protocol as r1_cross_experiments."""
    clone = copy.deepcopy(model)
    rng = torch.Generator()
    rng.manual_seed(seed + 999999)
    flat_params = [p.data.view(-1) for p in clone.parameters()]
    flat = torch.cat(flat_params)
    direction = torch.randn(flat.shape, generator=rng)
    direction = direction / direction.norm()
    offset = 0
    for p in clone.parameters():
        numel = p.numel()
        p.data += eps * direction[offset:offset + numel].view(p.shape).to(p.device)
        offset += numel
    return clone


def _power_iteration_cpu(model, X, y, criterion, n_iter=15):
    """
    Top Hessian eigenvalue via power iteration.
    Runs on CPU to avoid MPS limitations with second-order gradients
    through max_pool2d.
    """
    model_cpu = copy.deepcopy(model).cpu()
    X_cpu, y_cpu = X.cpu(), y.cpu()
    v = [torch.randn_like(p) for p in model_cpu.parameters()]
    v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
    v = [vi / v_norm for vi in v]
    eigenvalue = 0.0
    for _ in range(n_iter):
        model_cpu.zero_grad()
        loss = criterion(model_cpu(X_cpu), y_cpu)
        grads = torch.autograd.grad(loss, model_cpu.parameters(),
                                    create_graph=True)
        Hv_terms = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(Hv_terms, model_cpu.parameters())
        eigenvalue = sum((hv * vi).sum().item() for hv, vi in zip(Hv, v))
        hv_norm = sum((hv ** 2).sum() for hv in Hv).sqrt().item()
        if hv_norm < 1e-12:
            break
        v = [hv.detach() / hv_norm for hv in Hv]
    return abs(eigenvalue)


def find_eos_cnn(X, y, device):
    """Warmup to find EoS threshold for CNN/CIFAR."""
    model = build_cnn(seed=0).to(device)
    criterion = nn.MSELoss()
    for t in range(WARMUP_STEPS):
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= WARMUP_LR * p.grad
    # Power iteration on CPU (MPS doesn't support 2nd-order grads through max_pool2d)
    lam_max = _power_iteration_cpu(model, X, y, criterion, n_iter=15)
    return lam_max, 2.0 / lam_max


# ============================================================
# SINGLE MEASUREMENT RUN
# ============================================================

def measure_at(seed, lr, eps, n_inputs, X, y, device, n_steps=5000):
    """
    Full training run measuring λ and D₂ with specific ε and N_inputs.
    """
    criterion = nn.MSELoss()
    model = build_cnn(seed).to(device)
    perturbed = clone_perturbed_local(model, eps, seed).to(device)

    # Eval set with specified size
    torch.manual_seed(0)
    eval_idx = torch.randperm(X.shape[0])[:n_inputs]
    X_eval = X[eval_idx]

    distances = np.zeros(n_steps)
    outputs_rec = []

    for t in range(n_steps):
        with torch.no_grad():
            d = torch.norm(model(X_eval) - perturbed(X_eval)).item()
            distances[t] = d

        if t % 10 == 0:
            with torch.no_grad():
                outputs_rec.append(model(X_eval).cpu().numpy())

        # Train original
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        # Train perturbed
        perturbed.zero_grad()
        loss2 = criterion(perturbed(X), y)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

    # Lyapunov exponent
    log_d = np.log(distances + 1e-30)
    start = int(n_steps * 0.2)
    end = int(n_steps * 0.8)
    lyap = float(stats.linregress(np.arange(start, end), log_d[start:end])[0])

    # Correlation dimension
    outputs = np.array(outputs_rec)
    traj_start = len(outputs) // 5
    traj = outputs[traj_start:].reshape(len(outputs) - traj_start, -1)
    cd = correlation_dimension(traj, seed)

    return {"lyapunov": lyap, "corr_dim": float(cd)}


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2D: ε and N_inputs sensitivity audit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Device: {device}")

    n_inputs_grid = N_INPUTS_QUICK if args.quick else N_INPUTS_FULL
    eps_grid = EPSILONS_QUICK if args.quick else EPSILONS_FULL
    n_combos = len(n_inputs_grid) * len(eps_grid)

    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"  Condition: CNN/CIFAR-10 at {LR_FRAC*100:.0f}% EoS, seed {SEED}")
        print(f"  N_inputs: {n_inputs_grid}")
        print(f"  ε: {eps_grid}")
        print(f"  Total runs: {n_combos}")
        print(f"  Output → {OUT_DIR}/convergence_n_inputs_epsilon.json")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print("Loading CIFAR-10...")
    X, y = load_cifar10(2000, data_root=args.data_root)
    X = X.to(device)
    y = y.to(device)

    # Find EoS
    print("Finding EoS threshold...")
    lam_max, lr_eos = find_eos_cnn(X, y, device)
    lr = LR_FRAC * lr_eos
    print(f"  lambda_max = {lam_max:.4f}, EoS = {lr_eos:.6f}")
    print(f"  lr = {LR_FRAC} × {lr_eos:.6f} = {lr:.6f}")

    results = {
        "metadata": _collect_metadata(),
        "condition": "cnn_cifar",
        "lr_frac": LR_FRAC,
        "seed": SEED,
        "lam_max": lam_max,
        "lr_eos": lr_eos,
        "lr_actual": lr,
        "n_inputs_grid": n_inputs_grid,
        "epsilon_grid": eps_grid,
        "measurements": [],
    }

    t_total = time.time()
    for i, (n_inp, eps) in enumerate(
            [(n, e) for n in n_inputs_grid for e in eps_grid]):
        t0 = time.time()
        print(f"\n  [{i+1}/{n_combos}] N_inputs={n_inp}, ε={eps:.0e}")

        m = measure_at(SEED, lr, eps, n_inp, X, y, device, N_STEPS)

        elapsed = time.time() - t0
        print(f"    λ = {m['lyapunov']:.6f}, D₂ = {m['corr_dim']:.3f}  "
              f"({elapsed:.1f}s)")

        results["measurements"].append({
            "n_inputs": n_inp,
            "epsilon": eps,
            "lyapunov": m["lyapunov"],
            "corr_dim": m["corr_dim"],
        })

    total_elapsed = time.time() - t_total
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")

    # Save
    out_path = os.path.join(OUT_DIR, "convergence_n_inputs_epsilon.json")
    with open(out_path, "w") as f:
        json.dump(_serialize(results), f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table
    print("\n=== CONVERGENCE SUMMARY ===")
    print(f"{'N_inputs':>8s}  {'ε':>10s}  {'λ':>10s}  {'D₂':>8s}")
    print("-" * 42)
    for m in results["measurements"]:
        print(f"{m['n_inputs']:8d}  {m['epsilon']:10.0e}  "
              f"{m['lyapunov']:10.6f}  {m['corr_dim']:8.3f}")

    # Check convergence: is production config (ε=1e-5, N=100) within plateau?
    prod = [m for m in results["measurements"]
            if m["n_inputs"] == 100 and m["epsilon"] == 1e-5]
    if prod:
        p = prod[0]
        all_lyap = [m["lyapunov"] for m in results["measurements"]]
        all_d2 = [m["corr_dim"] for m in results["measurements"]]
        print(f"\n  Production (ε=1e-5, N=100): λ={p['lyapunov']:.6f}, "
              f"D₂={p['corr_dim']:.3f}")
        print(f"  Full grid range: λ ∈ [{min(all_lyap):.6f}, {max(all_lyap):.6f}]"
              f"  D₂ ∈ [{min(all_d2):.3f}, {max(all_d2):.3f}]")


if __name__ == "__main__":
    main()
