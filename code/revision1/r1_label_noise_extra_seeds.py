#!/usr/bin/env python3
"""
r1_label_noise_extra_seeds.py — Additional seeds for label-noise sweep
======================================================================

The original label-noise sweep (r1_label_noise_sweep.py) used seeds
[0, 1, 2]. The referee noted that 3 seeds gives ~50% relative uncertainty
on the standard deviation estimate. This script runs seeds [3, 4, 5, 6]
(4 additional seeds) to bring the total to 7 seeds per condition.

We focus on the noisiest points where variance is highest:
  - All noise levels for completeness, but especially p=0.50 and p=0.75
    for CNN (where σ was 0.68 and 0.57 respectively)

Same protocol as r1_label_noise_sweep.py — identical architectures,
learning rates, noise application, measurement pipeline. Only the
training seeds differ.

OUTPUT:
  data/main/revision1/label_noise_sweep_extra_seeds.json

After running, merge with the original label_noise_sweep.json to get
7-seed statistics.

USAGE:
  python -u code/revision1/r1_label_noise_extra_seeds.py
  python -u code/revision1/r1_label_noise_extra_seeds.py --dry-run
  python -u code/revision1/r1_label_noise_extra_seeds.py --condition cnn_cifar
  python -u code/revision1/r1_label_noise_extra_seeds.py --seeds 3 4 5 6 7 8 9

REQUIREMENTS: torch, torchvision, numpy, scipy
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

warnings.filterwarnings("ignore")

# Reuse protocol from original sweep
PROTOCOL_VERSION = "r1-phase3a-extra-seeds-2026-04-16"

# ============================================================
# NOISE LEVELS (same as original)
# ============================================================

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
NOISE_SEED = 137  # deterministic: same samples get relabeled across seeds

# ============================================================
# ARCHITECTURES (exact copies from r1_label_noise_sweep.py)
# ============================================================

class MLPCifar(nn.Module):
    """2-hidden-layer tanh MLP: 3072 -> W -> W -> 10."""
    def __init__(self, input_dim=3072, hidden_dim=85, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class SmallCNN(nn.Module):
    """Exactly Experiment K CNN. 268,650 parameters."""
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


# ============================================================
# DATA LOADING
# ============================================================

def load_cifar10_image(n_samples=2000, seed=42, data_root="./data"):
    """Load CIFAR-10 as (n, 3, 32, 32) images + one-hot labels."""
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
    y_onehot = torch.zeros(n_samples, 10)
    for i, label in enumerate(labels):
        y_onehot[i, label] = 1.0
    return X, y_onehot, labels


def apply_label_noise(y_onehot, raw_labels, noise_frac, n_classes=10):
    """
    Corrupt a fraction of labels uniformly at random.
    Uses NOISE_SEED so that the SAME subset of samples is relabeled
    regardless of the training seed.
    """
    if noise_frac <= 0.0:
        return y_onehot.clone(), 0

    n = len(raw_labels)
    rng = np.random.RandomState(NOISE_SEED)
    mask = rng.random(n) < noise_frac
    new_labels = rng.randint(0, n_classes, size=n)

    y_noisy = y_onehot.clone()
    n_corrupted = 0
    for i in range(n):
        if mask[i]:
            y_noisy[i] = 0.0
            y_noisy[i, new_labels[i]] = 1.0
            if new_labels[i] != raw_labels[i]:
                n_corrupted += 1

    return y_noisy, n_corrupted


# ============================================================
# CORE MEASUREMENTS (exact copies)
# ============================================================

def build_model(arch, seed, hidden_dim=85):
    torch.manual_seed(seed)
    if arch == "cnn":
        return SmallCNN()
    elif arch == "mlp":
        return MLPCifar(input_dim=3072, hidden_dim=hidden_dim, output_dim=10)
    raise ValueError(f"Unknown arch: {arch}")


def clone_perturbed(model, eps, seed):
    clone = copy.deepcopy(model)
    rng = torch.Generator()
    rng.manual_seed(seed + 999999)
    flat_params = [p.data.view(-1) for p in clone.parameters()]
    flat = torch.cat(flat_params)
    direction = torch.randn(flat.shape, generator=rng)
    direction = direction / direction.norm()
    direction = direction.to(flat.device)
    offset = 0
    for p in clone.parameters():
        numel = p.numel()
        p.data += eps * direction[offset:offset + numel].view(p.shape)
        offset += numel
    return clone


def compute_sharpness(model, X, y, criterion, n_iter=15):
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


def correlation_dimension(traj, seed):
    if len(traj) > 2000:
        idx = np.random.RandomState(seed).choice(
            len(traj), 2000, replace=False)
        traj_sub = traj[idx]
    else:
        traj_sub = traj
    n = len(traj_sub)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(traj_sub[i] - traj_sub[j]))
    dists = np.array(dists)
    if len(dists) == 0:
        return float("nan")
    log_eps = np.linspace(
        np.log(np.percentile(dists, 1) + 1e-15),
        np.log(np.percentile(dists, 95)),
        20,
    )
    log_C = [
        np.log(max(np.sum(dists < np.exp(le)) / (n * (n - 1) / 2), 1e-30))
        for le in log_eps
    ]
    log_C = np.array(log_C)
    return float(stats.linregress(log_eps[4:16], log_C[4:16])[0])


def find_eos_threshold(arch, X, y, device, hidden_dim=85,
                       warmup_lr=0.01, warmup_steps=1000, verbose=True):
    model = build_model(arch, seed=0, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    if verbose:
        print(f"  warmup: {warmup_steps} steps at lr={warmup_lr}")
    for t in range(warmup_steps):
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= warmup_lr * p.grad
        if verbose and t % 200 == 0:
            print(f"    step {t}: loss = {loss.item():.4f}")
    lam_max = compute_sharpness(model, X, y, criterion, n_iter=15)
    lr_eos = 2.0 / lam_max
    if verbose:
        print(f"  lambda_max = {lam_max:.4f}, EoS = {lr_eos:.6f}")
    return float(lam_max), float(lr_eos)


def run_single_seed(arch, seed, lr, X_train, y_train, X_eval, device,
                    hidden_dim=85, n_steps=5000, eps=1e-5):
    criterion = nn.MSELoss()
    model = build_model(arch, seed, hidden_dim=hidden_dim).to(device)
    perturbed = clone_perturbed(model, eps, seed).to(device)

    distances = np.zeros(n_steps)
    losses_rec = np.zeros(n_steps)
    gn_rec = np.zeros(n_steps)
    sharp_rec = []
    outputs_rec = []

    for t in range(n_steps):
        with torch.no_grad():
            d = torch.norm(model(X_eval) - perturbed(X_eval)).item()
            distances[t] = d

        if t % 10 == 0:
            with torch.no_grad():
                outputs_rec.append(model(X_eval).cpu().numpy())

        model.zero_grad()
        loss = criterion(model(X_train), y_train)
        losses_rec[t] = loss.item()
        loss.backward()
        with torch.no_grad():
            gn = sum((p.grad ** 2).sum() for p in model.parameters()
                     if p.grad is not None).sqrt().item()
            gn_rec[t] = gn
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        perturbed.zero_grad()
        loss2 = criterion(perturbed(X_train), y_train)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        if t > 0 and t % 500 == 0:
            sharp = compute_sharpness(model, X_train, y_train, criterion,
                                      n_iter=15)
            sharp_rec.append(sharp)

        if t > 0 and t % 1000 == 0:
            print(f"      step {t}/{n_steps}  loss={loss.item():.4f}  "
                  f"dist={d:.2e}", flush=True)

    log_d = np.log(distances + 1e-30)
    start = int(n_steps * 0.2)
    end = int(n_steps * 0.8)
    lyap = float(stats.linregress(
        np.arange(start, end), log_d[start:end])[0])

    outputs = np.array(outputs_rec)
    traj_start = len(outputs) // 5
    traj = outputs[traj_start:].reshape(len(outputs) - traj_start, -1)
    centered = traj - traj.mean(axis=0)
    try:
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        var_exp = (sv ** 2) / (sv ** 2).sum() * 100
        pc1 = float(var_exp[0])
        pc2 = float(var_exp[1]) if len(var_exp) > 1 else 0.0
    except Exception:
        pc1, pc2 = 100.0, 0.0
    cd = correlation_dimension(traj, seed)

    return {
        "lyapunov": lyap,
        "corr_dim": float(cd),
        "pc1": pc1,
        "pc2": pc2,
        "sharpness_series": [float(s) for s in sharp_rec],
        "grad_norm_series": [float(g) for g in gn_rec[::10]],
        "loss_series": [float(l) for l in losses_rec[::10]],
    }


# ============================================================
# CONDITION SPECS
# ============================================================

CONDITION_SPECS = {
    "cnn_cifar": {
        "arch": "cnn",
        "hidden_dim": None,
        "lr_frac": 0.30,
        "data_format": "image",
    },
    "mlp_cifar_w85": {
        "arch": "mlp",
        "hidden_dim": 85,
        "lr_frac": 0.90,
        "data_format": "flat",
    },
}


def _collect_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "script": os.path.basename(__file__),
        "protocol_version": PROTOCOL_VERSION,
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "git_commit": commit,
        "note": "Extra seeds to supplement original 3-seed label_noise_sweep.json",
    }


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return None
    return obj


# ============================================================
# MERGE UTILITY
# ============================================================

def merge_results(original_path, extra_path, output_path):
    """
    Merge original (seeds 0,1,2) and extra (seeds 3,4,5,6) results
    into a single file with updated summary statistics.
    """
    with open(original_path) as f:
        orig = json.load(f)
    with open(extra_path) as f:
        extra = json.load(f)

    merged = copy.deepcopy(orig)
    merged["revision1_metadata"]["merge_note"] = (
        f"Merged original seeds {orig['conditions'][list(orig['conditions'].keys())[0]].get('seeds', [0,1,2])} "
        f"with extra seeds from {os.path.basename(extra_path)}"
    )

    for cname in merged["conditions"]:
        if cname not in extra["conditions"]:
            continue

        orig_cond = merged["conditions"][cname]
        extra_cond = extra["conditions"][cname]

        orig_seeds = orig_cond.get("seeds", [0, 1, 2])
        extra_seeds = extra_cond.get("seeds", [3, 4, 5, 6])
        all_seeds = orig_seeds + extra_seeds
        orig_cond["seeds"] = all_seeds

        for p_key in orig_cond["results"]:
            if p_key not in extra_cond["results"]:
                continue

            orig_bucket = orig_cond["results"][p_key]
            extra_bucket = extra_cond["results"][p_key]

            for key in ["lyapunov", "corr_dim", "pc1", "pc2"]:
                orig_bucket[key].extend(extra_bucket[key])

            for key in ["sharpness_series", "grad_norm_series", "loss_series"]:
                if key in extra_bucket:
                    orig_bucket[key].extend(extra_bucket[key])

            # Recompute summary stats
            d2_vals = orig_bucket["corr_dim"]
            lam_vals = orig_bucket["lyapunov"]
            orig_bucket["d2_mean"] = float(np.mean(d2_vals))
            orig_bucket["d2_std"] = float(np.std(d2_vals))
            orig_bucket["lam_mean"] = float(np.mean(lam_vals))
            orig_bucket["lam_std"] = float(np.std(lam_vals))

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged results saved to {output_path}")

    # Print comparison table
    print("\n  COMPARISON: 3 seeds vs 7 seeds")
    print(f"  {'Condition':<20} {'p':>6} {'D₂(3)':>10} {'D₂(7)':>10} {'σ(3)':>8} {'σ(7)':>8}")
    print("  " + "-" * 64)

    for cname in merged["conditions"]:
        if cname not in extra["conditions"]:
            continue
        for p_key in sorted(merged["conditions"][cname]["results"].keys(),
                            key=float):
            bucket = merged["conditions"][cname]["results"][p_key]
            all_d2 = bucket["corr_dim"]
            d2_3 = np.mean(all_d2[:3])
            d2_7 = np.mean(all_d2)
            std_3 = np.std(all_d2[:3])
            std_7 = np.std(all_d2)
            print(f"  {cname:<20} {p_key:>6} {d2_3:>10.2f} {d2_7:>10.2f} "
                  f"{std_3:>8.2f} {std_7:>8.2f}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extra seeds for label-noise sweep (strengthens error bars)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=list(CONDITION_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6],
                        help="Training seeds (default: 3 4 5 6)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--merge", action="store_true",
                        help="After running, merge with original results")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    output_dir = args.output_dir or os.path.join(
        repo_root, "data", "main", "revision1")
    # Resolve data root — try multiple known locations
    data_root = args.data_root
    if data_root is None:
        for candidate in [
            os.path.join(repo_root, "data"),
            os.path.join(repo_root, "Torus Theory", "Pytorch", "data"),
        ]:
            if os.path.isdir(os.path.join(candidate, "cifar-10-batches-py")):
                data_root = candidate
                break
        if data_root is None:
            data_root = os.path.join(repo_root, "Torus Theory", "Pytorch", "data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "label_noise_sweep_extra_seeds.json")
    seeds = args.seeds

    conditions = (
        [args.condition] if args.condition
        else list(CONDITION_SPECS.keys())
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    total_runs = len(conditions) * len(NOISE_LEVELS) * len(seeds)

    print("=" * 64)
    print("Label-noise sweep: EXTRA SEEDS")
    print(f"  conditions: {conditions}")
    print(f"  noise levels: {NOISE_LEVELS}")
    print(f"  seeds: {seeds}")
    print(f"  n_steps: {args.n_steps}")
    print(f"  device: {device}")
    print(f"  output: {output_path}")
    print(f"  total training runs: {total_runs}")
    print("=" * 64)

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for cname in conditions:
            spec = CONDITION_SPECS[cname]
            print(f"\n  {cname} ({spec['arch']}, lr_frac={spec['lr_frac']}):")
            for p in NOISE_LEVELS:
                print(f"    p={p}: seeds {seeds}")
        return

    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    X_img, y_clean, raw_labels = load_cifar10_image(
        n_samples=2000, seed=42, data_root=data_root)
    X_flat = X_img.view(2000, -1)

    done = 0
    t0_global = time.time()

    results = {
        "experiment": "label_noise_sweep_extra_seeds",
        "revision1_metadata": _collect_metadata(),
        "conditions": {},
    }

    for cname in conditions:
        spec = CONDITION_SPECS[cname]
        arch = spec["arch"]
        hidden_dim = spec["hidden_dim"] or 85
        lr_frac = spec["lr_frac"]

        if spec["data_format"] == "image":
            X_train_base = X_img
        else:
            X_train_base = X_flat

        X_train_base = X_train_base.to(device)
        y_clean_dev = y_clean.to(device)

        print(f"\n{'='*64}")
        print(f"Condition: {cname} ({arch}, lr_frac={lr_frac})")
        print(f"{'='*64}")

        # EoS threshold on CLEAN data
        print("\nFinding EoS threshold (clean data, p=0.0)...")
        lam_max, lr_eos = find_eos_threshold(
            arch, X_train_base, y_clean_dev, device,
            hidden_dim=hidden_dim)
        lr = lr_frac * lr_eos
        print(f"  lr = {lr_frac} × {lr_eos:.6f} = {lr:.6f}")

        torch.manual_seed(0)
        n_eval = min(100, X_train_base.shape[0])
        eval_idx = torch.randperm(X_train_base.shape[0])[:n_eval]
        X_eval = X_train_base[eval_idx]

        n_params = build_model(arch, 0, hidden_dim=hidden_dim).count_params()

        cond_results = {
            "arch": arch,
            "n_params": int(n_params),
            "hidden_dim": hidden_dim if arch == "mlp" else None,
            "lr_frac": lr_frac,
            "lr": float(lr),
            "lam_max": float(lam_max),
            "lr_eos": float(lr_eos),
            "noise_levels": NOISE_LEVELS,
            "seeds": seeds,
            "noise_seed": NOISE_SEED,
            "results": {},
        }

        for p_noise in NOISE_LEVELS:
            print(f"\n  --- noise_frac = {p_noise} ---")

            y_noisy, n_corrupted = apply_label_noise(
                y_clean, raw_labels, p_noise)
            y_noisy_dev = y_noisy.to(device)
            effective_rate = n_corrupted / len(raw_labels)
            print(f"  labels corrupted: {n_corrupted}/{len(raw_labels)} "
                  f"(effective rate: {effective_rate:.3f})")

            bucket = {
                "lyapunov": [], "corr_dim": [],
                "pc1": [], "pc2": [],
                "sharpness_series": [], "grad_norm_series": [],
                "loss_series": [],
                "n_corrupted": n_corrupted,
                "effective_noise_rate": effective_rate,
            }

            for seed in seeds:
                done += 1
                elapsed = time.time() - t0_global
                eta_min = (elapsed / done * (total_runs - done)) / 60 \
                    if done > 1 else 0.0
                print(f"\n  [{done}/{total_runs}] {cname} "
                      f"p={p_noise} seed={seed}  "
                      f"ETA: {eta_min:.1f}min", flush=True)

                r = run_single_seed(
                    arch, seed=seed, lr=lr,
                    X_train=X_train_base, y_train=y_noisy_dev,
                    X_eval=X_eval, device=device,
                    hidden_dim=hidden_dim, n_steps=args.n_steps, eps=1e-5,
                )

                for key in ["lyapunov", "corr_dim", "pc1", "pc2"]:
                    bucket[key].append(r[key])
                for key in ["sharpness_series", "grad_norm_series",
                            "loss_series"]:
                    bucket[key].append(r[key])

                print(f"    λ={r['lyapunov']:.4f}  D₂={r['corr_dim']:.2f}  "
                      f"pc1={r['pc1']:.1f}%  pc2={r['pc2']:.1f}%")

            d2_vals = bucket["corr_dim"]
            lam_vals = bucket["lyapunov"]
            bucket["d2_mean"] = float(np.mean(d2_vals))
            bucket["d2_std"] = float(np.std(d2_vals))
            bucket["lam_mean"] = float(np.mean(lam_vals))
            bucket["lam_std"] = float(np.std(lam_vals))
            print(f"\n  p={p_noise} summary: "
                  f"D₂ = {bucket['d2_mean']:.2f} ± {bucket['d2_std']:.2f}, "
                  f"λ = {bucket['lam_mean']:.4f} ± {bucket['lam_std']:.4f}")

            cond_results["results"][str(p_noise)] = bucket

        results["conditions"][cname] = cond_results

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(_serialize(results), f, indent=2)
        print(f"\n  Saved intermediate results to {output_path}")

    # Final save
    with open(output_path, "w") as f:
        json.dump(_serialize(results), f, indent=2)

    elapsed_total = time.time() - t0_global
    print(f"\n{'='*64}")
    print(f"Extra seeds complete. {total_runs} runs in {elapsed_total/60:.1f} min")
    print(f"Output: {output_path}")
    print(f"{'='*64}")

    # Print summary
    print("\nSummary (extra seeds only):")
    print(f"{'Condition':<20} {'p':>6} {'D₂':>12} {'λ':>12}")
    print("-" * 52)
    for cname in conditions:
        cond_data = results["conditions"][cname]
        for p_noise in NOISE_LEVELS:
            key = str(p_noise)
            r = cond_data["results"][key]
            print(f"{cname:<20} {p_noise:>6.2f} "
                  f"{r['d2_mean']:>6.2f}±{r['d2_std']:<5.2f} "
                  f"{r['lam_mean']:>7.4f}±{r['lam_std']:<6.4f}")
        print()

    # Auto-merge if requested
    if args.merge:
        original_path = os.path.join(output_dir, "label_noise_sweep.json")
        merged_path = os.path.join(output_dir, "label_noise_sweep_merged.json")
        if os.path.exists(original_path):
            merge_results(original_path, output_path, merged_path)
        else:
            print(f"\nWARNING: Cannot merge — original not found at {original_path}")
            print("Run merge manually after placing the original file.")


if __name__ == "__main__":
    main()
