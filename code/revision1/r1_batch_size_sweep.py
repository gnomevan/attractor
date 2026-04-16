"""
r1_batch_size_sweep.py  —  Batch-size sweep: does the attractor survive SGD?
=============================================================================

Addresses PRL referee concern #1 (robustness beyond full-batch GD):
the strange attractor is measured under full-batch gradient descent,
but real training uses mini-batch SGD. This script sweeps batch size
from full-batch (B=2000) down to B=100, measuring D₂ and λ at each,
to test whether the fractal attractor structure persists under
stochastic gradient noise.

Design
------

Batch sizes B ∈ {2000, 1000, 500, 200, 100}.
  - B=2000 is the existing full-batch protocol (should reproduce)
  - B=1000, 500 are large-batch (modest gradient noise)
  - B=200, 100 are small-batch (substantial gradient noise)

Two conditions at their respective peak-D₂ learning rates:
  1. CNN/CIFAR-10 (268,650 params) at 30% EoS
  2. MLP/CIFAR-10 269K (w=85) at 90% EoS

Each at 5 batch sizes × 3 seeds = 15 runs per condition, 30 total.

Protocol
--------

Identical to r1_cross_experiments.py / r1_label_noise_sweep.py with
two key modifications:

1. **Mini-batch training**: at each step t, a random subset of B
   samples is drawn from the 2000-sample training set. Both the
   original and perturbed model copies see the SAME mini-batch at
   each step (paired-noise protocol). This ensures the Lyapunov
   divergence measurement isolates deterministic sensitivity, not
   stochastic forcing — the same technique used in stochastic
   dynamical systems.

   Implementation: a shared DataLoader-like index generator with
   a per-step RNG seeded by (seed * 100000 + t) ensures both
   models see identical batches and the batch sequence is
   deterministic across runs.

2. **Function-space recordings are unchanged**: the eval set is
   the same fixed 100-point subset, and outputs are recorded every
   10 steps. D₂ is computed on the same Grassberger-Procaccia
   pipeline. The stochastic noise is in the gradient updates only;
   the trajectory through function space is the clean signal.

The EoS threshold is measured ONCE at B=2000 (full-batch) and held
fixed across all batch sizes. The learning rate = lr_frac × lr_eos
is therefore constant; only the gradient noise level changes.

Output schema
-------------

JSON file:

  {
    "experiment": "batch_size_sweep",
    "conditions": {
      "cnn_cifar": {
        "arch": "cnn", "n_params": 268650,
        "lr_frac": 0.30, "lr": float, "lam_max": float, "lr_eos": float,
        "batch_sizes": [2000, 1000, 500, 200, 100],
        "seeds": [0, 1, 2],
        "results": {
          "2000": {"lyapunov": [...], "corr_dim": [...], ...},
          "1000": {"lyapunov": [...], "corr_dim": [...], ...},
          ...
        }
      },
      "mlp_cifar_w85": { ... }
    },
    "revision1_metadata": { ... }
  }

Usage
-----

   # Full run (30 training runs, ~2-3 hrs on T4)
   python -u r1_batch_size_sweep.py

   # Quick sanity (3 batch sizes instead of 5)
   python -u r1_batch_size_sweep.py --quick

   # Dry run
   python -u r1_batch_size_sweep.py --dry-run

   # Single condition
   python -u r1_batch_size_sweep.py --condition cnn_cifar

   # Custom seeds
   python -u r1_batch_size_sweep.py --seeds 0 1 2 3 4

Addresses: PRL referee concern about full-batch GD limitation.
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


# ============================================================
# PROTOCOL HASH
# ============================================================

PROTOCOL_VERSION = "r1-batch-sweep-2026-04-15"
_PROTOCOL_DESCRIPTOR = (
    "batch size sweep | "
    "mini-batch GD (paired noise), MSE, no momentum, no WD | "
    "eval=100 | n_steps=5000 | eps=1e-5 | "
    "perturb=flat-unit-random direction RNG(seed+999999) | "
    "warmup=1000 @ lr=0.01 seed=0 (full-batch) | sharpness=15 power iter | "
    "batch_sizes=[2000,1000,500,200,100] | "
    "paired mini-batches: idx RNG(seed*100000+t) per step | "
    "EoS fixed from full-batch (B=2000) | "
    "outputs every 10 steps | loss+gn every step (saved [::10]) | "
    "sharpness every 500 steps (t>0) | "
    "D2=GP 20 log-radii [percentile 1..95] fit[4:16] subsample<=2000 | "
    "Lyap=linregress log(dist) step window [0.2T,0.8T] | "
    f"version={PROTOCOL_VERSION}"
)
PROTOCOL_HASH = hashlib.sha256(_PROTOCOL_DESCRIPTOR.encode()).hexdigest()[:16]


# ============================================================
# BATCH SIZES
# ============================================================

BATCH_SIZES_FULL = [2000, 1000, 500, 200, 100]
BATCH_SIZES_QUICK = [2000, 500, 100]
N_SAMPLES = 2000  # full training set size


# ============================================================
# ARCHITECTURES (exact copies from r1_cross_experiments.py)
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
    y_onehot = torch.zeros(n_samples, 10)
    for i, label in enumerate(labels):
        y_onehot[i, label] = 1.0
    return X, y_onehot


# ============================================================
# CORE MEASUREMENTS (exact copies from r1_label_noise_sweep.py)
# ============================================================

def build_model(arch, seed, hidden_dim=85):
    """Deterministic model construction: seed THEN construct."""
    torch.manual_seed(seed)
    if arch == "cnn":
        return SmallCNN()
    elif arch == "mlp":
        return MLPCifar(input_dim=3072, hidden_dim=hidden_dim, output_dim=10)
    raise ValueError(f"Unknown arch: {arch}")


def clone_perturbed(model, eps, seed):
    """Clone and perturb along unit-norm random direction."""
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
    """
    Top Hessian eigenvalue via power iteration.

    CUDA: runs in-place on the model's device.
    MPS: copies to CPU (MPS lacks second-order grad support for pooling).
    """
    dev = next(model.parameters()).device
    on_mps = dev.type == "mps"
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
    """Grassberger-Procaccia D_2 matching legacy protocol exactly."""
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


# ============================================================
# EoS THRESHOLD (full-batch only)
# ============================================================

def find_eos_threshold(arch, X, y, device, hidden_dim=85,
                       warmup_lr=0.01, warmup_steps=1000, verbose=True):
    """Train a fresh seed-0 copy on FULL-BATCH data to measure lambda_max."""
    model = build_model(arch, seed=0, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    if verbose:
        print(f"  warmup: {warmup_steps} steps at lr={warmup_lr} (full-batch)")
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


# ============================================================
# MINI-BATCH INDEX GENERATOR (paired-noise protocol)
# ============================================================

def get_batch_indices(step, seed, batch_size, n_samples):
    """
    Deterministic batch index generator for paired-noise SGD.

    Both the original and perturbed model copies call this with the
    same (step, seed) and get identical indices, ensuring the
    stochastic component cancels in the Lyapunov divergence.

    For B=n_samples (full-batch), returns all indices (no-op).
    """
    if batch_size >= n_samples:
        return torch.arange(n_samples)
    rng = torch.Generator()
    rng.manual_seed(seed * 100000 + step)
    return torch.randperm(n_samples, generator=rng)[:batch_size]


# ============================================================
# SINGLE SEED RUN
# ============================================================

def run_single_seed(arch, seed, lr, batch_size, X_train, y_train, X_eval,
                    device, hidden_dim=85, n_steps=5000, eps=1e-5):
    """
    Execute one training trajectory with mini-batch GD and compute
    all measurements.

    The only difference from the full-batch protocol: at each step,
    a random subset of `batch_size` samples is drawn, and both
    the original and perturbed model train on that same subset.
    """
    n_samples = X_train.shape[0]
    criterion = nn.MSELoss()
    model = build_model(arch, seed, hidden_dim=hidden_dim).to(device)
    perturbed = clone_perturbed(model, eps, seed).to(device)

    distances = np.zeros(n_steps)
    losses_rec = np.zeros(n_steps)
    gn_rec = np.zeros(n_steps)
    sharp_rec = []
    outputs_rec = []

    for t in range(n_steps):
        # Function-space divergence on FIXED eval set (not the batch)
        with torch.no_grad():
            d = torch.norm(model(X_eval) - perturbed(X_eval)).item()
            distances[t] = d

        # Record function-space trajectory on fixed eval set
        if t % 10 == 0:
            with torch.no_grad():
                outputs_rec.append(model(X_eval).cpu().numpy())

        # === Paired mini-batch selection ===
        idx = get_batch_indices(t, seed, batch_size, n_samples)
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        # Train original on this batch
        model.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        losses_rec[t] = loss.item()
        loss.backward()
        with torch.no_grad():
            gn = sum((p.grad ** 2).sum() for p in model.parameters()
                     if p.grad is not None).sqrt().item()
            gn_rec[t] = gn
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        # Train perturbed copy on the SAME batch (paired noise)
        perturbed.zero_grad()
        loss2 = criterion(perturbed(X_batch), y_batch)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        # Sharpness snapshots (full-batch for comparability)
        if t > 0 and t % 500 == 0:
            sharp = compute_sharpness(model, X_train, y_train, criterion,
                                      n_iter=15)
            sharp_rec.append(sharp)

        # Progress
        if t > 0 and t % 1000 == 0:
            print(f"      step {t}/{n_steps}  loss={loss.item():.4f}  "
                  f"dist={d:.2e}  B={batch_size}", flush=True)

    # Lyapunov exponent (per step)
    log_d = np.log(distances + 1e-30)
    start = int(n_steps * 0.2)
    end = int(n_steps * 0.8)
    lyap = float(stats.linregress(
        np.arange(start, end), log_d[start:end])[0])

    # PCA + correlation dimension
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
        "lr_frac": 0.30,       # peak D₂ for CNN/CIFAR
        "data_format": "image",  # (n, 3, 32, 32)
    },
    "mlp_cifar_w85": {
        "arch": "mlp",
        "hidden_dim": 85,
        "lr_frac": 0.90,       # peak D₂ for MLP 269K
        "data_format": "flat",  # (n, 3072)
    },
}


# ============================================================
# METADATA
# ============================================================

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
        "protocol_hash": PROTOCOL_HASH,
        "protocol_descriptor": _PROTOCOL_DESCRIPTOR,
        "git_commit": commit,
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
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
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch-size sweep: does the strange attractor "
                    "survive mini-batch SGD?")
    parser.add_argument("--condition", type=str, default=None,
                        choices=list(CONDITION_SPECS.keys()),
                        help="Run only one condition (default: both)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Training seeds (default: 0 1 2)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                        help="Override batch sizes (default: protocol)")
    parser.add_argument("--quick", action="store_true",
                        help="Use 3 batch sizes instead of 5")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Write skeleton JSON with metadata only")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root for CIFAR-10 download")
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    output_dir = args.output_dir or os.path.join(
        repo_root, "data", "main", "revision1")
    data_root = args.data_root or os.path.join(repo_root, "data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "batch_size_sweep.json")

    if args.batch_sizes:
        batch_sizes = sorted(args.batch_sizes, reverse=True)
    else:
        batch_sizes = BATCH_SIZES_QUICK if args.quick else BATCH_SIZES_FULL
    seeds = args.seeds

    conditions = (
        [args.condition] if args.condition
        else list(CONDITION_SPECS.keys())
    )

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 64)
    print("Batch-size sweep: attractor robustness under mini-batch SGD")
    print(f"  conditions: {conditions}")
    print(f"  batch sizes: {batch_sizes}")
    print(f"  seeds: {seeds}")
    print(f"  n_steps: {args.n_steps}")
    print(f"  device: {device}")
    print(f"  output: {output_path}")
    print(f"  protocol_hash: {PROTOCOL_HASH}")
    print("=" * 64)

    total_runs = len(conditions) * len(batch_sizes) * len(seeds)
    print(f"  total training runs: {total_runs}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for cname in conditions:
            spec = CONDITION_SPECS[cname]
            print(f"\n  {cname} ({spec['arch']}, lr_frac={spec['lr_frac']}):")
            for B in batch_sizes:
                epochs_per_run = args.n_steps * B / N_SAMPLES
                print(f"    B={B:>5d}: seeds {seeds}  "
                      f"({epochs_per_run:.0f} effective epochs over "
                      f"{args.n_steps} steps)")
        print(f"\n  Total: {total_runs} runs")
        return

    # Build result structure
    results = {
        "experiment": "batch_size_sweep",
        "revision1_metadata": _collect_metadata(),
        "conditions": {},
    }

    if args.metadata_only:
        for cname in conditions:
            spec = CONDITION_SPECS[cname]
            results["conditions"][cname] = {
                "arch": spec["arch"],
                "lr_frac": spec["lr_frac"],
                "batch_sizes": batch_sizes,
                "seeds": seeds,
                "status": "metadata_only",
            }
        with open(output_path, "w") as f:
            json.dump(_serialize(results), f, indent=2)
        print(f"\nMetadata written to {output_path}")
        return

    # Load CIFAR-10 ONCE (both conditions use the same data)
    print("\nLoading CIFAR-10...")
    X_img, y_clean = load_cifar10_image(
        n_samples=N_SAMPLES, seed=42, data_root=data_root)
    X_flat = X_img.view(N_SAMPLES, -1)
    print(f"  X_img shape: {list(X_img.shape)}")
    print(f"  X_flat shape: {list(X_flat.shape)}")

    done = 0
    t0_global = time.time()

    for cname in conditions:
        spec = CONDITION_SPECS[cname]
        arch = spec["arch"]
        hidden_dim = spec["hidden_dim"] or 85
        lr_frac = spec["lr_frac"]

        # Select data format
        if spec["data_format"] == "image":
            X_train = X_img
        else:
            X_train = X_flat

        X_train = X_train.to(device)
        y_train = y_clean.to(device)

        print(f"\n{'='*64}")
        print(f"Condition: {cname} ({arch}, lr_frac={lr_frac})")
        print(f"{'='*64}")

        # EoS threshold on FULL-BATCH data (B=2000)
        print("\nFinding EoS threshold (full-batch, B=2000)...")
        lam_max, lr_eos = find_eos_threshold(
            arch, X_train, y_train, device,
            hidden_dim=hidden_dim)
        lr = lr_frac * lr_eos
        print(f"  lr = {lr_frac} × {lr_eos:.6f} = {lr:.6f}")

        # Eval set (same as r1_cross_experiments.py)
        torch.manual_seed(0)
        n_eval = min(100, X_train.shape[0])
        eval_idx = torch.randperm(X_train.shape[0])[:n_eval]
        X_eval = X_train[eval_idx]

        n_params = build_model(arch, 0, hidden_dim=hidden_dim).count_params()

        cond_results = {
            "arch": arch,
            "n_params": int(n_params),
            "hidden_dim": hidden_dim if arch == "mlp" else None,
            "lr_frac": lr_frac,
            "lr": float(lr),
            "lam_max": float(lam_max),
            "lr_eos": float(lr_eos),
            "batch_sizes": batch_sizes,
            "seeds": seeds,
            "results": {},
        }

        for B in batch_sizes:
            print(f"\n  --- batch_size = {B} "
                  f"({'full-batch' if B >= N_SAMPLES else 'mini-batch'}) ---")

            bucket = {
                "lyapunov": [], "corr_dim": [],
                "pc1": [], "pc2": [],
                "sharpness_series": [], "grad_norm_series": [],
                "loss_series": [],
                "batch_size": B,
                "gradient_noise_ratio": N_SAMPLES / B,  # effective noise scale
            }

            for seed in seeds:
                done += 1
                elapsed = time.time() - t0_global
                eta_min = (elapsed / done * (total_runs - done)) / 60 \
                    if done > 1 else 0.0
                print(f"\n  [{done}/{total_runs}] {cname} "
                      f"B={B} seed={seed}  "
                      f"ETA: {eta_min:.1f}min", flush=True)

                r = run_single_seed(
                    arch, seed=seed, lr=lr,
                    batch_size=B,
                    X_train=X_train, y_train=y_train,
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

            # Summary stats
            d2_vals = bucket["corr_dim"]
            lam_vals = bucket["lyapunov"]
            bucket["d2_mean"] = float(np.mean(d2_vals))
            bucket["d2_std"] = float(np.std(d2_vals))
            bucket["lam_mean"] = float(np.mean(lam_vals))
            bucket["lam_std"] = float(np.std(lam_vals))
            print(f"\n  B={B} summary: "
                  f"D₂ = {bucket['d2_mean']:.2f} ± {bucket['d2_std']:.2f}, "
                  f"λ = {bucket['lam_mean']:.4f} ± {bucket['lam_std']:.4f}")

            cond_results["results"][str(B)] = bucket

        results["conditions"][cname] = cond_results

        # Save after each condition (incremental)
        with open(output_path, "w") as f:
            json.dump(_serialize(results), f, indent=2)
        print(f"\n  Saved intermediate results to {output_path}")

    # Final save
    with open(output_path, "w") as f:
        json.dump(_serialize(results), f, indent=2)

    elapsed_total = time.time() - t0_global
    print(f"\n{'='*64}")
    print(f"Batch-size sweep complete. {total_runs} runs in "
          f"{elapsed_total/60:.1f} min")
    print(f"Output: {output_path}")
    print(f"{'='*64}")

    # Print summary table
    print("\nSummary:")
    print(f"{'Condition':<20} {'B':>6} {'D₂':>12} {'λ':>12}")
    print("-" * 52)
    for cname in conditions:
        cond_data = results["conditions"][cname]
        for B in batch_sizes:
            key = str(B)
            r = cond_data["results"][key]
            print(f"{cname:<20} {B:>6d} "
                  f"{r['d2_mean']:>6.2f}±{r['d2_std']:<5.2f} "
                  f"{r['lam_mean']:>7.4f}±{r['lam_std']:<6.4f}")
        print()


if __name__ == "__main__":
    main()
