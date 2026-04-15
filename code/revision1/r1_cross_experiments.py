"""
r1_cross_experiments.py  —  Revision 1 unified cross-experiment generator
==========================================================================

This script re-implements the three cross-architecture experiments in a
single file, with byte-exact protocol matching to the two original
generators it replaces:

  - ``Torus Theory/Pytorch/small_mlp_cifar.py``  (produced the committed
     ``cross_small_mlp_cifar_w50_seeds_0_1_2.json`` legacy file)
  - ``Torus Theory/Pytorch/cross_experiments.py`` (produced the
     committed ``cross_cnn_synthetic_seeds_0.json`` and the now-lost
     MLP-269K CIFAR-10 result)

It addresses PRL referee issues #1 and #2 by taking every cross-
architecture condition to N=10 seeds with a single, auditable protocol.
See ``paper/revision1_plan.md`` Phase 1 for the scope justification.

Conditions
----------

   --condition mlp_cifar_w50   : MLP (156,660 params), tanh, 3072-85-85-10  (wait, w50: 3072-50-50-10)
   --condition mlp_cifar_w85   : MLP (269,195 params), tanh, 3072-85-85-10
   --condition cnn_synthetic   : SmallCNN (268,650 params) on 3x32x32 synth

Protocol (identical across all three conditions unless noted)
-------------------------------------------------------------

  * Full-batch gradient descent, MSE loss, no momentum, no weight decay
  * Dataset: 2000-sample subset, ``seed=42`` (matches Experiment K)
      - mlp_cifar_w{50,85}: flattened CIFAR-10 (n, 3072)
      - cnn_synthetic:     synthetic 220-d structured data zero-padded
                           into (n, 3, 32, 32)
  * EoS warmup: ``torch.manual_seed(0)``, construct a fresh model,
    train 1000 steps at lr=0.01, then power iteration (15 steps) to
    find lambda_max. EoS threshold = 2/lambda_max.
  * Learning rate sweep: 12 fractions of the EoS threshold,
      [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
       0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
  * Per-seed run:
      - ``torch.manual_seed(seed)``, construct model
      - Clone model, perturb along unit-norm random direction with
        eps = 1e-5 (direction RNG uses seed + 999999)
      - Train for 5000 steps, full batch
      - Record function-space divergence distance at EVERY step on a
        fixed held-out eval set of 100 inputs
      - Record model outputs on the eval set every 10 steps
      - Record loss and gradient norm at every step (stored [::10])
      - Record top Hessian eigenvalue (15 power iterations) every
        100 steps after step 0
  * Lyapunov exponent: linear regression of ln(distance) on step index
    over the window [0.2T, 0.8T]. Units: inverse training steps.
  * Correlation dimension (Grassberger-Procaccia):
      - Post-transient trajectory = outputs[len//5:], flattened
      - Subsample to 2000 points max (seed-deterministic RNG)
      - Pairwise Euclidean distances
      - 20 log-spaced radii between 1st and 95th percentile
      - D_2 = slope of linregress on log(epsilon), log(C(epsilon))
        using fit indices [4:16]
  * PCA: SVD on centered trajectory; pc1, pc2 are percent variance.

Seed determinism notes
----------------------

The legacy ``small_mlp_cifar.py`` uses ``SmallMLP`` (class name) while
``cross_experiments.py`` uses ``LargeMLP``. Both are 3072-W-W-10 tanh
with identical layer construction order, so under identical RNG state
they produce bit-identical initial weights regardless of class name.
We use a single ``MLPCifar`` class here for both w50 and w85.

Legacy ``small_mlp_cifar.py`` records only {lyapunov, corr_dim, pc1,
pc2, sharpness_series}. Legacy ``cross_experiments.py`` additionally
records {grad_norm_series, loss_series}. We always record the superset;
``r1_merge.py`` tolerates missing fields in legacy seeds.

Output schema
-------------

JSON file matching the legacy schema:

  {
    "experiment": <condition name>,
    "lam_max": float, "lr_eos": float,
    "lr_fractions": [...], "test_lrs": [...],
    "n_params": int, "hidden_dim": int (MLP only),
    "n_samples": 2000, "data_shape": [...] (cnn_synthetic only),
    "seeds_run": [...],
    "revision1_metadata": {
       "script": "r1_cross_experiments.py",
       "git_commit": "...", "torch_version": "...",
       "numpy_version": "...", "protocol_hash": "..."
    },
    "lr_0": { "lyapunov": [...], "corr_dim": [...], "pc1": [...],
              "pc2": [...], "sharpness_series": [[...],[...]],
              "grad_norm_series": [[...],[...]],
              "loss_series": [[...],[...]] },
    "lr_1": { ... }, ...
  }

Usage
-----

   # Standard N=10 run of MLP-269K on CIFAR-10
   python -u r1_cross_experiments.py --condition mlp_cifar_w85 --seeds 0 1 2 3 4 5 6 7 8 9

   # Extend the legacy w50 (seeds 0,1,2) to seeds 3..9 only
   python -u r1_cross_experiments.py --condition mlp_cifar_w50 --seeds 3 4 5 6 7 8 9

   # Reproducibility check: run ONLY seed 0 of w50 and diff against the
   # committed file. Addresses the Phase 1 protocol-reconstruction risk.
   python -u r1_cross_experiments.py --condition mlp_cifar_w50 --seeds 0 \
       --reproduce-check data/main/cross_small_mlp_cifar_w50_seeds_0_1_2.json

   # CNN on synthetic, N=10 from scratch
   python -u r1_cross_experiments.py --condition cnn_synthetic --seeds 0 1 2 3 4 5 6 7 8 9

   # Dry run: print planned work without touching GPU
   python -u r1_cross_experiments.py --condition mlp_cifar_w85 --seeds 0 1 --dry-run

   # Metadata only: write a skeleton JSON with protocol + versions only
   python -u r1_cross_experiments.py --condition mlp_cifar_w85 --seeds 0 --metadata-only

   # Quick sanity: 5 LR points instead of 12
   python -u r1_cross_experiments.py --condition mlp_cifar_w85 --seeds 0 --quick

Output directory auto-resolves to ``data/main/revision1/`` under the
repo root, or to an explicit path via ``--output-dir``.

Addresses referee issues: #1 (seeds N=3 -> N=10),
#2 (suspiciously tight MLP variances), and the orphaned MLP-269K data
gap surfaced during revision 1 Phase 1.
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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# PROTOCOL HASH
# ============================================================
#
# A content-addressable fingerprint of the training + measurement
# protocol. Any change to this script that could affect numerical
# results SHOULD bump PROTOCOL_VERSION. The hash is saved alongside
# every output JSON so downstream code can verify protocol identity.

PROTOCOL_VERSION = "r1-2026-04-09"
_PROTOCOL_DESCRIPTOR = (
    "full-batch GD, MSE, no momentum, no WD | "
    "eval=100 | n_steps=5000 | eps=1e-5 | "
    "perturb=flat-unit-random direction RNG(seed+999999) | "
    "warmup=1000 @ lr=0.01 seed=0 | sharpness=15 power iter | "
    "lr_fractions=[.05,.10,.15,.20,.25,.30,.40,.50,.60,.70,.80,.90] | "
    "outputs every 10 steps | loss+gn every step (saved [::10]) | "
    "sharpness every 100 steps (t>0) | "
    "D2=GP 20 log-radii [percentile 1..95] fit[4:16] subsample<=2000 | "
    "Lyap=linregress log(dist) step window [0.2T,0.8T] | "
    f"version={PROTOCOL_VERSION}"
)
PROTOCOL_HASH = hashlib.sha256(_PROTOCOL_DESCRIPTOR.encode()).hexdigest()[:16]


# ============================================================
# ARCHITECTURES
# ============================================================

class MLPCifar(nn.Module):
    """
    2-hidden-layer tanh MLP: 3072 -> W -> W -> 10.

    Replaces both legacy classes:
      - ``small_mlp_cifar.SmallMLP`` (width 50 -> 156,660 params)
      - ``cross_experiments.LargeMLP`` (width 85 -> 269,195 params)

    Initialization is bit-identical to either legacy class for a given
    ``torch.manual_seed(seed)`` because PyTorch Linear layers are
    constructed in the same order and class name does not affect RNG.
    """

    def __init__(self, input_dim=3072, hidden_dim=50, output_dim=10):
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
    """Exactly Experiment K / cnn_seeds_v2.py CNN. 268,650 parameters."""

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
# CONDITION REGISTRY
# ============================================================

@dataclass
class Condition:
    name: str                 # output file stem / experiment label
    arch: str                 # "mlp" | "cnn"
    hidden_dim: int | None    # MLP width, None for CNN
    data: str                 # "cifar_flat" | "cifar_image" | "synth_image"
    output_stem: str          # filename stem, matches legacy convention

CONDITIONS = {
    "mlp_cifar_w50": Condition(
        name="small_mlp_cifar_w50", arch="mlp", hidden_dim=50,
        data="cifar_flat", output_stem="cross_small_mlp_cifar_w50"),
    "mlp_cifar_w85": Condition(
        name="small_mlp_cifar_w85", arch="mlp", hidden_dim=85,
        data="cifar_flat", output_stem="cross_small_mlp_cifar_w85"),
    "cnn_synthetic": Condition(
        name="cnn_synthetic", arch="cnn", hidden_dim=None,
        data="synth_image", output_stem="cross_cnn_synthetic"),
}


def build_model(cond: Condition, seed: int) -> nn.Module:
    """Deterministic model construction: seed THEN construct."""
    torch.manual_seed(seed)
    if cond.arch == "mlp":
        return MLPCifar(input_dim=3072,
                        hidden_dim=cond.hidden_dim,
                        output_dim=10)
    elif cond.arch == "cnn":
        return SmallCNN()
    raise ValueError(f"Unknown arch: {cond.arch}")


# ============================================================
# DATA LOADERS
# ============================================================

def _cifar10_raw(n_samples, seed, data_root):
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


def load_cifar10_flat(n_samples=2000, seed=42, data_root="./data"):
    X, y = _cifar10_raw(n_samples, seed, data_root)
    return X.view(n_samples, -1), y  # (n, 3072)


def load_cifar10_image(n_samples=2000, seed=42, data_root="./data"):
    return _cifar10_raw(n_samples, seed, data_root)  # (n, 3, 32, 32)


def generate_synthetic_images(n_samples=2000, seed=42):
    """
    Same 220-d structured synth data as Phase 1/2, zero-padded into
    (n, 3, 32, 32) so a CNN can ingest it. The first 220 entries of
    the flattened image carry the structured signal; the remaining
    2852 entries are zero. Identical to cross_experiments.py.
    """
    rng = np.random.RandomState(seed)
    n, k = n_samples, 10
    d_rand, d_quad = 200, 20
    centers = rng.randn(k, d_rand) * 2.0
    labels = rng.randint(0, k, size=n)
    X_rand = np.zeros((n, d_rand))
    for i in range(n):
        X_rand[i] = centers[labels[i]] + rng.randn(d_rand) * 0.5
    X_quad = X_rand[:, :d_quad] ** 2
    X_220 = np.concatenate([X_rand, X_quad], axis=1).astype(np.float32)
    X_3072 = np.zeros((n, 3072), dtype=np.float32)
    X_3072[:, :220] = X_220
    X = torch.tensor(X_3072).view(n, 3, 32, 32)
    y = np.zeros((n, k), dtype=np.float32)
    y[np.arange(n), labels] = 1.0
    return X, torch.tensor(y)


def load_condition_data(cond: Condition, n_samples=2000, data_root="./data"):
    if cond.data == "cifar_flat":
        return load_cifar10_flat(n_samples, seed=42, data_root=data_root)
    elif cond.data == "cifar_image":
        return load_cifar10_image(n_samples, seed=42, data_root=data_root)
    elif cond.data == "synth_image":
        return generate_synthetic_images(n_samples, seed=42)
    raise ValueError(f"Unknown data: {cond.data}")


# ============================================================
# CORE MEASUREMENTS
# ============================================================

def clone_perturbed(model, eps, seed):
    """
    Clone a model and perturb along a unit-norm random direction in
    flattened parameter space. The direction RNG is seeded with
    ``seed + 999999`` to match both legacy scripts.

    NOTE: legacy ``small_mlp_cifar.py`` omits ``.to(flat.device)`` on
    the direction tensor (works because that script is CPU-only).
    Legacy ``cross_experiments.py`` includes it (so the code works on
    GPU). We include it here; on CPU both variants are identical.
    """
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
    """Top Hessian eigenvalue via power iteration on the full loss."""
    v = [torch.randn_like(p) for p in model.parameters()]
    v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
    v = [vi / v_norm for vi in v]
    eigenvalue = 0.0
    for _ in range(n_iter):
        model.zero_grad()
        loss = criterion(model(X), y)
        grads = torch.autograd.grad(loss, model.parameters(),
                                    create_graph=True)
        Hv_terms = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(Hv_terms, model.parameters())
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
# WARMUP / EoS THRESHOLD
# ============================================================

def find_eos_threshold(cond: Condition, X, y, device,
                       warmup_lr=0.01, warmup_steps=1000, verbose=True):
    """Train a fresh seed-0 copy briefly to measure lambda_max."""
    model = build_model(cond, seed=0).to(device)
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


# ============================================================
# SINGLE SEED RUN
# ============================================================

def run_single_seed(cond: Condition, seed, lr, X, y, X_eval, device,
                    n_steps=5000, eps=1e-5):
    """Execute one training trajectory and compute all measurements."""
    criterion = nn.MSELoss()
    model = build_model(cond, seed).to(device)
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

        # Train original
        model.zero_grad()
        loss = criterion(model(X), y)
        losses_rec[t] = loss.item()
        loss.backward()
        with torch.no_grad():
            gn = sum((p.grad ** 2).sum() for p in model.parameters()
                     if p.grad is not None).sqrt().item()
            gn_rec[t] = gn
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        # Train perturbed copy
        perturbed.zero_grad()
        loss2 = criterion(perturbed(X), y)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        # Sharpness snapshots (matches legacy: t>0 and t%100==0)
        if t > 0 and t % 100 == 0:
            sharp = compute_sharpness(model, X, y, criterion, n_iter=15)
            sharp_rec.append(sharp)

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
# EXPERIMENT RUNNER
# ============================================================

LR_FRACTIONS_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                     0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
LR_FRACTIONS_QUICK = [0.05, 0.15, 0.30, 0.50, 0.90]


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


def run_condition(cond: Condition, seeds, n_steps, quick, device,
                  data_root, verbose=True):
    print("=" * 64)
    print(f"r1 cross experiment: {cond.name}")
    print(f"  seeds: {seeds}")
    print(f"  protocol_hash: {PROTOCOL_HASH}")
    print(f"  n_steps: {n_steps}   quick: {quick}")
    print("=" * 64)

    print(f"\nloading data for {cond.name}...")
    X, y = load_condition_data(cond, n_samples=2000, data_root=data_root)
    print(f"  X shape: {list(X.shape)}  y shape: {list(y.shape)}")
    X = X.to(device)
    y = y.to(device)

    n_params = build_model(cond, 0).count_params()
    print(f"  n_params: {n_params:,}")

    print("\nfinding EoS threshold...")
    lam_max, lr_eos = find_eos_threshold(cond, X, y, device, verbose=verbose)

    lr_fractions = LR_FRACTIONS_QUICK if quick else LR_FRACTIONS_FULL
    test_lrs = [frac * lr_eos for frac in lr_fractions]

    torch.manual_seed(0)
    n_eval = min(100, X.shape[0])
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    results = {
        "experiment": cond.name,
        "lam_max": lam_max,
        "lr_eos": lr_eos,
        "lr_fractions": lr_fractions,
        "test_lrs": [float(lr) for lr in test_lrs],
        "n_params": int(n_params),
        "n_samples": int(X.shape[0]),
        "data_shape": list(X.shape),
        "seeds_run": list(seeds),
        "revision1_metadata": _collect_metadata(),
    }
    if cond.arch == "mlp":
        results["hidden_dim"] = cond.hidden_dim

    total_runs = len(lr_fractions) * len(seeds)
    done = 0
    t0 = time.time()

    for li, (frac, lr) in enumerate(zip(lr_fractions, test_lrs)):
        lr_key = f"lr_{li}"
        bucket = {"lyapunov": [], "corr_dim": [],
                  "pc1": [], "pc2": [],
                  "sharpness_series": [], "grad_norm_series": [],
                  "loss_series": []}

        for seed in seeds:
            done += 1
            elapsed = time.time() - t0
            eta_min = (elapsed / done * (total_runs - done)) / 60 \
                if done > 1 else 0.0
            print(f"\n[{done}/{total_runs}] {cond.name} "
                  f"{frac:.0%} EoS lr={lr:.6f} seed={seed}  "
                  f"ETA: {eta_min:.1f}min", flush=True)
            r = run_single_seed(
                cond, seed=seed, lr=lr,
                X=X, y=y, X_eval=X_eval, device=device,
                n_steps=n_steps, eps=1e-5)
            for k in bucket:
                bucket[k].append(r[k])
            print(f"  -> lambda={r['lyapunov']:+.6f}  "
                  f"D2={r['corr_dim']:.3f}  "
                  f"PC1={r['pc1']:.1f}%  PC2={r['pc2']:.1f}%")

        results[lr_key] = bucket
        d2s = [d for d in bucket["corr_dim"]
               if d is not None and not np.isnan(d)]
        lyaps = bucket["lyapunov"]
        print(f"\n  {frac:.0%} EoS summary: "
              f"lambda = {np.mean(lyaps):+.6f} +- {np.std(lyaps):.6f},  "
              f"D2 = {np.nanmean(d2s):.3f} +- {np.nanstd(d2s):.3f}")

    return results


# ============================================================
# REPRODUCIBILITY CHECK
# ============================================================

def reproduce_check(new_results, legacy_path, tolerance=1e-6):
    """
    Compare a freshly-run condition against a committed legacy file.
    Exits with code 1 if any compared field differs by more than
    ``tolerance`` in relative value for floats.
    """
    print("\n" + "=" * 64)
    print(f"REPRODUCIBILITY CHECK against {legacy_path}")
    print("=" * 64)
    with open(legacy_path) as f:
        legacy = json.load(f)

    ok = True

    def diff(label, a, b):
        nonlocal ok
        if a is None or b is None:
            return
        if isinstance(a, float) and isinstance(b, float):
            denom = max(abs(a), abs(b), 1e-12)
            rel = abs(a - b) / denom
            if rel > tolerance:
                print(f"  MISMATCH  {label}: new={a:.8g} legacy={b:.8g} "
                      f"rel={rel:.2e}")
                ok = False
            else:
                print(f"  ok        {label}: {a:.6g}  (rel {rel:.1e})")
        else:
            if a != b:
                print(f"  MISMATCH  {label}: new={a} legacy={b}")
                ok = False

    for k in ("experiment", "lam_max", "lr_eos", "n_params", "n_samples"):
        if k in legacy:
            diff(k, new_results.get(k), legacy.get(k))

    # Compare seed-0 metrics per LR fraction. The new_results is
    # assumed to contain seed 0 as the first element of each bucket;
    # the legacy file stores per-seed lists indexed by seeds_run.
    legacy_seeds = legacy.get("seeds_run", [])
    if 0 not in legacy_seeds:
        print("  WARNING: legacy file has no seed 0; cannot compare.")
        return False
    legacy_seed0_idx = legacy_seeds.index(0)
    new_seeds = new_results.get("seeds_run", [])
    if 0 not in new_seeds:
        print("  WARNING: new results have no seed 0; cannot compare.")
        return False
    new_seed0_idx = new_seeds.index(0)

    for li, frac in enumerate(new_results["lr_fractions"]):
        key = f"lr_{li}"
        if key not in legacy:
            print(f"  skip   lr_{li} ({frac:.0%}): not in legacy")
            continue
        nb = new_results[key]
        lb = legacy[key]
        for field in ("lyapunov", "corr_dim", "pc1", "pc2"):
            if field not in lb:
                continue
            a = nb[field][new_seed0_idx]
            b = lb[field][legacy_seed0_idx]
            diff(f"{key}[{frac:.0%}].{field}", a, b)

    print("\n" + ("REPRODUCE CHECK PASSED" if ok
                  else "REPRODUCE CHECK FAILED"))
    return ok


# ============================================================
# OUTPUT
# ============================================================

def _default_output_dir(script_path):
    # revision1 layout: data/main/revision1/ under repo root
    here = os.path.dirname(os.path.abspath(script_path))
    repo = os.path.dirname(os.path.dirname(here))  # code/revision1 -> code -> repo
    return os.path.join(repo, "data", "main", "revision1")


def save_results(results, cond: Condition, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seeds = results.get("seeds_run", [])
    seeds_str = "_".join(str(s) for s in seeds)
    fname = f"{cond.output_stem}_seeds_{seeds_str}.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w") as f:
        json.dump(_serialize(results), f, indent=2)
    print(f"\nsaved -> {path}")
    return path


# ============================================================
# MAIN
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Revision 1 unified cross-experiment generator.")
    p.add_argument("--condition", required=True, choices=list(CONDITIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="List of integer seeds to run.")
    p.add_argument("--steps", type=int, default=5000,
                   help="Training steps per seed.")
    p.add_argument("--quick", action="store_true",
                   help="Use the 5-point LR grid instead of 12.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned work; do not touch GPU.")
    p.add_argument("--metadata-only", action="store_true",
                   help="Write a skeleton JSON with metadata only.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory. Defaults to "
                        "<repo>/data/main/revision1/")
    p.add_argument("--data-root", type=str, default="./data",
                   help="Where to look for/download CIFAR-10.")
    p.add_argument("--reproduce-check", type=str, default=None,
                   help="Path to a legacy JSON to diff against. Triggers "
                        "the reproducibility check after the run.")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = _parse_args()
    cond = CONDITIONS[args.condition]
    output_dir = args.output_dir or _default_output_dir(__file__)

    if args.dry_run:
        print("DRY RUN")
        print(f"  condition:   {cond.name}")
        print(f"  seeds:       {args.seeds}")
        print(f"  n_steps:     {args.steps}")
        print(f"  lr_frac:     "
              f"{LR_FRACTIONS_QUICK if args.quick else LR_FRACTIONS_FULL}")
        print(f"  output_dir:  {output_dir}")
        print(f"  protocol:    {PROTOCOL_HASH}")
        total = len(args.seeds) * (
            len(LR_FRACTIONS_QUICK) if args.quick else len(LR_FRACTIONS_FULL))
        print(f"  total runs:  {total}")
        return 0

    if args.metadata_only:
        meta = {
            "experiment": cond.name,
            "seeds_run": args.seeds,
            "revision1_metadata": _collect_metadata(),
            "note": "metadata-only skeleton; no training was performed",
        }
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{cond.output_stem}_seeds_"\
                f"{'_'.join(str(s) for s in args.seeds)}.metadata.json"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote metadata -> {path}")
        return 0

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    results = run_condition(
        cond, seeds=args.seeds, n_steps=args.steps, quick=args.quick,
        device=device, data_root=args.data_root)

    save_results(results, cond, output_dir)

    if args.reproduce_check:
        ok = reproduce_check(results, args.reproduce_check)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
