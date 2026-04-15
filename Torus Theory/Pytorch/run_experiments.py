"""
Chaos Onset in Gradient Descent — Phase 1 Experiments
=====================================================

Measures function-space Lyapunov exponents across learning rates
to identify the critical learning rate η_c where chaos begins.

USAGE:
    # Run the full Phase 1 suite (transition zone + broad sweep + sensitivity):
    python run_experiments.py --all

    # Run individual experiments:
    python run_experiments.py --transition     # 20 seeds, fine-grained η_c
    python run_experiments.py --broad          # 20 seeds, full Lyapunov curve
    python run_experiments.py --sensitivity    # perturbation ε sensitivity check

    # Quick test (fewer seeds, to verify everything works):
    python run_experiments.py --transition --seeds 3
    python run_experiments.py --broad --seeds 3

    # Custom learning rate range:
    python run_experiments.py --transition --lr-min 0.005 --lr-max 0.08 --n-lrs 40

OUTPUTS:
    results/transition_zone.npz    — data for transition zone figures
    results/broad_sweep.npz        — data for broad sweep figures
    results/sensitivity.npz        — data for perturbation sensitivity
    figures/                       — publication-quality plots (PNG + PDF)

REQUIREMENTS:
    pip install torch numpy matplotlib scipy
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    # Architecture
    "input_dim": 220,
    "hidden_dim": 50,
    "output_dim": 10,
    "activation": "tanh",  # "tanh" or "relu"

    # Data
    "n_samples": 2000,
    "n_classes": 10,
    "n_random_features": 200,
    "n_quadratic_features": 20,
    "data_seed": 42,

    # Training
    "n_train_steps": 5000,
    "loss_fn": "mse",

    # Lyapunov
    "perturbation_eps": 1e-8,
    "lyap_fit_start_frac": 0.2,   # skip first 20% (transient)
    "lyap_fit_end_frac": 0.8,     # skip last 20% (saturation)

    # Sharpness (power iteration)
    "sharpness_iters": 100,
}


# ============================================================
# 2. DATA GENERATION
# ============================================================

def generate_data(config):
    """
    Generate synthetic classification data. Deterministic given data_seed.

    Returns X (n_samples × input_dim) and y (n_samples × n_classes) as tensors.
    """
    rng = np.random.RandomState(config["data_seed"])

    n = config["n_samples"]
    k = config["n_classes"]
    d_rand = config["n_random_features"]
    d_quad = config["n_quadratic_features"]

    # Class centers in random-feature space
    centers = rng.randn(k, d_rand) * 2.0

    # Assign samples to classes and draw from Gaussian clusters
    labels = rng.randint(0, k, size=n)
    X_rand = np.zeros((n, d_rand))
    for i in range(n):
        X_rand[i] = centers[labels[i]] + rng.randn(d_rand) * 0.5

    # Quadratic features: square the first d_quad dimensions
    X_quad = X_rand[:, :d_quad] ** 2

    # Concatenate
    X = np.concatenate([X_rand, X_quad], axis=1).astype(np.float32)

    # One-hot labels
    y = np.zeros((n, k), dtype=np.float32)
    y[np.arange(n), labels] = 1.0

    return torch.tensor(X), torch.tensor(y)


# ============================================================
# 3. MODEL
# ============================================================

class MLP(nn.Module):
    """Two-hidden-layer MLP following Cohen et al. (2021) setup."""

    def __init__(self, input_dim, hidden_dim, output_dim, activation="tanh"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = torch.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def make_model(config, seed):
    """Create and initialize a model with a specific seed."""
    torch.manual_seed(seed)
    model = MLP(
        config["input_dim"],
        config["hidden_dim"],
        config["output_dim"],
        config["activation"],
    )
    return model


def clone_model_perturbed(model, eps, seed):
    """
    Create a copy of model with weights perturbed by eps in a random direction.
    The perturbation direction is deterministic given seed.
    """
    clone = MLP(
        model.fc1.in_features,
        model.fc1.out_features,
        model.fc3.out_features,
        "tanh" if model.act == torch.tanh else "relu",
    )

    # Copy weights exactly
    clone.load_state_dict(model.state_dict())

    # Generate perturbation direction (unit norm in full parameter space)
    rng = torch.Generator()
    rng.manual_seed(seed + 999999)

    all_params = []
    for p in clone.parameters():
        all_params.append(p.data.view(-1))
    flat = torch.cat(all_params)

    direction = torch.randn(flat.shape, generator=rng)
    direction = direction / direction.norm()

    # Apply perturbation
    offset = 0
    for p in clone.parameters():
        numel = p.numel()
        p.data += eps * direction[offset:offset + numel].view(p.shape)
        offset += numel

    return clone


# ============================================================
# 4. TRAINING + LYAPUNOV COMPUTATION
# ============================================================

def compute_lyapunov(config, lr, seed, X, y, X_eval=None):
    """
    Train two networks (original + perturbed) and compute
    the function-space Lyapunov exponent.

    Returns:
        lyap_exponent (float): estimated Lyapunov exponent
        final_loss (float): final training loss of the original network
        distances (np.array): function-space distance at each step
    """
    device = X.device

    # Build models
    model = make_model(config, seed).to(device)
    perturbed = clone_model_perturbed(
        model, config["perturbation_eps"], seed
    ).to(device)

    if X_eval is None:
        X_eval = X  # use training data as evaluation set

    criterion = nn.MSELoss()
    n_steps = config["n_train_steps"]
    distances = np.zeros(n_steps)

    for t in range(n_steps):
        # Measure function-space distance BEFORE this step's update
        with torch.no_grad():
            f1 = model(X_eval)
            f2 = perturbed(X_eval)
            d = torch.norm(f1 - f2).item()
            distances[t] = d

        # Training step for original
        model.zero_grad()
        loss1 = criterion(model(X), y)
        loss1.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad

        # Identical training step for perturbed
        perturbed.zero_grad()
        loss2 = criterion(perturbed(X), y)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                p -= lr * p.grad

    # Estimate Lyapunov exponent from log-distance slope
    log_d = np.log(distances + 1e-30)

    start = int(n_steps * config["lyap_fit_start_frac"])
    end = int(n_steps * config["lyap_fit_end_frac"])
    if end <= start + 10:
        end = min(start + 100, n_steps)

    t_range = np.arange(start, end)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        t_range, log_d[start:end]
    )

    final_loss = loss1.item()
    return slope, final_loss, distances


# ============================================================
# 5. SHARPNESS (λ_max via power iteration)
# ============================================================

def compute_sharpness(model, X, y, n_iter=100):
    """
    Compute the largest eigenvalue of the Hessian of the loss
    using power iteration (Hessian-vector products via autograd).
    """
    criterion = nn.MSELoss()

    # Random initial vector
    v = [torch.randn_like(p) for p in model.parameters()]
    v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
    v = [vi / v_norm for vi in v]

    for _ in range(n_iter):
        # Compute Hv via double backward
        model.zero_grad()
        loss = criterion(model(X), y)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Hessian-vector product: H @ v
        Hv_terms = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(Hv_terms, model.parameters())

        # Eigenvalue estimate: v^T H v
        eigenvalue = sum((hv * vi).sum().item() for hv, vi in zip(Hv, v))

        # Normalize for next iteration
        hv_norm = sum((hv ** 2).sum() for hv in Hv).sqrt().item()
        if hv_norm < 1e-12:
            break
        v = [hv.detach() / hv_norm for hv in Hv]

    return abs(eigenvalue)


# ============================================================
# 6. EXPERIMENT RUNNERS
# ============================================================

def run_transition_zone(config, n_seeds=20, n_lrs=40,
                        lr_min=0.005, lr_max=0.08):
    """Phase 1A: Fine-grained transition zone to pin down η_c."""

    print("=" * 60)
    print(f"TRANSITION ZONE: {n_seeds} seeds × {n_lrs} learning rates")
    print(f"  LR range: [{lr_min}, {lr_max}]")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    lrs = np.linspace(lr_min, lr_max, n_lrs)
    all_lyaps = np.zeros((n_seeds, n_lrs))
    all_losses = np.zeros((n_seeds, n_lrs))

    # Compute sharpness at lowest LR for EoS reference
    print("  Computing sharpness (λ_max)...")
    ref_model = make_model(config, seed=0).to(device)
    # Train briefly at low LR to get reasonable sharpness estimate
    criterion = nn.MSELoss()
    for t in range(1000):
        ref_model.zero_grad()
        loss = criterion(ref_model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in ref_model.parameters():
                p -= 0.01 * p.grad
    lam_max = compute_sharpness(ref_model, X, y, config["sharpness_iters"])
    lr_eos = 2.0 / lam_max
    print(f"  λ_max ≈ {lam_max:.4f}, 2/λ_max ≈ {lr_eos:.4f}")

    total = n_seeds * n_lrs
    done = 0
    t0 = time.time()

    for s in range(n_seeds):
        for j, lr in enumerate(lrs):
            lyap, loss_val, _ = compute_lyapunov(config, lr, seed=s, X=X, y=y)
            all_lyaps[s, j] = lyap
            all_losses[s, j] = loss_val
            done += 1

            if done % 20 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] seed={s} lr={lr:.4f} "
                      f"lyap={lyap:.6f} loss={loss_val:.6f} "
                      f"ETA: {eta:.0f}s")

    # Compute per-seed η_c (zero-crossing by interpolation)
    eta_c_per_seed = []
    for s in range(n_seeds):
        lyaps_s = all_lyaps[s]
        # Find first crossing from negative to positive
        for j in range(len(lrs) - 1):
            if lyaps_s[j] <= 0 and lyaps_s[j + 1] > 0:
                # Linear interpolation
                frac = -lyaps_s[j] / (lyaps_s[j + 1] - lyaps_s[j])
                eta_c = lrs[j] + frac * (lrs[j + 1] - lrs[j])
                eta_c_per_seed.append(eta_c)
                break
        else:
            # No clean crossing found; use first positive
            pos = np.where(lyaps_s > 0)[0]
            if len(pos) > 0:
                eta_c_per_seed.append(lrs[pos[0]])

    eta_c_arr = np.array(eta_c_per_seed)
    mean_lyaps = all_lyaps.mean(axis=0)
    std_lyaps = all_lyaps.std(axis=0)

    # Mean curve zero-crossing
    eta_c_from_mean = None
    for j in range(len(lrs) - 1):
        if mean_lyaps[j] <= 0 and mean_lyaps[j + 1] > 0:
            frac = -mean_lyaps[j] / (mean_lyaps[j + 1] - mean_lyaps[j])
            eta_c_from_mean = lrs[j] + frac * (lrs[j + 1] - lrs[j])
            break

    print()
    print("RESULTS:")
    print(f"  η_c per seed: {eta_c_arr}")
    print(f"  η_c mean ± std: {eta_c_arr.mean():.5f} ± {eta_c_arr.std():.5f}")
    if eta_c_from_mean:
        print(f"  η_c from mean curve: {eta_c_from_mean:.5f}")
    print(f"  2/λ_max (EoS): {lr_eos:.5f}")
    print(f"  η_c / (2/λ_max): {eta_c_arr.mean() / lr_eos * 100:.1f}%")

    results = {
        "lrs": lrs,
        "all_lyaps": all_lyaps,
        "all_losses": all_losses,
        "mean_lyaps": mean_lyaps,
        "std_lyaps": std_lyaps,
        "seeds": np.arange(n_seeds),
        "lr_eos": lr_eos,
        "lam_max": lam_max,
        "eta_c_per_seed": eta_c_arr,
        "eta_c_mean": eta_c_arr.mean() if len(eta_c_arr) > 0 else np.nan,
        "eta_c_std": eta_c_arr.std() if len(eta_c_arr) > 0 else np.nan,
        "eta_c_from_mean": eta_c_from_mean if eta_c_from_mean else np.nan,
        "config_str": str(config),
    }

    os.makedirs("results", exist_ok=True)
    np.savez("results/transition_zone.npz", **results)
    print("  Saved → results/transition_zone.npz")

    return results


def run_broad_sweep(config, n_seeds=20, n_lrs=25,
                    lr_min=0.01, lr_max=0.42):
    """Phase 1B: Full Lyapunov curve across the EoS threshold."""

    print("=" * 60)
    print(f"BROAD SWEEP: {n_seeds} seeds × {n_lrs} learning rates")
    print(f"  LR range: [{lr_min}, {lr_max}]")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    lrs = np.linspace(lr_min, lr_max, n_lrs)
    all_lyaps = np.zeros((n_seeds, n_lrs))
    all_losses = np.zeros((n_seeds, n_lrs))

    # Sharpness
    print("  Computing sharpness (λ_max)...")
    ref_model = make_model(config, seed=0).to(device)
    criterion = nn.MSELoss()
    for t in range(1000):
        ref_model.zero_grad()
        loss = criterion(ref_model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in ref_model.parameters():
                p -= 0.01 * p.grad
    lam_max = compute_sharpness(ref_model, X, y, config["sharpness_iters"])
    lr_eos = 2.0 / lam_max
    print(f"  λ_max ≈ {lam_max:.4f}, 2/λ_max ≈ {lr_eos:.4f}")

    total = n_seeds * n_lrs
    done = 0
    t0 = time.time()

    for s in range(n_seeds):
        for j, lr in enumerate(lrs):
            lyap, loss_val, _ = compute_lyapunov(config, lr, seed=s, X=X, y=y)
            all_lyaps[s, j] = lyap
            all_losses[s, j] = loss_val
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] seed={s} lr={lr:.4f} "
                      f"lyap={lyap:.6f} ETA: {eta:.0f}s")

    results = {
        "lrs": lrs,
        "all_lyaps": all_lyaps,
        "all_losses": all_losses,
        "mean_lyaps": all_lyaps.mean(axis=0),
        "std_lyaps": all_lyaps.std(axis=0),
        "seeds": np.arange(n_seeds),
        "lr_eos": lr_eos,
        "lam_max": lam_max,
    }

    os.makedirs("results", exist_ok=True)
    np.savez("results/broad_sweep.npz", **results)
    print("  Saved → results/broad_sweep.npz")

    return results


def run_sensitivity(config, n_seeds=5):
    """Phase 1C: Check that Lyapunov estimates are stable across ε."""

    print("=" * 60)
    print("PERTURBATION SENSITIVITY ANALYSIS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    epsilons = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    test_lrs = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20]

    results_arr = np.zeros((len(epsilons), len(test_lrs), n_seeds))

    total = len(epsilons) * len(test_lrs) * n_seeds
    done = 0
    t0 = time.time()

    for ei, eps in enumerate(epsilons):
        cfg = dict(config)
        cfg["perturbation_eps"] = eps
        for li, lr in enumerate(test_lrs):
            for s in range(n_seeds):
                lyap, _, _ = compute_lyapunov(cfg, lr, seed=s, X=X, y=y)
                results_arr[ei, li, s] = lyap
                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - t0
                    eta_t = elapsed / done * (total - done)
                    print(f"  [{done}/{total}] ε={eps:.0e} lr={lr:.2f} "
                          f"lyap={lyap:.6f} ETA: {eta_t:.0f}s")

    results = {
        "epsilons": np.array(epsilons),
        "test_lrs": np.array(test_lrs),
        "lyap_results": results_arr,  # shape: (n_eps, n_lrs, n_seeds)
        "n_seeds": n_seeds,
    }

    os.makedirs("results", exist_ok=True)
    np.savez("results/sensitivity.npz", **results)
    print("  Saved → results/sensitivity.npz")

    return results


# ============================================================
# 7. PLOTTING
# ============================================================

def plot_transition_zone(npz_path="results/transition_zone.npz"):
    """Generate publication-quality transition zone figure."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.linewidth": 0.8,
    })

    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]
    all_lyaps = d["all_lyaps"]
    mean_lyaps = d["mean_lyaps"]
    std_lyaps = d["std_lyaps"]
    lr_eos = float(d["lr_eos"])
    n_seeds = all_lyaps.shape[0]

    eta_c_mean = float(d["eta_c_mean"]) if "eta_c_mean" in d else None
    eta_c_std = float(d["eta_c_std"]) if "eta_c_std" in d else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: individual seeds + mean
    colors = plt.cm.Set2(np.linspace(0, 1, n_seeds))
    for s in range(n_seeds):
        ax1.scatter(lrs, all_lyaps[s], s=12, alpha=0.35, color=colors[s % 8],
                    label=f"Seed {s}" if s < 8 else None, zorder=2)
    ax1.plot(lrs, mean_lyaps, "k-s", ms=4, lw=1.5, label="Mean", zorder=3)
    ax1.fill_between(lrs, mean_lyaps - std_lyaps, mean_lyaps + std_lyaps,
                     alpha=0.15, color="gray")
    ax1.axhline(0, color="k", lw=0.5, ls="-")
    if lr_eos <= lrs[-1] * 1.5:
        ax1.axvline(lr_eos, color="orange", ls="--", lw=1.2,
                    label=f"2/λ_max = {lr_eos:.3f}")
    ax1.set_xlabel("Learning rate η")
    ax1.set_ylabel("Lyapunov exponent")
    ax1.set_title(f"Transition zone: {n_seeds} seeds")
    ax1.legend(fontsize=8, ncol=2)

    # Right: mean with zero-crossing
    ax2.plot(lrs, mean_lyaps, "k-o", ms=4, lw=1.5, zorder=3)
    ax2.fill_between(lrs, mean_lyaps - std_lyaps, mean_lyaps + std_lyaps,
                     alpha=0.2, color="steelblue")
    ax2.axhline(0, color="k", lw=0.5)
    if eta_c_mean and not np.isnan(eta_c_mean):
        ax2.axvline(eta_c_mean, color="green", lw=1.5,
                    label=f"η_c = {eta_c_mean:.4f} ± {eta_c_std:.4f}")
        if eta_c_std and not np.isnan(eta_c_std):
            ax2.axvspan(eta_c_mean - eta_c_std, eta_c_mean + eta_c_std,
                        alpha=0.15, color="green")
    if lr_eos <= lrs[-1] * 3:
        ax2.axvline(lr_eos, color="orange", ls="--", lw=1.2,
                    label=f"2/λ_max = {lr_eos:.3f}")
    ax2.set_xlabel("Learning rate η")
    ax2.set_ylabel("Lyapunov exponent (mean ± std)")
    ax2.set_title("Critical η: zero crossing")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/transition_zone.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/transition_zone.pdf", bbox_inches="tight")
    print("  Saved → figures/transition_zone.png + .pdf")
    plt.close()


def plot_broad_sweep(npz_path="results/broad_sweep.npz"):
    """Generate publication-quality broad sweep figure."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.linewidth": 0.8,
    })

    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]
    all_lyaps = d["all_lyaps"]
    lr_eos = float(d["lr_eos"])
    n_seeds = all_lyaps.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.tab10(np.linspace(0, 1, min(n_seeds, 10)))
    for s in range(n_seeds):
        ax1.plot(lrs, all_lyaps[s], "-o", ms=3, alpha=0.5, lw=0.8,
                 color=colors[s % 10],
                 label=f"Seed {s}" if s < 10 else None)
    ax1.axvline(lr_eos, color="orange", ls="--", lw=1.5,
                label=f"2/λ_max = {lr_eos:.3f}")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("Learning rate η")
    ax1.set_ylabel("Lyapunov exponent (function space)")
    ax1.set_title(f"Reproducibility: {n_seeds} seeds")
    ax1.legend(fontsize=7, ncol=2)

    mean = all_lyaps.mean(axis=0)
    std = all_lyaps.std(axis=0)
    ax2.plot(lrs, mean, "k-o", ms=4, lw=1.5)
    ax2.fill_between(lrs, mean - std, mean + std, alpha=0.2, color="steelblue")
    ax2.axvline(lr_eos, color="orange", ls="--", lw=1.5,
                label=f"2/λ_max = {lr_eos:.3f}")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Learning rate η")
    ax2.set_ylabel("Lyapunov exponent (mean ± std)")
    ax2.set_title("Mean across seeds with error bands")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/broad_sweep.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/broad_sweep.pdf", bbox_inches="tight")
    print("  Saved → figures/broad_sweep.png + .pdf")
    plt.close()


def plot_sensitivity(npz_path="results/sensitivity.npz"):
    """Generate perturbation sensitivity figure."""
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    epsilons = d["epsilons"]
    test_lrs = d["test_lrs"]
    lyap_results = d["lyap_results"]  # (n_eps, n_lrs, n_seeds)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(test_lrs)))

    for li, lr in enumerate(test_lrs):
        means = lyap_results[:, li, :].mean(axis=1)
        stds = lyap_results[:, li, :].std(axis=1)
        ax.errorbar(epsilons, means, yerr=stds, fmt="-o", ms=4,
                    color=colors[li], label=f"η = {lr:.2f}", capsize=3)

    ax.set_xscale("log")
    ax.set_xlabel("Perturbation magnitude ε")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title("Sensitivity to perturbation size")
    ax.legend(fontsize=9)
    ax.axhline(0, color="k", lw=0.5)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/sensitivity.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/sensitivity.pdf", bbox_inches="tight")
    print("  Saved → figures/sensitivity.png + .pdf")
    plt.close()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chaos Onset in Gradient Descent — Phase 1 Experiments"
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all Phase 1 experiments")
    parser.add_argument("--transition", action="store_true",
                        help="Run transition zone experiment")
    parser.add_argument("--broad", action="store_true",
                        help="Run broad sweep experiment")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run perturbation sensitivity analysis")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing .npz files")

    # Customization
    parser.add_argument("--seeds", type=int, default=20,
                        help="Number of random seeds (default: 20)")
    parser.add_argument("--n-lrs", type=int, default=40,
                        help="Number of learning rates (default: 40)")
    parser.add_argument("--lr-min", type=float, default=None,
                        help="Min learning rate")
    parser.add_argument("--lr-max", type=float, default=None,
                        help="Max learning rate")
    parser.add_argument("--train-steps", type=int, default=5000,
                        help="Training steps per run (default: 5000)")
    parser.add_argument("--activation", default="tanh",
                        choices=["tanh", "relu"],
                        help="Activation function (default: tanh)")

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["n_train_steps"] = args.train_steps
    config["activation"] = args.activation

    if args.plot_only:
        print("Generating plots from existing results...")
        if os.path.exists("results/transition_zone.npz"):
            plot_transition_zone()
        if os.path.exists("results/broad_sweep.npz"):
            plot_broad_sweep()
        if os.path.exists("results/sensitivity.npz"):
            plot_sensitivity()
        return

    run_any = args.all or args.transition or args.broad or args.sensitivity
    if not run_any:
        parser.print_help()
        print("\nQuick start:")
        print("  python run_experiments.py --transition --seeds 3  # fast test")
        print("  python run_experiments.py --all                   # full suite")
        return

    if args.all or args.transition:
        lr_min = args.lr_min if args.lr_min else 0.005
        lr_max = args.lr_max if args.lr_max else 0.08
        run_transition_zone(config, n_seeds=args.seeds, n_lrs=args.n_lrs,
                           lr_min=lr_min, lr_max=lr_max)
        plot_transition_zone()

    if args.all or args.broad:
        lr_min = args.lr_min if args.lr_min else 0.01
        lr_max = args.lr_max if args.lr_max else 0.42
        run_broad_sweep(config, n_seeds=args.seeds,
                        n_lrs=args.n_lrs if not args.transition else 25,
                        lr_min=lr_min, lr_max=lr_max)
        plot_broad_sweep()

    if args.all or args.sensitivity:
        run_sensitivity(config, n_seeds=min(args.seeds, 5))
        plot_sensitivity()

    print()
    print("Done. Results in results/, figures in figures/")


if __name__ == "__main__":
    main()
