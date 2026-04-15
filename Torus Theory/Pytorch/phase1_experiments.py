"""
Chaos Onset in Gradient Descent — Phase 1 Statistical Hardening
================================================================

Matches the EXACT protocol from the original run_experiments.py:
- Architecture: Input(220) → Linear(220,50) → Tanh → Linear(50,50) → Tanh → Linear(50,10)
- Data: 2000 samples, 10 classes, 200 Gaussian features + 20 quadratic (X[:,:20]**2)
- Perturbation: unit-norm random direction, magnitude ε, seed = init_seed + 999999
- Sharpness: λ_max computed after 1000 warmup steps at lr=0.01
- Lyapunov fit: middle 60% of log-distance vs step (linregress)
- Distance: recorded EVERY step, BEFORE the gradient update

CORRECTION from sensitivity analysis (Experiment B):
    ε = 1e-8 was in the noise floor — all LRs returned ~+0.00004 uniformly,
    a numerical artifact rather than genuine dynamics. Signal converges at
    ε ≥ 1e-6 and plateaus through 1e-3. Default changed to ε = 1e-5.
    
    At proper ε, chaos is NON-MONOTONIC with learning rate:
    - η ≈ 0.005: stable (λ < 0)
    - η ≈ 0.020: peak chaos (λ ≈ +0.0002)
    - η ≈ 0.080: strongly contracting (λ ≈ -0.0006)
    
    Transition zone range expanded to [0.005, 0.15] to capture the full arc.

Two experiments:
  A) 20-seed transition zone sweep  → tighten η_c, map chaos window
  B) ε sensitivity sweep            → validate measurement (COMPLETED)

USAGE:
    python phase1_experiments.py --transition --seeds 3         # quick test
    python phase1_experiments.py --transition                   # full 20-seed run
    python phase1_experiments.py --sensitivity                  # ε sweep
    python phase1_experiments.py --all                          # everything
    python phase1_experiments.py --broad                        # broad sweep too
    python phase1_experiments.py --plot-only                    # regenerate figs

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
    "activation": "tanh",

    # Data
    "n_samples": 2000,
    "n_classes": 10,
    "n_random_features": 200,
    "n_quadratic_features": 20,
    "data_seed": 42,

    # Training
    "n_train_steps": 5000,

    # Lyapunov
    # NOTE: ε = 1e-8 was in the noise floor (sensitivity analysis showed
    # uniform +0.00003 across all LRs — numerical artifact, not dynamics).
    # Signal converges at ε ≥ 1e-6 and plateaus through 1e-3.
    # Using 1e-5 as default: safely in the plateau, well below nonlinear regime.
    "perturbation_eps": 1e-5,
    "lyap_fit_start_frac": 0.2,
    "lyap_fit_end_frac": 0.8,

    # Sharpness
    "sharpness_iters": 100,
    "sharpness_warmup_steps": 1000,
    "sharpness_warmup_lr": 0.01,
}


# ============================================================
# 2. DATA GENERATION — matches original exactly
# ============================================================

def generate_data(config):
    """
    Deterministic synthetic data.
    200 Gaussian cluster features + 20 quadratic features (squares of first 20).
    """
    rng = np.random.RandomState(config["data_seed"])

    n = config["n_samples"]
    k = config["n_classes"]
    d_rand = config["n_random_features"]
    d_quad = config["n_quadratic_features"]

    # Class centers
    centers = rng.randn(k, d_rand) * 2.0

    # Assign and sample
    labels = rng.randint(0, k, size=n)
    X_rand = np.zeros((n, d_rand))
    for i in range(n):
        X_rand[i] = centers[labels[i]] + rng.randn(d_rand) * 0.5

    # Quadratic features: square the first d_quad dimensions
    X_quad = X_rand[:, :d_quad] ** 2

    # Concatenate → (2000, 220)
    X = np.concatenate([X_rand, X_quad], axis=1).astype(np.float32)

    # One-hot labels → (2000, 10)
    y = np.zeros((n, k), dtype=np.float32)
    y[np.arange(n), labels] = 1.0

    return torch.tensor(X), torch.tensor(y)


# ============================================================
# 3. MODEL — matches original exactly
# ============================================================

class MLP(nn.Module):
    """Two-hidden-layer MLP, following Cohen et al. (2021) setup."""

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
    return MLP(
        config["input_dim"],
        config["hidden_dim"],
        config["output_dim"],
        config["activation"],
    )


def clone_model_perturbed(model, eps, seed):
    """
    Clone model, perturb by eps along a unit-norm random direction.
    Direction seeded at seed + 999999 for reproducibility.
    """
    act_name = "tanh" if model.act == torch.tanh else "relu"
    clone = MLP(
        model.fc1.in_features,
        model.fc1.out_features,
        model.fc3.out_features,
        act_name,
    )
    clone.load_state_dict(model.state_dict())

    # Unit-norm random direction in parameter space
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
# 4. LYAPUNOV COMPUTATION — matches original exactly
# ============================================================

def compute_lyapunov(config, lr, seed, X, y, eps_override=None):
    """
    Train original + perturbed networks, compute function-space Lyapunov exponent.

    Returns:
        lyap_exponent: slope of log(distance) vs step over middle 60%
        final_loss: final training loss
        distances: np.array of function-space distances at every step
    """
    device = X.device
    eps = eps_override if eps_override is not None else config["perturbation_eps"]

    model = make_model(config, seed).to(device)
    perturbed = clone_model_perturbed(model, eps, seed).to(device)

    criterion = nn.MSELoss()
    n_steps = config["n_train_steps"]
    distances = np.zeros(n_steps)

    for t in range(n_steps):
        # Measure distance BEFORE update
        with torch.no_grad():
            f1 = model(X)
            f2 = perturbed(X)
            d = torch.norm(f1 - f2).item()
            distances[t] = d

        # Training step — original
        model.zero_grad()
        loss1 = criterion(model(X), y)
        loss1.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad

        # Training step — perturbed (identical)
        perturbed.zero_grad()
        loss2 = criterion(perturbed(X), y)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                p -= lr * p.grad

    # Fit Lyapunov exponent: slope of log(distance) over middle 60%
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
# 5. SHARPNESS — matches original (warmup then power iteration)
# ============================================================

def compute_sharpness(model, X, y, n_iter=100):
    """Largest Hessian eigenvalue via power iteration."""
    criterion = nn.MSELoss()

    v = [torch.randn_like(p) for p in model.parameters()]
    v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
    v = [vi / v_norm for vi in v]

    eigenvalue = 0.0
    for _ in range(n_iter):
        model.zero_grad()
        loss = criterion(model(X), y)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        Hv_terms = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(Hv_terms, model.parameters())

        eigenvalue = sum((hv * vi).sum().item() for hv, vi in zip(Hv, v))

        hv_norm = sum((hv ** 2).sum() for hv in Hv).sqrt().item()
        if hv_norm < 1e-12:
            break
        v = [hv.detach() / hv_norm for hv in Hv]

    return abs(eigenvalue)


def compute_eos_threshold(config, X, y, device):
    """
    Compute λ_max after 1000 warmup steps at lr=0.01 (matching original).
    Returns (λ_max, 2/λ_max).
    """
    print("  Computing sharpness (λ_max) after warmup...")
    ref_model = make_model(config, seed=0).to(device)
    criterion = nn.MSELoss()

    warmup_steps = config["sharpness_warmup_steps"]
    warmup_lr = config["sharpness_warmup_lr"]

    for t in range(warmup_steps):
        ref_model.zero_grad()
        loss = criterion(ref_model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in ref_model.parameters():
                p -= warmup_lr * p.grad

    lam_max = compute_sharpness(ref_model, X, y, config["sharpness_iters"])
    lr_eos = 2.0 / lam_max
    print(f"  λ_max ≈ {lam_max:.4f}, 2/λ_max ≈ {lr_eos:.4f}")
    return lam_max, lr_eos


# ============================================================
# 6. EXPERIMENT A: Transition Zone (20 seeds)
# ============================================================

def run_transition_zone(config, n_seeds=20, n_lrs=50,
                        lr_min=0.005, lr_max=0.15):
    """Fine-grained transition zone to pin down η_c."""

    print("=" * 60)
    print(f"EXPERIMENT A: TRANSITION ZONE — {n_seeds} seeds × {n_lrs} LRs")
    print(f"  LR range: [{lr_min}, {lr_max}]")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    n_params = sum(p.numel() for p in make_model(config, 0).parameters())
    print(f"  Parameters: {n_params}")

    lam_max, lr_eos = compute_eos_threshold(config, X, y, device)

    lrs = np.linspace(lr_min, lr_max, n_lrs)
    all_lyaps = np.zeros((n_seeds, n_lrs))
    all_losses = np.zeros((n_seeds, n_lrs))

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
                      f"lyap={lyap:+.6f} loss={loss_val:.6f} "
                      f"ETA: {eta:.0f}s")

    # Per-seed η_c by interpolation
    eta_c_per_seed = []
    for s in range(n_seeds):
        lyaps_s = all_lyaps[s]
        for j in range(len(lrs) - 1):
            if lyaps_s[j] <= 0 and lyaps_s[j + 1] > 0:
                frac = -lyaps_s[j] / (lyaps_s[j + 1] - lyaps_s[j])
                eta_c = lrs[j] + frac * (lrs[j + 1] - lrs[j])
                eta_c_per_seed.append(eta_c)
                break
        else:
            pos = np.where(lyaps_s > 0)[0]
            if len(pos) > 0:
                eta_c_per_seed.append(lrs[pos[0]])

    eta_c_arr = np.array(eta_c_per_seed)
    mean_lyaps = all_lyaps.mean(axis=0)
    std_lyaps = all_lyaps.std(axis=0)

    # Mean-curve zero crossing
    eta_c_from_mean = None
    for j in range(len(lrs) - 1):
        if mean_lyaps[j] <= 0 and mean_lyaps[j + 1] > 0:
            frac = -mean_lyaps[j] / (mean_lyaps[j + 1] - mean_lyaps[j])
            eta_c_from_mean = lrs[j] + frac * (lrs[j + 1] - lrs[j])
            break

    # Fraction chaotic at each LR
    frac_chaotic = (all_lyaps > 0).mean(axis=0)

    # 50% chaotic crossing
    eta_c_50pct = None
    for j in range(len(lrs) - 1):
        if frac_chaotic[j] < 0.5 and frac_chaotic[j + 1] >= 0.5:
            f1, f2 = frac_chaotic[j], frac_chaotic[j + 1]
            eta_c_50pct = lrs[j] + (0.5 - f1) * (lrs[j + 1] - lrs[j]) / (f2 - f1)
            break

    print()
    print("RESULTS:")
    print(f"  η_c per seed (n={len(eta_c_arr)}): {eta_c_arr}")
    if len(eta_c_arr) > 0:
        print(f"  η_c mean ± std: {eta_c_arr.mean():.5f} ± {eta_c_arr.std():.5f}")
        ci95 = 1.96 * eta_c_arr.std() / np.sqrt(len(eta_c_arr))
        print(f"  η_c 95% CI: [{eta_c_arr.mean()-ci95:.5f}, {eta_c_arr.mean()+ci95:.5f}]")
        print(f"  η_c / (2/λ_max): {eta_c_arr.mean() / lr_eos * 100:.1f}%")
    if eta_c_from_mean is not None:
        print(f"  η_c from mean curve: {eta_c_from_mean:.5f}")
    if eta_c_50pct is not None:
        print(f"  η_c from 50% chaotic: {eta_c_50pct:.5f}")
    print(f"  2/λ_max (EoS): {lr_eos:.5f}")

    # Per-LR summary table
    print()
    print(f"  {'η':>8s}  {'η/EoS':>6s}  {'mean λ':>10s}  {'±std':>8s}  {'chaotic':>8s}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}")
    for j, lr in enumerate(lrs):
        print(f"  {lr:8.5f}  {lr/lr_eos:6.3f}  {mean_lyaps[j]:+10.6f}  "
              f"{std_lyaps[j]:8.6f}  {frac_chaotic[j]:8.3f}")

    # Save
    results = {
        "lrs": lrs,
        "all_lyaps": all_lyaps,
        "all_losses": all_losses,
        "mean_lyaps": mean_lyaps,
        "std_lyaps": std_lyaps,
        "frac_chaotic": frac_chaotic,
        "seeds": np.arange(n_seeds),
        "lr_eos": lr_eos,
        "lam_max": lam_max,
        "eta_c_per_seed": eta_c_arr,
        "eta_c_mean": eta_c_arr.mean() if len(eta_c_arr) > 0 else np.nan,
        "eta_c_std": eta_c_arr.std() if len(eta_c_arr) > 0 else np.nan,
        "eta_c_from_mean": eta_c_from_mean if eta_c_from_mean else np.nan,
        "eta_c_50pct": eta_c_50pct if eta_c_50pct else np.nan,
        "config_str": str(config),
    }

    os.makedirs("results", exist_ok=True)
    np.savez("results/transition_zone.npz", **results)
    print(f"\n  Saved → results/transition_zone.npz")

    return results


# ============================================================
# 7. EXPERIMENT A2: Broad Sweep (20 seeds)
# ============================================================

def run_broad_sweep(config, n_seeds=20, n_lrs=25,
                    lr_min=0.01, lr_max=0.42):
    """Full Lyapunov curve across the EoS threshold."""

    print("=" * 60)
    print(f"BROAD SWEEP: {n_seeds} seeds × {n_lrs} LRs")
    print(f"  LR range: [{lr_min}, {lr_max}]")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    lam_max, lr_eos = compute_eos_threshold(config, X, y, device)

    lrs = np.linspace(lr_min, lr_max, n_lrs)
    all_lyaps = np.zeros((n_seeds, n_lrs))
    all_losses = np.zeros((n_seeds, n_lrs))

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
                      f"lyap={lyap:+.6f} ETA: {eta:.0f}s")

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
    print(f"  Saved → results/broad_sweep.npz")

    return results


# ============================================================
# 8. EXPERIMENT B: Perturbation Sensitivity (ε sweep)
# ============================================================

def run_sensitivity(config, n_seeds=5):
    """
    Test whether Lyapunov exponents are stable across perturbation magnitudes.
    
    PASS: λ flat across ε (true dynamical invariant)
    CONCERN: λ drifts monotonically with ε (nonlinear contamination)
    FAIL: λ changes sign with ε (measurement is meaningless)
    """

    print("=" * 60)
    print("EXPERIMENT B: PERTURBATION SENSITIVITY")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    epsilons = [1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
    test_lrs = [0.005, 0.010, 0.020, 0.040, 0.080]

    results_arr = np.zeros((len(epsilons), len(test_lrs), n_seeds))

    total = len(epsilons) * len(test_lrs) * n_seeds
    done = 0
    t0 = time.time()

    for ei, eps in enumerate(epsilons):
        for li, lr in enumerate(test_lrs):
            for s in range(n_seeds):
                lyap, _, _ = compute_lyapunov(
                    config, lr, seed=s, X=X, y=y, eps_override=eps
                )
                results_arr[ei, li, s] = lyap
                done += 1
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - t0
                    eta_t = elapsed / done * (total - done)
                    print(f"  [{done}/{total}] ε={eps:.0e} lr={lr:.3f} "
                          f"seed={s} lyap={lyap:+.6f} ETA: {eta_t:.0f}s")

    # Analysis
    print()
    print("SENSITIVITY ANALYSIS:")
    for li, lr in enumerate(test_lrs):
        print(f"\n  η = {lr:.3f}")
        print(f"  {'ε':>10s}  {'mean λ':>10s}  {'std λ':>8s}  {'sign':>5s}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*5}")
        means = []
        for ei, eps in enumerate(epsilons):
            vals = results_arr[ei, li, :]
            m = vals.mean()
            s = vals.std()
            means.append(m)
            print(f"  {eps:10.0e}  {m:+10.7f}  {s:8.6f}  {'  +' if m > 0 else '  -'}")

        # Diagnostic
        means = np.array(means)
        log_eps = np.log10(epsilons)
        corr = np.corrcoef(log_eps, means)[0, 1]
        signs = means > 0
        sign_changes = np.sum(np.diff(signs.astype(int)) != 0)
        cv = np.std(means) / (np.mean(np.abs(means)) + 1e-15)

        if sign_changes > 0:
            verdict = "⚠ SIGN CHANGE"
        elif abs(corr) > 0.8 and cv > 0.3:
            verdict = "⚠ MONOTONIC DRIFT"
        elif cv < 0.2:
            verdict = "✓ STABLE"
        else:
            verdict = "~ MODERATE"
        print(f"  Corr(log ε, λ): {corr:+.3f} | CV: {cv:.3f} | {verdict}")

    results = {
        "epsilons": np.array(epsilons),
        "test_lrs": np.array(test_lrs),
        "lyap_results": results_arr,
        "n_seeds": n_seeds,
    }

    os.makedirs("results", exist_ok=True)
    np.savez("results/sensitivity.npz", **results)
    print(f"\n  Saved → results/sensitivity.npz")

    return results


# ============================================================
# 9. PLOTTING
# ============================================================

def plot_transition_zone(npz_path="results/transition_zone.npz"):
    """Publication-quality transition zone figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    frac_chaotic = d["frac_chaotic"] if "frac_chaotic" in d else (all_lyaps > 0).mean(axis=0)

    # ── Figure 1: Transition zone (2 panels) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: individual seeds + mean
    colors = plt.cm.Set2(np.linspace(0, 1, min(n_seeds, 8)))
    for s in range(n_seeds):
        ax1.scatter(lrs, all_lyaps[s], s=12, alpha=0.35,
                    color=colors[s % 8], zorder=2,
                    label=f"Seed {s}" if s < 8 else None)
    ax1.plot(lrs, mean_lyaps, "k-s", ms=4, lw=1.5, label="Mean", zorder=3)
    ax1.fill_between(lrs, mean_lyaps - std_lyaps, mean_lyaps + std_lyaps,
                     alpha=0.15, color="gray")
    ax1.axhline(0, color="k", lw=0.5)
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

    # ── Figure 2: Fraction chaotic + KAM interleaving ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: fraction chaotic
    bar_colors = ['#d62728' if f > 0.5 else '#1f77b4' for f in frac_chaotic]
    ax1.bar(range(len(lrs)), frac_chaotic, color=bar_colors,
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(0.5, color='gray', ls='--', lw=0.8)
    ax1.set_xticks(range(0, len(lrs), max(1, len(lrs)//10)))
    ax1.set_xticklabels([f'{lrs[i]:.3f}' for i in range(0, len(lrs), max(1, len(lrs)//10))],
                        rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel("Learning rate η")
    ax1.set_ylabel("Fraction of seeds with λ > 0")
    ax1.set_title(f"Chaos prevalence across seeds (n={n_seeds})")
    ax1.set_ylim(-0.05, 1.05)

    # Right: KAM interleaving heatmap
    for j in range(len(lrs)):
        sorted_lyaps = np.sort(all_lyaps[:, j])
        for si, lyap in enumerate(sorted_lyaps):
            color = '#d62728' if lyap > 0 else '#1f77b4'
            ax2.scatter(j, si, c=color, s=20, marker='s', edgecolors='none')
    ax2.set_xticks(range(0, len(lrs), max(1, len(lrs)//10)))
    ax2.set_xticklabels([f'{lrs[i]:.3f}' for i in range(0, len(lrs), max(1, len(lrs)//10))],
                        rotation=45, ha='right', fontsize=8)
    ax2.set_xlabel("Learning rate η")
    ax2.set_ylabel("Seed (sorted by λ)")
    ax2.set_title("KAM interleaving: red=chaotic, blue=ordered")

    plt.tight_layout()
    plt.savefig("figures/transition_kam.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/transition_kam.pdf", bbox_inches="tight")
    print("  Saved → figures/transition_kam.png + .pdf")
    plt.close()


def plot_broad_sweep(npz_path="results/broad_sweep.npz"):
    """Publication-quality broad sweep figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    ax1.set_ylabel("Lyapunov exponent")
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
    """Perturbation sensitivity figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    epsilons = d["epsilons"]
    test_lrs = d["test_lrs"]
    lyap_results = d["lyap_results"]

    # ── Panel plot: one subplot per LR ──
    n_lrs = len(test_lrs)
    fig, axes = plt.subplots(1, n_lrs, figsize=(4 * n_lrs, 4), sharey=True)
    if n_lrs == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_lrs))
    for li, (ax, lr) in enumerate(zip(axes, test_lrs)):
        means = lyap_results[:, li, :].mean(axis=1)
        stds = lyap_results[:, li, :].std(axis=1)
        ax.errorbar(epsilons, means, yerr=stds, fmt="-o", ms=5,
                    color=colors[li], capsize=3, lw=1.5)

        # Individual points
        for ei, eps in enumerate(epsilons):
            vals = lyap_results[ei, li, :]
            jitter = np.random.RandomState(42).uniform(0.85, 1.15, len(vals))
            ax.scatter([eps * j for j in jitter], vals,
                       c=[colors[li]], alpha=0.3, s=12, zorder=1)

        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('ε')
        ax.set_title(f'η = {lr:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Lyapunov Exponent')
    fig.suptitle('Perturbation Sensitivity: Is λ Independent of ε?', fontsize=14, y=1.02)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/sensitivity.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/sensitivity.pdf", bbox_inches="tight")
    print("  Saved → figures/sensitivity.png + .pdf")
    plt.close()

    # ── Heatmap ──
    fig, ax = plt.subplots(figsize=(8, 5))
    heatmap = lyap_results.mean(axis=2)  # (n_eps, n_lrs)
    vmax = np.abs(heatmap).max()
    im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, origin='lower')

    ax.set_xticks(range(n_lrs))
    ax.set_xticklabels([f'{lr:.3f}' for lr in test_lrs])
    ax.set_yticks(range(len(epsilons)))
    ax.set_yticklabels([f'{eps:.0e}' for eps in epsilons])
    ax.set_xlabel('Learning Rate η')
    ax.set_ylabel('Perturbation ε')
    ax.set_title('Mean λ: stable = consistent color down each column')

    for i in range(len(epsilons)):
        for j in range(n_lrs):
            val = heatmap[i, j]
            ax.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(val) > vmax * 0.5 else 'black')

    plt.colorbar(im, label='Lyapunov Exponent')
    plt.tight_layout()
    plt.savefig("figures/sensitivity_heatmap.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/sensitivity_heatmap.pdf", bbox_inches="tight")
    print("  Saved → figures/sensitivity_heatmap.png + .pdf")
    plt.close()


# ============================================================
# 10. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chaos Onset — Phase 1 Statistical Hardening"
    )
    parser.add_argument("--all", action="store_true",
                        help="Run transition + broad + sensitivity")
    parser.add_argument("--transition", action="store_true",
                        help="Experiment A: transition zone")
    parser.add_argument("--broad", action="store_true",
                        help="Experiment A2: broad sweep")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Experiment B: ε sensitivity")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate figures from .npz files")

    parser.add_argument("--seeds", type=int, default=20,
                        help="Number of seeds (default: 20)")
    parser.add_argument("--n-lrs", type=int, default=40,
                        help="Number of learning rates (default: 40)")
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--lr-max", type=float, default=None)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--activation", default="tanh",
                        choices=["tanh", "relu"])

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
        print("  python phase1_experiments.py --sensitivity --seeds 3    # fast (~30 min)")
        print("  python phase1_experiments.py --transition --seeds 3     # test (~1 hr)")
        print("  python phase1_experiments.py --all                      # full suite")
        return

    if args.all or args.sensitivity:
        run_sensitivity(config, n_seeds=min(args.seeds, 5))
        plot_sensitivity()

    if args.all or args.transition:
        lr_min = args.lr_min if args.lr_min else 0.005
        lr_max = args.lr_max if args.lr_max else 0.15
        n_lrs = args.n_lrs if args.n_lrs != 40 else 50  # default 50 for wider range
        run_transition_zone(config, n_seeds=args.seeds, n_lrs=n_lrs,
                           lr_min=lr_min, lr_max=lr_max)
        plot_transition_zone()

    if args.all or args.broad:
        lr_min = args.lr_min if args.lr_min else 0.01
        lr_max = args.lr_max if args.lr_max else 0.42
        run_broad_sweep(config, n_seeds=args.seeds,
                        n_lrs=args.n_lrs if not args.transition else 25,
                        lr_min=lr_min, lr_max=lr_max)
        plot_broad_sweep()

    print()
    print("Done. Results in results/, figures in figures/")


if __name__ == "__main__":
    main()
