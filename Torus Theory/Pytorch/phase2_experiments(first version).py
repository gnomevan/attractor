"""
Chaos Onset — Phase 2: Dynamical Systems Characterization
===========================================================

Uses the same architecture/data/training as Phase 1, but records different
measurements to characterize the GEOMETRY of the training dynamics.

Four experiments:
  C) Power spectrum of training loss — discrete peaks (torus) vs broadband (chaos)
  D) Takens embedding of loss time series — visualize the attractor shape
  E) Bifurcation diagram — loss vs η at fine spacing
  F) Correlation dimension — fractal dimension of training trajectory vs η

All share the Phase 1 infrastructure (model, data, training loop).

USAGE:
    python phase2_experiments.py --all                          # everything
    python phase2_experiments.py --power-spectrum               # Experiment C
    python phase2_experiments.py --takens                       # Experiment D
    python phase2_experiments.py --bifurcation                  # Experiment E
    python phase2_experiments.py --dimension                    # Experiment F
    python phase2_experiments.py --plot-only                    # regenerate figs

REQUIREMENTS:
    pip install torch numpy matplotlib scipy
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import stats, signal

# ============================================================
# SHARED INFRASTRUCTURE (identical to Phase 1)
# ============================================================

DEFAULT_CONFIG = {
    "input_dim": 220,
    "hidden_dim": 50,
    "output_dim": 10,
    "activation": "tanh",
    "n_samples": 2000,
    "n_classes": 10,
    "n_random_features": 200,
    "n_quadratic_features": 20,
    "data_seed": 42,
    "n_train_steps": 5000,
    "perturbation_eps": 1e-5,
    "sharpness_iters": 100,
    "sharpness_warmup_steps": 1000,
    "sharpness_warmup_lr": 0.01,
}


def generate_data(config):
    rng = np.random.RandomState(config["data_seed"])
    n, k = config["n_samples"], config["n_classes"]
    d_rand, d_quad = config["n_random_features"], config["n_quadratic_features"]
    centers = rng.randn(k, d_rand) * 2.0
    labels = rng.randint(0, k, size=n)
    X_rand = np.zeros((n, d_rand))
    for i in range(n):
        X_rand[i] = centers[labels[i]] + rng.randn(d_rand) * 0.5
    X_quad = X_rand[:, :d_quad] ** 2
    X = np.concatenate([X_rand, X_quad], axis=1).astype(np.float32)
    y = np.zeros((n, k), dtype=np.float32)
    y[np.arange(n), labels] = 1.0
    return torch.tensor(X), torch.tensor(y)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation="tanh"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = torch.tanh if activation == "tanh" else torch.relu

    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


def make_model(config, seed):
    torch.manual_seed(seed)
    return MLP(config["input_dim"], config["hidden_dim"],
               config["output_dim"], config["activation"])


# ============================================================
# TRAINING WITH FULL RECORDING
# ============================================================

def train_and_record(config, lr, seed, X, y, n_steps=None,
                     record_outputs=False, X_eval=None, output_every=1):
    """
    Train a network and record the full loss time series.
    Optionally record network outputs on evaluation data.

    Returns dict with:
        losses: np.array of shape (n_steps,)
        outputs: np.array of shape (n_record, n_eval, output_dim) if record_outputs
    """
    device = X.device
    if n_steps is None:
        n_steps = config["n_train_steps"]
    if X_eval is None:
        X_eval = X

    model = make_model(config, seed).to(device)
    criterion = nn.MSELoss()

    losses = np.zeros(n_steps)
    outputs = [] if record_outputs else None

    for t in range(n_steps):
        with torch.no_grad():
            loss_val = criterion(model(X), y).item()
            losses[t] = loss_val

            if record_outputs and t % output_every == 0:
                out = model(X_eval).cpu().numpy()
                outputs.append(out)

        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad

    result = {"losses": losses, "lr": lr, "seed": seed}
    if record_outputs:
        result["outputs"] = np.array(outputs)
    return result


# ============================================================
# EXPERIMENT C: Power Spectrum of Training Loss
# ============================================================

def run_power_spectrum(config, n_seeds=5, n_steps=20000):
    """
    Record loss at every step for long training, compute power spectral density.

    Prediction from torus framework:
    - Low η (stable regime): discrete spectral peaks (periodic/quasiperiodic)
    - Chaos window (~0.01-0.04): peaks broadening, possibly broadband
    - High η (basin regime): broadband but low-power (contracting dynamics)
    """
    # LRs spanning all three regimes identified in Phase 1
    test_lrs = [0.005, 0.010, 0.015, 0.020, 0.030, 0.040,
                0.060, 0.080, 0.100, 0.150, 0.200, 0.300]

    print("=" * 60)
    print(f"EXPERIMENT C: POWER SPECTRUM — {n_seeds} seeds × {len(test_lrs)} LRs")
    print(f"  Steps per run: {n_steps}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    cfg = dict(config)
    cfg["n_train_steps"] = n_steps

    all_results = {}  # lr -> list of (freqs, psd) per seed

    total = len(test_lrs) * n_seeds
    done = 0
    t0 = time.time()

    for lr in test_lrs:
        all_results[lr] = {"psds": [], "losses": [], "peak_freqs": []}
        for s in range(n_seeds):
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 1 else 0
            print(f"  [{done}/{total}] lr={lr:.3f} seed={s} ETA: {eta:.0f}s",
                  end="", flush=True)

            result = train_and_record(cfg, lr, s, X, y, n_steps=n_steps)
            losses = result["losses"]

            # Skip early transient (first 20%), use rest for spectral analysis
            start = n_steps // 5
            loss_signal = losses[start:]

            # Detrend: remove linear trend to focus on oscillations
            loss_detrended = signal.detrend(loss_signal, type='linear')

            # Power spectral density via Welch's method
            freqs, psd = signal.welch(loss_detrended, fs=1.0,
                                       nperseg=min(1024, len(loss_detrended) // 4),
                                       noverlap=None)

            all_results[lr]["psds"].append(psd)
            all_results[lr]["losses"].append(losses)

            # Find dominant peaks
            peaks, properties = signal.find_peaks(psd, height=np.median(psd) * 5,
                                                   distance=3)
            peak_freqs = freqs[peaks][:5]  # top 5 peaks
            all_results[lr]["peak_freqs"].append(peak_freqs)

            print(f"  → {len(peaks)} peaks, max_psd={psd.max():.2e}")

    # Save
    save_data = {
        "test_lrs": np.array(test_lrs),
        "freqs": freqs,
        "n_seeds": n_seeds,
        "n_steps": n_steps,
    }
    for li, lr in enumerate(test_lrs):
        save_data[f"psds_{li}"] = np.array(all_results[lr]["psds"])
        # Save a subsample of losses (every 10th step) to keep file small
        save_data[f"losses_{li}"] = np.array(all_results[lr]["losses"])[:, ::10]

    os.makedirs("results", exist_ok=True)
    np.savez("results/power_spectrum.npz", **save_data)
    print(f"\n  Saved → results/power_spectrum.npz")

    return all_results


# ============================================================
# EXPERIMENT D: Takens Delay Embedding
# ============================================================

def optimal_delay(x, max_lag=500):
    """Estimate optimal delay τ as first minimum of autocorrelation."""
    n = len(x)
    x_centered = x - x.mean()
    acf = np.correlate(x_centered, x_centered, mode='full')
    acf = acf[n-1:]  # positive lags only
    acf = acf / acf[0]  # normalize

    for i in range(1, min(max_lag, len(acf) - 1)):
        if acf[i] < acf[i-1] and acf[i] < acf[i+1]:
            return i
    return max_lag // 4  # fallback


def delay_embed(x, dim, tau):
    """Construct delay embedding: [x(t), x(t+τ), ..., x(t+(d-1)τ)]"""
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError(f"Signal too short for dim={dim}, tau={tau}")
    embedded = np.zeros((n, dim))
    for d in range(dim):
        embedded[:, d] = x[d * tau: d * tau + n]
    return embedded


def correlation_dimension(points, n_scales=20):
    """
    Grassberger-Procaccia correlation dimension estimate.
    Subsample if too many points for O(n²) distance computation.
    """
    n = len(points)
    if n > 2000:
        idx = np.random.RandomState(42).choice(n, 2000, replace=False)
        points = points[idx]
        n = 2000

    # Pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(points[i] - points[j]))
    dists = np.array(dists)

    if len(dists) == 0:
        return float('nan'), [], []

    # Correlation integral at multiple scales
    log_eps = np.linspace(np.log(np.percentile(dists, 1) + 1e-15),
                          np.log(np.percentile(dists, 95)),
                          n_scales)
    epsilons = np.exp(log_eps)

    log_C = []
    for eps in epsilons:
        C = np.sum(dists < eps) / (n * (n - 1) / 2)
        if C > 0:
            log_C.append(np.log(C))
        else:
            log_C.append(-30)

    log_C = np.array(log_C)

    # Fit slope over middle 60%
    start = len(log_eps) // 5
    end = 4 * len(log_eps) // 5
    if end - start > 2:
        slope, _, _, _, _ = stats.linregress(log_eps[start:end], log_C[start:end])
    else:
        slope = float('nan')

    return slope, log_eps, log_C


def run_takens(config, n_seeds=5, n_steps=20000):
    """
    Takens delay embedding of training loss time series.

    For each LR: embed loss(t) in delay coordinates, visualize attractor,
    compute correlation dimension.
    """
    test_lrs = [0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.200]
    embed_dim = 3  # for visualization
    embed_dims_for_dim = [3, 5, 7, 9]  # for dimension estimation

    print("=" * 60)
    print(f"EXPERIMENT D: TAKENS EMBEDDING — {n_seeds} seeds × {len(test_lrs)} LRs")
    print(f"  Steps per run: {n_steps}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    cfg = dict(config)
    cfg["n_train_steps"] = n_steps

    all_results = {}

    total = len(test_lrs) * n_seeds
    done = 0
    t0 = time.time()

    for lr in test_lrs:
        all_results[lr] = {
            "embeddings_3d": [],  # for visualization
            "taus": [],
            "corr_dims": [],     # dimension vs embed_dim
        }

        for s in range(n_seeds):
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 1 else 0
            print(f"  [{done}/{total}] lr={lr:.3f} seed={s} ETA: {eta:.0f}s",
                  end="", flush=True)

            result = train_and_record(cfg, lr, s, X, y, n_steps=n_steps)
            losses = result["losses"]

            # Skip transient
            start = n_steps // 5
            loss_signal = losses[start:]

            # Detrend
            loss_detrended = signal.detrend(loss_signal, type='linear')

            # Optimal delay
            tau = optimal_delay(loss_detrended)
            all_results[lr]["taus"].append(tau)

            # 3D embedding for visualization (store subsampled)
            emb3 = delay_embed(loss_detrended, 3, tau)
            step = max(1, len(emb3) // 2000)
            all_results[lr]["embeddings_3d"].append(emb3[::step])

            # Correlation dimension at multiple embedding dimensions
            dims = []
            for ed in embed_dims_for_dim:
                try:
                    emb = delay_embed(loss_detrended, ed, tau)
                    d, _, _ = correlation_dimension(emb)
                    dims.append(d)
                except ValueError:
                    dims.append(float('nan'))
            all_results[lr]["corr_dims"].append(dims)

            print(f"  → τ={tau}, D₂={dims[0]:.2f}" if not np.isnan(dims[0]) else f"  → τ={tau}")

    # Save
    save_data = {
        "test_lrs": np.array(test_lrs),
        "embed_dims": np.array(embed_dims_for_dim),
        "n_seeds": n_seeds,
        "n_steps": n_steps,
    }
    for li, lr in enumerate(test_lrs):
        save_data[f"taus_{li}"] = np.array(all_results[lr]["taus"])
        save_data[f"corr_dims_{li}"] = np.array(all_results[lr]["corr_dims"])
        # Save first seed's 3D embedding for visualization
        save_data[f"emb3d_{li}"] = all_results[lr]["embeddings_3d"][0]

    os.makedirs("results", exist_ok=True)
    np.savez("results/takens_embedding.npz", **save_data)
    print(f"\n  Saved → results/takens_embedding.npz")

    return all_results


# ============================================================
# EXPERIMENT E: Bifurcation Diagram
# ============================================================

def run_bifurcation(config, n_seeds=5, n_lrs=200, n_steps=10000,
                    lr_min=0.001, lr_max=0.20, record_last_n=200):
    """
    At each LR, record the final `record_last_n` loss values.
    Plot loss vs η — the classic bifurcation diagram.

    Expect:
    - Low η: single converged value (fixed point)
    - Medium η: possibly 2 or 4 alternating values (period-doubling)
    - Chaos window: filled band
    - High η: broader filled band (chaotic exploration within basin)
    """
    print("=" * 60)
    print(f"EXPERIMENT E: BIFURCATION DIAGRAM — {n_seeds} seeds × {n_lrs} LRs")
    print(f"  LR range: [{lr_min}, {lr_max}], steps: {n_steps}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    cfg = dict(config)
    cfg["n_train_steps"] = n_steps

    lrs = np.linspace(lr_min, lr_max, n_lrs)
    # Store last N loss values for each (lr, seed)
    all_final_losses = np.zeros((n_seeds, n_lrs, record_last_n))

    total = n_seeds * n_lrs
    done = 0
    t0 = time.time()

    for s in range(n_seeds):
        for j, lr in enumerate(lrs):
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 1 else 0
                print(f"  [{done}/{total}] seed={s} lr={lr:.4f} ETA: {eta:.0f}s")

            result = train_and_record(cfg, lr, s, X, y, n_steps=n_steps)
            all_final_losses[s, j, :] = result["losses"][-record_last_n:]

    # Save
    results = {
        "lrs": lrs,
        "all_final_losses": all_final_losses,
        "n_seeds": n_seeds,
        "n_steps": n_steps,
        "record_last_n": record_last_n,
    }

    os.makedirs("results", exist_ok=True)
    np.savez("results/bifurcation.npz", **results)
    print(f"  Saved → results/bifurcation.npz")

    return results


# ============================================================
# EXPERIMENT F: Correlation Dimension of Function-Space Trajectory
# ============================================================

def run_dimension(config, n_seeds=5, n_steps=10000):
    """
    Record network output f_θ(X_eval) at regular intervals during training.
    This gives a trajectory in R^(n_eval × output_dim).
    Compute correlation dimension to quantify attractor complexity.

    Predict:
    - Low η: D ≈ 1 (convergence path)
    - Chaos window: D > 1, possibly non-integer
    - High η: D possibly higher (more complex basin exploration)
    """
    test_lrs = [0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.200]
    n_eval = 100  # subsample of training data for output recording
    output_every = 5  # record every 5th step

    print("=" * 60)
    print(f"EXPERIMENT F: CORRELATION DIMENSION — {n_seeds} seeds × {len(test_lrs)} LRs")
    print(f"  Steps: {n_steps}, recording every {output_every} steps")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    # Subsample evaluation points
    torch.manual_seed(0)
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    cfg = dict(config)
    cfg["n_train_steps"] = n_steps

    all_results = {}

    total = len(test_lrs) * n_seeds
    done = 0
    t0 = time.time()

    for lr in test_lrs:
        all_results[lr] = {"corr_dims": [], "pca_variance": []}

        for s in range(n_seeds):
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 1 else 0
            print(f"  [{done}/{total}] lr={lr:.3f} seed={s} ETA: {eta:.0f}s",
                  end="", flush=True)

            result = train_and_record(cfg, lr, s, X, y, n_steps=n_steps,
                                      record_outputs=True, X_eval=X_eval,
                                      output_every=output_every)

            # outputs shape: (n_record, n_eval, output_dim)
            outputs = result["outputs"]

            # Skip transient (first 20%)
            start = len(outputs) // 5
            outputs = outputs[start:]

            # Flatten to trajectory: each point is the full output vector
            # Shape: (n_record, n_eval * output_dim)
            trajectory = outputs.reshape(len(outputs), -1)

            # PCA variance (how many dimensions needed?)
            centered = trajectory - trajectory.mean(axis=0)
            try:
                _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
                var_explained = (s_vals ** 2) / (s_vals ** 2).sum()
                cumvar = np.cumsum(var_explained)
                all_results[lr]["pca_variance"].append(var_explained[:20])
            except:
                all_results[lr]["pca_variance"].append(np.zeros(20))

            # Correlation dimension
            # Subsample trajectory points for efficiency
            if len(trajectory) > 2000:
                idx = np.random.RandomState(s).choice(len(trajectory), 2000, replace=False)
                traj_sub = trajectory[idx]
            else:
                traj_sub = trajectory

            d, _, _ = correlation_dimension(traj_sub)
            all_results[lr]["corr_dims"].append(d)

            pc1 = var_explained[0] if len(var_explained) > 0 else 0
            print(f"  → D₂={d:.2f}, PC1={pc1*100:.1f}%")

    # Save
    save_data = {
        "test_lrs": np.array(test_lrs),
        "n_seeds": n_seeds,
        "n_steps": n_steps,
        "n_eval": n_eval,
        "output_every": output_every,
    }
    for li, lr in enumerate(test_lrs):
        save_data[f"corr_dims_{li}"] = np.array(all_results[lr]["corr_dims"])
        save_data[f"pca_var_{li}"] = np.array(all_results[lr]["pca_variance"])

    os.makedirs("results", exist_ok=True)
    np.savez("results/dimension.npz", **save_data)
    print(f"\n  Saved → results/dimension.npz")

    return all_results


# ============================================================
# PLOTTING
# ============================================================

def plot_power_spectrum(npz_path="results/power_spectrum.npz"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    test_lrs = d["test_lrs"]
    freqs = d["freqs"]
    n_lrs = len(test_lrs)

    # Determine grid layout
    ncols = 4
    nrows = (n_lrs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=True)
    axes = axes.flatten()

    for li, lr in enumerate(test_lrs):
        ax = axes[li]
        psds = d[f"psds_{li}"]
        mean_psd = psds.mean(axis=0)

        # Individual seeds
        for s in range(len(psds)):
            ax.semilogy(freqs, psds[s], alpha=0.3, lw=0.5)
        ax.semilogy(freqs, mean_psd, 'k-', lw=1.5, label='Mean')

        ax.set_title(f'η = {lr:.3f}', fontsize=10)
        ax.set_xlabel('Frequency (cycles/step)', fontsize=8)
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for li in range(n_lrs, len(axes)):
        axes[li].set_visible(False)

    axes[0].set_ylabel('PSD', fontsize=10)
    fig.suptitle('Power Spectra of Training Loss\nDiscrete peaks = toroidal, Broadband = chaotic',
                 fontsize=13, y=1.02)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/power_spectrum.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/power_spectrum.pdf", bbox_inches="tight")
    print("  Saved → figures/power_spectrum.png + .pdf")
    plt.close()

    # Summary figure: spectral entropy vs LR
    fig, ax = plt.subplots(figsize=(8, 4))
    entropies = []
    for li, lr in enumerate(test_lrs):
        psds = d[f"psds_{li}"]
        mean_psd = psds.mean(axis=0)
        # Spectral entropy (normalized)
        psd_norm = mean_psd / mean_psd.sum()
        psd_norm = psd_norm[psd_norm > 0]
        entropy = -np.sum(psd_norm * np.log(psd_norm))
        max_entropy = np.log(len(psd_norm))
        entropies.append(entropy / max_entropy)

    ax.plot(test_lrs, entropies, 'ko-', ms=6, lw=2)
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Normalized Spectral Entropy', fontsize=12)
    ax.set_title('Spectral Entropy: 0 = pure tone (torus), 1 = white noise (chaos)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/spectral_entropy.png", dpi=200, bbox_inches="tight")
    print("  Saved → figures/spectral_entropy.png")
    plt.close()


def plot_takens(npz_path="results/takens_embedding.npz"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    d = np.load(npz_path, allow_pickle=True)
    test_lrs = d["test_lrs"]
    embed_dims = d["embed_dims"]
    n_lrs = len(test_lrs)

    # 3D embeddings
    ncols = 4
    nrows = (n_lrs + ncols - 1) // ncols
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))

    for li, lr in enumerate(test_lrs):
        ax = fig.add_subplot(nrows, ncols, li + 1, projection='3d')
        emb = d[f"emb3d_{li}"]
        colors = np.linspace(0, 1, len(emb))
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colors, cmap='viridis',
                   s=1, alpha=0.5)
        ax.set_title(f'η = {lr:.3f}', fontsize=10)
        ax.set_xlabel('L(t)', fontsize=7)
        ax.set_ylabel('L(t+τ)', fontsize=7)
        ax.set_zlabel('L(t+2τ)', fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle('Takens Delay Embedding of Training Loss\nColor = training time (yellow→purple)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/takens_embeddings.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/takens_embeddings.pdf", bbox_inches="tight")
    print("  Saved → figures/takens_embeddings.png + .pdf")
    plt.close()

    # Correlation dimension vs LR
    fig, ax = plt.subplots(figsize=(8, 5))
    for di, ed in enumerate(embed_dims):
        means = []
        stds = []
        for li in range(n_lrs):
            dims = d[f"corr_dims_{li}"][:, di]
            means.append(np.nanmean(dims))
            stds.append(np.nanstd(dims))
        ax.errorbar(test_lrs, means, yerr=stds, fmt='o-', ms=5, capsize=3,
                    label=f'd_embed = {ed}')

    ax.axhline(1, color='gray', ls=':', lw=0.8, label='D=1 (curve)')
    ax.axhline(2, color='gray', ls='--', lw=0.8, label='D=2 (surface)')
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Correlation Dimension D₂', fontsize=12)
    ax.set_title('Attractor Dimension of Training Loss Dynamics', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/takens_dimension.png", dpi=200, bbox_inches="tight")
    print("  Saved → figures/takens_dimension.png")
    plt.close()


def plot_bifurcation(npz_path="results/bifurcation.npz"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]
    all_final = d["all_final_losses"]  # (n_seeds, n_lrs, record_last_n)
    n_seeds = all_final.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full bifurcation diagram (one seed)
    ax = axes[0]
    for j, lr in enumerate(lrs):
        vals = all_final[0, j, :]  # seed 0
        ax.scatter([lr] * len(vals), vals, c='black', s=0.1, alpha=0.3)
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Training Loss (final 200 steps)', fontsize=12)
    ax.set_title('Bifurcation Diagram (Seed 0)', fontsize=13)
    ax.set_yscale('log')

    # Right: loss range (max - min of final values) vs LR
    ax = axes[1]
    for s in range(min(n_seeds, 5)):
        ranges = all_final[s].max(axis=1) - all_final[s].min(axis=1)
        ax.plot(lrs, ranges, '-', alpha=0.5, lw=0.8, label=f'Seed {s}')

    mean_range = (all_final.max(axis=2) - all_final.min(axis=2)).mean(axis=0)
    ax.plot(lrs, mean_range, 'k-', lw=2, label='Mean')
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Loss oscillation amplitude', fontsize=12)
    ax.set_title('Oscillation Amplitude vs Learning Rate', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/bifurcation.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/bifurcation.pdf", bbox_inches="tight")
    print("  Saved → figures/bifurcation.png + .pdf")
    plt.close()


def plot_dimension(npz_path="results/dimension.npz"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    test_lrs = d["test_lrs"]
    n_lrs = len(test_lrs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: correlation dimension vs LR
    means = []
    stds = []
    for li in range(n_lrs):
        dims = d[f"corr_dims_{li}"]
        means.append(np.nanmean(dims))
        stds.append(np.nanstd(dims))

    ax1.errorbar(test_lrs, means, yerr=stds, fmt='ko-', ms=6, capsize=3, lw=2)
    ax1.axhline(1, color='gray', ls=':', lw=0.8, label='D=1 (line)')
    ax1.axhline(2, color='gray', ls='--', lw=0.8, label='D=2 (surface)')
    ax1.set_xlabel('Learning Rate η', fontsize=12)
    ax1.set_ylabel('Correlation Dimension', fontsize=12)
    ax1.set_title('Function-Space Trajectory Dimension', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: PCA variance explained (PC1%) vs LR
    pc1_means = []
    pc1_stds = []
    for li in range(n_lrs):
        pca = d[f"pca_var_{li}"]
        pc1s = pca[:, 0] if pca.ndim == 2 else [0]
        pc1_means.append(np.mean(pc1s))
        pc1_stds.append(np.std(pc1s))

    ax2.errorbar(test_lrs, [m*100 for m in pc1_means],
                 yerr=[s*100 for s in pc1_stds],
                 fmt='ko-', ms=6, capsize=3, lw=2)
    ax2.set_xlabel('Learning Rate η', fontsize=12)
    ax2.set_ylabel('Variance in PC1 (%)', fontsize=12)
    ax2.set_title('Dimensionality: Lower PC1% = Higher-Dimensional Dynamics', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/dimension_analysis.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/dimension_analysis.pdf", bbox_inches="tight")
    print("  Saved → figures/dimension_analysis.png + .pdf")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chaos Onset — Phase 2 Dynamical Characterization"
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--power-spectrum", action="store_true", help="Experiment C")
    parser.add_argument("--takens", action="store_true", help="Experiment D")
    parser.add_argument("--bifurcation", action="store_true", help="Experiment E")
    parser.add_argument("--dimension", action="store_true", help="Experiment F")
    parser.add_argument("--plot-only", action="store_true")

    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20000,
                        help="Training steps for C/D (default: 20000)")
    parser.add_argument("--bif-steps", type=int, default=10000,
                        help="Training steps for bifurcation (default: 10000)")
    parser.add_argument("--bif-lrs", type=int, default=200,
                        help="Number of LRs for bifurcation (default: 200)")

    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)

    if args.plot_only:
        print("Generating plots from existing results...")
        for name, func in [("power_spectrum", plot_power_spectrum),
                           ("takens_embedding", plot_takens),
                           ("bifurcation", plot_bifurcation),
                           ("dimension", plot_dimension)]:
            path = f"results/{name}.npz"
            if os.path.exists(path):
                func(path)
        return

    run_any = args.all or args.power_spectrum or args.takens or args.bifurcation or args.dimension
    if not run_any:
        parser.print_help()
        print("\nQuick start:")
        print("  python phase2_experiments.py --power-spectrum --seeds 2    # fast test")
        print("  python phase2_experiments.py --all --seeds 3              # moderate")
        print("  python phase2_experiments.py --all                        # full suite")
        return

    if args.all or args.power_spectrum:
        run_power_spectrum(config, n_seeds=args.seeds, n_steps=args.steps)
        plot_power_spectrum()

    if args.all or args.takens:
        run_takens(config, n_seeds=args.seeds, n_steps=args.steps)
        plot_takens()

    if args.all or args.bifurcation:
        run_bifurcation(config, n_seeds=args.seeds, n_lrs=args.bif_lrs,
                        n_steps=args.bif_steps)
        plot_bifurcation()

    if args.all or args.dimension:
        run_dimension(config, n_seeds=args.seeds, n_steps=args.steps)
        plot_dimension()

    print("\nDone. Results in results/, figures in figures/")


if __name__ == "__main__":
    main()
