"""
Geometric Structure of the Chaos Transition
============================================

This experiment goes beyond scalar Lyapunov exponents to ask:
WHAT HAPPENS in each dimension when training becomes chaotic?

Instead of measuring total distance ‖f₁(X) - f₂(X)‖, we track the
per-dimension correlation between two networks (original + perturbed)
across all 10 output dimensions, over the course of training.

Three signatures can appear:

  PRESERVATION:  correlation stays near +1
    → the two networks learn the same thing in this dimension
    → the torus is intact along this axis

  INVERSION:     correlation goes toward -1
    → the two networks learn MIRROR IMAGES in this dimension
    → this is period-doubling: the eigenvalue crossed -1
    → the torus has flipped inside-out along this axis

  DECORRELATION: correlation goes toward 0
    → the two networks learn unrelated things in this dimension
    → this is genuine chaos: no structure preserved
    → the torus has shattered along this axis

The key prediction from the torus framework: the transition is NOT
uniform across dimensions. At moderate learning rates, you should see
a MIX — some dimensions preserving, some inverting, some decorrelating.
The "chaos" is geometrically structured. The donut doesn't just break;
it folds and mirrors as it breaks.

USAGE:
    python geometric_structure.py --run
    python geometric_structure.py --run --seeds 5 --lr 0.03
    python geometric_structure.py --sweep          # across learning rates
    python geometric_structure.py --plot-only

OUTPUTS:
    results/geometric_*.npz
    figures/geometric_*.png
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn

# Import model and data generation from run_experiments.py
# (or copy the relevant functions here if running standalone)
try:
    from run_experiments import (
        DEFAULT_CONFIG, generate_data, make_model,
        clone_model_perturbed, compute_sharpness
    )
except ImportError:
    # ---- Standalone fallback: include necessary functions ----

    DEFAULT_CONFIG = {
        "input_dim": 220, "hidden_dim": 50, "output_dim": 10,
        "activation": "tanh", "n_samples": 2000, "n_classes": 10,
        "n_random_features": 200, "n_quadratic_features": 20,
        "data_seed": 42, "n_train_steps": 5000, "loss_fn": "mse",
        "perturbation_eps": 1e-8, "lyap_fit_start_frac": 0.2,
        "lyap_fit_end_frac": 0.8, "sharpness_iters": 100,
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

    def clone_model_perturbed(model, eps, seed):
        clone = MLP(model.fc1.in_features, model.fc1.out_features,
                     model.fc3.out_features,
                     "tanh" if model.act == torch.tanh else "relu")
        clone.load_state_dict(model.state_dict())
        rng = torch.Generator(); rng.manual_seed(seed + 999999)
        params = []; [params.append(p.data.view(-1)) for p in clone.parameters()]
        flat = torch.cat(params)
        d = torch.randn(flat.shape, generator=rng); d = d / d.norm()
        off = 0
        for p in clone.parameters():
            n = p.numel(); p.data += eps * d[off:off+n].view(p.shape); off += n
        return clone


# ============================================================
# CORE MEASUREMENT: Per-Dimension Correlation Over Training
# ============================================================

def compute_geometric_structure(config, lr, seed, X, y, record_every=10):
    """
    Train two networks (original + perturbed) and track the
    per-output-dimension correlation between their outputs over time.

    Returns:
        correlations: (n_records, n_outputs) — Pearson correlation per dim per step
        cosine_sims:  (n_records, n_outputs) — cosine similarity per dim per step
        distances:    (n_records, n_outputs) — L2 distance per dim per step
        total_dist:   (n_records,) — total function-space distance
        pca_explained:(n_records, n_outputs) — PCA variance explained of diff matrix
        record_steps: (n_records,) — which training steps were recorded
    """
    device = X.device
    n_out = config["output_dim"]
    n_steps = config["n_train_steps"]

    model = make_model(config, seed).to(device)
    perturbed = clone_model_perturbed(
        model, config["perturbation_eps"], seed
    ).to(device)

    criterion = nn.MSELoss()

    # Pre-allocate recording arrays
    n_records = n_steps // record_every
    correlations = np.zeros((n_records, n_out))
    cosine_sims = np.zeros((n_records, n_out))
    distances = np.zeros((n_records, n_out))
    total_dist = np.zeros(n_records)
    pca_explained = np.zeros((n_records, n_out))
    record_steps = np.zeros(n_records, dtype=int)

    rec_idx = 0

    for t in range(n_steps):
        if t % record_every == 0 and rec_idx < n_records:
            with torch.no_grad():
                f1 = model(X).cpu().numpy()       # (n_samples, n_out)
                f2 = perturbed(X).cpu().numpy()    # (n_samples, n_out)

                # Total distance
                total_dist[rec_idx] = np.linalg.norm(f1 - f2)

                # Per-dimension analysis
                for d in range(n_out):
                    v1 = f1[:, d]
                    v2 = f2[:, d]

                    # Pearson correlation
                    if np.std(v1) > 1e-12 and np.std(v2) > 1e-12:
                        correlations[rec_idx, d] = np.corrcoef(v1, v2)[0, 1]
                    else:
                        correlations[rec_idx, d] = 1.0  # both constant = same

                    # Cosine similarity
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 1e-12 and n2 > 1e-12:
                        cosine_sims[rec_idx, d] = np.dot(v1, v2) / (n1 * n2)
                    else:
                        cosine_sims[rec_idx, d] = 1.0

                    # Per-dimension distance
                    distances[rec_idx, d] = np.linalg.norm(v1 - v2)

                # PCA of the difference matrix
                diff = f1 - f2  # (n_samples, n_out)
                if np.linalg.norm(diff) > 1e-15:
                    U, S, Vt = np.linalg.svd(diff, full_matrices=False)
                    var_explained = S ** 2
                    total_var = var_explained.sum()
                    if total_var > 1e-15:
                        pca_explained[rec_idx] = var_explained / total_var
                    else:
                        pca_explained[rec_idx] = np.zeros(n_out)

                record_steps[rec_idx] = t
                rec_idx += 1

        # Training step (identical for both)
        model.zero_grad()
        loss1 = criterion(model(X), y)
        loss1.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad

        perturbed.zero_grad()
        loss2 = criterion(perturbed(X), y)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                p -= lr * p.grad

    return {
        "correlations": correlations[:rec_idx],
        "cosine_sims": cosine_sims[:rec_idx],
        "distances": distances[:rec_idx],
        "total_dist": total_dist[:rec_idx],
        "pca_explained": pca_explained[:rec_idx],
        "record_steps": record_steps[:rec_idx],
    }


# ============================================================
# EXPERIMENT: Geometric Structure at a Single Learning Rate
# ============================================================

def run_single_lr(config, lr, n_seeds=5):
    """
    Run geometric structure analysis at a single learning rate,
    averaging across seeds.
    """
    print(f"\n  Geometric structure at η = {lr:.4f}, {n_seeds} seeds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    all_results = []
    for s in range(n_seeds):
        t0 = time.time()
        res = compute_geometric_structure(config, lr, seed=s, X=X, y=y)
        elapsed = time.time() - t0
        print(f"    Seed {s}: {elapsed:.1f}s")
        all_results.append(res)

    # Stack across seeds
    n_records = all_results[0]["correlations"].shape[0]
    n_out = config["output_dim"]

    corr_all = np.stack([r["correlations"] for r in all_results])  # (seeds, steps, dims)
    cos_all = np.stack([r["cosine_sims"] for r in all_results])
    dist_all = np.stack([r["distances"] for r in all_results])
    tdist_all = np.stack([r["total_dist"] for r in all_results])
    pca_all = np.stack([r["pca_explained"] for r in all_results])
    steps = all_results[0]["record_steps"]

    return {
        "lr": lr,
        "n_seeds": n_seeds,
        "record_steps": steps,
        "correlations": corr_all,       # (seeds, steps, dims)
        "cosine_sims": cos_all,
        "distances": dist_all,
        "total_dist": tdist_all,
        "pca_explained": pca_all,
    }


# ============================================================
# EXPERIMENT: Sweep Across Learning Rates
# ============================================================

def run_geometric_sweep(config, n_seeds=5, lrs=None):
    """
    Run geometric structure analysis across multiple learning rates
    to see how the balance of preservation/inversion/decorrelation shifts.
    """
    if lrs is None:
        lrs = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06,
               0.08, 0.10, 0.15, 0.20, 0.30]

    print("=" * 60)
    print(f"GEOMETRIC STRUCTURE SWEEP: {n_seeds} seeds × {len(lrs)} LRs")
    print("=" * 60)

    all_lr_results = []
    for lr in lrs:
        res = run_single_lr(config, lr, n_seeds)
        all_lr_results.append(res)

    # For each LR, compute the FINAL-state correlation signature
    # (average over last 20% of training, across seeds)
    summary = {
        "lrs": np.array(lrs),
        "n_seeds": n_seeds,
    }

    n_out = config["output_dim"]
    n_lrs = len(lrs)

    # Final correlation per dimension per LR (averaged over seeds)
    final_corr = np.zeros((n_lrs, n_out))
    final_cos = np.zeros((n_lrs, n_out))

    # Classification: what fraction of dims are preserved/inverted/chaotic
    frac_preserved = np.zeros(n_lrs)
    frac_inverted = np.zeros(n_lrs)
    frac_chaotic = np.zeros(n_lrs)

    for li, res in enumerate(all_lr_results):
        corr = res["correlations"]  # (seeds, steps, dims)
        n_steps = corr.shape[1]
        late_start = int(n_steps * 0.8)

        # Average correlations over late training and seeds
        late_corr = corr[:, late_start:, :].mean(axis=(0, 1))  # (dims,)
        final_corr[li] = late_corr

        cos = res["cosine_sims"]
        late_cos = cos[:, late_start:, :].mean(axis=(0, 1))
        final_cos[li] = late_cos

        # Classify each dimension
        for d in range(n_out):
            c = late_corr[d]
            if c > 0.5:
                frac_preserved[li] += 1
            elif c < -0.5:
                frac_inverted[li] += 1
            else:
                frac_chaotic[li] += 1

    frac_preserved /= n_out
    frac_inverted /= n_out
    frac_chaotic /= n_out

    summary["final_corr"] = final_corr          # (n_lrs, n_out)
    summary["final_cos"] = final_cos
    summary["frac_preserved"] = frac_preserved   # (n_lrs,)
    summary["frac_inverted"] = frac_inverted
    summary["frac_chaotic"] = frac_chaotic

    # Save full time series for selected LRs (for detailed plots)
    # Save all of them - they're not that big
    for li, res in enumerate(all_lr_results):
        for key in ["correlations", "cosine_sims", "distances",
                     "total_dist", "pca_explained", "record_steps"]:
            summary[f"lr{li}_{key}"] = res[key]

    os.makedirs("results", exist_ok=True)
    np.savez("results/geometric_sweep.npz", **summary)
    print(f"\n  Saved → results/geometric_sweep.npz")

    # Print summary table
    print(f"\n  {'η':>8s}  {'Preserved':>10s}  {'Inverted':>10s}  {'Chaotic':>10s}")
    print("  " + "-" * 44)
    for li, lr in enumerate(lrs):
        print(f"  {lr:8.4f}  {frac_preserved[li]:10.0%}  "
              f"{frac_inverted[li]:10.0%}  {frac_chaotic[li]:10.0%}")

    return summary


# ============================================================
# PLOTTING
# ============================================================

def plot_geometric_timeseries(npz_path="results/geometric_sweep.npz"):
    """
    Plot per-dimension correlations over training time for selected LRs.
    Shows how different dimensions preserve, invert, or decorrelate.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]

    # Pick representative LRs: one low, one near transition, one high
    indices = []
    targets = [0.01, 0.03, 0.08, 0.20]
    for target in targets:
        idx = np.argmin(np.abs(lrs - target))
        if idx not in indices:
            indices.append(idx)

    n_plots = len(indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    cmap = plt.cm.tab10

    for pi, li in enumerate(indices):
        ax = axes[pi]
        lr = lrs[li]

        key_corr = f"lr{li}_correlations"
        key_steps = f"lr{li}_record_steps"
        if key_corr not in d:
            continue

        corr = d[key_corr]      # (seeds, steps, dims)
        steps = d[key_steps]

        # Average across seeds
        mean_corr = corr.mean(axis=0)  # (steps, dims)
        n_out = mean_corr.shape[1]

        for dim in range(n_out):
            ax.plot(steps, mean_corr[:, dim], color=cmap(dim / 10),
                    lw=1.2, alpha=0.7, label=f"Dim {dim}" if pi == 0 else None)

        ax.axhline(1, color="k", lw=0.3, ls=":")
        ax.axhline(0, color="k", lw=0.5, ls="-")
        ax.axhline(-1, color="k", lw=0.3, ls=":")

        # Shade regions
        ax.axhspan(0.5, 1.05, alpha=0.05, color="blue")
        ax.axhspan(-1.05, -0.5, alpha=0.05, color="red")

        ax.set_xlabel("Training step")
        if pi == 0:
            ax.set_ylabel("Per-dimension correlation")
        ax.set_title(f"η = {lr:.3f}")
        ax.set_ylim(-1.1, 1.1)

    if n_plots > 0:
        axes[0].legend(fontsize=7, ncol=2, loc="lower left")

    plt.suptitle("Per-dimension correlation over training\n"
                 "(blue zone = preserved, red zone = inverted, "
                 "middle = decorrelated/chaotic)",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/geometric_timeseries.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/geometric_timeseries.pdf", bbox_inches="tight")
    print("  Saved → figures/geometric_timeseries.png + .pdf")
    plt.close()


def plot_geometric_summary(npz_path="results/geometric_sweep.npz"):
    """
    Plot the fraction of preserved/inverted/chaotic dimensions vs learning rate.
    This is the key figure: a stacked area chart showing the geometric
    composition of the transition.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 12})

    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]
    frac_p = d["frac_preserved"]
    frac_i = d["frac_inverted"]
    frac_c = d["frac_chaotic"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: stacked area chart
    ax1.fill_between(lrs, 0, frac_p, alpha=0.6,
                     color="#3B8BD4", label="Preserved (corr > 0.5)")
    ax1.fill_between(lrs, frac_p, frac_p + frac_i, alpha=0.6,
                     color="#D05538", label="Inverted (corr < −0.5)")
    ax1.fill_between(lrs, frac_p + frac_i, 1.0, alpha=0.6,
                     color="#888880", label="Decorrelated (|corr| < 0.5)")

    ax1.set_xlabel("Learning rate η")
    ax1.set_ylabel("Fraction of output dimensions")
    ax1.set_title("Geometric composition of the transition")
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(lrs[0], lrs[-1])

    # Right: per-dimension final correlation heatmap
    final_corr = d["final_corr"]  # (n_lrs, n_out)
    im = ax2.imshow(final_corr.T, aspect="auto", cmap="RdBu_r",
                    vmin=-1, vmax=1,
                    extent=[lrs[0], lrs[-1], final_corr.shape[1] - 0.5, -0.5])
    ax2.set_xlabel("Learning rate η")
    ax2.set_ylabel("Output dimension")
    ax2.set_title("Final correlation per dimension")
    plt.colorbar(im, ax=ax2, label="Pearson correlation", shrink=0.8)

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/geometric_summary.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/geometric_summary.pdf", bbox_inches="tight")
    print("  Saved → figures/geometric_summary.png + .pdf")
    plt.close()


def plot_pca_structure(npz_path="results/geometric_sweep.npz"):
    """
    Plot PCA variance explained of the difference matrix over training.
    Shows whether the divergence is concentrated in a few directions
    (structured) or spread evenly (random/chaotic).
    """
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]

    indices = []
    for target in [0.01, 0.03, 0.08, 0.20]:
        idx = np.argmin(np.abs(lrs - target))
        if idx not in indices:
            indices.append(idx)

    fig, axes = plt.subplots(1, len(indices), figsize=(4.5 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for pi, li in enumerate(indices):
        ax = axes[pi]
        lr = lrs[li]

        key_pca = f"lr{li}_pca_explained"
        key_steps = f"lr{li}_record_steps"
        if key_pca not in d:
            continue

        pca = d[key_pca]  # (seeds, steps, dims)
        steps = d[key_steps]

        # Average across seeds, take late training
        mean_pca = pca.mean(axis=0)  # (steps, dims)
        late = mean_pca[int(len(mean_pca) * 0.8):]
        avg_late = late.mean(axis=0)  # (dims,)

        ax.bar(range(len(avg_late)), avg_late, color="#5DCAA5", edgecolor="#0F6E56",
               lw=0.5)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance explained")
        ax.set_title(f"η = {lr:.3f}")

        # Add text: is divergence concentrated or spread?
        top1 = avg_late[0] if len(avg_late) > 0 else 0
        ax.text(0.95, 0.95, f"PC1: {top1:.0%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10)

    plt.suptitle("PCA of difference matrix\n"
                 "(concentrated = structured divergence, "
                 "spread = random/chaotic)",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/pca_structure.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/pca_structure.pdf", bbox_inches="tight")
    print("  Saved → figures/pca_structure.png + .pdf")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Geometric Structure of the Chaos Transition"
    )
    parser.add_argument("--run", action="store_true",
                        help="Run at a single learning rate")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep across learning rates")
    parser.add_argument("--plot-only", action="store_true",
                        help="Generate plots from existing results")

    parser.add_argument("--lr", type=float, default=0.03,
                        help="Learning rate for single-LR run (default: 0.03)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds (default: 5)")
    parser.add_argument("--train-steps", type=int, default=5000,
                        help="Training steps (default: 5000)")

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["n_train_steps"] = args.train_steps

    if args.plot_only:
        print("Generating plots from existing results...")
        if os.path.exists("results/geometric_sweep.npz"):
            plot_geometric_timeseries()
            plot_geometric_summary()
            plot_pca_structure()
        return

    if args.sweep:
        run_geometric_sweep(config, n_seeds=args.seeds)
        plot_geometric_timeseries()
        plot_geometric_summary()
        plot_pca_structure()
    elif args.run:
        res = run_single_lr(config, args.lr, n_seeds=args.seeds)
        os.makedirs("results", exist_ok=True)
        np.savez(f"results/geometric_lr{args.lr:.4f}.npz", **{
            k: v for k, v in res.items() if isinstance(v, np.ndarray)
        })
        print(f"  Saved → results/geometric_lr{args.lr:.4f}.npz")
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python geometric_structure.py --sweep --seeds 3   # fast test")
        print("  python geometric_structure.py --sweep --seeds 10  # full run")


if __name__ == "__main__":
    main()
