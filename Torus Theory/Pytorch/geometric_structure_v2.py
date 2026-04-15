"""
Geometric Structure of the Chaos Transition — v2
=================================================

WHAT CHANGED FROM v1:
The first version used ε=1e-8 perturbations. The divergence never grew
large enough relative to the signal for correlations to move off +1.0.

This version uses TWO complementary approaches:

APPROACH 1: DIFFERENT SEEDS (large separation)
  Train N networks from completely different random initializations.
  Compare all pairs. At low learning rates, they should all converge to
  the same function (correlation ~1). At high learning rates, some pairs
  may find mirrored solutions (correlation ~-1), orthogonal solutions
  (correlation ~0), or the same solution by different paths (+1).

  This directly tests: does the loss landscape have mirror-image basins
  that become accessible at higher learning rates?

APPROACH 2: LYAPUNOV VECTOR DIRECTION TRACKING (with renormalization)
  Train the original + perturbed pair, but every K steps RENORMALIZE the
  perturbation back to a fixed magnitude. Before renormalizing, record
  the DIRECTION of the difference vector. Track how that direction
  rotates over training.

  If the direction flips (rotates by ~180°) in certain output dimensions,
  that's the inversion signature. If it wanders randomly, that's chaos.
  If it stays fixed, that's structured divergence along a single mode.

USAGE:
    python geometric_structure_v2.py --seeds-compare --seeds 10
    python geometric_structure_v2.py --lyapunov-vectors --seeds 5
    python geometric_structure_v2.py --all --seeds 10
    python geometric_structure_v2.py --plot-only

OUTPUTS:
    results/seeds_comparison.npz
    results/lyapunov_vectors.npz
    figures/seeds_comparison_*.png
    figures/lyapunov_vectors_*.png
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations

# ---- Model & data (standalone) ----

DEFAULT_CONFIG = {
    "input_dim": 220, "hidden_dim": 50, "output_dim": 10,
    "activation": "tanh", "n_samples": 2000, "n_classes": 10,
    "n_random_features": 200, "n_quadratic_features": 20,
    "data_seed": 42, "n_train_steps": 5000, "loss_fn": "mse",
    "perturbation_eps": 1e-8, "sharpness_iters": 100,
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

def train_model(model, X, y, lr, n_steps):
    """Train a model and return it. Also returns loss history."""
    criterion = nn.MSELoss()
    losses = []
    for t in range(n_steps):
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
        if t % 100 == 0:
            losses.append(loss.item())
    return model, losses


# ============================================================
# APPROACH 1: COMPARE DIFFERENT SEEDS
# ============================================================

def run_seeds_comparison(config, n_seeds=10, lrs=None):
    """
    Train n_seeds networks at each learning rate, then compute
    all pairwise output correlations. This reveals whether different
    basins produce mirrored, orthogonal, or identical functions.
    """
    if lrs is None:
        lrs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    print("=" * 60)
    print(f"SEEDS COMPARISON: {n_seeds} seeds × {len(lrs)} LRs")
    print(f"  Comparing all {n_seeds*(n_seeds-1)//2} pairs per LR")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    n_out = config["output_dim"]
    n_pairs = n_seeds * (n_seeds - 1) // 2

    # Store results
    # Per-dimension correlations for all pairs at each LR
    all_pair_corr = np.zeros((len(lrs), n_pairs, n_out))
    # Per-dimension cosine similarity
    all_pair_cos = np.zeros((len(lrs), n_pairs, n_out))
    # Overall output correlation (flattened)
    all_pair_overall = np.zeros((len(lrs), n_pairs))
    # Final losses
    all_losses = np.zeros((len(lrs), n_seeds))

    total = len(lrs) * n_seeds
    done = 0
    t0 = time.time()

    for li, lr in enumerate(lrs):
        # Train all seeds
        outputs = []  # list of (n_samples, n_out) arrays
        for s in range(n_seeds):
            model = make_model(config, seed=s).to(device)
            model, losses = train_model(model, X, y, lr, config["n_train_steps"])
            with torch.no_grad():
                out = model(X).cpu().numpy()
            outputs.append(out)
            all_losses[li, s] = losses[-1] if losses else float("nan")
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] lr={lr:.3f} seed={s} "
                  f"loss={all_losses[li,s]:.6f} ETA: {eta:.0f}s")

        # Compare all pairs
        pair_idx = 0
        for si, sj in combinations(range(n_seeds), 2):
            f1 = outputs[si]  # (n_samples, n_out)
            f2 = outputs[sj]

            # Per-dimension correlation
            for d in range(n_out):
                v1, v2 = f1[:, d], f2[:, d]
                s1, s2 = np.std(v1), np.std(v2)
                if s1 > 1e-12 and s2 > 1e-12:
                    all_pair_corr[li, pair_idx, d] = np.corrcoef(v1, v2)[0, 1]
                else:
                    all_pair_corr[li, pair_idx, d] = 1.0

                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-12 and n2 > 1e-12:
                    all_pair_cos[li, pair_idx, d] = np.dot(v1, v2) / (n1 * n2)
                else:
                    all_pair_cos[li, pair_idx, d] = 1.0

            # Overall correlation (flatten outputs)
            flat1, flat2 = f1.ravel(), f2.ravel()
            if np.std(flat1) > 1e-12 and np.std(flat2) > 1e-12:
                all_pair_overall[li, pair_idx] = np.corrcoef(flat1, flat2)[0, 1]
            else:
                all_pair_overall[li, pair_idx] = 1.0

            pair_idx += 1

    # Classify
    frac_preserved = np.zeros(len(lrs))
    frac_inverted = np.zeros(len(lrs))
    frac_chaotic = np.zeros(len(lrs))

    for li in range(len(lrs)):
        # Average correlation across pairs, per dimension
        mean_corr = all_pair_corr[li].mean(axis=0)  # (n_out,)
        for d in range(n_out):
            c = mean_corr[d]
            if c > 0.5:
                frac_preserved[li] += 1
            elif c < -0.5:
                frac_inverted[li] += 1
            else:
                frac_chaotic[li] += 1
        frac_preserved[li] /= n_out
        frac_inverted[li] /= n_out
        frac_chaotic[li] /= n_out

    # Print summary
    print(f"\n  {'η':>8s}  {'Preserved':>10s}  {'Inverted':>10s}  "
          f"{'Decorrel':>10s}  {'Overall r':>10s}")
    print("  " + "-" * 54)
    for li, lr in enumerate(lrs):
        overall = all_pair_overall[li].mean()
        print(f"  {lr:8.4f}  {frac_preserved[li]:10.0%}  "
              f"{frac_inverted[li]:10.0%}  {frac_chaotic[li]:10.0%}  "
              f"{overall:10.4f}")

    results = {
        "lrs": np.array(lrs),
        "n_seeds": n_seeds,
        "all_pair_corr": all_pair_corr,       # (n_lrs, n_pairs, n_out)
        "all_pair_cos": all_pair_cos,
        "all_pair_overall": all_pair_overall,  # (n_lrs, n_pairs)
        "all_losses": all_losses,
        "frac_preserved": frac_preserved,
        "frac_inverted": frac_inverted,
        "frac_chaotic": frac_chaotic,
    }

    os.makedirs("results", exist_ok=True)
    np.savez("results/seeds_comparison.npz", **results)
    print(f"\n  Saved → results/seeds_comparison.npz")
    return results


# ============================================================
# APPROACH 2: LYAPUNOV VECTOR DIRECTION TRACKING
# ============================================================

def run_lyapunov_vectors(config, n_seeds=5, lrs=None,
                         renorm_every=50):
    """
    Track the DIRECTION of the Lyapunov vector (the direction in which
    two nearby networks diverge) over training, with periodic
    renormalization of the perturbation magnitude.

    At each renormalization step:
    1. Record the direction of Δf = f₁(X) - f₂(X) in output space
    2. Record the magnitude ‖Δf‖
    3. Rescale the perturbed network's weights back to ε distance
       from the original

    This reveals:
    - Whether the divergence direction ROTATES over training
    - Whether certain output dimensions FLIP SIGN (inversion)
    - Whether the direction is stable (structured) or wandering (chaotic)
    """
    if lrs is None:
        lrs = [0.01, 0.02, 0.03, 0.05, 0.08, 0.15, 0.25]

    print("=" * 60)
    print(f"LYAPUNOV VECTOR TRACKING: {n_seeds} seeds × {len(lrs)} LRs")
    print(f"  Renormalization every {renorm_every} steps")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)

    n_steps = config["n_train_steps"]
    n_out = config["output_dim"]
    n_renorms = n_steps // renorm_every
    eps = config["perturbation_eps"]

    # For each LR and seed, store:
    # - direction vectors at each renormalization (n_renorms, n_out)
    # - growth factors (n_renorms,)
    # - consecutive direction cosine similarity (n_renorms - 1,)

    all_results = {}

    total = len(lrs) * n_seeds
    done = 0
    t0 = time.time()

    for li, lr in enumerate(lrs):
        lr_directions = []      # (seeds, n_renorms, n_out)
        lr_growth = []          # (seeds, n_renorms)
        lr_dir_stability = []   # (seeds, n_renorms-1) cosine sim of consecutive dirs
        lr_dim_flips = []       # (seeds, n_renorms, n_out) sign of each dim

        for s in range(n_seeds):
            model = make_model(config, seed=s).to(device)

            # Create perturbed copy
            torch.manual_seed(s + 999999)
            perturbed = MLP(config["input_dim"], config["hidden_dim"],
                           config["output_dim"], config["activation"]).to(device)
            perturbed.load_state_dict(model.state_dict())

            # Apply initial perturbation
            direction = []
            for p in perturbed.parameters():
                d = torch.randn_like(p)
                direction.append(d)
            dir_norm = sum((d**2).sum() for d in direction).sqrt()
            for p, d in zip(perturbed.parameters(), direction):
                p.data += eps * d / dir_norm

            criterion = nn.MSELoss()
            directions = np.zeros((n_renorms, n_out))
            growth_factors = np.zeros(n_renorms)
            dim_signs = np.zeros((n_renorms, n_out))

            renorm_idx = 0

            for t in range(n_steps):
                if t % renorm_every == 0 and renorm_idx < n_renorms:
                    with torch.no_grad():
                        f1 = model(X)        # (n_samples, n_out)
                        f2 = perturbed(X)
                        diff = (f2 - f1).cpu().numpy()  # (n_samples, n_out)

                        # Per-dimension: mean signed difference
                        # (captures direction, not just magnitude)
                        dim_means = diff.mean(axis=0)  # (n_out,)
                        dim_norm = np.linalg.norm(dim_means)

                        if dim_norm > 1e-15:
                            directions[renorm_idx] = dim_means / dim_norm
                        else:
                            directions[renorm_idx] = 0

                        dim_signs[renorm_idx] = np.sign(dim_means)

                        # Total growth since last renormalization
                        total_diff = np.linalg.norm(diff)
                        growth_factors[renorm_idx] = total_diff

                        # RENORMALIZE: rescale perturbed weights back to ε distance
                        param_diff = []
                        for p1, p2 in zip(model.parameters(),
                                          perturbed.parameters()):
                            param_diff.append(p2.data - p1.data)

                        param_dist = sum((d**2).sum()
                                         for d in param_diff).sqrt().item()

                        if param_dist > 1e-15:
                            scale = eps / param_dist
                            for p1, p2, d in zip(model.parameters(),
                                                  perturbed.parameters(),
                                                  param_diff):
                                p2.data = p1.data + d * scale

                    renorm_idx += 1

                # Training step for both
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

            # Compute direction stability (cosine sim between consecutive directions)
            dir_stab = np.zeros(n_renorms - 1)
            for i in range(n_renorms - 1):
                d1 = directions[i]
                d2 = directions[i + 1]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-12 and n2 > 1e-12:
                    dir_stab[i] = np.dot(d1, d2) / (n1 * n2)

            lr_directions.append(directions)
            lr_growth.append(growth_factors)
            lr_dir_stability.append(dir_stab)
            lr_dim_flips.append(dim_signs)

            done += 1
            elapsed = time.time() - t0
            eta_t = elapsed / done * (total - done)
            print(f"  [{done}/{total}] lr={lr:.3f} seed={s} "
                  f"mean_dir_stability={dir_stab.mean():.4f} ETA: {eta_t:.0f}s")

        all_results[f"lr{li}_directions"] = np.array(lr_directions)
        all_results[f"lr{li}_growth"] = np.array(lr_growth)
        all_results[f"lr{li}_dir_stability"] = np.array(lr_dir_stability)
        all_results[f"lr{li}_dim_signs"] = np.array(lr_dim_flips)

    all_results["lrs"] = np.array(lrs)
    all_results["n_seeds"] = n_seeds
    all_results["renorm_every"] = renorm_every

    os.makedirs("results", exist_ok=True)
    np.savez("results/lyapunov_vectors.npz", **all_results)
    print(f"\n  Saved → results/lyapunov_vectors.npz")
    return all_results


# ============================================================
# PLOTTING
# ============================================================

def plot_seeds_comparison(npz_path="results/seeds_comparison.npz"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]
    pair_corr = d["all_pair_corr"]      # (n_lrs, n_pairs, n_out)
    pair_overall = d["all_pair_overall"]  # (n_lrs, n_pairs)
    frac_p = d["frac_preserved"]
    frac_i = d["frac_inverted"]
    frac_c = d["frac_chaotic"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Top-left: stacked area (preserved / inverted / decorrelated)
    ax = axes[0, 0]
    ax.fill_between(lrs, 0, frac_p, alpha=0.6, color="#3B8BD4",
                    label="Preserved (r > 0.5)")
    ax.fill_between(lrs, frac_p, frac_p + frac_i, alpha=0.6,
                    color="#D05538", label="Inverted (r < −0.5)")
    ax.fill_between(lrs, frac_p + frac_i, 1.0, alpha=0.6,
                    color="#888880", label="Decorrelated (|r| < 0.5)")
    ax.set_xlabel("Learning rate η")
    ax.set_ylabel("Fraction of dimensions")
    ax.set_title("Geometric composition (different seeds)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    # Top-right: heatmap of mean per-dim correlation
    ax = axes[0, 1]
    mean_corr = pair_corr.mean(axis=1)  # (n_lrs, n_out)
    im = ax.imshow(mean_corr.T, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1,
                   extent=[lrs[0], lrs[-1], mean_corr.shape[1]-0.5, -0.5])
    ax.set_xlabel("Learning rate η")
    ax.set_ylabel("Output dimension")
    ax.set_title("Mean pairwise correlation per dimension")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

    # Bottom-left: overall pairwise correlation distribution
    ax = axes[1, 0]
    positions = np.arange(len(lrs))
    bp_data = [pair_overall[li] for li in range(len(lrs))]
    bp = ax.boxplot(bp_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#5DCAA5")
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{lr:.3f}" for lr in lrs], rotation=45, fontsize=8)
    ax.set_xlabel("Learning rate η")
    ax.set_ylabel("Overall pairwise correlation")
    ax.set_title("Distribution of pairwise correlations")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(-1, color="k", lw=0.3, ls=":")

    # Bottom-right: per-dimension scatter for a few LRs
    ax = axes[1, 1]
    n_out = pair_corr.shape[2]
    show_lrs = [0, len(lrs)//3, 2*len(lrs)//3, len(lrs)-1]
    colors = ["#3B8BD4", "#5DCAA5", "#EF9F27", "#D05538"]
    for pi, li in enumerate(show_lrs):
        if li >= len(lrs):
            continue
        mean_c = pair_corr[li].mean(axis=0)
        ax.scatter(np.arange(n_out) + pi*0.15, mean_c, s=40,
                   color=colors[pi], label=f"η={lrs[li]:.3f}",
                   edgecolors="white", linewidths=0.5, zorder=3)
    ax.axhline(0.5, color="blue", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(-0.5, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Output dimension")
    ax.set_ylabel("Mean pairwise correlation")
    ax.set_title("Per-dimension correlations at selected LRs")
    ax.legend(fontsize=9)
    ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/seeds_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig("figures/seeds_comparison.pdf", bbox_inches="tight")
    print("  Saved → figures/seeds_comparison.png")
    plt.close()


def plot_lyapunov_vectors(npz_path="results/lyapunov_vectors.npz"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]
    renorm_every = int(d["renorm_every"])

    # Pick representative LRs
    indices = []
    for target in [0.01, 0.03, 0.08, 0.20]:
        idx = np.argmin(np.abs(lrs - target))
        if idx not in indices:
            indices.append(idx)

    n_plots = len(indices)

    # Figure 1: Direction stability over training
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    if n_plots == 1: axes = [axes]

    for pi, li in enumerate(indices):
        ax = axes[pi]
        lr = lrs[li]
        key = f"lr{li}_dir_stability"
        if key not in d: continue
        stab = d[key]  # (seeds, n_renorms-1)
        steps = np.arange(stab.shape[1]) * renorm_every

        for s in range(stab.shape[0]):
            ax.plot(steps, stab[s], alpha=0.3, lw=0.8, color="#3B8BD4")
        ax.plot(steps, stab.mean(axis=0), color="#0C447C", lw=2,
                label="Mean")
        ax.axhline(1, color="k", lw=0.3, ls=":")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(-1, color="k", lw=0.3, ls=":")
        ax.set_xlabel("Training step")
        if pi == 0: ax.set_ylabel("Direction cosine similarity\n(consecutive steps)")
        ax.set_title(f"η = {lr:.3f}")
        ax.set_ylim(-1.1, 1.1)

    plt.suptitle("Lyapunov vector direction stability\n"
                 "(+1 = same direction, −1 = flipped, 0 = orthogonal)",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/lyapunov_vectors_stability.png",
                dpi=200, bbox_inches="tight")
    print("  Saved → figures/lyapunov_vectors_stability.png")
    plt.close()

    # Figure 2: Per-dimension sign tracking
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    if n_plots == 1: axes = [axes]

    for pi, li in enumerate(indices):
        ax = axes[pi]
        lr = lrs[li]
        key = f"lr{li}_dim_signs"
        if key not in d: continue
        signs = d[key]  # (seeds, n_renorms, n_out)
        # Average sign across seeds (gives a value between -1 and +1)
        mean_signs = signs.mean(axis=0)  # (n_renorms, n_out)
        steps = np.arange(mean_signs.shape[0]) * renorm_every

        im = ax.imshow(mean_signs.T, aspect="auto", cmap="RdBu_r",
                       vmin=-1, vmax=1,
                       extent=[0, steps[-1], mean_signs.shape[1]-0.5, -0.5])
        ax.set_xlabel("Training step")
        if pi == 0: ax.set_ylabel("Output dimension")
        ax.set_title(f"η = {lr:.3f}")

    plt.suptitle("Per-dimension sign of divergence direction\n"
                 "(blue = positive, red = negative, white = mixed across seeds)",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    plt.savefig("figures/lyapunov_vectors_signs.png",
                dpi=200, bbox_inches="tight")
    print("  Saved → figures/lyapunov_vectors_signs.png")
    plt.close()

    # Figure 3: Growth factor over training
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    if n_plots == 1: axes = [axes]

    for pi, li in enumerate(indices):
        ax = axes[pi]
        lr = lrs[li]
        key = f"lr{li}_growth"
        if key not in d: continue
        growth = d[key]  # (seeds, n_renorms)
        steps = np.arange(growth.shape[1]) * renorm_every

        for s in range(growth.shape[0]):
            ax.semilogy(steps, growth[s] + 1e-15, alpha=0.3, lw=0.8,
                        color="#EF9F27")
        ax.semilogy(steps, growth.mean(axis=0) + 1e-15,
                    color="#854F0B", lw=2, label="Mean")
        ax.set_xlabel("Training step")
        if pi == 0: ax.set_ylabel("Growth factor (per renorm interval)")
        ax.set_title(f"η = {lr:.3f}")

    plt.suptitle("Perturbation growth between renormalizations\n"
                 "(increasing = chaos, decreasing = stability)",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    plt.savefig("figures/lyapunov_vectors_growth.png",
                dpi=200, bbox_inches="tight")
    print("  Saved → figures/lyapunov_vectors_growth.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Geometric Structure of Chaos Transition — v2"
    )
    parser.add_argument("--seeds-compare", action="store_true",
                        help="Compare networks from different seeds")
    parser.add_argument("--lyapunov-vectors", action="store_true",
                        help="Track Lyapunov vector directions")
    parser.add_argument("--all", action="store_true",
                        help="Run both experiments")
    parser.add_argument("--plot-only", action="store_true",
                        help="Generate plots from existing results")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=5000)

    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)
    config["n_train_steps"] = args.train_steps

    if args.plot_only:
        if os.path.exists("results/seeds_comparison.npz"):
            plot_seeds_comparison()
        if os.path.exists("results/lyapunov_vectors.npz"):
            plot_lyapunov_vectors()
        return

    if not (args.seeds_compare or args.lyapunov_vectors or args.all):
        parser.print_help()
        print("\nQuick start:")
        print("  python geometric_structure_v2.py --seeds-compare --seeds 5")
        print("  python geometric_structure_v2.py --all --seeds 10")
        return

    if args.all or args.seeds_compare:
        run_seeds_comparison(config, n_seeds=args.seeds)
        plot_seeds_comparison()

    if args.all or args.lyapunov_vectors:
        run_lyapunov_vectors(config, n_seeds=min(args.seeds, 5))
        plot_lyapunov_vectors()

    print("\nDone. Check results/ and figures/")


if __name__ == "__main__":
    main()
