"""
Experiment L: Topological Data Analysis of CNN Training Trajectory
====================================================================

The CIFAR-10 CNN at 30% EoS shows D₂ = 3.64 — a multi-dimensional
attractor in function space. This experiment tests whether that 
attractor has TOROIDAL TOPOLOGY.

Torus signatures in persistent homology:
  - 2-torus (D≈2): H₁ = 2 persistent generators (two independent loops)
  - 3-torus (D≈3): H₁ = 3 persistent generators, H₂ = 3
  - Strange attractor: many short-lived H₁ features, no persistent H₂

The experiment:
  1. Train CNN at several fractions of EoS (5%, 15%, 20%, 30%, 40%, 60%, 90%)
  2. Record function-space trajectory (network outputs on eval set)
  3. PCA to reduce to manageable dimensions
  4. Compute persistent homology (H₀, H₁, H₂)
  5. Compare persistence diagrams across learning rates

Two TDA approaches:
  A) Point cloud TDA on the function-space trajectory
  B) Delay embedding TDA on the sharpness time series

Approach B is complementary: if sharpness oscillations are the coupled
modes generating the multi-dimensional dynamics, the delay-embedded
sharpness should reconstruct the same topology.

USAGE:
    python experiment_L_tda.py --seeds 3              # full run
    python experiment_L_tda.py --seeds 1 --quick      # fast test (fewer points)
    python experiment_L_tda.py --plot-only             # from saved results

REQUIREMENTS:
    pip install torch torchvision numpy matplotlib scipy
    pip install ripser          # preferred TDA library (fast C++ backend)
    # OR falls back to from-scratch implementation if ripser unavailable
"""

import argparse, os, time, warnings
import numpy as np
import torch
import torch.nn as nn
from scipy import stats, signal

# Try to import ripser; fall back to from-scratch if unavailable
try:
    from ripser import ripser
    HAS_RIPSER = True
    print("Using ripser for persistent homology (fast)")
except ImportError:
    HAS_RIPSER = False
    print("ripser not found — using from-scratch implementation (slower, H₁ only)")


# ============================================================
# CNN + DATA (identical to Phase 3 Experiment K)
# ============================================================

class SmallCNN(nn.Module):
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


def load_cifar10_subset(n_samples=2000, seed=42):
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
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


def compute_sharpness_cnn(model, X, y, criterion, n_iter=20):
    v = [torch.randn_like(p) for p in model.parameters()]
    v_norm = sum((vi**2).sum() for vi in v).sqrt()
    v = [vi / v_norm for vi in v]
    eigenvalue = 0.0
    for _ in range(n_iter):
        model.zero_grad()
        loss = criterion(model(X), y)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        Hv_terms = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(Hv_terms, model.parameters())
        eigenvalue = sum((hv * vi).sum().item() for hv, vi in zip(Hv, v))
        hv_norm = sum((hv**2).sum() for hv in Hv).sqrt().item()
        if hv_norm < 1e-12: break
        v = [hv.detach() / hv_norm for hv in Hv]
    return abs(eigenvalue)


# ============================================================
# FROM-SCRATCH PERSISTENT HOMOLOGY (fallback if no ripser)
# ============================================================

def persistent_homology_scratch(distance_matrix, max_dim=1):
    """
    Simplified Vietoris-Rips persistent homology.
    Computes H₀ (connected components) via union-find.
    Computes H₁ (loops) via boundary matrix reduction.
    Limited to small point clouds (<500 points).
    """
    n = len(distance_matrix)

    # All pairwise edges, sorted by distance
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((distance_matrix[i, j], i, j))
    edges.sort()

    # H₀: union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    h0_births = [0.0] * n
    h0_deaths = []
    h1_pairs = []

    for dist, i, j in edges:
        if find(i) != find(j):
            # Merging components — H₀ death
            h0_deaths.append(dist)
            union(i, j)
        else:
            # Closing a loop — potential H₁ birth
            # Simplified: record as H₁ birth at this distance
            h1_pairs.append(dist)

    # H₁: approximate persistence by looking for gaps in the loop-closing distances
    # This is a simplification — proper H₁ requires boundary matrix reduction
    h1_births_deaths = []
    if len(h1_pairs) > 0:
        # Group nearby loop-closings and find persistent ones
        h1_sorted = sorted(h1_pairs)
        # Simple heuristic: gaps between consecutive loop distances
        for k in range(len(h1_sorted) - 1):
            birth = h1_sorted[k]
            death = h1_sorted[k + 1]
            if death - birth > 0:
                h1_births_deaths.append((birth, death))

    return {
        "h0_deaths": h0_deaths,
        "h1_pairs": h1_births_deaths,
    }


# ============================================================
# TDA PIPELINE
# ============================================================

def compute_persistence(points, max_dim=2, max_points=400):
    """
    Compute persistent homology of a point cloud.
    Uses ripser if available, otherwise falls back to from-scratch.
    
    Returns dict with persistence diagrams for each dimension.
    """
    n = len(points)

    # Subsample if too many points (ripser is O(n³) in memory)
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, max_points, replace=False)
        points = points[idx]
        n = max_points

    if HAS_RIPSER:
        result = ripser(points, maxdim=max_dim)
        diagrams = {}
        for dim in range(max_dim + 1):
            dgm = result['dgms'][dim]
            # Filter out infinite death times
            finite = dgm[np.isfinite(dgm[:, 1])]
            diagrams[f'h{dim}'] = finite
        return diagrams
    else:
        # From-scratch: compute distance matrix, run simplified algorithm
        from scipy.spatial.distance import pdist, squareform
        dist_mat = squareform(pdist(points))
        result = persistent_homology_scratch(dist_mat, max_dim=1)

        diagrams = {}
        # H₀
        births = np.zeros(len(result['h0_deaths']))
        deaths = np.array(result['h0_deaths'])
        diagrams['h0'] = np.column_stack([births, deaths]) if len(deaths) > 0 else np.zeros((0, 2))

        # H₁ (approximate)
        if result['h1_pairs']:
            h1 = np.array(result['h1_pairs'])
            diagrams['h1'] = h1
        else:
            diagrams['h1'] = np.zeros((0, 2))

        diagrams['h2'] = np.zeros((0, 2))  # can't compute H₂ from scratch easily
        return diagrams


def persistence_summary(diagrams):
    """Extract key statistics from persistence diagrams."""
    summary = {}
    for dim_key in ['h0', 'h1', 'h2']:
        dgm = diagrams.get(dim_key, np.zeros((0, 2)))
        if len(dgm) > 0:
            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[lifetimes > 0]
            if len(lifetimes) > 0:
                sorted_lt = np.sort(lifetimes)[::-1]
                summary[dim_key] = {
                    'n_features': len(lifetimes),
                    'max_lifetime': float(sorted_lt[0]),
                    'top_3': sorted_lt[:3].tolist(),
                    'mean_lifetime': float(np.mean(lifetimes)),
                    'total_persistence': float(np.sum(lifetimes)),
                    # Gap ratio: lifetime[0] / lifetime[1] — large gap = clear topology
                    'gap_ratio': float(sorted_lt[0] / sorted_lt[1]) if len(sorted_lt) > 1 else float('inf'),
                }
            else:
                summary[dim_key] = {'n_features': 0}
        else:
            summary[dim_key] = {'n_features': 0}
    return summary


def delay_embed(x, dim, tau):
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError(f"Signal too short for dim={dim}, tau={tau}")
    emb = np.zeros((n, dim))
    for d in range(dim):
        emb[:, d] = x[d * tau: d * tau + n]
    return emb


def optimal_delay(x, max_lag=500):
    n = len(x)
    x_c = x - x.mean()
    acf = np.correlate(x_c, x_c, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-30)
    for i in range(1, min(max_lag, len(acf) - 1)):
        if acf[i] < acf[i-1] and acf[i] < acf[i+1]:
            return i
    return max_lag // 4


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_tda_experiment(n_seeds=3, n_steps=5000, n_samples=2000, quick=False):
    """
    Train CNN at multiple EoS fractions, record trajectories,
    compute persistent homology.
    """
    # LRs spanning the D₂ transition
    lr_fractions = [0.05, 0.15, 0.20, 0.30, 0.40, 0.60, 0.90]

    # Parameters
    n_eval = 50 if quick else 100
    output_every = 10 if quick else 5
    max_tda_points = 200 if quick else 400
    pca_dim = 6 if quick else 10  # reduce to this many dims before TDA
    sharpness_every = 50

    print("=" * 60)
    print(f"EXPERIMENT L: TOPOLOGICAL DATA ANALYSIS OF CNN TRAJECTORY")
    print(f"  {n_seeds} seeds × {len(lr_fractions)} LRs, {n_steps} steps")
    print(f"  n_eval={n_eval}, output_every={output_every}")
    print(f"  PCA to {pca_dim}D before TDA, max {max_tda_points} points")
    print(f"  TDA backend: {'ripser' if HAS_RIPSER else 'from-scratch (H₀+H₁ only)'}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load data
    print("  Loading CIFAR-10...")
    X, y = load_cifar10_subset(n_samples=n_samples, seed=42)
    X, y = X.to(device), y.to(device)

    # Find EoS threshold
    print("  Finding EoS threshold...")
    torch.manual_seed(0)
    ref_model = SmallCNN().to(device)
    criterion = nn.MSELoss()
    for t in range(1000):
        ref_model.zero_grad()
        loss = criterion(ref_model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in ref_model.parameters():
                if p.grad is not None:
                    p -= 0.01 * p.grad
    lam_max = compute_sharpness_cnn(ref_model, X, y, criterion, n_iter=50)
    lr_eos = 2.0 / lam_max
    print(f"  λ_max ≈ {lam_max:.4f}, EoS ≈ {lr_eos:.4f}")
    del ref_model

    test_lrs = [frac * lr_eos for frac in lr_fractions]

    # Eval subset
    torch.manual_seed(0)
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    total_runs = len(lr_fractions) * n_seeds
    done = 0
    t0 = time.time()

    all_results = {
        "lam_max": lam_max, "lr_eos": lr_eos,
        "lr_fractions": lr_fractions, "test_lrs": [float(lr) for lr in test_lrs],
        "n_eval": n_eval, "output_every": output_every, "pca_dim": pca_dim,
    }

    for li, (frac, lr) in enumerate(zip(lr_fractions, test_lrs)):
        lr_results = {
            "persistence_summaries": [],
            "sharpness_persistence": [],
            "corr_dims": [],
            "pc_variances": [],
        }

        for s in range(n_seeds):
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total_runs - done) if done > 1 else 0
            print(f"\n  [{done}/{total_runs}] {frac:.0%} EoS, η={lr:.4f}, seed={s} "
                  f"ETA: {eta:.0f}s")

            # --- Train and record trajectory ---
            torch.manual_seed(s)
            model = SmallCNN().to(device)
            outputs_rec = []
            sharpness_rec = []

            for t in range(n_steps):
                if t % output_every == 0:
                    with torch.no_grad():
                        outputs_rec.append(model(X_eval).cpu().numpy())

                model.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()

                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p -= lr * p.grad

                if t % sharpness_every == 0 and t > 0:
                    sharp = compute_sharpness_cnn(model, X, y, criterion, n_iter=15)
                    sharpness_rec.append(sharp)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # --- Process trajectory ---
            outputs = np.array(outputs_rec)  # (n_record, n_eval, 10)
            start = len(outputs) // 5  # skip transient
            traj = outputs[start:]
            traj_flat = traj.reshape(len(traj), -1)  # (n_points, n_eval * 10)

            print(f"    Trajectory: {traj_flat.shape[0]} points × {traj_flat.shape[1]} dims")

            # PCA reduction
            centered = traj_flat - traj_flat.mean(axis=0)
            try:
                U, sv, Vt = np.linalg.svd(centered, full_matrices=False)
                var_exp = (sv**2) / (sv**2).sum()
                traj_pca = centered @ Vt[:pca_dim].T  # project to pca_dim dimensions
                lr_results["pc_variances"].append(var_exp[:20].tolist())
                print(f"    PCA: PC1={var_exp[0]*100:.1f}%, "
                      f"top-{pca_dim} cumulative={var_exp[:pca_dim].sum()*100:.1f}%")
            except Exception as e:
                print(f"    PCA failed: {e}")
                continue

            # --- TDA on function-space trajectory (Approach A) ---
            print(f"    Computing persistence on {pca_dim}D trajectory...")
            try:
                max_dim = 2 if HAS_RIPSER else 1
                diagrams = compute_persistence(traj_pca, max_dim=max_dim,
                                                max_points=max_tda_points)
                summary = persistence_summary(diagrams)
                lr_results["persistence_summaries"].append(summary)

                # Report
                for dk in ['h0', 'h1', 'h2']:
                    s_data = summary.get(dk, {})
                    n_feat = s_data.get('n_features', 0)
                    if n_feat > 0:
                        top3 = s_data.get('top_3', [])
                        gap = s_data.get('gap_ratio', 0)
                        print(f"    {dk.upper()}: {n_feat} features, "
                              f"top lifetimes={[f'{x:.4f}' for x in top3]}, "
                              f"gap ratio={gap:.2f}")
                    else:
                        print(f"    {dk.upper()}: no features")

            except Exception as e:
                print(f"    TDA failed: {e}")
                lr_results["persistence_summaries"].append({})

            # --- TDA on delay-embedded sharpness (Approach B) ---
            if len(sharpness_rec) > 20:
                sh = np.array(sharpness_rec)
                sh_start = len(sh) // 5
                sh_signal = sh[sh_start:]
                sh_detrended = signal.detrend(sh_signal, type='linear')

                tau = optimal_delay(sh_detrended, max_lag=len(sh_detrended) // 3)
                embed_dim = min(7, (len(sh_detrended) - 1) // tau)
                if embed_dim >= 3:
                    try:
                        sh_embedded = delay_embed(sh_detrended, embed_dim, tau)
                        print(f"    Sharpness embedding: {sh_embedded.shape}, τ={tau}")

                        sh_diagrams = compute_persistence(sh_embedded, max_dim=min(2, embed_dim-1),
                                                           max_points=max_tda_points)
                        sh_summary = persistence_summary(sh_diagrams)
                        lr_results["sharpness_persistence"].append(sh_summary)

                        for dk in ['h0', 'h1']:
                            s_data = sh_summary.get(dk, {})
                            n_feat = s_data.get('n_features', 0)
                            if n_feat > 0:
                                print(f"    Sharpness {dk.upper()}: {n_feat} features, "
                                      f"top={s_data.get('top_3', [])[:2]}")
                    except Exception as e:
                        print(f"    Sharpness TDA failed: {e}")
                        lr_results["sharpness_persistence"].append({})
                else:
                    print(f"    Sharpness series too short for embedding (n={len(sh_signal)}, τ={tau})")
                    lr_results["sharpness_persistence"].append({})
            else:
                lr_results["sharpness_persistence"].append({})

        all_results[f"lr_{li}"] = lr_results

    # ── Summary ──
    print("\n" + "=" * 60)
    print("EXPERIMENT L SUMMARY")
    print("=" * 60)
    print(f"\n{'frac_EoS':>8s}  {'H₁ feat':>8s}  {'H₁ top life':>12s}  {'H₁ gap':>8s}  {'H₂ feat':>8s}  {'torus?':>8s}")
    print("-" * 60)
    for li, frac in enumerate(lr_fractions):
        r = all_results[f"lr_{li}"]
        if r["persistence_summaries"]:
            # Average across seeds
            h1_feats = []
            h1_tops = []
            h1_gaps = []
            h2_feats = []
            for ps in r["persistence_summaries"]:
                if ps:
                    h1 = ps.get('h1', {})
                    h1_feats.append(h1.get('n_features', 0))
                    h1_tops.append(h1.get('max_lifetime', 0))
                    h1_gaps.append(h1.get('gap_ratio', 0))
                    h2 = ps.get('h2', {})
                    h2_feats.append(h2.get('n_features', 0))

            mean_h1_feat = np.mean(h1_feats) if h1_feats else 0
            mean_h1_top = np.mean(h1_tops) if h1_tops else 0
            mean_h1_gap = np.mean(h1_gaps) if h1_gaps else 0
            mean_h2_feat = np.mean(h2_feats) if h2_feats else 0

            # Torus diagnostic: ≥2 persistent H₁ features with gap ratio > 3
            torus_signal = "YES" if mean_h1_feat >= 2 and mean_h1_gap > 3 else "maybe" if mean_h1_feat >= 2 else "no"

            print(f"  {frac:6.0%}    {mean_h1_feat:8.1f}  {mean_h1_top:12.4f}  {mean_h1_gap:8.1f}  {mean_h2_feat:8.1f}  {torus_signal:>8s}")
        else:
            print(f"  {frac:6.0%}    —")

    # Save
    import json
    def _ser(obj):
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)): return float(obj)
        elif isinstance(obj, (np.int64, np.int32)): return int(obj)
        elif isinstance(obj, list): return [_ser(v) for v in obj]
        elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)): return None
        return obj

    os.makedirs("results", exist_ok=True)
    with open("results/tda_cifar10.json", "w") as f:
        json.dump(_ser(all_results), f, indent=2)
    print(f"\n  Saved → results/tda_cifar10.json")

    return all_results


# ============================================================
# PLOTTING
# ============================================================

def plot_tda(json_path="results/tda_cifar10.json"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json

    with open(json_path) as f:
        results = json.load(f)

    lr_fractions = results["lr_fractions"]
    n_lrs = len(lr_fractions)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect per-LR averages
    h1_counts, h1_tops, h1_gaps, h2_counts = [], [], [], []
    for li in range(n_lrs):
        r = results[f"lr_{li}"]
        ps_list = r["persistence_summaries"]
        h1c, h1t, h1g, h2c = [], [], [], []
        for ps in ps_list:
            if ps:
                h1 = ps.get('h1', {})
                h1c.append(h1.get('n_features', 0))
                h1t.append(h1.get('max_lifetime', 0))
                h1g.append(h1.get('gap_ratio', 0) if h1.get('gap_ratio') is not None else 0)
                h2 = ps.get('h2', {})
                h2c.append(h2.get('n_features', 0))
        h1_counts.append(np.mean(h1c) if h1c else 0)
        h1_tops.append(np.mean(h1t) if h1t else 0)
        h1_gaps.append(np.mean(h1g) if h1g else 0)
        h2_counts.append(np.mean(h2c) if h2c else 0)

    # Top-left: H₁ feature count vs EoS fraction
    ax = axes[0, 0]
    ax.plot(lr_fractions, h1_counts, 'ko-', ms=8, lw=2)
    ax.axhline(2, color='red', ls='--', lw=1, label='2 = torus')
    ax.axhline(3, color='orange', ls='--', lw=1, label='3 = 3-torus')
    ax.set_xlabel('Fraction of EoS')
    ax.set_ylabel('Number of persistent H₁ features')
    ax.set_title('H₁ Count: How many independent loops?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: H₁ maximum lifetime vs EoS fraction
    ax = axes[0, 1]
    ax.plot(lr_fractions, h1_tops, 'ko-', ms=8, lw=2)
    ax.set_xlabel('Fraction of EoS')
    ax.set_ylabel('Max H₁ lifetime')
    ax.set_title('H₁ Persistence: How robust are the loops?')
    ax.grid(True, alpha=0.3)

    # Bottom-left: H₁ gap ratio
    ax = axes[1, 0]
    ax.plot(lr_fractions, h1_gaps, 'ko-', ms=8, lw=2)
    ax.axhline(3, color='red', ls='--', lw=1, label='Gap > 3 = clear topology')
    ax.set_xlabel('Fraction of EoS')
    ax.set_ylabel('H₁ gap ratio (top / 2nd)')
    ax.set_title('Topological Signal Clarity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: H₂ feature count
    ax = axes[1, 1]
    ax.plot(lr_fractions, h2_counts, 'ko-', ms=8, lw=2)
    ax.axhline(1, color='red', ls='--', lw=1, label='1 = torus void')
    ax.set_xlabel('Fraction of EoS')
    ax.set_ylabel('Number of persistent H₂ features')
    ax.set_title('H₂ Count: Is there an enclosed void?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Persistent Homology of CNN Training Trajectory\n'
                 f'CIFAR-10 CNN ({results.get("lam_max", "?"):.1f} λ_max, '
                 f'EoS = {results.get("lr_eos", "?"):.3f})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/tda_cifar10.png", dpi=200, bbox_inches="tight")
    print(f"  Saved → figures/tda_cifar10.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment L: TDA of CNN Training")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--quick", action="store_true", help="Fewer points, faster run")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_tda()
        return

    run_tda_experiment(n_seeds=args.seeds, n_steps=args.steps, quick=args.quick)
    plot_tda()
    print("\nDone.")


if __name__ == "__main__":
    main()
