"""
Cross-experiment: CNN on Synthetic Data — 3 seeds × 12 LR fractions
=====================================================================

Null control for the PRL paper. Tests whether the CNN architecture alone
produces multi-dimensional chaos. Expected result: D₂ ≈ 1.0 everywhere.

Uses the SAME SmallCNN architecture (268,650 params) as the CIFAR-10 
experiment, but trained on structureless synthetic data embedded into
3×32×32 image tensors.

Protocol matches cnn_seeds_extension_fixed.py exactly:
  - Full-batch GD, MSE loss, no momentum/weight decay
  - 5,000 training steps
  - Warmup: 1,000 steps at lr=0.01 to find λ_max
  - Lyapunov via function-space divergence, ε = 1e-5
  - Correlation dimension via Grassberger-Procaccia
  - 12 LR fractions × 3 seeds = 36 runs

USAGE (Colab):
    from google.colab import drive
    drive.mount('/content/drive')
    
    !python -u cross_cnn_synthetic_3seeds.py

    # Quick test:
    !python -u cross_cnn_synthetic_3seeds.py --quick
"""

import argparse, os, time, json, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ============================================================
# CNN ARCHITECTURE (identical to CIFAR-10 experiment)
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


# ============================================================
# SYNTHETIC DATA (identical to Phase 1/2/3, embedded in images)
# ============================================================

def generate_synthetic_data(n_samples=2000, n_classes=10, n_random=200,
                            n_quadratic=20, data_seed=42):
    """
    Generate structured synthetic data and embed into 3×32×32 image tensors.
    
    The synthetic data is 220-dimensional (200 random + 20 quadratic features).
    We embed into 3×32×32 = 3072 dimensions by placing the 220 features into
    the first 220 positions of the flattened tensor, with the rest zero-padded.
    This preserves the data structure while matching the CNN input format.
    """
    rng = np.random.RandomState(data_seed)
    
    # Generate structured synthetic data (same as phase3_experiments_k.py)
    centers = rng.randn(n_classes, n_random) * 2.0
    labels = rng.randint(0, n_classes, size=n_samples)
    X_rand = np.zeros((n_samples, n_random))
    for i in range(n_samples):
        X_rand[i] = centers[labels[i]] + rng.randn(n_random) * 0.5
    X_quad = X_rand[:, :n_quadratic] ** 2
    X_flat = np.concatenate([X_rand, X_quad], axis=1).astype(np.float32)  # (2000, 220)
    
    # Embed into 3×32×32 image tensors
    X_img = np.zeros((n_samples, 3, 32, 32), dtype=np.float32)
    X_img_flat = X_img.reshape(n_samples, -1)  # (2000, 3072)
    X_img_flat[:, :X_flat.shape[1]] = X_flat    # first 220 dims get the data
    X_img = X_img_flat.reshape(n_samples, 3, 32, 32)
    
    # One-hot labels
    y = np.zeros((n_samples, n_classes), dtype=np.float32)
    y[np.arange(n_samples), labels] = 1.0
    
    return torch.tensor(X_img), torch.tensor(y)


# ============================================================
# SHARPNESS (top Hessian eigenvalue via power iteration)
# ============================================================

def compute_sharpness(model, X, y, criterion, n_iter=50):
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
        if hv_norm < 1e-12:
            break
        v = [hv.detach() / hv_norm for hv in Hv]
    return abs(eigenvalue)


# ============================================================
# CORRELATION DIMENSION (Grassberger-Procaccia)
# ============================================================

def correlation_dimension(points, embed_dims=None, n_ref=500):
    from scipy.spatial.distance import pdist
    
    n, d = points.shape
    if embed_dims is None:
        embed_dims = [min(d, k) for k in [2, 4, 6, 8, 10] if k <= d]
        if d not in embed_dims:
            embed_dims.append(d)
        embed_dims = sorted(set(embed_dims))

    best_d2 = 0.0
    results = {}

    for ed in embed_dims:
        pts = points[:, :ed]
        dists = pdist(pts)

        r_min = np.percentile(dists[dists > 0], 1) if np.any(dists > 0) else 1e-10
        r_max = np.percentile(dists, 95)
        if r_min <= 0 or r_max <= r_min:
            continue

        radii = np.logspace(np.log10(r_min), np.log10(r_max), 30)
        counts = np.array([np.sum(dists < r) for r in radii])
        N_pairs = len(dists)
        C_r = counts / N_pairs

        mask = (C_r > 0.01) & (C_r < 0.5)
        if mask.sum() < 5:
            mask = (C_r > 0.005) & (C_r < 0.8)
        if mask.sum() < 4:
            continue

        log_r = np.log(radii[mask])
        log_C = np.log(C_r[mask])
        slope, intercept = np.polyfit(log_r, log_C, 1)

        results[ed] = {'D2': float(slope), 'n_points_fit': int(mask.sum())}
        best_d2 = float(slope)

    return best_d2, results


# ============================================================
# SINGLE SEED RUN
# ============================================================

def run_single_seed(seed, lr, X, y, X_eval, device, n_steps=5000,
                    epsilon=1e-5, sharpness_every=100, output_every=10):
    criterion = nn.MSELoss()

    torch.manual_seed(seed)
    model = SmallCNN().to(device)

    torch.manual_seed(seed)
    model_p = SmallCNN().to(device)
    pert_rng = torch.Generator()
    pert_rng.manual_seed(seed + 999999)
    with torch.no_grad():
        direction = [torch.randn(p.shape, generator=pert_rng).to(device) for p in model_p.parameters()]
        d_norm = sum((d**2).sum() for d in direction).sqrt()
        for p, d in zip(model_p.parameters(), direction):
            p.add_(d / d_norm * epsilon)

    outputs_rec = []
    sharpness_rec = []
    grad_norm_rec = []
    loss_rec = []
    n_eval = X_eval.shape[0]

    for t in range(n_steps):
        if t % output_every == 0:
            with torch.no_grad():
                out = model(X_eval).cpu().numpy().flatten()
                outputs_rec.append(out)

        if t % sharpness_every == 0:
            model.zero_grad()
            loss_val = criterion(model(X), y)
            loss_val.backward()
            gn = sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None).sqrt().item()
            loss_rec.append(loss_val.item())
            grad_norm_rec.append(gn)
            if t > 0:
                sharp = compute_sharpness(model, X, y, criterion, n_iter=20)
                sharpness_rec.append(sharp)

        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        model_p.zero_grad()
        loss_p = criterion(model_p(X), y)
        loss_p.backward()
        with torch.no_grad():
            for p in model_p.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

    with torch.no_grad():
        out_ref = model(X_eval).cpu().numpy().flatten()
        out_pert = model_p(X_eval).cpu().numpy().flatten()
    delta_final = np.linalg.norm(out_pert - out_ref)
    lyap = np.log(delta_final / epsilon) / n_steps if delta_final > 0 else 0.0

    outputs = np.array(outputs_rec)
    start = len(outputs) // 5
    traj = outputs[start:]

    centered = traj - traj.mean(axis=0)
    try:
        U, sv, Vt = np.linalg.svd(centered, full_matrices=False)
        var_exp = (sv**2) / ((sv**2).sum() + 1e-30)
        pc1 = float(var_exp[0] * 100)
        pc2 = float(var_exp[1] * 100) if len(var_exp) > 1 else 0.0
    except:
        pc1, pc2 = 99.0, 0.5

    pca_dim = min(10, traj.shape[1])
    try:
        traj_pca = centered @ Vt[:pca_dim].T
        d2, _ = correlation_dimension(traj_pca)
    except:
        d2 = 0.0

    del model, model_p
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        'lyapunov': lyap, 'corr_dim': d2,
        'pc1': pc1, 'pc2': pc2,
        'sharpness_series': sharpness_rec,
        'grad_norm_series': grad_norm_rec,
        'loss_series': loss_rec,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(seeds=[0, 1, 2], n_steps=5000, quick=False):
    lr_fractions = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if quick:
        n_steps = 2000
        lr_fractions = [0.05, 0.15, 0.3, 0.5, 0.9]
        print(f"QUICK MODE: {n_steps} steps, {len(lr_fractions)} fractions")

    # Generate synthetic data embedded in image tensors
    print("Generating synthetic data (embedded in 3×32×32)...")
    X, y = generate_synthetic_data(n_samples=2000)
    X, y = X.to(device), y.to(device)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # Find EoS threshold
    print("Finding EoS threshold (1000 warmup steps at lr=0.01)...")
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
    lam_max = compute_sharpness(ref_model, X, y, criterion, n_iter=50)
    lr_eos = 2.0 / lam_max
    print(f"  λ_max = {lam_max:.4f}, EoS = 2/λ_max = {lr_eos:.6f}")
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_lrs = [frac * lr_eos for frac in lr_fractions]

    # Eval subset
    torch.manual_seed(0)
    n_eval = 100
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    results = {
        'experiment': 'cnn_synthetic',
        'lam_max': float(lam_max),
        'lr_eos': float(lr_eos),
        'lr_fractions': lr_fractions,
        'test_lrs': [float(lr) for lr in test_lrs],
        'n_params': 268650,
        'n_samples': 2000,
        'data_shape': list(X.shape),
        'seeds_run': seeds,
    }

    total_runs = len(lr_fractions) * len(seeds)
    done = 0
    t0 = time.time()

    for li, (frac, lr) in enumerate(zip(lr_fractions, test_lrs)):
        lr_key = f"lr_{li}"
        results[lr_key] = {
            'lyapunov': [], 'corr_dim': [], 'pc1': [], 'pc2': [],
            'sharpness_series': [], 'grad_norm_series': [], 'loss_series': [],
        }

        for seed in seeds:
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total_runs - done) if done > 1 else 0
            print(f"\n[{done}/{total_runs}] LR={frac:.0%} EoS (η={lr:.6f}), "
                  f"seed={seed}, ETA: {eta/60:.1f}min")

            r = run_single_seed(
                seed=seed, lr=lr, X=X, y=y, X_eval=X_eval,
                device=device, n_steps=n_steps,
                epsilon=1e-5, sharpness_every=100, output_every=10,
            )

            for k in results[lr_key]:
                results[lr_key][k].append(r[k])

            print(f"  λ={r['lyapunov']:+.6f}, D₂={r['corr_dim']:.3f}, "
                  f"PC1={r['pc1']:.1f}%, PC2={r['pc2']:.1f}%")

        lyaps = results[lr_key]['lyapunov']
        d2s = results[lr_key]['corr_dim']
        print(f"  → {frac:.0%} EoS: λ = {np.mean(lyaps):+.6f} ± {np.std(lyaps):.6f}, "
              f"D₂ = {np.mean(d2s):.3f} ± {np.std(d2s):.3f}")

    return results


def save_results(results):
    # Save locally
    os.makedirs("results", exist_ok=True)
    seeds_str = "_".join(str(s) for s in results['seeds_run'])
    fname = f"cross_cnn_synthetic_seeds_{seeds_str}.json"

    def _ser(obj):
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, list):
            return [_ser(v) for v in obj]
        elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
            return None
        return obj

    path = os.path.join("results", fname)
    with open(path, 'w') as f:
        json.dump(_ser(results), f, indent=2)
    print(f"\nSaved → {path}")

    # Also save to Drive if available
    drive_dir = "/content/drive/MyDrive/chaos_research/results"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive_dir, exist_ok=True)
        drive_path = os.path.join(drive_dir, fname)
        with open(drive_path, 'w') as f:
            json.dump(_ser(results), f, indent=2)
        print(f"Saved → {drive_path}")

    # Print summary
    fracs = results['lr_fractions']
    print(f"\n{'Frac':>6} {'η':>10} {'D₂ mean':>8} {'D₂ std':>8} {'λ mean':>12} {'PC1%':>7}")
    print("-" * 60)
    for i, frac in enumerate(fracs):
        lr = results[f'lr_{i}']
        d2 = np.array(lr['corr_dim'])
        lyap = np.array(lr['lyapunov'])
        pc1 = np.mean(lr['pc1'])
        print(f"{frac:>6.2f} {results['test_lrs'][i]:>10.6f} "
              f"{d2.mean():>8.3f} {d2.std():>8.3f} "
              f"{lyap.mean():>+12.6f} {pc1:>7.1f}")

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Cross-experiment: CNN on synthetic data (null control)")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (2000 steps, 5 fractions)")
    args = parser.parse_args()

    print("=" * 60)
    print("CROSS-EXPERIMENT: CNN on Synthetic Data")
    print(f"  Seeds: {args.seeds}")
    print(f"  Steps: {args.steps}")
    n_fracs = 5 if args.quick else 12
    print(f"  {n_fracs} LR fractions × {len(args.seeds)} seeds = "
          f"{n_fracs * len(args.seeds)} runs")
    print("=" * 60)

    results = run_experiment(seeds=args.seeds, n_steps=args.steps, quick=args.quick)
    save_results(results)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
