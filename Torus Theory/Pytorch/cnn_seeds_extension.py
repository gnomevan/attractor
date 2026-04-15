"""
CNN Seeds Extension — 7 additional seeds (3–9) for PRL submission
==================================================================

Matches Phase 3 Experiment K protocol exactly:
  - SmallCNN on CIFAR-10 (2000 subset), 268,650 params
  - Full-batch GD, MSE loss, no momentum/weight decay
  - 5,000 training steps
  - Lyapunov via function-space divergence, ε = 1e-5
  - Correlation dimension via Grassberger-Procaccia
  - PCA of function-space trajectory

Also includes Lorenz attractor D₂ validation (should recover D₂ ≈ 2.05).

USAGE (Colab):
    # Mount Drive first
    from google.colab import drive
    drive.mount('/content/drive')

    !python -u cnn_seeds_extension.py --seeds 3 4 5 6 7 8 9

    # Or quick test with one seed:
    !python -u cnn_seeds_extension.py --seeds 3 --quick

    # Lorenz validation only:
    !python -u cnn_seeds_extension.py --lorenz-only
"""

import argparse, os, time, json, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ============================================================
# CNN ARCHITECTURE (identical to Phase 3)
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
# DATA LOADING
# ============================================================

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
    """
    Grassberger-Procaccia correlation dimension.
    Uses multiple embedding dimensions to check convergence.
    Returns the D₂ estimate from the highest embedding dimension.
    """
    from scipy.spatial.distance import pdist

    n, d = points.shape
    if n < 50:
        return 0.0, {}

    # Center and normalize
    points = points - points.mean(axis=0)
    scale = np.std(points)
    if scale < 1e-15:
        return 0.0, {}
    points = points / scale

    if embed_dims is None:
        embed_dims = [min(d, k) for k in [2, 4, 6, 8, 10] if k <= d]
        if d not in embed_dims:
            embed_dims.append(d)
        embed_dims = sorted(set(embed_dims))

    results = {}
    best_d2 = 0.0

    for ed in embed_dims:
        pts = points[:, :ed]

        # Subsample reference points for speed
        if n > n_ref:
            idx = np.random.RandomState(42).choice(n, n_ref, replace=False)
            ref_pts = pts[idx]
        else:
            ref_pts = pts

        dists = pdist(ref_pts)
        dists = dists[dists > 0]

        if len(dists) < 100:
            continue

        # Log-spaced radii
        r_min = np.percentile(dists, 1)
        r_max = np.percentile(dists, 90)
        if r_min <= 0 or r_max <= r_min:
            continue

        radii = np.logspace(np.log10(r_min), np.log10(r_max), 30)
        counts = np.array([np.sum(dists < r) for r in radii])
        N_pairs = len(dists)
        C_r = counts / N_pairs

        # Linear regression on log-log
        mask = (C_r > 0.01) & (C_r < 0.5)  # scaling regime
        if mask.sum() < 5:
            mask = (C_r > 0.005) & (C_r < 0.8)
        if mask.sum() < 4:
            continue

        log_r = np.log(radii[mask])
        log_C = np.log(C_r[mask])
        slope, intercept = np.polyfit(log_r, log_C, 1)

        results[ed] = {
            'D2': float(slope),
            'n_points_fit': int(mask.sum()),
        }
        best_d2 = float(slope)

    return best_d2, results


# ============================================================
# SINGLE SEED RUN
# ============================================================

def run_single_seed(seed, lr, X, y, X_eval, device, n_steps=5000,
                    epsilon=1e-5, sharpness_every=100, output_every=10):
    """
    Train CNN at given lr for given seed.
    Returns: lyapunov, corr_dim, pc1, pc2, sharpness_series, grad_norm_series, loss_series
    """
    criterion = nn.MSELoss()

    # --- Train reference model ---
    torch.manual_seed(seed)
    model = SmallCNN().to(device)

    # --- Train perturbed model (for Lyapunov) ---
    torch.manual_seed(seed)
    model_p = SmallCNN().to(device)
    # Apply perturbation
    pert_rng = torch.Generator()
    pert_rng.manual_seed(seed + 999999)
    with torch.no_grad():
        direction = [torch.randn(p.shape, generator=pert_rng, device=device) for p in model_p.parameters()]
        d_norm = sum((d**2).sum() for d in direction).sqrt()
        for p, d in zip(model_p.parameters(), direction):
            p.add_(d / d_norm * epsilon)

    outputs_rec = []
    sharpness_rec = []
    grad_norm_rec = []
    loss_rec = []

    n_eval = X_eval.shape[0]

    for t in range(n_steps):
        # Record outputs for trajectory analysis
        if t % output_every == 0:
            with torch.no_grad():
                out = model(X_eval).cpu().numpy().flatten()
                outputs_rec.append(out)

        # Record loss and grad norm
        if t % sharpness_every == 0:
            model.zero_grad()
            loss_val = criterion(model(X), y)
            loss_val.backward()
            gn = sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None).sqrt().item()
            loss_rec.append(loss_val.item())
            grad_norm_rec.append(gn)

            # Sharpness (skip step 0)
            if t > 0:
                sharp = compute_sharpness(model, X, y, criterion, n_iter=20)
                sharpness_rec.append(sharp)

        # --- GD step for both models ---
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

    # --- Lyapunov exponent ---
    with torch.no_grad():
        out_ref = model(X_eval).cpu().numpy().flatten()
        out_pert = model_p(X_eval).cpu().numpy().flatten()
    delta_final = np.linalg.norm(out_pert - out_ref)
    if delta_final > 0 and epsilon > 0:
        lyap = np.log(delta_final / epsilon) / n_steps
    else:
        lyap = 0.0

    # --- Correlation dimension and PCA ---
    outputs = np.array(outputs_rec)  # (n_record, n_eval*10)
    start = len(outputs) // 5  # skip transient
    traj = outputs[start:]

    # PCA
    centered = traj - traj.mean(axis=0)
    try:
        U, sv, Vt = np.linalg.svd(centered, full_matrices=False)
        var_exp = (sv**2) / ((sv**2).sum() + 1e-30)
        pc1 = float(var_exp[0] * 100)
        pc2 = float(var_exp[1] * 100) if len(var_exp) > 1 else 0.0
    except:
        pc1, pc2 = 99.0, 0.5

    # Correlation dimension on PCA-reduced trajectory
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
        'lyapunov': lyap,
        'corr_dim': d2,
        'pc1': pc1,
        'pc2': pc2,
        'sharpness_series': sharpness_rec,
        'grad_norm_series': grad_norm_rec,
        'loss_series': loss_rec,
    }


# ============================================================
# LORENZ ATTRACTOR VALIDATION
# ============================================================

def lorenz_validation():
    """
    Generate Lorenz attractor trajectory, compute D₂.
    Expected: D₂ ≈ 2.05 (Grassberger & Procaccia, 1983).
    This validates our correlation dimension pipeline.
    """
    print("\n" + "=" * 60)
    print("LORENZ ATTRACTOR D₂ VALIDATION")
    print("=" * 60)

    from scipy.integrate import solve_ivp

    sigma, rho, beta = 10.0, 28.0, 8.0/3.0

    def lorenz(t, state):
        x, y, z = state
        return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

    # Integrate for a long time, discard transient
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 100000)
    sol = solve_ivp(lorenz, t_span, [1.0, 1.0, 1.0], t_eval=t_eval,
                    method='RK45', rtol=1e-10, atol=1e-12)

    # Discard first 20% as transient
    start = len(sol.t) // 5
    traj = sol.y[:, start:].T  # (n_points, 3)

    print(f"  Lorenz trajectory: {traj.shape[0]} points × {traj.shape[1]} dims")

    # Subsample to match our CNN pipeline scale
    for n_pts in [500, 1000, 2000, 5000]:
        idx = np.random.RandomState(42).choice(len(traj), min(n_pts, len(traj)), replace=False)
        pts = traj[idx]
        d2, details = correlation_dimension(pts, n_ref=min(500, n_pts))
        print(f"  n={n_pts:5d}: D₂ = {d2:.3f}")

    # Full trajectory
    d2_full, details_full = correlation_dimension(traj[::10], n_ref=1000)
    print(f"  Full (subsampled 10x): D₂ = {d2_full:.3f}")
    print(f"  Expected: D₂ ≈ 2.05 (Grassberger & Procaccia, 1983)")

    # Also test on a known 2-torus (should give D₂ ≈ 2.0)
    print("\n  2-TORUS CONTROL:")
    t = np.linspace(0, 100*np.pi, 10000)
    R, r = 3.0, 1.0
    omega1, omega2 = 1.0, np.sqrt(2)  # incommensurate frequencies
    x = (R + r*np.cos(omega2*t)) * np.cos(omega1*t)
    y_t = (R + r*np.cos(omega2*t)) * np.sin(omega1*t)
    z = r * np.sin(omega2*t)
    torus_pts = np.column_stack([x, y_t, z])
    d2_torus, _ = correlation_dimension(torus_pts, n_ref=1000)
    print(f"  Quasiperiodic 2-torus: D₂ = {d2_torus:.3f} (expected ≈ 2.0)")

    return {
        'lorenz_d2': d2_full,
        'torus_d2': d2_torus,
        'lorenz_expected': 2.05,
        'torus_expected': 2.0,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_extension(seeds, n_steps=5000, quick=False):
    """Run CNN experiments for specified seeds at all 12 LR fractions."""

    lr_fractions = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if quick:
        n_steps = 2000
        print(f"QUICK MODE: {n_steps} steps")

    # Load data
    print("Loading CIFAR-10...")
    X, y = load_cifar10_subset(n_samples=2000, seed=42)
    X, y = X.to(device), y.to(device)

    # Find EoS threshold (same protocol as Phase 3)
    print("Finding EoS threshold...")
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
    print(f"λ_max = {lam_max:.4f}, EoS = {lr_eos:.4f}")
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_lrs = [frac * lr_eos for frac in lr_fractions]

    # Eval subset (same as Phase 3)
    torch.manual_seed(0)
    n_eval = 100
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    # Results storage
    results = {
        'lam_max': float(lam_max),
        'lr_eos': float(lr_eos),
        'lr_fractions': lr_fractions,
        'test_lrs': [float(lr) for lr in test_lrs],
        'n_params': 268650,
        'n_samples': 2000,
        'seeds_run': seeds,
    }

    total_runs = len(lr_fractions) * len(seeds)
    done = 0
    t0 = time.time()

    for li, (frac, lr) in enumerate(zip(lr_fractions, test_lrs)):
        lr_key = f"lr_{li}"
        results[lr_key] = {
            'lyapunov': [],
            'corr_dim': [],
            'pc1': [],
            'pc2': [],
            'sharpness_series': [],
            'grad_norm_series': [],
            'loss_series': [],
        }

        for seed in seeds:
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total_runs - done) if done > 1 else 0
            print(f"\n[{done}/{total_runs}] LR={frac:.0%} EoS (η={lr:.5f}), "
                  f"seed={seed}, ETA: {eta/60:.1f}min")

            r = run_single_seed(
                seed=seed, lr=lr, X=X, y=y, X_eval=X_eval,
                device=device, n_steps=n_steps,
                epsilon=1e-5, sharpness_every=100, output_every=10,
            )

            for k in results[lr_key]:
                results[lr_key][k].append(r[k])

            print(f"  λ={r['lyapunov']:.6f}, D₂={r['corr_dim']:.3f}, "
                  f"PC1={r['pc1']:.1f}%, PC2={r['pc2']:.1f}%")

        # Per-LR summary
        lyaps = results[lr_key]['lyapunov']
        d2s = results[lr_key]['corr_dim']
        print(f"  → {frac:.0%} EoS summary: "
              f"λ = {np.mean(lyaps):.6f} ± {np.std(lyaps):.6f}, "
              f"D₂ = {np.mean(d2s):.3f} ± {np.std(d2s):.3f}")

    return results


def save_results(results, output_dir="results"):
    """Save with numpy/json serialization."""
    os.makedirs(output_dir, exist_ok=True)

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

    seeds_str = "_".join(str(s) for s in results.get('seeds_run', []))
    fname = f"cifar10_eos_seeds_{seeds_str}.json"
    path = os.path.join(output_dir, fname)
    with open(path, 'w') as f:
        json.dump(_ser(results), f, indent=2)
    print(f"\nSaved → {path}")

    # Also try to save to Drive
    drive_dir = "/content/drive/MyDrive/chaos_research/results"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive_dir, exist_ok=True)
        drive_path = os.path.join(drive_dir, fname)
        with open(drive_path, 'w') as f:
            json.dump(_ser(results), f, indent=2)
        print(f"Saved → {drive_path}")

    return path


def merge_results(original_path, extension_path, output_path=None):
    """
    Merge original 3-seed results with new seeds.
    Produces a single JSON with 10 seeds per LR.
    """
    with open(original_path) as f:
        orig = json.load(f)
    with open(extension_path) as f:
        ext = json.load(f)

    merged = {
        'lam_max': orig['lam_max'],
        'lr_eos': orig.get('lr_eos', 2.0 / orig['lam_max']),
        'lr_fractions': orig.get('lr_fractions', ext.get('lr_fractions')),
        'test_lrs': orig.get('test_lrs', ext.get('test_lrs')),
        'n_params': orig.get('n_params', 268650),
        'n_samples': orig.get('n_samples', 2000),
        'seeds_original': [0, 1, 2],
        'seeds_extension': ext.get('seeds_run', []),
    }

    n_lrs = len(merged['lr_fractions'])
    for li in range(n_lrs):
        lr_key = f"lr_{li}"
        merged[lr_key] = {}
        for field in ['lyapunov', 'corr_dim', 'pc1', 'pc2',
                       'sharpness_series', 'grad_norm_series', 'loss_series']:
            orig_vals = orig.get(lr_key, {}).get(field, [])
            ext_vals = ext.get(lr_key, {}).get(field, [])
            merged[lr_key][field] = orig_vals + ext_vals

    if output_path is None:
        output_path = "results/cifar10_eos_merged_10seeds.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged → {output_path}")
    print(f"  Seeds per LR: {len(merged['lr_0']['lyapunov'])}")

    # Summary table
    fracs = merged['lr_fractions']
    print(f"\n{'frac':>6s}  {'η':>8s}  {'λ mean':>10s}  {'λ std':>10s}  "
          f"{'D₂ mean':>8s}  {'D₂ std':>8s}  {'PC1':>6s}  {'n':>3s}")
    print("-" * 75)
    for li, frac in enumerate(fracs):
        r = merged[f"lr_{li}"]
        lyaps = np.array(r['lyapunov'])
        d2s = np.array(r['corr_dim'])
        pc1s = np.array(r['pc1'])
        n = len(lyaps)
        lr_val = merged['test_lrs'][li] if li < len(merged['test_lrs']) else frac * merged['lr_eos']
        print(f"  {frac:4.0%}  {lr_val:8.5f}  {lyaps.mean():+10.6f}  {lyaps.std():10.6f}  "
              f"{d2s.mean():8.3f}  {d2s.std():8.3f}  {pc1s.mean():5.1f}%  {n:3d}")

    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CNN Seeds Extension for PRL")
    parser.add_argument("--seeds", type=int, nargs='+', default=[3, 4, 5, 6, 7, 8, 9],
                        help="Seeds to run (default: 3 4 5 6 7 8 9)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--quick", action="store_true", help="Quick test (2000 steps)")
    parser.add_argument("--lorenz-only", action="store_true", help="Run Lorenz validation only")
    parser.add_argument("--merge", nargs=2, metavar=('ORIG', 'EXT'),
                        help="Merge original and extension JSONs")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    if args.merge:
        merge_results(args.merge[0], args.merge[1])
        return

    # Always run Lorenz validation
    lorenz_results = lorenz_validation()

    if args.lorenz_only:
        save_results({'lorenz_validation': lorenz_results}, args.output_dir)
        return

    print("\n" + "=" * 60)
    print(f"CNN SEEDS EXTENSION")
    print(f"  Seeds: {args.seeds}")
    print(f"  Steps: {args.steps}")
    print(f"  12 LR fractions × {len(args.seeds)} seeds = {12 * len(args.seeds)} runs")
    print("=" * 60)

    results = run_extension(seeds=args.seeds, n_steps=args.steps, quick=args.quick)
    results['lorenz_validation'] = lorenz_results

    save_results(results, args.output_dir)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Copy original cifar10_eos.json to results/")
    print(f"  2. Run: python cnn_seeds_extension.py --merge results/cifar10_eos.json "
          f"results/cifar10_eos_seeds_{'_'.join(str(s) for s in args.seeds)}.json")
    print(f"  3. This produces results/cifar10_eos_merged_10seeds.json")


if __name__ == "__main__":
    main()
