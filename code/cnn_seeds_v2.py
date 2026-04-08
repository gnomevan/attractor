"""
CNN Seeds Extension v2 — Protocol-matched to Experiment K
===========================================================

Runs 7 additional seeds (3–9) for the CIFAR-10 CNN, matching the
original phase3_experiments_k.py EXACTLY:

  - Perturbation via copy.deepcopy + flat random direction
  - Lyapunov via linregress on log(distance) over steps [0.2T, 0.8T]
  - Distance computed on X_eval at EVERY training step
  - Correlation dimension: 20 log-spaced radii, percentile(1) to
    percentile(95), fit indices [4:16]
  - Sharpness: 15 power iterations
  - Grad norm/loss recorded every step, subsampled [::10] when saving

Also includes Lorenz attractor D₂ validation using the SAME correlation
dimension pipeline (should recover D₂ ≈ 2.05).

USAGE (Lightning.ai or Colab):
    python -u cnn_seeds_v2.py --seeds 3 4 5 6 7 8 9
    python -u cnn_seeds_v2.py --seeds 3 --quick          # test one seed
    python -u cnn_seeds_v2.py --lorenz-only               # validation only
    python -u cnn_seeds_v2.py --merge results/cifar10_eos.json results/cifar10_eos_seeds_3_4_5_6_7_8_9.json
"""

import argparse, os, time, json, copy, warnings
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# CNN ARCHITECTURE (identical to Experiment K)
# ============================================================

class SmallCNN(nn.Module):
    """Small CNN following Cohen et al. (2021) CIFAR-10 setup."""
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


def make_cnn(seed):
    """Create CNN with deterministic initialization."""
    torch.manual_seed(seed)
    return SmallCNN()


def clone_cnn_perturbed(model, eps, seed):
    """
    Clone CNN and perturb along unit-norm random direction.
    EXACTLY matches original Experiment K protocol:
      1. copy.deepcopy to guarantee identical weights
      2. Flatten ALL params into one vector
      3. Generate one random direction from seeded Generator
      4. Normalize to unit norm
      5. Add eps * direction
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


# ============================================================
# DATA LOADING (identical to Experiment K)
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
# SHARPNESS (identical to Experiment K: n_iter default matches usage)
# ============================================================

def compute_sharpness_cnn(model, X, y, criterion, n_iter=20):
    """Power iteration for top Hessian eigenvalue."""
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
# CORRELATION DIMENSION (identical to Experiment K)
# ============================================================

def correlation_dimension_k(traj, seed):
    """
    Correlation dimension matching Experiment K EXACTLY:
      - Subsample to 2000 if needed (using seed-based RNG)
      - All pairwise distances (O(n²) loop)
      - 20 log-spaced radii from percentile(1) to percentile(95)
      - Fit indices [4:16]
    """
    if len(traj) > 2000:
        traj_sub = traj[np.random.RandomState(seed).choice(len(traj), 2000, replace=False)]
    else:
        traj_sub = traj

    n = len(traj_sub)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(traj_sub[i] - traj_sub[j]))
    dists = np.array(dists)

    if len(dists) == 0:
        return float('nan')

    log_eps = np.linspace(
        np.log(np.percentile(dists, 1) + 1e-15),
        np.log(np.percentile(dists, 95)),
        20
    )
    log_C = [
        np.log(max(np.sum(dists < np.exp(le)) / (n * (n - 1) / 2), 1e-30))
        for le in log_eps
    ]
    log_C = np.array(log_C)

    cd = stats.linregress(log_eps[4:16], log_C[4:16])[0]
    return cd


# ============================================================
# SINGLE SEED RUN (matches Experiment K training loop exactly)
# ============================================================

def run_single_seed(seed, lr, X, y, X_eval, device, n_steps=5000, eps=1e-5):
    """
    Train CNN at given lr for given seed.
    Protocol matches original Experiment K line-by-line:
      - Distance recorded every step on X_eval
      - Outputs recorded every 10 steps
      - Sharpness every 100 steps (t > 0), 15 iterations
      - Loss and grad norm every step
      - Lyapunov via linregress on log(distance) from 0.2T to 0.8T
    """
    criterion = nn.MSELoss()

    # Create model and perturbed clone
    model = make_cnn(seed).to(device)
    perturbed = clone_cnn_perturbed(model, eps, seed).to(device)

    distances = np.zeros(n_steps)
    losses_rec = np.zeros(n_steps)
    gn_rec = np.zeros(n_steps)
    sharp_rec = []
    outputs_rec = []

    for t in range(n_steps):
        # Distance at every step (on X_eval)
        with torch.no_grad():
            d = torch.norm(model(X_eval) - perturbed(X_eval)).item()
            distances[t] = d

        # Train original — record loss and grad norm
        model.zero_grad()
        loss = criterion(model(X), y)
        losses_rec[t] = loss.item()
        loss.backward()
        with torch.no_grad():
            gn = sum((p.grad**2).sum() for p in model.parameters()
                     if p.grad is not None).sqrt().item()
            gn_rec[t] = gn

        # Record outputs every 10 steps
        if t % 10 == 0:
            with torch.no_grad():
                outputs_rec.append(model(X_eval).cpu().numpy())

        # Sharpness every 100 steps (skip t=0), 15 iterations
        if t % 100 == 0 and t > 0:
            sharp = compute_sharpness_cnn(model, X, y, criterion, n_iter=15)
            sharp_rec.append(sharp)

        # GD update — original
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        # GD update — perturbed
        perturbed.zero_grad()
        loss2 = criterion(perturbed(X), y)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

    # --- Lyapunov exponent via linregress (matches original exactly) ---
    log_d = np.log(distances + 1e-30)
    start, end = int(n_steps * 0.2), int(n_steps * 0.8)
    lyap = stats.linregress(np.arange(start, end), log_d[start:end])[0]

    # --- Trajectory analysis ---
    outputs_arr = np.array(outputs_rec)  # (n_record, n_eval, 10)
    traj_start = len(outputs_arr) // 5
    traj = outputs_arr[traj_start:].reshape(len(outputs_arr) - traj_start, -1)

    # PCA
    centered = traj - traj.mean(axis=0)
    try:
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        ve = (sv**2) / (sv**2).sum()
        pc1 = float(ve[0] * 100)
        pc2 = float(ve[1] * 100) if len(ve) > 1 else 0.0
    except:
        pc1, pc2 = 100.0, 0.0

    # Correlation dimension (matches original exactly)
    cd = correlation_dimension_k(traj, seed)

    del model, perturbed
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        'lyapunov': lyap,
        'corr_dim': cd,
        'pc1': pc1,
        'pc2': pc2,
        'sharpness_series': sharp_rec,
        'grad_norm_series': gn_rec[::10].tolist(),  # subsample to match original
        'loss_series': losses_rec[::10].tolist(),     # subsample to match original
    }


# ============================================================
# LORENZ ATTRACTOR VALIDATION
# ============================================================

def lorenz_validation():
    """
    Validate the correlation dimension pipeline on known systems.
    Uses the SAME correlation_dimension_k function as the CNN experiments.
    """
    print("\n" + "=" * 60)
    print("CORRELATION DIMENSION VALIDATION")
    print("=" * 60)

    from scipy.integrate import solve_ivp

    # --- Lorenz attractor (expected D₂ ≈ 2.05) ---
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    def lorenz(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    sol = solve_ivp(lorenz, (0, 200), [1.0, 1.0, 1.0],
                    t_eval=np.linspace(0, 200, 100000),
                    method='RK45', rtol=1e-10, atol=1e-12)

    start = len(sol.t) // 5
    lorenz_traj = sol.y[:, start:].T  # (n_points, 3)

    # Subsample to match CNN trajectory scale (~800 points after transient skip)
    for n_pts in [400, 800, 1600]:
        idx = np.random.RandomState(42).choice(len(lorenz_traj), n_pts, replace=False)
        pts = lorenz_traj[idx]
        d2 = correlation_dimension_k(pts, seed=42)
        print(f"  Lorenz n={n_pts}: D₂ = {d2:.3f} (expected ≈ 2.05)")

    # --- Quasiperiodic 2-torus (expected D₂ ≈ 2.0) ---
    t = np.linspace(0, 100 * np.pi, 10000)
    R, r = 3.0, 1.0
    omega1, omega2 = 1.0, np.sqrt(2)
    x = (R + r * np.cos(omega2 * t)) * np.cos(omega1 * t)
    y_t = (R + r * np.cos(omega2 * t)) * np.sin(omega1 * t)
    z = r * np.sin(omega2 * t)
    torus_traj = np.column_stack([x, y_t, z])

    for n_pts in [400, 800, 1600]:
        idx = np.random.RandomState(42).choice(len(torus_traj), n_pts, replace=False)
        pts = torus_traj[idx]
        d2 = correlation_dimension_k(pts, seed=42)
        print(f"  2-torus n={n_pts}: D₂ = {d2:.3f} (expected ≈ 2.0)")

    # Use the 800-point versions as the "official" validation
    idx_l = np.random.RandomState(42).choice(len(lorenz_traj), 800, replace=False)
    idx_t = np.random.RandomState(42).choice(len(torus_traj), 800, replace=False)
    d2_lorenz = correlation_dimension_k(lorenz_traj[idx_l], seed=42)
    d2_torus = correlation_dimension_k(torus_traj[idx_t], seed=42)

    print(f"\n  VALIDATION SUMMARY:")
    print(f"    Lorenz:  D₂ = {d2_lorenz:.3f} (expected 2.05)")
    print(f"    2-torus: D₂ = {d2_torus:.3f} (expected 2.00)")

    return {
        'lorenz_d2': float(d2_lorenz),
        'torus_d2': float(d2_torus),
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

    # Load data (identical to Experiment K)
    print("Loading CIFAR-10...")
    X, y = load_cifar10_subset(n_samples=2000, seed=42)
    X, y = X.to(device), y.to(device)

    n_params = make_cnn(0).count_params()
    print(f"CNN parameters: {n_params}")

    # Find EoS threshold (identical to Experiment K)
    print("Finding EoS threshold...")
    torch.manual_seed(0)
    ref_model = make_cnn(0).to(device)
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
    print(f"λ_max = {lam_max:.4f}, EoS = {lr_eos:.4f}")
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_lrs = [frac * lr_eos for frac in lr_fractions]

    # Eval subset (identical to Experiment K)
    torch.manual_seed(0)
    n_eval = min(100, 2000)
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    results = {
        'lam_max': float(lam_max),
        'lr_eos': float(lr_eos),
        'lr_fractions': lr_fractions,
        'test_lrs': [float(lr) for lr in test_lrs],
        'n_params': n_params,
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
            print(f"\n[{done}/{total_runs}] {frac:.0%} EoS, η={lr:.4f}, seed={seed} "
                  f"ETA: {eta / 60:.1f}min", end="", flush=True)

            r = run_single_seed(
                seed=seed, lr=lr, X=X, y=y, X_eval=X_eval,
                device=device, n_steps=n_steps, eps=1e-5,
            )

            for k in results[lr_key]:
                results[lr_key][k].append(r[k])

            print(f"  → λ={r['lyapunov']:+.6f}, D₂={r['corr_dim']:.2f}, "
                  f"PC1={r['pc1']:.1f}%, PC2={r['pc2']:.1f}%")

        # Per-LR summary
        lyaps = results[lr_key]['lyapunov']
        d2s = [d for d in results[lr_key]['corr_dim'] if not np.isnan(d)]
        print(f"\n  → {frac:.0%} EoS summary: "
              f"λ = {np.mean(lyaps):+.6f} ± {np.std(lyaps):.6f}, "
              f"D₂ = {np.nanmean(d2s):.3f} ± {np.nanstd(d2s):.3f}")

    return results


# ============================================================
# SERIALIZATION AND MERGING
# ============================================================

def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, list):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return None
    return obj


def save_results(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    seeds_str = "_".join(str(s) for s in results.get('seeds_run', []))
    fname = f"cifar10_eos_seeds_{seeds_str}.json"
    path = os.path.join(output_dir, fname)
    with open(path, 'w') as f:
        json.dump(_serialize(results), f, indent=2)
    print(f"\nSaved → {path}")

    # Also save to Drive/persistent storage if available
    for drive_root in ["/content/drive/MyDrive", "/teamspace/studios/this_studio"]:
        drive_dir = os.path.join(drive_root, "chaos_research", "results")
        if os.path.isdir(drive_root):
            os.makedirs(drive_dir, exist_ok=True)
            drive_path = os.path.join(drive_dir, fname)
            with open(drive_path, 'w') as f:
                json.dump(_serialize(results), f, indent=2)
            print(f"Saved → {drive_path}")

    return path


def merge_results(original_path, extension_path, output_path=None):
    """Merge original 3-seed results with new seeds."""
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
        json.dump(_serialize(merged), f, indent=2)
    print(f"\nMerged → {output_path}")
    print(f"  Seeds per LR: {len(merged['lr_0']['lyapunov'])}")

    # Summary table
    fracs = merged['lr_fractions']
    print(f"\n{'frac':>6s}  {'η':>8s}  {'λ mean':>10s}  {'λ std':>10s}  "
          f"{'D₂ mean':>8s}  {'D₂ std':>8s}  {'PC1':>6s}  {'n':>3s}")
    print("-" * 75)
    for li, frac in enumerate(fracs):
        r = merged[f"lr_{li}"]
        lyaps = np.array([v for v in r['lyapunov'] if v is not None])
        d2s = np.array([v for v in r['corr_dim'] if v is not None])
        pc1s = np.array([v for v in r['pc1'] if v is not None])
        n = len(lyaps)
        lr_val = merged['test_lrs'][li]
        print(f"  {frac:4.0%}  {lr_val:8.5f}  {lyaps.mean():+10.6f}  {lyaps.std():10.6f}  "
              f"{np.nanmean(d2s):8.3f}  {np.nanstd(d2s):8.3f}  {pc1s.mean():5.1f}%  {n:3d}")

    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CNN Seeds Extension v2 (protocol-matched)")
    parser.add_argument("--seeds", type=int, nargs='+', default=[3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lorenz-only", action="store_true")
    parser.add_argument("--skip-lorenz", action="store_true")
    parser.add_argument("--merge", nargs=2, metavar=('ORIG', 'EXT'))
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    if args.merge:
        merge_results(args.merge[0], args.merge[1])
        return

    # Lorenz validation (unless skipped)
    lorenz_results = None
    if not args.skip_lorenz:
        lorenz_results = lorenz_validation()

    if args.lorenz_only:
        if lorenz_results:
            save_results({'lorenz_validation': lorenz_results, 'seeds_run': []},
                         args.output_dir)
        return

    print("\n" + "=" * 60)
    print(f"CNN SEEDS EXTENSION v2 (protocol-matched)")
    print(f"  Seeds: {args.seeds}")
    print(f"  Steps: {args.steps}")
    print(f"  12 LR fractions × {len(args.seeds)} seeds = {12 * len(args.seeds)} runs")
    print("=" * 60)

    results = run_extension(seeds=args.seeds, n_steps=args.steps, quick=args.quick)
    if lorenz_results:
        results['lorenz_validation'] = lorenz_results

    save_results(results, args.output_dir)

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    fracs = results['lr_fractions']
    print(f"\n{'frac':>6s}  {'λ mean':>10s}  {'D₂ mean':>8s}  {'PC1':>6s}")
    print("-" * 40)
    for li, frac in enumerate(fracs):
        r = results[f"lr_{li}"]
        print(f"  {frac:4.0%}  {np.mean(r['lyapunov']):+10.6f}  "
              f"{np.nanmean(r['corr_dim']):8.3f}  {np.mean(r['pc1']):5.1f}%")

    print(f"\nTo merge with original data:")
    seeds_str = "_".join(str(s) for s in args.seeds)
    print(f"  python cnn_seeds_v2.py --merge PATH/cifar10_eos.json "
          f"results/cifar10_eos_seeds_{seeds_str}.json")


if __name__ == "__main__":
    main()
