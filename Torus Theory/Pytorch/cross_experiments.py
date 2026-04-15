"""
Cross-Experiment Controls — Isolating Architecture vs. Data
=============================================================

The main paper compares:
  - MLP (14K params) on synthetic data → D₂ ≈ 0.9
  - CNN (269K params) on CIFAR-10     → D₂ ≈ 3.67

Two variables changed simultaneously: architecture AND data.
This script runs the two missing controls:

  Experiment A: Large MLP on CIFAR-10
    - Flatten 32×32×3 images to 3072-d vectors
    - MLP with 2 hidden layers, width 85 → 269,195 params
      (matches CNN param count within 0.2%)
    - Same learning rate fractions of 2/λ_max
    - If D₂ stays low: convolutional structure matters
    - If D₂ goes high: data complexity alone is sufficient

  Experiment B: CNN on synthetic "images"
    - Embed 220-d synthetic data into 3×32×32 tensors (zero-padded)
    - Same CNN architecture (269K params)
    - Same learning rate fractions of 2/λ_max
    - If D₂ stays low: real data complexity matters
    - If D₂ goes high: CNN architecture alone is sufficient

Protocol matches Experiment K / cnn_seeds_v2.py exactly:
  - Full-batch GD, MSE loss, no momentum, no weight decay
  - Lyapunov via function-space divergence, ε = 1e-5
  - Correlation dimension: Grassberger-Procaccia, 20 log-spaced radii,
    fit indices [4:16]
  - PCA of function-space trajectory
  - Sharpness: power iteration, 15 iterations
  - 5,000 training steps

USAGE:

  Lightning.ai (recommended):
    python -u cross_experiments.py --both --seeds 3
    python -u cross_experiments.py --both --seeds 1 --quick   # test first

  Kaggle (outputs to /kaggle/working/ automatically):
    !python -u cross_experiments.py --both --seeds 3

  Run just one experiment:
    python -u cross_experiments.py --mlp-cifar --seeds 3
    python -u cross_experiments.py --cnn-synthetic --seeds 3

  Platform auto-detected. Results saved to:
    Kaggle:       /kaggle/working/results/
    Lightning.ai: /teamspace/studios/this_studio/chaos_research/results/
                  (also local ./results/)
    Local:        ./results/

REQUIREMENTS:
    pip install torch torchvision numpy scipy
"""

import argparse, os, time, json, copy, warnings
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# ARCHITECTURES
# ============================================================

class LargeMLP(nn.Module):
    """
    MLP sized to match CNN param count (~269K).
    Input: 3072 (flattened 32×32×3 CIFAR-10 image)
    Architecture: 3072 → 85 → 85 → 10
    Params: 3072×85 + 85 + 85×85 + 85 + 85×10 + 10 = 269,195

    Uses tanh activation to match the original MLP experiments.
    """
    def __init__(self, input_dim=3072, hidden_dim=85, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # flatten images
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class SmallCNN(nn.Module):
    """Identical to Experiment K CNN."""
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
# DATA LOADING
# ============================================================

def _cifar10_data_root():
    """Find CIFAR-10 data root for this platform."""
    # Kaggle: may be pre-loaded as a dataset
    if os.path.isdir('/kaggle/input/cifar10-python'):
        return '/kaggle/input/cifar10-python'
    elif os.path.isdir('/kaggle/working'):
        return '/kaggle/working/data'
    else:
        return './data'


def _load_cifar10_raw(n_samples=2000, seed=42):
    """Load CIFAR-10 subset as image tensors. Shared by both loaders."""
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=_cifar10_data_root(), train=True, download=True,
        transform=transform
    )

    # Same deterministic subset as Experiment K
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), n_samples, replace=False)

    images, labels = [], []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    X = torch.stack(images)          # (n, 3, 32, 32)
    y = torch.zeros(n_samples, 10)
    for i, label in enumerate(labels):
        y[i, label] = 1.0

    return X, y


def load_cifar10_flat(n_samples=2000, seed=42):
    """
    Load CIFAR-10 subset as flattened vectors for MLP.
    Returns X as (n_samples, 3072) — same images as Experiment K,
    just flattened instead of kept as 3×32×32.
    """
    X, y = _load_cifar10_raw(n_samples, seed)
    return X.view(n_samples, -1), y    # (n, 3072)


def load_cifar10_images(n_samples=2000, seed=42):
    """Load CIFAR-10 subset as images (for CNN). Identical to Experiment K."""
    return _load_cifar10_raw(n_samples, seed)


def generate_synthetic_images(n_samples=2000, seed=42):
    """
    Generate synthetic structured data embedded in 3×32×32 image tensors.
    Same data generation as Phase 1/2 (220-d structured), embedded into
    the first 220 positions of a flattened 3072-d tensor, rest zero.

    This lets the CNN architecture process the synthetic data, though
    the spatial structure (convolutions, pooling) operates on data
    that has no meaningful spatial correlations.
    """
    rng = np.random.RandomState(seed)
    n, k = n_samples, 10
    d_rand, d_quad = 200, 20

    # Same generation as Phase 1/2
    centers = rng.randn(k, d_rand) * 2.0
    labels = rng.randint(0, k, size=n)
    X_rand = np.zeros((n, d_rand))
    for i in range(n):
        X_rand[i] = centers[labels[i]] + rng.randn(d_rand) * 0.5
    X_quad = X_rand[:, :d_quad] ** 2
    X_220 = np.concatenate([X_rand, X_quad], axis=1).astype(np.float32)

    # Embed into 3×32×32 = 3072 dimensions
    X_3072 = np.zeros((n, 3072), dtype=np.float32)
    X_3072[:, :220] = X_220

    # Reshape to image format
    X = torch.tensor(X_3072).view(n, 3, 32, 32)

    y = np.zeros((n, k), dtype=np.float32)
    y[np.arange(n), labels] = 1.0

    return X, torch.tensor(y)


# ============================================================
# CORE MEASUREMENT FUNCTIONS (protocol-matched)
# ============================================================

def make_model(model_type, seed):
    """Create model with deterministic initialization."""
    torch.manual_seed(seed)
    if model_type == "mlp_cifar":
        return LargeMLP(input_dim=3072, hidden_dim=85, output_dim=10)
    elif model_type == "cnn_synthetic":
        return SmallCNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def clone_perturbed(model, eps, seed):
    """Clone and perturb along unit-norm random direction (matches Experiment K)."""
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
    """Power iteration for top Hessian eigenvalue (matches Experiment K)."""
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


def correlation_dimension(traj, seed):
    """Grassberger-Procaccia D₂ (matches Experiment K exactly)."""
    if len(traj) > 2000:
        traj_sub = traj[np.random.RandomState(seed).choice(
            len(traj), 2000, replace=False)]
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
# SINGLE SEED RUN
# ============================================================

def run_single_seed(model_type, seed, lr, X, y, X_eval, device,
                    n_steps=5000, eps=1e-5):
    """
    Train model at given lr, recording all measurements.
    Protocol matches Experiment K / cnn_seeds_v2.py.
    """
    criterion = nn.MSELoss()

    model = make_model(model_type, seed).to(device)
    perturbed = clone_perturbed(model, eps, seed).to(device)

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

        # Record outputs every 10 steps
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

        # Train perturbed
        perturbed.zero_grad()
        loss2 = criterion(perturbed(X), y)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        # Sharpness every 100 steps (after step 0)
        if t > 0 and t % 100 == 0:
            sharp = compute_sharpness(model, X, y, criterion, n_iter=15)
            sharp_rec.append(sharp)

    # Lyapunov exponent
    log_d = np.log(distances + 1e-30)
    start = int(n_steps * 0.2)
    end = int(n_steps * 0.8)
    lyap = stats.linregress(np.arange(start, end), log_d[start:end])[0]

    # PCA + correlation dimension on output trajectory
    outputs = np.array(outputs_rec)
    traj_start = len(outputs) // 5
    traj = outputs[traj_start:].reshape(len(outputs) - traj_start, -1)

    centered = traj - traj.mean(axis=0)
    try:
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        var_exp = (sv ** 2) / (sv ** 2).sum() * 100
        pc1 = var_exp[0]
        pc2 = var_exp[1] if len(var_exp) > 1 else 0.0
    except:
        pc1, pc2 = 100.0, 0.0

    cd = correlation_dimension(traj, seed)

    return {
        'lyapunov': float(lyap),
        'corr_dim': float(cd),
        'pc1': float(pc1),
        'pc2': float(pc2),
        'sharpness_series': [float(s) for s in sharp_rec],
        'grad_norm_series': [float(g) for g in gn_rec[::10]],
        'loss_series': [float(l) for l in losses_rec[::10]],
    }


# ============================================================
# WARMUP: FIND EoS THRESHOLD
# ============================================================

def find_eos_threshold(model_type, X, y, device, warmup_lr=0.01,
                       warmup_steps=1000):
    """
    Train briefly to find λ_max, compute 2/λ_max as EoS threshold.
    Matches Experiment K protocol.
    """
    torch.manual_seed(0)
    model = make_model(model_type, 0).to(device)
    criterion = nn.MSELoss()

    print(f"  Warmup: {warmup_steps} steps at lr={warmup_lr}")
    for t in range(warmup_steps):
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= warmup_lr * p.grad
        if t % 200 == 0:
            print(f"    step {t}: loss = {loss.item():.4f}")

    lam_max = compute_sharpness(model, X, y, criterion, n_iter=15)
    lr_eos = 2.0 / lam_max
    print(f"  λ_max = {lam_max:.4f}, EoS threshold = {lr_eos:.6f}")

    return lam_max, lr_eos


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(model_type, X, y, device, seeds, n_steps=5000,
                   quick=False):
    """
    Run cross-experiment for given model type and data.
    """
    n_params = make_model(model_type, 0).count_params()
    print(f"\n{'=' * 60}")
    print(f"CROSS-EXPERIMENT: {model_type.upper()}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Data shape: {list(X.shape)}")
    print(f"  Seeds: {seeds}")
    print(f"{'=' * 60}")

    X_dev = X.to(device)
    y_dev = y.to(device)

    # Find EoS threshold
    lam_max, lr_eos = find_eos_threshold(model_type, X_dev, y_dev, device)

    # Learning rate fractions — focused sweep around the interesting region
    if quick:
        lr_fractions = [0.05, 0.15, 0.30, 0.50, 0.90]
    else:
        lr_fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                        0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    test_lrs = [frac * lr_eos for frac in lr_fractions]

    # Eval subset
    torch.manual_seed(0)
    n_eval = min(100, X.shape[0])
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X_dev[eval_idx]

    results = {
        'experiment': model_type,
        'lam_max': float(lam_max),
        'lr_eos': float(lr_eos),
        'lr_fractions': lr_fractions,
        'test_lrs': [float(lr) for lr in test_lrs],
        'n_params': n_params,
        'n_samples': int(X.shape[0]),
        'data_shape': list(X.shape),
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
            print(f"\n[{done}/{total_runs}] {frac:.0%} EoS, "
                  f"η={lr:.6f}, seed={seed} "
                  f"ETA: {eta / 60:.1f}min", flush=True)

            r = run_single_seed(
                model_type=model_type, seed=seed, lr=lr,
                X=X_dev, y=y_dev, X_eval=X_eval,
                device=device, n_steps=n_steps, eps=1e-5,
            )

            for k in results[lr_key]:
                results[lr_key][k].append(r[k])

            print(f"  → λ={r['lyapunov']:+.6f}, D₂={r['corr_dim']:.2f}, "
                  f"PC1={r['pc1']:.1f}%, PC2={r['pc2']:.1f}%")

        # Per-LR summary
        lyaps = results[lr_key]['lyapunov']
        d2s = [d for d in results[lr_key]['corr_dim']
               if d is not None and not np.isnan(d)]
        print(f"\n  → {frac:.0%} EoS summary: "
              f"λ = {np.mean(lyaps):+.6f} ± {np.std(lyaps):.6f}, "
              f"D₂ = {np.nanmean(d2s):.3f} ± {np.nanstd(d2s):.3f}")

    return results


# ============================================================
# SERIALIZATION AND OUTPUT
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


def detect_platform():
    """Auto-detect compute platform."""
    if os.path.isdir("/kaggle/working"):
        return "kaggle"
    elif os.path.isdir("/teamspace/studios/this_studio"):
        return "lightning"
    elif os.path.isdir("/content/drive"):
        return "colab"
    else:
        return "local"


def get_output_dir():
    """Get the primary output directory for this platform."""
    platform = detect_platform()
    if platform == "kaggle":
        return "/kaggle/working/results"
    elif platform == "lightning":
        return "/teamspace/studios/this_studio/chaos_research/results"
    else:
        return "results"


def save_results(results, output_dir=None):
    if output_dir is None:
        output_dir = get_output_dir()

    os.makedirs(output_dir, exist_ok=True)
    exp_name = results.get('experiment', 'unknown')
    seeds_str = "_".join(str(s) for s in results.get('seeds_run', []))
    fname = f"cross_{exp_name}_seeds_{seeds_str}.json"
    path = os.path.join(output_dir, fname)
    with open(path, 'w') as f:
        json.dump(_serialize(results), f, indent=2)
    print(f"\nSaved → {path}")

    platform = detect_platform()

    # On Lightning.ai, also save to local ./results/ as backup
    if platform == "lightning":
        local_dir = "results"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, fname)
        with open(local_path, 'w') as f:
            json.dump(_serialize(results), f, indent=2)
        print(f"Backup → {local_path}")

    # On Kaggle, also save to local ./results/ (visible in output tab)
    if platform == "kaggle":
        local_dir = "results"
        if os.path.abspath(local_dir) != os.path.abspath(output_dir):
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, fname)
            with open(local_path, 'w') as f:
                json.dump(_serialize(results), f, indent=2)

    return path


def print_summary(results):
    """Print a clean summary table."""
    exp = results.get('experiment', '?')
    fracs = results['lr_fractions']
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {exp.upper()}")
    print(f"  Params: {results['n_params']:,}")
    print(f"  λ_max = {results['lam_max']:.4f}, "
          f"EoS = {results['lr_eos']:.6f}")
    print(f"{'=' * 60}")
    print(f"{'frac':>6s}  {'η':>10s}  {'λ mean':>10s}  "
          f"{'D₂ mean':>8s}  {'D₂ std':>8s}  {'PC1':>6s}")
    print("-" * 60)
    for li, frac in enumerate(fracs):
        r = results[f"lr_{li}"]
        lyaps = np.array([v for v in r['lyapunov'] if v is not None])
        d2s = np.array([v for v in r['corr_dim']
                        if v is not None and not np.isnan(v)])
        pc1s = np.array([v for v in r['pc1'] if v is not None])
        lr_val = results['test_lrs'][li]
        print(f"  {frac:4.0%}  {lr_val:10.6f}  {lyaps.mean():+10.6f}  "
              f"{np.nanmean(d2s):8.3f}  {np.nanstd(d2s):8.3f}  "
              f"{pc1s.mean():5.1f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-experiments: isolate architecture vs. data effects")
    parser.add_argument("--mlp-cifar", action="store_true",
                        help="Experiment A: Large MLP on CIFAR-10")
    parser.add_argument("--cnn-synthetic", action="store_true",
                        help="Experiment B: CNN on synthetic data")
    parser.add_argument("--both", action="store_true",
                        help="Run both experiments")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (default: 3)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--quick", action="store_true",
                        help="Fewer LRs for quick testing")
    args = parser.parse_args()

    if not (args.mlp_cifar or args.cnn_synthetic or args.both):
        parser.print_help()
        print("\nExamples:")
        print("  python -u cross_experiments.py --both --seeds 3")
        print("  python -u cross_experiments.py --mlp-cifar --seeds 3")
        print("  python -u cross_experiments.py --both --seeds 1 --quick")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    platform = detect_platform()
    output_dir = get_output_dir()
    print(f"Device: {device}")
    print(f"Platform: {platform}")
    print(f"Output dir: {output_dir}")

    seeds = list(range(args.seeds))
    all_results = {}

    # ---- Experiment A: Large MLP on CIFAR-10 ----
    if args.mlp_cifar or args.both:
        print("\n" + "=" * 60)
        print("LOADING CIFAR-10 (flattened for MLP)...")
        X_flat, y = load_cifar10_flat(n_samples=2000, seed=42)
        print(f"  X shape: {list(X_flat.shape)}, y shape: {list(y.shape)}")

        m = LargeMLP()
        print(f"  LargeMLP params: {m.count_params():,}")

        results_a = run_experiment(
            "mlp_cifar", X_flat, y, device, seeds,
            n_steps=args.steps, quick=args.quick
        )
        save_results(results_a)
        print_summary(results_a)
        all_results['mlp_cifar'] = results_a

    # ---- Experiment B: CNN on synthetic data ----
    if args.cnn_synthetic or args.both:
        print("\n" + "=" * 60)
        print("GENERATING SYNTHETIC DATA (as 3×32×32 images)...")
        X_syn, y_syn = generate_synthetic_images(n_samples=2000, seed=42)
        print(f"  X shape: {list(X_syn.shape)}, y shape: {list(y_syn.shape)}")

        m = SmallCNN()
        print(f"  SmallCNN params: {m.count_params():,}")

        results_b = run_experiment(
            "cnn_synthetic", X_syn, y_syn, device, seeds,
            n_steps=args.steps, quick=args.quick
        )
        save_results(results_b)
        print_summary(results_b)
        all_results['cnn_synthetic'] = results_b

    # ---- Comparison ----
    if len(all_results) == 2:
        print("\n" + "=" * 60)
        print("COMPARISON: Architecture vs. Data")
        print("=" * 60)
        print(f"\nOriginal results (for reference):")
        print(f"  MLP (14K) on synthetic:  D₂ ≈ 0.9")
        print(f"  CNN (269K) on CIFAR-10:  D₂ ≈ 3.67")
        print(f"\nCross-experiment results:")

        # Find 30% EoS result for each
        for name, res in all_results.items():
            fracs = res['lr_fractions']
            # Find closest to 0.30
            idx_30 = min(range(len(fracs)),
                         key=lambda i: abs(fracs[i] - 0.30))
            r = res[f"lr_{idx_30}"]
            d2s = [d for d in r['corr_dim']
                   if d is not None and not np.isnan(d)]
            lyaps = r['lyapunov']
            frac = fracs[idx_30]
            print(f"  {name} at {frac:.0%} EoS: "
                  f"D₂ = {np.nanmean(d2s):.2f} ± {np.nanstd(d2s):.2f}, "
                  f"λ = {np.mean(lyaps):+.6f}")

        print(f"\nInterpretation guide:")
        print(f"  If MLP+CIFAR D₂ ≈ 1 and CNN+synthetic D₂ ≈ 1:")
        print(f"    → BOTH architecture AND data needed")
        print(f"  If MLP+CIFAR D₂ >> 1:")
        print(f"    → Data complexity alone sufficient")
        print(f"  If CNN+synthetic D₂ >> 1:")
        print(f"    → CNN architecture alone sufficient")

    print("\nDone.")


if __name__ == "__main__":
    main()
