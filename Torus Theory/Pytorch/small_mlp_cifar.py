"""
Cross-Experiment C: Small MLP (14K) on CIFAR-10
=================================================

The original paper compares:
  - MLP (14K) on synthetic → D₂ ≈ 0.9
  - CNN (269K) on CIFAR-10 → D₂ ≈ 3.67

Cross-experiments A & B showed:
  - MLP (269K) on CIFAR-10 → D₂ ≈ 4.3 at 90% EoS (!)
  - CNN (269K) on synthetic → D₂ ≈ 1.0 everywhere

New question: Is the MLP-on-CIFAR result about data complexity or
parameter count? This script tests the original 14K MLP on CIFAR-10.

  If D₂ > 1 at high LR: data complexity alone drives the transition,
    regardless of parameter count
  If D₂ ≈ 0.9: you need sufficient parameters AND complex data

Architecture: 3072 → 50 → 50 → 10 (tanh), matching Phase 1 except
for input dimension (3072 instead of 220).
Params: 3072×50 + 50 + 50×50 + 50 + 50×10 + 10 = 156,660

Note: this is actually larger than the original 14K MLP (which had
220 inputs). With 3072-d CIFAR input, the first layer alone has 153K
params. So this tests "MLP structure + CIFAR data" but NOT at 14K.

For a true 14K-param MLP on CIFAR-10, we'd need width ~4, which is
too narrow to learn anything. Instead we test two sizes:
  - Width 50 (156K params): comparable to CNN, MLP structure
  - Width 4 (12K params): true small-param regime

Designed to run on CPU (Mac). No GPU needed.

USAGE:
    python -u small_mlp_cifar.py --seeds 3
    python -u small_mlp_cifar.py --seeds 1 --quick    # test first
    python -u small_mlp_cifar.py --seeds 3 --width 4  # tiny version
"""

import argparse, os, time, json, copy, warnings
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# ARCHITECTURE
# ============================================================

class SmallMLP(nn.Module):
    """
    2-hidden-layer tanh MLP matching Phase 1 structure,
    but accepting flattened CIFAR-10 input (3072-d).
    """
    def __init__(self, input_dim=3072, hidden_dim=50, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# DATA
# ============================================================

def load_cifar10_flat(n_samples=2000, seed=42):
    """Load CIFAR-10 subset as flattened vectors. Same subset as Experiment K."""
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
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

    X = torch.stack(images).view(n_samples, -1)  # (n, 3072)
    y = torch.zeros(n_samples, 10)
    for i, label in enumerate(labels):
        y[i, label] = 1.0

    return X, y


# ============================================================
# MEASUREMENT FUNCTIONS (protocol-matched to Experiment K)
# ============================================================

def clone_perturbed(model, eps, seed):
    """Clone and perturb along unit-norm random direction."""
    clone = copy.deepcopy(model)
    rng = torch.Generator()
    rng.manual_seed(seed + 999999)
    flat_params = [p.data.view(-1) for p in clone.parameters()]
    flat = torch.cat(flat_params)
    direction = torch.randn(flat.shape, generator=rng)
    direction = direction / direction.norm()
    offset = 0
    for p in clone.parameters():
        numel = p.numel()
        p.data += eps * direction[offset:offset + numel].view(p.shape)
        offset += numel
    return clone


def compute_sharpness(model, X, y, criterion, n_iter=15):
    """Power iteration for top Hessian eigenvalue."""
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
    """Grassberger-Procaccia D₂ (matches Experiment K)."""
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
    return stats.linregress(log_eps[4:16], log_C[4:16])[0]


# ============================================================
# SINGLE SEED RUN
# ============================================================

def run_single_seed(hidden_dim, seed, lr, X, y, X_eval,
                    n_steps=5000, eps=1e-5):
    """Train and measure. Protocol matches Experiment K."""
    criterion = nn.MSELoss()

    torch.manual_seed(seed)
    model = SmallMLP(input_dim=X.shape[1], hidden_dim=hidden_dim)
    perturbed = clone_perturbed(model, eps, seed)

    distances = np.zeros(n_steps)
    outputs_rec = []
    sharp_rec = []

    for t in range(n_steps):
        with torch.no_grad():
            d = torch.norm(model(X_eval) - perturbed(X_eval)).item()
            distances[t] = d

        if t % 10 == 0:
            with torch.no_grad():
                outputs_rec.append(model(X_eval).cpu().numpy())

        # Train original
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
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

        # Sharpness every 100 steps
        if t > 0 and t % 100 == 0:
            sharp = compute_sharpness(model, X, y, criterion, n_iter=15)
            sharp_rec.append(sharp)

        # Progress
        if t % 1000 == 0 and t > 0:
            print(f"    step {t}/{n_steps}, loss={loss.item():.4f}, "
                  f"dist={d:.2e}", flush=True)

    # Lyapunov
    log_d = np.log(distances + 1e-30)
    start = int(n_steps * 0.2)
    end = int(n_steps * 0.8)
    lyap = stats.linregress(np.arange(start, end), log_d[start:end])[0]

    # PCA + D₂
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
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-experiment C: Small MLP on CIFAR-10 (CPU)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--width", type=int, default=50,
                        help="Hidden layer width (default: 50 → 156K params; "
                             "try 4 → 12K params)")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer LRs for quick test")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    hidden_dim = args.width

    print(f"{'=' * 60}")
    print(f"CROSS-EXPERIMENT C: Small MLP on CIFAR-10")
    print(f"  Hidden dim: {hidden_dim}")
    m = SmallMLP(input_dim=3072, hidden_dim=hidden_dim)
    n_params = m.count_params()
    print(f"  Parameters: {n_params:,}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: CPU")
    print(f"{'=' * 60}")

    # Load data
    print("\nLoading CIFAR-10...")
    X, y = load_cifar10_flat(n_samples=2000, seed=42)
    print(f"  X: {list(X.shape)}, y: {list(y.shape)}")

    # Find EoS threshold
    print("\nFinding EoS threshold (1000 warmup steps at lr=0.01)...")
    torch.manual_seed(0)
    warmup_model = SmallMLP(input_dim=3072, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    for t in range(1000):
        warmup_model.zero_grad()
        loss = criterion(warmup_model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in warmup_model.parameters():
                if p.grad is not None:
                    p -= 0.01 * p.grad
        if t % 200 == 0:
            print(f"  step {t}: loss = {loss.item():.4f}")

    lam_max = compute_sharpness(warmup_model, X, y, criterion, n_iter=15)
    lr_eos = 2.0 / lam_max
    print(f"  λ_max = {lam_max:.4f}, EoS = {lr_eos:.6f}")
    del warmup_model

    # Learning rates
    if args.quick:
        lr_fractions = [0.05, 0.15, 0.30, 0.50, 0.90]
    else:
        lr_fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                        0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    test_lrs = [frac * lr_eos for frac in lr_fractions]

    # Eval subset
    torch.manual_seed(0)
    n_eval = min(100, X.shape[0])
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    results = {
        'experiment': f'small_mlp_cifar_w{hidden_dim}',
        'lam_max': float(lam_max),
        'lr_eos': float(lr_eos),
        'lr_fractions': lr_fractions,
        'test_lrs': [float(lr) for lr in test_lrs],
        'n_params': n_params,
        'hidden_dim': hidden_dim,
        'n_samples': 2000,
        'seeds_run': seeds,
    }

    total_runs = len(lr_fractions) * len(seeds)
    done = 0
    t0 = time.time()

    for li, (frac, lr) in enumerate(zip(lr_fractions, test_lrs)):
        lr_key = f"lr_{li}"
        results[lr_key] = {
            'lyapunov': [], 'corr_dim': [], 'pc1': [], 'pc2': [],
            'sharpness_series': [],
        }

        for seed in seeds:
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total_runs - done) if done > 1 else 0
            print(f"\n[{done}/{total_runs}] {frac:.0%} EoS, "
                  f"η={lr:.6f}, seed={seed}  "
                  f"ETA: {eta / 60:.1f}min", flush=True)

            r = run_single_seed(
                hidden_dim=hidden_dim, seed=seed, lr=lr,
                X=X, y=y, X_eval=X_eval,
                n_steps=args.steps, eps=1e-5,
            )

            for k in results[lr_key]:
                results[lr_key][k].append(r[k])

            print(f"  → λ={r['lyapunov']:+.6f}, D₂={r['corr_dim']:.2f}, "
                  f"PC1={r['pc1']:.1f}%, PC2={r['pc2']:.1f}%")

        # Summary
        lyaps = results[lr_key]['lyapunov']
        d2s = [d for d in results[lr_key]['corr_dim']
               if d is not None and not np.isnan(d)]
        print(f"\n  → {frac:.0%} EoS: "
              f"λ = {np.mean(lyaps):+.6f} ± {np.std(lyaps):.6f}, "
              f"D₂ = {np.nanmean(d2s):.3f} ± {np.nanstd(d2s):.3f}")

    # Save
    os.makedirs("results", exist_ok=True)
    seeds_str = "_".join(str(s) for s in seeds)
    fname = f"results/cross_small_mlp_cifar_w{hidden_dim}_seeds_{seeds_str}.json"

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

    with open(fname, 'w') as f:
        json.dump(_ser(results), f, indent=2)
    print(f"\nSaved → {fname}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: MLP (w={hidden_dim}, {n_params:,} params) on CIFAR-10")
    print(f"  λ_max = {lam_max:.4f}, EoS = {lr_eos:.6f}")
    print(f"{'=' * 60}")
    print(f"{'frac':>6s}  {'η':>10s}  {'λ':>10s}  {'D₂':>8s}  {'PC1':>6s}")
    print("-" * 48)
    for li, frac in enumerate(lr_fractions):
        r = results[f"lr_{li}"]
        lr_val = test_lrs[li]
        lyaps = np.array(r['lyapunov'])
        d2s = np.array([d for d in r['corr_dim']
                        if d is not None and not np.isnan(d)])
        pc1s = np.array(r['pc1'])
        print(f"  {frac:4.0%}  {lr_val:10.6f}  {np.mean(lyaps):+10.6f}  "
              f"{np.nanmean(d2s):8.3f}  {np.mean(pc1s):5.1f}%")

    print(f"\nFor reference:")
    print(f"  Original MLP (14K) + synthetic:  D₂ ≈ 0.9 everywhere")
    print(f"  CNN (269K) + CIFAR-10 at 30%%:    D₂ ≈ 3.67")
    print(f"  MLP (269K) + CIFAR-10 at 90%%:    D₂ ≈ 4.3 (1 seed)")


if __name__ == "__main__":
    main()
