"""
Chaos Onset — Phase 3: Architecture Scaling
=============================================

The critical question from Phase 2: D₂ ≈ 0.9 everywhere in the 14K-parameter
MLP. Is that a property of gradient descent, or of the architecture being
too small? Phase 3 tests whether wider, deeper, or differently-activated
networks produce D₂ > 1 — the threshold for multi-dimensional dynamics.

Four experiments:
  G) Width scaling: hidden=50, 100, 200, 400 (same depth)
  H) Depth scaling: 2, 3, 4, 5 hidden layers (same width)
  I) ReLU activation: same architecture, different nonlinearity
  J) Combined: best width × best depth × both activations
  K) CIFAR-10 CNN at near-EoS learning rates (the critical test)

Experiment K is the strongest test of the torus hypothesis. It uses
real data, a convolutional architecture, and learning rates near the
EoS threshold where Cohen et al. documented sharpness oscillations.
Those oscillations are the candidate for coupled oscillatory modes
that could produce toroidal geometry (D₂ > 1).

Each experiment measures:
  - 20-seed Lyapunov curve (does the chaos window shift?)
  - PC1/PC2 variance ratio (does off-axis dynamics grow?)
  - Correlation dimension (does D₂ cross 1?)
  - Sharpness spectral flatness (does the transition sharpen?)

USAGE:
    python phase3_experiments.py --width --seeds 5              # width scaling
    python phase3_experiments.py --depth --seeds 5              # depth scaling  
    python phase3_experiments.py --relu --seeds 5               # ReLU comparison
    python phase3_experiments.py --cifar --seeds 3              # CIFAR-10 near EoS
    python phase3_experiments.py --all --seeds 5                # everything
    python phase3_experiments.py --plot-only

REQUIREMENTS:
    pip install torch torchvision numpy matplotlib scipy
"""

import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from scipy import stats, signal


# ============================================================
# DATA (identical to Phase 1/2)
# ============================================================

DEFAULT_CONFIG = {
    "n_samples": 2000, "n_classes": 10, "n_random_features": 200,
    "n_quadratic_features": 20, "data_seed": 42,
    "perturbation_eps": 1e-5,
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


# ============================================================
# FLEXIBLE MODEL (variable width, depth, activation)
# ============================================================

class FlexMLP(nn.Module):
    """MLP with configurable width, depth, and activation."""
    def __init__(self, input_dim=220, hidden_dim=50, output_dim=10,
                 n_hidden_layers=2, activation="tanh"):
        super().__init__()
        self.act_fn = torch.tanh if activation == "tanh" else torch.relu

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))
        return self.layers[-1](x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def make_flex_model(input_dim, hidden_dim, output_dim, n_hidden_layers, activation, seed):
    torch.manual_seed(seed)
    return FlexMLP(input_dim, hidden_dim, output_dim, n_hidden_layers, activation)


def clone_model_perturbed(model, eps, seed):
    """Clone and perturb along unit-norm random direction."""
    import copy
    clone = copy.deepcopy(model)
    rng = torch.Generator()
    rng.manual_seed(seed + 999999)
    flat_params = []
    for p in clone.parameters():
        flat_params.append(p.data.view(-1))
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
# MEASUREMENTS
# ============================================================

def compute_lyapunov(model_cfg, lr, seed, X, y, eps=1e-5, n_steps=5000):
    """Compute function-space Lyapunov exponent for a flexible architecture."""
    device = X.device
    model = make_flex_model(**model_cfg, seed=seed).to(device)
    perturbed = clone_model_perturbed(model, eps, seed).to(device)
    criterion = nn.MSELoss()

    distances = np.zeros(n_steps)
    for t in range(n_steps):
        with torch.no_grad():
            d = torch.norm(model(X) - perturbed(X)).item()
            distances[t] = d

        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        perturbed.zero_grad()
        loss2 = criterion(perturbed(X), y)
        loss2.backward()
        with torch.no_grad():
            for p in perturbed.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

    log_d = np.log(distances + 1e-30)
    start = int(n_steps * 0.2)
    end = int(n_steps * 0.8)
    slope, _, _, _, _ = stats.linregress(np.arange(start, end), log_d[start:end])
    return slope, distances


def compute_trajectory_stats(model_cfg, lr, seed, X, y, n_steps=5000,
                              n_eval=100, output_every=5):
    """Record output trajectory and compute PCA + correlation dimension."""
    device = X.device
    model = make_flex_model(**model_cfg, seed=seed).to(device)
    criterion = nn.MSELoss()

    torch.manual_seed(0)
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    outputs = []
    for t in range(n_steps):
        if t % output_every == 0:
            with torch.no_grad():
                outputs.append(model(X_eval).cpu().numpy())

        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

    outputs = np.array(outputs)
    start = len(outputs) // 5
    traj = outputs[start:].reshape(len(outputs) - start, -1)

    # PCA
    centered = traj - traj.mean(axis=0)
    try:
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        var_exp = (sv**2) / (sv**2).sum()
    except:
        var_exp = np.zeros(20)

    # Correlation dimension
    if len(traj) > 2000:
        idx = np.random.RandomState(seed).choice(len(traj), 2000, replace=False)
        traj_sub = traj[idx]
    else:
        traj_sub = traj

    n = len(traj_sub)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(traj_sub[i] - traj_sub[j]))
    dists = np.array(dists)

    if len(dists) > 0:
        log_eps = np.linspace(np.log(np.percentile(dists, 1) + 1e-15),
                              np.log(np.percentile(dists, 95)), 20)
        log_C = [np.log(max(np.sum(dists < np.exp(le)) / (n*(n-1)/2), 1e-30)) for le in log_eps]
        log_C = np.array(log_C)
        s, e = 4, 16
        corr_dim = stats.linregress(log_eps[s:e], log_C[s:e])[0]
    else:
        corr_dim = float('nan')

    return {"var_explained": var_exp[:20], "corr_dim": corr_dim}


def compute_sharpness_spectrum(model_cfg, lr, seed, X, y, n_steps=10000,
                                sharpness_every=50):
    """Train and record sharpness time series, compute spectral flatness."""
    device = X.device
    model = make_flex_model(**model_cfg, seed=seed).to(device)
    criterion = nn.MSELoss()

    sharpness_vals = []
    for t in range(n_steps):
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        if t % sharpness_every == 0 and t > 0:
            # Quick power iteration
            v = [torch.randn_like(p) for p in model.parameters()]
            v_norm = sum((vi**2).sum() for vi in v).sqrt()
            v = [vi / v_norm for vi in v]
            eigenvalue = 0.0
            for _ in range(20):
                model.zero_grad()
                lo = criterion(model(X), y)
                grads = torch.autograd.grad(lo, model.parameters(), create_graph=True)
                Hv = sum((g * vi).sum() for g, vi in zip(grads, v))
                hv = torch.autograd.grad(Hv, model.parameters())
                eigenvalue = sum((h * vi).sum().item() for h, vi in zip(hv, v))
                hn = sum((h**2).sum() for h in hv).sqrt().item()
                if hn < 1e-12: break
                v = [h.detach() / hn for h in hv]
            sharpness_vals.append(abs(eigenvalue))

    sh = np.array(sharpness_vals)
    start = len(sh) // 5
    sh_sig = sh[start:]
    if len(sh_sig) > 10:
        sh_det = signal.detrend(sh_sig, type='linear')
        _, psd = signal.welch(sh_det, fs=1.0/sharpness_every,
                              nperseg=min(64, len(sh_det)//2))
        psd_pos = psd[psd > 0]
        flatness = np.exp(np.mean(np.log(psd_pos))) / np.mean(psd_pos) if len(psd_pos) > 0 else 0
    else:
        flatness = float('nan')

    return {"sharpness_flatness": flatness, "final_sharpness": sh[-1] if len(sh) > 0 else 0}


# ============================================================
# EXPERIMENT RUNNERS
# ============================================================

def run_scaling_experiment(name, architectures, X, y, config,
                           n_seeds=5, n_lyap_lrs=20, n_steps=5000):
    """
    Generic runner for architecture scaling experiments.
    
    architectures: list of dicts with keys matching FlexMLP constructor
                   plus a 'label' key for display
    """
    lyap_lrs = np.linspace(0.005, 0.15, n_lyap_lrs)
    # Subset of LRs for expensive measurements
    detail_lrs = [0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120]

    print("=" * 60)
    print(f"PHASE 3: {name}")
    print(f"  Architectures: {len(architectures)}")
    print(f"  Lyapunov: {n_seeds} seeds × {n_lyap_lrs} LRs")
    print(f"  Detail (PCA/dim/sharpness): {n_seeds} seeds × {len(detail_lrs)} LRs")
    print("=" * 60)

    device = X.device
    all_results = {}

    for arch in architectures:
        label = arch.pop('label')
        model_cfg = arch
        n_params = make_flex_model(**model_cfg, seed=0).count_params()
        print(f"\n--- {label} ({n_params} params) ---")

        arch_results = {
            "label": label, "n_params": n_params, "model_cfg": str(model_cfg),
            "lyap_lrs": lyap_lrs.tolist(), "detail_lrs": detail_lrs,
        }

        # 1. Lyapunov sweep
        print(f"  Lyapunov sweep ({n_seeds} seeds × {n_lyap_lrs} LRs)...")
        all_lyaps = np.zeros((n_seeds, n_lyap_lrs))
        total = n_seeds * n_lyap_lrs
        done = 0
        t0 = time.time()
        for s in range(n_seeds):
            for j, lr in enumerate(lyap_lrs):
                done += 1
                lyap, _ = compute_lyapunov(model_cfg, lr, s, X, y,
                                           eps=config["perturbation_eps"],
                                           n_steps=n_steps)
                all_lyaps[s, j] = lyap
                if done % 20 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(f"    [{done}/{total}] lr={lr:.3f} λ={lyap:+.6f} ETA: {eta:.0f}s")

        arch_results["all_lyaps"] = all_lyaps
        arch_results["mean_lyaps"] = all_lyaps.mean(axis=0)
        arch_results["frac_chaotic"] = (all_lyaps > 0).mean(axis=0)

        # Chaos window stats
        mean_l = all_lyaps.mean(axis=0)
        pos_mask = mean_l > 0
        peak_idx = np.argmax(mean_l)
        arch_results["peak_lr"] = float(lyap_lrs[peak_idx])
        arch_results["peak_lyap"] = float(mean_l[peak_idx])
        pos_lrs = lyap_lrs[pos_mask]
        if len(pos_lrs) > 0:
            arch_results["window"] = [float(pos_lrs[0]), float(pos_lrs[-1])]
        else:
            arch_results["window"] = None
        print(f"    Peak: η={lyap_lrs[peak_idx]:.3f}, λ={mean_l[peak_idx]:+.6f}")

        # 2. Trajectory stats at detail LRs
        print(f"  Trajectory stats ({n_seeds} seeds × {len(detail_lrs)} LRs)...")
        corr_dims = {}
        pca_pc1 = {}
        pca_pc2 = {}
        for lr in detail_lrs:
            dims, p1s, p2s = [], [], []
            for s in range(n_seeds):
                ts = compute_trajectory_stats(model_cfg, lr, s, X, y, n_steps=n_steps)
                dims.append(ts["corr_dim"])
                p1s.append(ts["var_explained"][0] * 100)
                p2s.append(ts["var_explained"][1] * 100 if len(ts["var_explained"]) > 1 else 0)
            corr_dims[lr] = dims
            pca_pc1[lr] = p1s
            pca_pc2[lr] = p2s
            print(f"    η={lr:.3f}: D₂={np.nanmean(dims):.3f}, PC1={np.mean(p1s):.1f}%, PC2={np.mean(p2s):.1f}%")

        arch_results["corr_dims"] = {str(k): v for k, v in corr_dims.items()}
        arch_results["pca_pc1"] = {str(k): v for k, v in pca_pc1.items()}
        arch_results["pca_pc2"] = {str(k): v for k, v in pca_pc2.items()}

        # 3. Sharpness spectral flatness at 2 LRs (expensive)
        print(f"  Sharpness spectrum (2 LRs)...")
        for lr in [0.005, 0.030]:
            sf = compute_sharpness_spectrum(model_cfg, lr, 0, X, y, n_steps=min(n_steps, 5000))
            arch_results[f"sharpness_flatness_lr{lr}"] = sf["sharpness_flatness"]
            arch_results[f"final_sharpness_lr{lr}"] = sf["final_sharpness"]
            print(f"    η={lr:.3f}: flatness={sf['sharpness_flatness']:.4f}, λ_max={sf['final_sharpness']:.2f}")

        # Restore label for next iteration
        model_cfg['label'] = label
        all_results[label] = arch_results

    return all_results


def run_width_scaling(config, n_seeds=5, n_steps=5000):
    X, y = generate_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X, y = X.to(device), y.to(device)

    architectures = [
        {"input_dim": 220, "hidden_dim": 50,  "output_dim": 10, "n_hidden_layers": 2, "activation": "tanh", "label": "h=50 (baseline)"},
        {"input_dim": 220, "hidden_dim": 100, "output_dim": 10, "n_hidden_layers": 2, "activation": "tanh", "label": "h=100"},
        {"input_dim": 220, "hidden_dim": 200, "output_dim": 10, "n_hidden_layers": 2, "activation": "tanh", "label": "h=200"},
        {"input_dim": 220, "hidden_dim": 400, "output_dim": 10, "n_hidden_layers": 2, "activation": "tanh", "label": "h=400"},
    ]

    results = run_scaling_experiment("WIDTH SCALING", architectures, X, y, config,
                                     n_seeds=n_seeds, n_steps=n_steps)
    import json
    os.makedirs("results", exist_ok=True)
    with open("results/width_scaling.json", "w") as f:
        json.dump({k: _serialize(v) for k, v in results.items()}, f, indent=2)
    print(f"\n  Saved → results/width_scaling.json")
    return results


def run_depth_scaling(config, n_seeds=5, n_steps=5000):
    X, y = generate_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X, y = X.to(device), y.to(device)

    architectures = [
        {"input_dim": 220, "hidden_dim": 50, "output_dim": 10, "n_hidden_layers": 2, "activation": "tanh", "label": "depth=2 (baseline)"},
        {"input_dim": 220, "hidden_dim": 50, "output_dim": 10, "n_hidden_layers": 3, "activation": "tanh", "label": "depth=3"},
        {"input_dim": 220, "hidden_dim": 50, "output_dim": 10, "n_hidden_layers": 4, "activation": "tanh", "label": "depth=4"},
        {"input_dim": 220, "hidden_dim": 50, "output_dim": 10, "n_hidden_layers": 5, "activation": "tanh", "label": "depth=5"},
    ]

    results = run_scaling_experiment("DEPTH SCALING", architectures, X, y, config,
                                     n_seeds=n_seeds, n_steps=n_steps)
    import json
    os.makedirs("results", exist_ok=True)
    with open("results/depth_scaling.json", "w") as f:
        json.dump({k: _serialize(v) for k, v in results.items()}, f, indent=2)
    print(f"\n  Saved → results/depth_scaling.json")
    return results


def run_relu_comparison(config, n_seeds=5, n_steps=5000):
    X, y = generate_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X, y = X.to(device), y.to(device)

    architectures = [
        {"input_dim": 220, "hidden_dim": 50, "output_dim": 10, "n_hidden_layers": 2, "activation": "tanh", "label": "tanh (baseline)"},
        {"input_dim": 220, "hidden_dim": 50, "output_dim": 10, "n_hidden_layers": 2, "activation": "relu", "label": "relu"},
        {"input_dim": 220, "hidden_dim": 200, "output_dim": 10, "n_hidden_layers": 2, "activation": "relu", "label": "relu h=200"},
        {"input_dim": 220, "hidden_dim": 200, "output_dim": 10, "n_hidden_layers": 4, "activation": "relu", "label": "relu h=200 d=4"},
    ]

    results = run_scaling_experiment("RELU COMPARISON", architectures, X, y, config,
                                     n_seeds=n_seeds, n_steps=n_steps)
    import json
    os.makedirs("results", exist_ok=True)
    with open("results/relu_comparison.json", "w") as f:
        json.dump({k: _serialize(v) for k, v in results.items()}, f, indent=2)
    print(f"\n  Saved → results/relu_comparison.json")
    return results


def _serialize(obj):
    """Make results JSON-serializable."""
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
    return obj


# ============================================================
# PLOTTING
# ============================================================

def plot_scaling(json_path, title, fig_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json

    with open(json_path) as f:
        results = json.load(f)

    labels = list(results.keys())
    n_arch = len(labels)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_arch))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Lyapunov curves
    ax = axes[0, 0]
    for i, label in enumerate(labels):
        r = results[label]
        lrs = r["lyap_lrs"]
        mean_l = r["mean_lyaps"]
        ax.plot(lrs, mean_l, 'o-', ms=3, lw=1.5, color=colors[i],
                label=f'{label} ({r["n_params"]}p)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('η'); ax.set_ylabel('Mean Lyapunov exponent')
    ax.set_title('Chaos Window vs Architecture')
    ax.legend(fontsize=8)

    # Top-right: Fraction chaotic
    ax = axes[0, 1]
    for i, label in enumerate(labels):
        r = results[label]
        ax.plot(r["lyap_lrs"], r["frac_chaotic"], 'o-', ms=3, lw=1.5,
                color=colors[i], label=label)
    ax.axhline(0.5, color='k', ls='--', lw=0.5)
    ax.set_xlabel('η'); ax.set_ylabel('Fraction chaotic')
    ax.set_title('Fraction of Seeds with λ > 0')
    ax.legend(fontsize=8)

    # Bottom-left: Correlation dimension at detail LRs
    ax = axes[1, 0]
    for i, label in enumerate(labels):
        r = results[label]
        detail_lrs = sorted([float(k) for k in r["corr_dims"].keys()])
        means = [np.nanmean(r["corr_dims"][str(lr)]) for lr in detail_lrs]
        stds = [np.nanstd(r["corr_dims"][str(lr)]) for lr in detail_lrs]
        ax.errorbar(detail_lrs, means, yerr=stds, fmt='o-', ms=4, capsize=3,
                    color=colors[i], label=label)
    ax.axhline(1, color='gray', ls=':'); ax.axhline(2, color='gray', ls='--')
    ax.set_xlabel('η'); ax.set_ylabel('Correlation Dimension D₂')
    ax.set_title('Attractor Dimension'); ax.legend(fontsize=8)

    # Bottom-right: PC2 variance
    ax = axes[1, 1]
    for i, label in enumerate(labels):
        r = results[label]
        detail_lrs = sorted([float(k) for k in r["pca_pc2"].keys()])
        means = [np.mean(r["pca_pc2"][str(lr)]) for lr in detail_lrs]
        ax.plot(detail_lrs, means, 'o-', ms=4, lw=1.5, color=colors[i], label=label)
    ax.set_xlabel('η'); ax.set_ylabel('PC2 Variance (%)')
    ax.set_title('Off-Axis Dynamics (higher = more complex)'); ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{fig_name}.png", dpi=200, bbox_inches="tight")
    print(f"  Saved → figures/{fig_name}.png")
    plt.close()

    # Summary table
    print(f"\n  {'Architecture':>25s}  {'Params':>8s}  {'Peak η':>7s}  {'Peak λ':>10s}  {'Window':>15s}")
    for label in labels:
        r = results[label]
        w = f"[{r['window'][0]:.3f},{r['window'][1]:.3f}]" if r['window'] else "none"
        print(f"  {label:>25s}  {r['n_params']:>8d}  {r['peak_lr']:>7.3f}  {r['peak_lyap']:>+10.6f}  {w:>15s}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Architecture Scaling")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--width", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--cifar", action="store_true", help="Experiment K: CIFAR-10 CNN near EoS")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)

    if args.plot_only:
        for name, title, fig in [
            ("width_scaling", "Width Scaling", "width_scaling"),
            ("depth_scaling", "Depth Scaling", "depth_scaling"),
            ("relu_comparison", "ReLU Comparison", "relu_comparison"),
        ]:
            path = f"results/{name}.json"
            if os.path.exists(path):
                plot_scaling(path, title, fig)
        if os.path.exists("results/cifar10_eos.json"):
            plot_cifar10_eos()
        return

    run_any = args.all or args.width or args.depth or args.relu or args.cifar
    if not run_any:
        parser.print_help()
        print("\n  python phase3_experiments.py --width --seeds 3")
        print("  python phase3_experiments.py --cifar --seeds 3    # the critical test")
        print("  python phase3_experiments.py --all --seeds 5")
        return

    if args.all or args.width:
        run_width_scaling(config, n_seeds=args.seeds, n_steps=args.steps)
        plot_scaling("results/width_scaling.json", "Width Scaling", "width_scaling")

    if args.all or args.depth:
        run_depth_scaling(config, n_seeds=args.seeds, n_steps=args.steps)
        plot_scaling("results/depth_scaling.json", "Depth Scaling", "depth_scaling")

    if args.all or args.relu:
        run_relu_comparison(config, n_seeds=args.seeds, n_steps=args.steps)
        plot_scaling("results/relu_comparison.json", "ReLU Comparison", "relu_comparison")

    if args.all or args.cifar:
        run_cifar10_eos(n_seeds=args.seeds, n_steps=args.steps)
        plot_cifar10_eos()

    print("\nDone.")


# ============================================================
# EXPERIMENT K: CIFAR-10 CNN at Near-EoS Learning Rates
# ============================================================
# This is the strongest test of the torus hypothesis. Cohen et al.'s
# sharpness oscillations occur at η ≈ 2/λ_max. Those oscillations are
# the candidate for coupled oscillatory modes that could produce
# toroidal geometry. We need: real data, convolutional architecture,
# and learning rates near the EoS threshold.

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
    torch.manual_seed(seed)
    return SmallCNN()


def clone_cnn_perturbed(model, eps, seed):
    """Clone CNN and perturb along unit-norm random direction."""
    import copy
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


def compute_sharpness_cnn(model, X, y, criterion, n_iter=20):
    """Power iteration for CNN."""
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


def load_cifar10_subset(n_samples=2000, seed=42):
    """
    Load a subset of CIFAR-10 for full-batch training.
    Downloads automatically on first run.
    """
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Deterministic subset
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), n_samples, replace=False)

    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    X = torch.stack(images)  # (n_samples, 3, 32, 32)

    # One-hot labels for MSE loss (matching Phase 1 protocol)
    y = torch.zeros(n_samples, 10)
    for i, label in enumerate(labels):
        y[i, label] = 1.0

    return X, y


def run_cifar10_eos(n_seeds=5, n_steps=5000, n_samples=2000, eps=1e-5):
    """
    Experiment K: CIFAR-10 CNN at near-EoS learning rates.

    Protocol:
    1. Load CIFAR-10 subset (2000 images, full-batch)
    2. Train CNN briefly, compute λ_max to find EoS threshold
    3. Sweep learning rates from 0.1× to 1.0× of 2/λ_max
    4. At each LR: Lyapunov exponents, PCA/correlation dimension,
       sharpness time series
    5. Compare dynamics near EoS vs far below — does D₂ > 1 emerge?
    """
    print("=" * 60)
    print(f"EXPERIMENT K: CIFAR-10 CNN at Near-EoS Learning Rates")
    print(f"  {n_seeds} seeds, {n_steps} steps, {n_samples} images")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load data
    print("  Loading CIFAR-10...")
    X, y = load_cifar10_subset(n_samples=n_samples, seed=42)
    X, y = X.to(device), y.to(device)

    n_params = make_cnn(0).count_params()
    print(f"  CNN parameters: {n_params}")

    # Find EoS threshold
    print("  Finding EoS threshold (warmup + power iteration)...")
    torch.manual_seed(0)
    ref_model = make_cnn(0).to(device)
    criterion = nn.MSELoss()

    # Warmup at low lr
    warmup_lr = 0.01
    for t in range(1000):
        ref_model.zero_grad()
        loss = criterion(ref_model(X), y)
        loss.backward()
        with torch.no_grad():
            for p in ref_model.parameters():
                if p.grad is not None:
                    p -= warmup_lr * p.grad

    lam_max = compute_sharpness_cnn(ref_model, X, y, criterion, n_iter=50)
    lr_eos = 2.0 / lam_max
    print(f"  λ_max ≈ {lam_max:.4f}")
    print(f"  EoS threshold (2/λ_max) ≈ {lr_eos:.4f}")
    del ref_model

    # Learning rates: span from 0.1× to 1.0× EoS
    # Include some below chaos window and some near EoS
    lr_fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 0.95]
    test_lrs = [frac * lr_eos for frac in lr_fractions]
    print(f"  Test LRs (fraction of EoS): {lr_fractions}")
    print(f"  Test LRs (absolute): {[f'{lr:.4f}' for lr in test_lrs]}")

    # Subsample for output recording
    torch.manual_seed(0)
    n_eval = min(100, n_samples)
    eval_idx = torch.randperm(X.shape[0])[:n_eval]
    X_eval = X[eval_idx]

    results = {
        "lam_max": lam_max, "lr_eos": lr_eos,
        "lr_fractions": lr_fractions, "test_lrs": test_lrs,
        "n_params": n_params, "n_samples": n_samples,
    }

    total_runs = len(test_lrs) * n_seeds
    done = 0
    t0 = time.time()

    for li, (frac, lr) in enumerate(zip(lr_fractions, test_lrs)):
        lr_results = {
            "lyapunov": [], "corr_dim": [], "pc1": [], "pc2": [],
            "sharpness_series": [], "grad_norm_series": [], "loss_series": [],
        }

        for s in range(n_seeds):
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total_runs - done) if done > 1 else 0
            print(f"  [{done}/{total_runs}] {frac:.0%} EoS, η={lr:.4f}, seed={s} "
                  f"ETA: {eta:.0f}s", end="", flush=True)

            # --- Lyapunov exponent ---
            model = make_cnn(s).to(device)
            perturbed = clone_cnn_perturbed(model, eps, s).to(device)

            distances = np.zeros(n_steps)
            losses_rec = np.zeros(n_steps)
            gn_rec = np.zeros(n_steps)
            sharp_rec = []
            outputs_rec = []

            for t in range(n_steps):
                with torch.no_grad():
                    d = torch.norm(model(X_eval) - perturbed(X_eval)).item()
                    distances[t] = d

                # Train original
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

                # Sharpness every 100 steps
                if t % 100 == 0 and t > 0:
                    sharp = compute_sharpness_cnn(model, X, y, criterion, n_iter=15)
                    sharp_rec.append(sharp)

                # SGD update original
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

            # Lyapunov exponent
            log_d = np.log(distances + 1e-30)
            start, end = int(n_steps * 0.2), int(n_steps * 0.8)
            lyap = stats.linregress(np.arange(start, end), log_d[start:end])[0]
            lr_results["lyapunov"].append(lyap)

            # Trajectory dimension
            outputs_arr = np.array(outputs_rec)
            traj_start = len(outputs_arr) // 5
            traj = outputs_arr[traj_start:].reshape(len(outputs_arr) - traj_start, -1)

            centered = traj - traj.mean(axis=0)
            try:
                _, sv, _ = np.linalg.svd(centered, full_matrices=False)
                ve = (sv**2) / (sv**2).sum()
                lr_results["pc1"].append(float(ve[0] * 100))
                lr_results["pc2"].append(float(ve[1] * 100) if len(ve) > 1 else 0)
            except:
                lr_results["pc1"].append(100.0)
                lr_results["pc2"].append(0.0)
                ve = np.array([1.0])

            # Correlation dimension
            if len(traj) > 2000:
                traj_sub = traj[np.random.RandomState(s).choice(len(traj), 2000, replace=False)]
            else:
                traj_sub = traj
            n_pts = len(traj_sub)
            dists_cd = []
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    dists_cd.append(np.linalg.norm(traj_sub[i] - traj_sub[j]))
            dists_cd = np.array(dists_cd)
            if len(dists_cd) > 0:
                log_eps = np.linspace(np.log(np.percentile(dists_cd, 1) + 1e-15),
                                      np.log(np.percentile(dists_cd, 95)), 20)
                log_C = [np.log(max(np.sum(dists_cd < np.exp(le)) / (n_pts*(n_pts-1)/2), 1e-30))
                         for le in log_eps]
                cd = stats.linregress(log_eps[4:16], np.array(log_C)[4:16])[0]
            else:
                cd = float('nan')
            lr_results["corr_dim"].append(cd)

            # Store sharpness and grad norm series (subsampled)
            lr_results["sharpness_series"].append(sharp_rec)
            lr_results["grad_norm_series"].append(gn_rec[::10].tolist())
            lr_results["loss_series"].append(losses_rec[::10].tolist())

            del model, perturbed
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"  → λ={lyap:+.6f}, D₂={cd:.2f}, PC1={ve[0]*100:.1f}%, "
                  f"PC2={ve[1]*100:.1f}%" if len(ve) > 1 else f"  → λ={lyap:+.6f}")

        results[f"lr_{li}"] = lr_results

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT K SUMMARY")
    print("=" * 60)
    print(f"  {'frac_EoS':>8s}  {'η':>8s}  {'mean_λ':>10s}  {'D₂':>6s}  {'PC1%':>6s}  {'PC2%':>6s}")
    for li, (frac, lr) in enumerate(zip(lr_fractions, test_lrs)):
        r = results[f"lr_{li}"]
        ml = np.mean(r["lyapunov"])
        md = np.nanmean(r["corr_dim"])
        mp1 = np.mean(r["pc1"])
        mp2 = np.mean(r["pc2"])
        print(f"  {frac:8.0%}  {lr:8.4f}  {ml:+10.6f}  {md:6.2f}  {mp1:6.1f}  {mp2:6.1f}")

    # Check: does D₂ exceed 1 anywhere?
    max_d2 = max(np.nanmean(results[f"lr_{li}"]["corr_dim"]) for li in range(len(test_lrs)))
    max_pc2 = max(np.mean(results[f"lr_{li}"]["pc2"]) for li in range(len(test_lrs)))
    print(f"\n  Max D₂ across all LRs: {max_d2:.3f}")
    print(f"  Max PC2% across all LRs: {max_pc2:.1f}%")
    if max_d2 > 1.1:
        print(f"  *** D₂ > 1 DETECTED — multi-dimensional dynamics present ***")
    else:
        print(f"  D₂ ≤ 1 — dynamics remain one-dimensional at this architecture")

    # Save
    import json
    os.makedirs("results", exist_ok=True)
    with open("results/cifar10_eos.json", "w") as f:
        json.dump(_serialize(results), f, indent=2)
    print(f"\n  Saved → results/cifar10_eos.json")

    return results


def plot_cifar10_eos(json_path="results/cifar10_eos.json"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json

    with open(json_path) as f:
        results = json.load(f)

    lr_fractions = results["lr_fractions"]
    test_lrs = results["test_lrs"]
    lr_eos = results["lr_eos"]
    n_lrs = len(lr_fractions)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect per-LR means
    lyaps, dims, pc1s, pc2s = [], [], [], []
    lyap_stds, dim_stds = [], []
    for li in range(n_lrs):
        r = results[f"lr_{li}"]
        lyaps.append(np.mean(r["lyapunov"]))
        lyap_stds.append(np.std(r["lyapunov"]))
        dims.append(np.nanmean(r["corr_dim"]))
        dim_stds.append(np.nanstd(r["corr_dim"]))
        pc1s.append(np.mean(r["pc1"]))
        pc2s.append(np.mean(r["pc2"]))

    # Top-left: Lyapunov vs fraction of EoS
    ax = axes[0, 0]
    ax.errorbar(lr_fractions, lyaps, yerr=lyap_stds, fmt='ko-', ms=6, capsize=3, lw=2)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.axvline(1.0, color='orange', ls='--', lw=1.5, label='EoS threshold')
    ax.set_xlabel('Fraction of 2/λ_max')
    ax.set_ylabel('Lyapunov Exponent')
    ax.set_title('Chaos vs Distance to EoS')
    ax.legend()

    # Top-right: Correlation dimension
    ax = axes[0, 1]
    ax.errorbar(lr_fractions, dims, yerr=dim_stds, fmt='ko-', ms=6, capsize=3, lw=2)
    ax.axhline(1, color='gray', ls=':', lw=0.8, label='D=1')
    ax.axhline(2, color='gray', ls='--', lw=0.8, label='D=2')
    ax.axvline(1.0, color='orange', ls='--', lw=1.5)
    ax.set_xlabel('Fraction of 2/λ_max')
    ax.set_ylabel('Correlation Dimension D₂')
    ax.set_title('Does D₂ Cross 1 Near EoS?')
    ax.legend()

    # Bottom-left: PC1 and PC2
    ax = axes[1, 0]
    ax.plot(lr_fractions, pc2s, 'ko-', ms=6, lw=2, label='PC2%')
    ax.axvline(1.0, color='orange', ls='--', lw=1.5)
    ax.set_xlabel('Fraction of 2/λ_max')
    ax.set_ylabel('PC2 Variance (%)')
    ax.set_title('Off-Axis Dynamics Growth Near EoS')
    ax.legend()

    # Bottom-right: Sharpness time series at 2 extremes
    ax = axes[1, 1]
    for li, label_str in [(0, f'{lr_fractions[0]:.0%} EoS'),
                           (n_lrs - 1, f'{lr_fractions[-1]:.0%} EoS')]:
        r = results[f"lr_{li}"]
        if r["sharpness_series"] and len(r["sharpness_series"][0]) > 0:
            sh = r["sharpness_series"][0]  # first seed
            ax.plot(range(len(sh)), sh, '-', lw=1.5, label=label_str)
    ax.axhline(results["lam_max"], color='red', ls=':', lw=1, label=f'λ_max={results["lam_max"]:.1f}')
    ax.set_xlabel('Measurement index (every 100 steps)')
    ax.set_ylabel('Top Hessian Eigenvalue')
    ax.set_title('Sharpness Dynamics: Far vs Near EoS')
    ax.legend(fontsize=9)

    fig.suptitle(f'CIFAR-10 CNN ({results["n_params"]} params) at Near-EoS Learning Rates',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/cifar10_eos.png", dpi=200, bbox_inches="tight")
    print(f"  Saved → figures/cifar10_eos.png")
    plt.close()


if __name__ == "__main__":
    main()
