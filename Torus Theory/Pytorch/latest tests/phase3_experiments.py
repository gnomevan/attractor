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

Each experiment measures:
  - 20-seed Lyapunov curve (does the chaos window shift?)
  - PC1/PC2 variance ratio (does off-axis dynamics grow?)
  - Correlation dimension (does D₂ cross 1?)
  - Sharpness spectral flatness (does the transition sharpen?)

USAGE:
    python phase3_experiments.py --width --seeds 5              # width scaling
    python phase3_experiments.py --depth --seeds 5              # depth scaling  
    python phase3_experiments.py --relu --seeds 5               # ReLU comparison
    python phase3_experiments.py --all --seeds 5                # everything
    python phase3_experiments.py --plot-only

REQUIREMENTS:
    pip install torch numpy matplotlib scipy
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
        return

    run_any = args.all or args.width or args.depth or args.relu
    if not run_any:
        parser.print_help()
        print("\n  python phase3_experiments.py --width --seeds 3")
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

    print("\nDone.")


if __name__ == "__main__":
    main()
