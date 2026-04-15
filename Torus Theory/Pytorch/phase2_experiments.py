"""
Chaos Onset — Phase 2: Dynamical Systems Characterization (v2)
================================================================

Revised after Experiment C showed scalar training loss is dominated by
monotonic convergence (steep 1/f^α, no discrete peaks, spectral entropy
varies by only 0.3%).

KEY CHANGE: Three observables recorded during training:
  1. Loss — for bifurcation diagram only
  2. Gradient norm — oscillatory dynamics visible here
  3. Sharpness (top Hessian eigenvalue, every N steps) — the EoS observable

USAGE:
    python phase2_experiments.py --all --seeds 3
    python phase2_experiments.py --power-spectrum --seeds 2
    python phase2_experiments.py --plot-only
"""

import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from scipy import stats, signal

DEFAULT_CONFIG = {
    "input_dim": 220, "hidden_dim": 50, "output_dim": 10, "activation": "tanh",
    "n_samples": 2000, "n_classes": 10, "n_random_features": 200,
    "n_quadratic_features": 20, "data_seed": 42, "n_train_steps": 5000,
    "perturbation_eps": 1e-5, "sharpness_iters": 100,
    "sharpness_warmup_steps": 1000, "sharpness_warmup_lr": 0.01,
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

def compute_sharpness_fast(model, X, y, criterion, n_iter=20):
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

def train_and_record_full(config, lr, seed, X, y, n_steps=None,
                          sharpness_every=50, record_outputs=False,
                          X_eval=None, output_every=5):
    device = X.device
    if n_steps is None: n_steps = config["n_train_steps"]
    if X_eval is None: X_eval = X
    model = make_model(config, seed).to(device)
    criterion = nn.MSELoss()
    losses = np.zeros(n_steps)
    grad_norms = np.zeros(n_steps)
    sharpness_vals, sharpness_steps = [], []
    outputs = [] if record_outputs else None

    for t in range(n_steps):
        model.zero_grad()
        loss = criterion(model(X), y)
        losses[t] = loss.item()
        loss.backward()
        with torch.no_grad():
            gn = sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None).sqrt().item()
            grad_norms[t] = gn
        if record_outputs and t % output_every == 0:
            with torch.no_grad():
                outputs.append(model(X_eval).cpu().numpy())
        # SGD update (must happen before sharpness, which calls zero_grad)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad
        if t % sharpness_every == 0:
            sharp = compute_sharpness_fast(model, X, y, criterion, n_iter=20)
            sharpness_vals.append(sharp)
            sharpness_steps.append(t)

    result = {"losses": losses, "grad_norms": grad_norms,
              "sharpness": np.array(sharpness_vals),
              "sharpness_steps": np.array(sharpness_steps),
              "lr": lr, "seed": seed}
    if record_outputs: result["outputs"] = np.array(outputs)
    return result

# ============================================================
# EXPERIMENT C: Power Spectrum (gradient norm + sharpness)
# ============================================================

def run_power_spectrum(config, n_seeds=5, n_steps=20000):
    test_lrs = [0.005, 0.010, 0.015, 0.020, 0.030, 0.040,
                0.060, 0.080, 0.100, 0.150, 0.200, 0.300]
    print("=" * 60)
    print(f"EXPERIMENT C: POWER SPECTRUM (grad norm + sharpness)")
    print(f"  {n_seeds} seeds x {len(test_lrs)} LRs, {n_steps} steps")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)
    cfg = dict(config); cfg["n_train_steps"] = n_steps

    total = len(test_lrs) * n_seeds; done = 0; t0 = time.time()
    save_data = {"test_lrs": np.array(test_lrs), "n_seeds": n_seeds, "n_steps": n_steps}

    for li, lr in enumerate(test_lrs):
        gn_psds, sh_psds, loss_psds = [], [], []
        for s in range(n_seeds):
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 1 else 0
            print(f"  [{done}/{total}] lr={lr:.3f} seed={s} ETA: {eta:.0f}s", end="", flush=True)

            result = train_and_record_full(cfg, lr, s, X, y, n_steps=n_steps, sharpness_every=50)
            start = n_steps // 5

            # Gradient norm: log-transform then detrend
            gn_log = np.log(result["grad_norms"][start:] + 1e-30)
            gn_det = signal.detrend(gn_log, type='linear')
            f_gn, p_gn = signal.welch(gn_det, fs=1.0, nperseg=min(1024, len(gn_det)//4))
            gn_psds.append(p_gn)

            # Sharpness
            sh = result["sharpness"]; sh_start = len(sh) // 5
            sh_sig = sh[sh_start:]
            if len(sh_sig) > 10:
                sh_det = signal.detrend(sh_sig, type='linear')
                f_sh, p_sh = signal.welch(sh_det, fs=1.0/50, nperseg=min(64, len(sh_det)//2))
                sh_psds.append(p_sh)

            # Loss: log-transform then detrend
            loss_log = np.log(result["losses"][start:] + 1e-30)
            loss_det = signal.detrend(loss_log, type='linear')
            f_loss, p_loss = signal.welch(loss_det, fs=1.0, nperseg=min(1024, len(loss_det)//4))
            loss_psds.append(p_loss)

            peaks, _ = signal.find_peaks(p_gn, height=np.median(p_gn)*5, distance=3)
            print(f"  -> gn:{len(peaks)} peaks, sharp:{sh[-1]:.2f}")

        save_data[f"gn_psds_{li}"] = np.array(gn_psds)
        save_data[f"loss_psds_{li}"] = np.array(loss_psds)
        if sh_psds: save_data[f"sh_psds_{li}"] = np.array(sh_psds)

    save_data["freqs_gn"] = f_gn; save_data["freqs_loss"] = f_loss
    if sh_psds: save_data["freqs_sh"] = f_sh
    os.makedirs("results", exist_ok=True)
    np.savez("results/power_spectrum_v2.npz", **save_data)
    print(f"\n  Saved -> results/power_spectrum_v2.npz")

# ============================================================
# EXPERIMENT D: Takens Delay Embedding (gradient norm)
# ============================================================

def optimal_delay(x, max_lag=500):
    n = len(x); x_c = x - x.mean()
    acf = np.correlate(x_c, x_c, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-30)
    for i in range(1, min(max_lag, len(acf)-1)):
        if acf[i] < acf[i-1] and acf[i] < acf[i+1]: return i
    return max_lag // 4

def delay_embed(x, dim, tau):
    n = len(x) - (dim-1)*tau
    if n <= 0: raise ValueError(f"Too short for dim={dim}, tau={tau}")
    emb = np.zeros((n, dim))
    for d in range(dim): emb[:, d] = x[d*tau:d*tau+n]
    return emb

def correlation_dimension(points, n_scales=20):
    n = len(points)
    if n > 2000:
        points = points[np.random.RandomState(42).choice(n, 2000, replace=False)]
        n = 2000
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(points[i] - points[j]))
    dists = np.array(dists)
    if len(dists) == 0: return float('nan'), [], []
    log_eps = np.linspace(np.log(np.percentile(dists,1)+1e-15), np.log(np.percentile(dists,95)), n_scales)
    log_C = [np.log(max(np.sum(dists < np.exp(le)) / (n*(n-1)/2), 1e-30)) for le in log_eps]
    log_C = np.array(log_C)
    s, e = len(log_eps)//5, 4*len(log_eps)//5
    slope = stats.linregress(log_eps[s:e], log_C[s:e])[0] if e-s > 2 else float('nan')
    return slope, log_eps, log_C

def run_takens(config, n_seeds=5, n_steps=20000):
    test_lrs = [0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.200]
    embed_dims = [3, 5, 7, 9]
    print("=" * 60)
    print(f"EXPERIMENT D: TAKENS EMBEDDING (gradient norm)")
    print(f"  {n_seeds} seeds x {len(test_lrs)} LRs, {n_steps} steps")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)
    cfg = dict(config); cfg["n_train_steps"] = n_steps

    total = len(test_lrs)*n_seeds; done = 0; t0 = time.time()
    save_data = {"test_lrs": np.array(test_lrs), "embed_dims": np.array(embed_dims),
                 "n_seeds": n_seeds, "n_steps": n_steps}

    for li, lr in enumerate(test_lrs):
        taus, cdims = [], []
        emb3d_first = None
        for s in range(n_seeds):
            done += 1; elapsed = time.time()-t0
            eta = elapsed/done*(total-done) if done > 1 else 0
            print(f"  [{done}/{total}] lr={lr:.3f} seed={s} ETA: {eta:.0f}s", end="", flush=True)

            result = train_and_record_full(cfg, lr, s, X, y, n_steps=n_steps,
                                           sharpness_every=n_steps+1)
            gn = result["grad_norms"]
            start = n_steps // 5
            gn_log = np.log(gn[start:] + 1e-30)
            gn_det = signal.detrend(gn_log, type='linear')

            tau = optimal_delay(gn_det)
            taus.append(tau)
            if s == 0:
                e3 = delay_embed(gn_det, 3, tau)
                step = max(1, len(e3)//2000)
                emb3d_first = e3[::step]

            dims = []
            for ed in embed_dims:
                try:
                    emb = delay_embed(gn_det, ed, tau)
                    d, _, _ = correlation_dimension(emb)
                    dims.append(d)
                except ValueError:
                    dims.append(float('nan'))
            cdims.append(dims)
            print(f"  -> tau={tau}, D2={dims[0]:.2f}")

        save_data[f"taus_{li}"] = np.array(taus)
        save_data[f"corr_dims_{li}"] = np.array(cdims)
        if emb3d_first is not None: save_data[f"emb3d_{li}"] = emb3d_first

    os.makedirs("results", exist_ok=True)
    np.savez("results/takens_v2.npz", **save_data)
    print(f"\n  Saved -> results/takens_v2.npz")

# ============================================================
# EXPERIMENT E: Bifurcation Diagram (loss + gradient norm)
# ============================================================

def run_bifurcation(config, n_seeds=5, n_lrs=200, n_steps=10000,
                    lr_min=0.001, lr_max=0.20, record_last_n=200):
    print("=" * 60)
    print(f"EXPERIMENT E: BIFURCATION ({n_seeds} seeds x {n_lrs} LRs)")
    print(f"  [{lr_min}, {lr_max}], {n_steps} steps, last {record_last_n}")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)
    cfg = dict(config); cfg["n_train_steps"] = n_steps

    lrs = np.linspace(lr_min, lr_max, n_lrs)
    all_loss = np.zeros((n_seeds, n_lrs, record_last_n))
    all_gn = np.zeros((n_seeds, n_lrs, record_last_n))

    total = n_seeds*n_lrs; done = 0; t0 = time.time()
    for s in range(n_seeds):
        for j, lr in enumerate(lrs):
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time()-t0
                eta = elapsed/done*(total-done) if done > 1 else 0
                print(f"  [{done}/{total}] seed={s} lr={lr:.4f} ETA: {eta:.0f}s")
            result = train_and_record_full(cfg, lr, s, X, y, n_steps=n_steps,
                                           sharpness_every=n_steps+1)
            all_loss[s, j, :] = result["losses"][-record_last_n:]
            all_gn[s, j, :] = result["grad_norms"][-record_last_n:]

    os.makedirs("results", exist_ok=True)
    np.savez("results/bifurcation.npz", lrs=lrs, all_final_losses=all_loss,
             all_final_gradnorms=all_gn, n_seeds=n_seeds, n_steps=n_steps,
             record_last_n=record_last_n)
    print(f"  Saved -> results/bifurcation.npz")

# ============================================================
# EXPERIMENT F: Correlation Dimension (function-space trajectory)
# ============================================================

def run_dimension(config, n_seeds=5, n_steps=10000):
    test_lrs = [0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.200]
    n_eval = 100; output_every = 5
    print("=" * 60)
    print(f"EXPERIMENT F: CORRELATION DIMENSION (function space)")
    print(f"  {n_seeds} seeds x {len(test_lrs)} LRs, {n_steps} steps")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X, y = generate_data(config)
    X, y = X.to(device), y.to(device)
    torch.manual_seed(0)
    X_eval = X[torch.randperm(X.shape[0])[:n_eval]]
    cfg = dict(config); cfg["n_train_steps"] = n_steps

    total = len(test_lrs)*n_seeds; done = 0; t0 = time.time()
    save_data = {"test_lrs": np.array(test_lrs), "n_seeds": n_seeds,
                 "n_steps": n_steps, "n_eval": n_eval, "output_every": output_every}

    for li, lr in enumerate(test_lrs):
        cdims, pcavars = [], []
        for s in range(n_seeds):
            done += 1; elapsed = time.time()-t0
            eta = elapsed/done*(total-done) if done > 1 else 0
            print(f"  [{done}/{total}] lr={lr:.3f} seed={s} ETA: {eta:.0f}s", end="", flush=True)

            result = train_and_record_full(cfg, lr, s, X, y, n_steps=n_steps,
                                           sharpness_every=n_steps+1,
                                           record_outputs=True, X_eval=X_eval,
                                           output_every=output_every)
            outputs = result["outputs"]
            start = len(outputs)//5
            traj = outputs[start:].reshape(len(outputs)-start, -1)

            centered = traj - traj.mean(axis=0)
            try:
                _, sv, _ = np.linalg.svd(centered, full_matrices=False)
                ve = (sv**2) / (sv**2).sum()
                pcavars.append(ve[:20])
            except:
                ve = np.zeros(20); pcavars.append(ve)

            if len(traj) > 2000:
                traj_sub = traj[np.random.RandomState(s).choice(len(traj), 2000, replace=False)]
            else:
                traj_sub = traj
            d, _, _ = correlation_dimension(traj_sub)
            cdims.append(d)
            print(f"  -> D2={d:.2f}, PC1={ve[0]*100:.1f}%")

        save_data[f"corr_dims_{li}"] = np.array(cdims)
        save_data[f"pca_var_{li}"] = np.array(pcavars)

    os.makedirs("results", exist_ok=True)
    np.savez("results/dimension.npz", **save_data)
    print(f"\n  Saved -> results/dimension.npz")

# ============================================================
# PLOTTING (keeping external for clarity)
# ============================================================

def plot_power_spectrum(npz_path="results/power_spectrum_v2.npz"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(npz_path, allow_pickle=True)
    test_lrs = d["test_lrs"]; freqs_gn = d["freqs_gn"]; n_lrs = len(test_lrs)

    ncols = 4; nrows = (n_lrs+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), sharey=True)
    axes = axes.flatten()
    for li, lr in enumerate(test_lrs):
        ax = axes[li]; psds = d[f"gn_psds_{li}"]
        for s in range(len(psds)):
            ax.semilogy(freqs_gn, psds[s], alpha=0.3, lw=0.5, color='steelblue')
        ax.semilogy(freqs_gn, psds.mean(axis=0), 'k-', lw=1.5)
        ax.set_title(f'η={lr:.3f}', fontsize=10); ax.set_xlim(0, 0.5); ax.grid(True, alpha=0.3)
    for li in range(n_lrs, len(axes)): axes[li].set_visible(False)
    axes[0].set_ylabel('PSD (log grad norm)')
    fig.suptitle('Power Spectra of Gradient Norm', fontsize=13, y=1.02)
    plt.tight_layout(); os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/power_spectrum_gradnorm.png", dpi=200, bbox_inches="tight")
    print("  Saved -> figures/power_spectrum_gradnorm.png"); plt.close()

    # Comparison: loss vs grad norm vs sharpness
    compare_idx = [i for i, lr in enumerate(test_lrs) if lr in [0.005, 0.020, 0.040, 0.080, 0.150, 0.300]]
    if compare_idx:
        nc = len(compare_idx); n_rows = 2 + (1 if f"sh_psds_{compare_idx[0]}" in d else 0)
        fig, axes = plt.subplots(n_rows, nc, figsize=(3.5*nc, 3.5*n_rows), sharex=True)
        for col, li in enumerate(compare_idx):
            lr = test_lrs[li]
            axes[0, col].semilogy(d["freqs_loss"], d[f"loss_psds_{li}"].mean(axis=0), 'k-', lw=1)
            axes[0, col].set_title(f'η={lr:.3f}', fontsize=10)
            if col == 0: axes[0, col].set_ylabel('Loss PSD')
            axes[1, col].semilogy(freqs_gn, d[f"gn_psds_{li}"].mean(axis=0), 'k-', lw=1)
            if col == 0: axes[1, col].set_ylabel('Grad Norm PSD')
            if n_rows > 2 and f"sh_psds_{li}" in d:
                axes[2, col].semilogy(d["freqs_sh"], d[f"sh_psds_{li}"].mean(axis=0), 'k-', lw=1)
                if col == 0: axes[2, col].set_ylabel('Sharpness PSD')
            axes[-1, col].set_xlabel('Frequency')
        fig.suptitle('Observable Comparison: Loss vs Grad Norm vs Sharpness', fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig("figures/power_spectrum_comparison.png", dpi=200, bbox_inches="tight")
        print("  Saved -> figures/power_spectrum_comparison.png"); plt.close()

    # Spectral entropy
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, key_prefix in [('Gradient norm', 'gn_psds'), ('Loss (log-detrended)', 'loss_psds')]:
        ents = []
        for li in range(n_lrs):
            mp = d[f"{key_prefix}_{li}"].mean(axis=0)
            pn = mp / mp.sum(); pn = pn[pn > 0]
            ents.append(-np.sum(pn * np.log(pn)) / np.log(len(pn)))
        fmt = 'ko-' if 'gn' in key_prefix else 's--'
        ax.plot(test_lrs, ents, fmt, ms=5, lw=1.5, label=label, color='black' if 'gn' in key_prefix else 'gray')
    ax.set_xlabel('Learning Rate η'); ax.set_ylabel('Normalized Spectral Entropy')
    ax.set_title('Spectral Complexity: Grad Norm vs Loss'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("figures/spectral_entropy_v2.png", dpi=200, bbox_inches="tight")
    print("  Saved -> figures/spectral_entropy_v2.png"); plt.close()

def plot_takens(npz_path="results/takens_v2.npz"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(npz_path, allow_pickle=True)
    test_lrs = d["test_lrs"]; embed_dims = d["embed_dims"]; n_lrs = len(test_lrs)

    ncols = 4; nrows = (n_lrs+ncols-1)//ncols
    fig = plt.figure(figsize=(5*ncols, 4*nrows))
    for li in range(n_lrs):
        ax = fig.add_subplot(nrows, ncols, li+1, projection='3d')
        key = f"emb3d_{li}"
        if key in d:
            emb = d[key]; c = np.linspace(0, 1, len(emb))
            ax.scatter(emb[:,0], emb[:,1], emb[:,2], c=c, cmap='viridis', s=1, alpha=0.5)
        ax.set_title(f'η={test_lrs[li]:.3f}', fontsize=10)
    fig.suptitle('Takens Embedding of Gradient Norm', fontsize=13, y=1.02)
    plt.tight_layout(); os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/takens_gradnorm.png", dpi=200, bbox_inches="tight")
    print("  Saved -> figures/takens_gradnorm.png"); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for di, ed in enumerate(embed_dims):
        means = [np.nanmean(d[f"corr_dims_{li}"][:, di]) for li in range(n_lrs)]
        stds = [np.nanstd(d[f"corr_dims_{li}"][:, di]) for li in range(n_lrs)]
        ax.errorbar(test_lrs, means, yerr=stds, fmt='o-', ms=5, capsize=3, label=f'd={ed}')
    ax.axhline(1, color='gray', ls=':'); ax.axhline(2, color='gray', ls='--')
    ax.set_xlabel('Learning Rate η'); ax.set_ylabel('Correlation Dimension D₂')
    ax.set_title('Attractor Dimension of Gradient Norm Dynamics')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("figures/takens_dimension_v2.png", dpi=200, bbox_inches="tight")
    print("  Saved -> figures/takens_dimension_v2.png"); plt.close()

def plot_bifurcation(npz_path="results/bifurcation.npz"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(npz_path, allow_pickle=True)
    lrs = d["lrs"]; all_loss = d["all_final_losses"]; all_gn = d["all_final_gradnorms"]
    n_seeds = all_loss.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for j, lr in enumerate(lrs):
        axes[0,0].scatter([lr]*len(all_loss[0,j,:]), all_loss[0,j,:], c='black', s=0.1, alpha=0.3)
        axes[0,1].scatter([lr]*len(all_gn[0,j,:]), all_gn[0,j,:], c='darkblue', s=0.1, alpha=0.3)
    axes[0,0].set_ylabel('Loss'); axes[0,0].set_title('Loss Bifurcation (Seed 0)'); axes[0,0].set_yscale('log')
    axes[0,1].set_ylabel('Grad Norm'); axes[0,1].set_title('Grad Norm Bifurcation (Seed 0)'); axes[0,1].set_yscale('log')

    for s in range(min(n_seeds, 5)):
        axes[1,0].plot(lrs, all_loss[s].max(axis=1)-all_loss[s].min(axis=1), '-', alpha=0.4, lw=0.8)
        axes[1,1].plot(lrs, all_gn[s].max(axis=1)-all_gn[s].min(axis=1), '-', alpha=0.4, lw=0.8)
    axes[1,0].plot(lrs, (all_loss.max(axis=2)-all_loss.min(axis=2)).mean(axis=0), 'k-', lw=2)
    axes[1,1].plot(lrs, (all_gn.max(axis=2)-all_gn.min(axis=2)).mean(axis=0), 'k-', lw=2)
    axes[1,0].set_xlabel('η'); axes[1,0].set_ylabel('Loss Oscillation'); axes[1,0].set_yscale('log')
    axes[1,1].set_xlabel('η'); axes[1,1].set_ylabel('Grad Norm Oscillation'); axes[1,1].set_yscale('log')

    fig.suptitle('Bifurcation: Loss vs Gradient Norm', fontsize=14, y=1.01)
    plt.tight_layout(); os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/bifurcation.png", dpi=200, bbox_inches="tight")
    print("  Saved -> figures/bifurcation.png"); plt.close()

def plot_dimension(npz_path="results/dimension.npz"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(npz_path, allow_pickle=True)
    test_lrs = d["test_lrs"]; n_lrs = len(test_lrs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    means = [np.nanmean(d[f"corr_dims_{li}"]) for li in range(n_lrs)]
    stds = [np.nanstd(d[f"corr_dims_{li}"]) for li in range(n_lrs)]
    ax1.errorbar(test_lrs, means, yerr=stds, fmt='ko-', ms=6, capsize=3, lw=2)
    ax1.axhline(1, color='gray', ls=':'); ax1.axhline(2, color='gray', ls='--')
    ax1.set_xlabel('η'); ax1.set_ylabel('Correlation Dimension')
    ax1.set_title('Function-Space Trajectory Dimension'); ax1.grid(True, alpha=0.3)

    pc1 = [np.mean(d[f"pca_var_{li}"][:,0])*100 for li in range(n_lrs)]
    pc1s = [np.std(d[f"pca_var_{li}"][:,0])*100 for li in range(n_lrs)]
    ax2.errorbar(test_lrs, pc1, yerr=pc1s, fmt='ko-', ms=6, capsize=3, lw=2)
    ax2.set_xlabel('η'); ax2.set_ylabel('PC1 Variance (%)')
    ax2.set_title('Lower = Higher-Dimensional Dynamics'); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/dimension_analysis.png", dpi=200, bbox_inches="tight")
    print("  Saved -> figures/dimension_analysis.png"); plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Dynamical Characterization (v2)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--power-spectrum", action="store_true")
    parser.add_argument("--takens", action="store_true")
    parser.add_argument("--bifurcation", action="store_true")
    parser.add_argument("--dimension", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--bif-steps", type=int, default=10000)
    parser.add_argument("--bif-lrs", type=int, default=200)
    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)

    if args.plot_only:
        for name, func in [("power_spectrum_v2", plot_power_spectrum), ("takens_v2", plot_takens),
                           ("bifurcation", plot_bifurcation), ("dimension", plot_dimension)]:
            if os.path.exists(f"results/{name}.npz"): func(f"results/{name}.npz")
        return

    run_any = args.all or args.power_spectrum or args.takens or args.bifurcation or args.dimension
    if not run_any:
        parser.print_help()
        print("\n  python phase2_experiments.py --power-spectrum --seeds 2")
        print("  python phase2_experiments.py --all --seeds 3")
        return

    if args.all or args.power_spectrum:
        run_power_spectrum(config, n_seeds=args.seeds, n_steps=args.steps); plot_power_spectrum()
    if args.all or args.takens:
        run_takens(config, n_seeds=args.seeds, n_steps=args.steps); plot_takens()
    if args.all or args.bifurcation:
        run_bifurcation(config, n_seeds=args.seeds, n_lrs=args.bif_lrs, n_steps=args.bif_steps); plot_bifurcation()
    if args.all or args.dimension:
        run_dimension(config, n_seeds=args.seeds, n_steps=args.steps); plot_dimension()
    print("\nDone.")

if __name__ == "__main__":
    main()
