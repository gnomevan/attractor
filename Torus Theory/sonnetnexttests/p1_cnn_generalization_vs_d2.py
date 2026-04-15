"""
P1: Strange Attractor Geometry vs. Generalization
===================================================
Tests the core prediction: test accuracy at convergence should peak
in the 20-40% EoS range where D2 is highest — not at peak Lyapunov
(~15% EoS) and not above the chaos window.

Architecture: CNN ~269K params, CIFAR-10
Method:    Full-batch gradient descent, MSE loss
Outputs:   lr_vs_d2_accuracy.npz, lr_vs_d2_accuracy.pdf

Run on Colab T4. Mount Drive first. Expected runtime: ~4-6 hours.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, grad

# ── Drive / output directory ────────────────────────────────────────────────
DRIVE_DIR = '/content/drive/MyDrive/chaos_generalization'
os.makedirs(DRIVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)


# ── Architecture ─────────────────────────────────────────────────────────────
class CNN(nn.Module):
    """
    ~268K parameter CNN matching prior D2 experiments.
    Conv(3→32) → Conv(32→64) → Conv(64→128) → FC(2048→85) → FC(85→10)
    3× AvgPool2d(2): 32×32 → 16×16 → 8×8 → 4×4 → flatten=2048
    Total params: 268,273 (matches prior experiment's 268,650 to within 0.1%)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.Tanh(),
            nn.AvgPool2d(2),                              # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.Tanh(),
            nn.AvgPool2d(2),                              # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.Tanh(),
            nn.AvgPool2d(2),                              # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 85), nn.Tanh(),
            nn.Linear(85, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Data ─────────────────────────────────────────────────────────────────────
def load_cifar10(n_train=5000, n_test=1000, n_traj=500, seed=0):
    """
    Returns:
        X_train, Y_train  — full-batch training tensors (on DEVICE)
        X_test,  Y_test   — held-out accuracy evaluation (on DEVICE)
        X_traj,  Y_traj   — small fixed set for trajectory tracking (on CPU)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=True, download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)

    # Training subset
    train_idx = rng.choice(len(train_ds), n_train, replace=False)
    X_tr = torch.stack([train_ds[i][0] for i in train_idx]).to(DEVICE)
    Y_tr_labels = torch.tensor([train_ds[i][1] for i in train_idx])
    Y_tr = torch.zeros(n_train, 10, device=DEVICE)
    Y_tr.scatter_(1, Y_tr_labels.unsqueeze(1).to(DEVICE), 1.0)  # one-hot, MSE

    # Test subset
    test_idx = rng.choice(len(test_ds), n_test, replace=False)
    X_te = torch.stack([test_ds[i][0] for i in test_idx]).to(DEVICE)
    Y_te = torch.tensor([test_ds[i][1] for i in test_idx], device=DEVICE)

    # Trajectory probe set (CPU, small)
    traj_idx = test_idx[:n_traj]
    X_tj = torch.stack([test_ds[i][0] for i in traj_idx])

    print(f"Data: train={n_train}, test={n_test}, traj_probe={n_traj}", flush=True)
    return X_tr, Y_tr, X_te, Y_te, X_tj


# ── Sharpness (top Hessian eigenvalue via power iteration) ───────────────────
@torch.no_grad()
def compute_sharpness(model, X, Y, n_iter=50):
    """
    Estimates λ_max of the loss Hessian via power iteration on Hessian-vector
    products. Uses full-batch data passed in.
    """
    criterion = nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]

    def hvp(v_flat):
        # v_flat: 1D tensor matching total param count
        model.zero_grad()
        loss = criterion(model(X), Y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g_flat = torch.cat([g.reshape(-1) for g in grads])
        # dot with v
        gv = (g_flat * v_flat).sum()
        hvp_grads = torch.autograd.grad(gv, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hvp_grads]).detach()

    n_params = sum(p.numel() for p in params)
    v = torch.randn(n_params, device=DEVICE)
    v = v / v.norm()

    eigenvalue = 0.0
    for _ in range(n_iter):
        hv = hvp(v)
        eigenvalue = (hv * v).sum().item()
        v = hv / (hv.norm() + 1e-12)

    return float(eigenvalue)


# ── Correlation dimension (Grassberger-Procaccia) ────────────────────────────
def correlation_dimension(trajectory, n_proj=40, r_low_pct=5, r_high_pct=30,
                           n_r=30, seed=42):
    """
    Estimates D2 from a trajectory array of shape (T, d).

    Steps:
      1. Random projection to n_proj dims (avoids curse of dimensionality)
      2. Compute log C(r) vs log r via the Grassberger-Procaccia algorithm
      3. Fit slope in the scaling region [r_low_pct, r_high_pct] percentile
         of pairwise distances

    Returns D2 estimate (float) and diagnostic arrays (log_r, log_C).
    """
    rng = np.random.default_rng(seed)
    T, d = trajectory.shape

    # 1. Project
    P = rng.standard_normal((d, n_proj)) / np.sqrt(n_proj)
    traj_proj = trajectory @ P               # (T, n_proj)

    # 2. Pairwise distances (subsampled for speed if T large)
    T_use = min(T, 800)
    idx = rng.choice(T, T_use, replace=False)
    pts = traj_proj[idx]                     # (T_use, n_proj)

    # Efficient pairwise distance via broadcasting
    diff = pts[:, None, :] - pts[None, :, :]  # (T_use, T_use, n_proj)
    dists = np.sqrt((diff ** 2).sum(-1))       # (T_use, T_use)

    # Upper triangle only
    upper = dists[np.triu_indices(T_use, k=1)]

    # 3. Correlation sum
    r_min = np.percentile(upper, r_low_pct)
    r_max = np.percentile(upper, r_high_pct)
    if r_min <= 0:
        r_min = upper[upper > 0].min() if (upper > 0).any() else 1e-8
    r_grid = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_r))

    log_C = []
    for r in r_grid:
        C = (upper < r).mean()
        log_C.append(np.log(C + 1e-12))
    log_C = np.array(log_C)
    log_r = np.log(r_grid)

    # 4. Linear fit → slope = D2
    finite = np.isfinite(log_C) & np.isfinite(log_r)
    if finite.sum() < 4:
        return np.nan, log_r, log_C

    coeffs = np.polyfit(log_r[finite], log_C[finite], 1)
    D2 = float(coeffs[0])
    return D2, log_r, log_C


# ── Single training run ───────────────────────────────────────────────────────
def train_one_lr(lr, X_train, Y_train, X_test, Y_test, X_traj,
                 n_steps=3000, traj_every=10, warmup_steps=500,
                 seed=0):
    """
    Trains CNN with given learning rate from fixed init.
    Returns dict with D2, test_accuracy, sharpness_history, etc.
    """
    torch.manual_seed(seed)
    model = CNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    trajectory = []   # function-space points: outputs on X_traj
    losses = []
    sharpness_log = []  # track periodically (expensive)

    X_traj_dev = X_traj.to(DEVICE)

    t0 = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, Y_train)
        loss.backward()
        optimizer.step()

        # Record trajectory point
        if step >= warmup_steps and step % traj_every == 0:
            with torch.no_grad():
                pts = model(X_traj_dev).cpu().numpy().reshape(-1)
            trajectory.append(pts)

        losses.append(loss.item())

        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"  lr={lr:.4f}  step={step}/{n_steps}  "
                  f"loss={loss.item():.4f}  ({elapsed:.0f}s)", flush=True)

    # Test accuracy
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        test_acc = (preds == Y_test).float().mean().item()

    # D2
    trajectory = np.array(trajectory)     # (T, d)
    D2 = np.nan
    log_r = log_C = None
    if len(trajectory) >= 50:
        D2, log_r, log_C = correlation_dimension(trajectory)

    return {
        'lr': lr,
        'D2': D2,
        'test_acc': test_acc,
        'losses': np.array(losses),
        'trajectory_shape': trajectory.shape,
        'log_r': log_r,
        'log_C': log_C,
    }


# ── Learning rate sweep ───────────────────────────────────────────────────────
def lr_sweep(lr_fractions, eos_threshold, X_tr, Y_tr, X_te, Y_te, X_tj,
             n_steps=3000, n_seeds=3):
    """
    Sweeps learning rates expressed as fractions of the EoS threshold.
    Averages D2 and test_acc over n_seeds.
    """
    results = []

    for frac in lr_fractions:
        lr = frac * eos_threshold
        seed_D2 = []
        seed_acc = []

        for seed in range(n_seeds):
            print(f"\n[LR={lr:.5f}  ({frac*100:.0f}% EoS)  seed={seed}]", flush=True)
            r = train_one_lr(lr, X_tr, Y_tr, X_te, Y_te, X_tj,
                             n_steps=n_steps, seed=seed)
            seed_D2.append(r['D2'])
            seed_acc.append(r['test_acc'])
            print(f"  D2={r['D2']:.3f}  test_acc={r['test_acc']:.4f}", flush=True)

        results.append({
            'lr_frac': frac,
            'lr':      lr,
            'D2_mean': np.nanmean(seed_D2),
            'D2_std':  np.nanstd(seed_D2),
            'acc_mean': np.mean(seed_acc),
            'acc_std':  np.std(seed_acc),
            'D2_seeds': seed_D2,
            'acc_seeds': seed_acc,
        })

        # Save checkpoint after each LR
        _save_results(results)

    return results


def _save_results(results):
    path = os.path.join(DRIVE_DIR, 'p1_cnn_results.npz')
    np.savez(path,
             lr_fracs=np.array([r['lr_frac'] for r in results]),
             lrs=np.array([r['lr'] for r in results]),
             D2_mean=np.array([r['D2_mean'] for r in results]),
             D2_std=np.array([r['D2_std'] for r in results]),
             acc_mean=np.array([r['acc_mean'] for r in results]),
             acc_std=np.array([r['acc_std'] for r in results]))
    print(f"  → Saved checkpoint to {path}", flush=True)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_results(results):
    lr_fracs = np.array([r['lr_frac'] for r in results])
    D2_mean  = np.array([r['D2_mean'] for r in results])
    D2_std   = np.array([r['D2_std'] for r in results])
    acc_mean = np.array([r['acc_mean'] for r in results])
    acc_std  = np.array([r['acc_std'] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: LR fraction vs D2 and accuracy (dual axis)
    ax1 = axes[0]
    ax2 = ax1.twinx()
    l1 = ax1.errorbar(lr_fracs * 100, D2_mean, yerr=D2_std,
                      color='steelblue', marker='o', label='D₂')
    l2 = ax2.errorbar(lr_fracs * 100, acc_mean, yerr=acc_std,
                      color='tomato', marker='s', linestyle='--',
                      label='Test accuracy')
    ax1.set_xlabel('Learning rate (% of EoS threshold)')
    ax1.set_ylabel('Correlation dimension D₂', color='steelblue')
    ax2.set_ylabel('Test accuracy', color='tomato')
    ax1.set_title('D₂ and Generalization vs. Learning Rate')
    lines = [l1, l2]
    labels = ['D₂', 'Test accuracy']
    ax1.legend(lines, labels, loc='upper left')
    ax1.axvspan(20, 40, alpha=0.1, color='green', label='Predicted optimal zone')

    # Right: D2 vs accuracy scatter
    ax = axes[1]
    sc = ax.scatter(D2_mean, acc_mean, c=lr_fracs * 100,
                    cmap='viridis', s=80, zorder=3)
    ax.errorbar(D2_mean, acc_mean,
                xerr=D2_std, yerr=acc_std,
                fmt='none', color='gray', alpha=0.5, zorder=2)
    plt.colorbar(sc, ax=ax, label='LR (% EoS)')
    ax.set_xlabel('Correlation dimension D₂')
    ax.set_ylabel('Test accuracy')
    ax.set_title('D₂ vs. Test Accuracy\n(color = learning rate)')

    plt.tight_layout()
    path = os.path.join(DRIVE_DIR, 'p1_cnn_d2_vs_accuracy.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {path}", flush=True)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== P1: CNN Generalization vs. D2 ===", flush=True)

    # Load data
    X_tr, Y_tr, X_te, Y_te, X_tj = load_cifar10(
        n_train=5000, n_test=2000, n_traj=500)

    # ── Step 1: Estimate EoS threshold ──────────────────────────────────────
    # Run a quick warmup at moderate LR, then measure λ_max.
    # EoS threshold = 2 / λ_max
    print("\n[Step 1] Estimating EoS threshold via sharpness...", flush=True)
    torch.manual_seed(0)
    probe_model = CNN().to(DEVICE)
    probe_opt   = torch.optim.SGD(probe_model.parameters(), lr=0.01, momentum=0.0)
    criterion   = nn.MSELoss()

    for step in range(1000):
        probe_opt.zero_grad()
        loss = criterion(probe_model(X_tr), Y_tr)
        loss.backward()
        probe_opt.step()
        if step % 200 == 0:
            print(f"  warmup step {step}  loss={loss.item():.4f}", flush=True)

    lambda_max = compute_sharpness(probe_model, X_tr, Y_tr)
    eos_threshold = 2.0 / lambda_max
    print(f"\nλ_max = {lambda_max:.4f}")
    print(f"EoS threshold η* = 2/λ_max = {eos_threshold:.5f}", flush=True)

    np.save(os.path.join(DRIVE_DIR, 'p1_eos_threshold.npy'),
            np.array([lambda_max, eos_threshold]))

    # ── Step 2: LR sweep ────────────────────────────────────────────────────
    # 14 fractions spanning 5% to 80% of EoS
    # Key prediction: accuracy peaks at 20-40% EoS, coinciding with D2 peak
    lr_fractions = np.array([
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.50, 0.60, 0.70, 0.80
    ])

    print(f"\n[Step 2] Sweeping {len(lr_fractions)} learning rates "
          f"({lr_fractions.min()*100:.0f}%–{lr_fractions.max()*100:.0f}% EoS)...",
          flush=True)

    results = lr_sweep(
        lr_fractions, eos_threshold,
        X_tr, Y_tr, X_te, Y_te, X_tj,
        n_steps=3000,
        n_seeds=3,
    )

    # ── Step 3: Plot ─────────────────────────────────────────────────────────
    print("\n[Step 3] Generating plots...", flush=True)
    plot_results(results)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'LR frac':>10}  {'LR':>8}  {'D2':>6}  {'Test acc':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['lr_frac']*100:>9.0f}%  "
              f"{r['lr']:>8.5f}  "
              f"{r['D2_mean']:>6.3f}  "
              f"{r['acc_mean']:>10.4f}")

    best_acc_idx = np.argmax([r['acc_mean'] for r in results])
    best_D2_idx  = np.nanargmax([r['D2_mean'] for r in results])
    print(f"\nPeak accuracy at {results[best_acc_idx]['lr_frac']*100:.0f}% EoS "
          f"(D2={results[best_acc_idx]['D2_mean']:.3f})")
    print(f"Peak D2 at {results[best_D2_idx]['lr_frac']*100:.0f}% EoS "
          f"(acc={results[best_D2_idx]['acc_mean']:.4f})")

    alignment = abs(best_acc_idx - best_D2_idx) <= 1
    print(f"\nPrediction alignment: {'✓ CONFIRMED' if alignment else '✗ DISCONFIRMED'}")
    print("(Prediction: peak accuracy and peak D2 at the same ~20-40% EoS LR)")


if __name__ == '__main__':
    main()
