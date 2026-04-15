"""
P3: Depth Widens the Chaos Window → Larger Generalization Gap
==============================================================
Tests the prediction: deeper/wider networks show a larger gap between
low-LR and optimal-LR generalization, because depth provides more
coupled modes (wider chaos window → larger exploration budget).

Architectures:
  MLP:  2-layer tanh, hidden=50, ~14K params   (D2 ≈ 0.9 at peak)
  CNN:  as in P1, ~269K params                  (D2 ≈ 3.6 at peak)

Key comparison:
  generalization_gap = acc(optimal_LR) - acc(low_LR)
  Prediction: gap_CNN >> gap_MLP

Also plots: generalization curves (acc vs LR/EoS%) for both architectures.

Data:
  MLP: 1,600 synthetic structured examples (220-dim, 10 classes)
  CNN: CIFAR-10 subset (same as P1)

Run on Colab T4. Expected runtime: ~2-3 hours.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DRIVE_DIR = '/content/drive/MyDrive/chaos_generalization'
os.makedirs(DRIVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Architectures
# ═══════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    2-hidden-layer tanh MLP matching prior experiments.
    Input: 220 dim. Hidden: 50. Output: 10. ~14,110 params.
    """
    def __init__(self, input_dim=220, hidden_dim=50, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """
    ~268K parameter CNN (matches P1 experiment and prior D2 experiments).
    Conv(3→32) → Conv(32→64) → Conv(64→128) → FC(2048→85) → FC(85→10)
    Total params: 268,273
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 85), nn.Tanh(),
            nn.Linear(85, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data for MLP (matching prior experiments exactly)
# ═══════════════════════════════════════════════════════════════════════════

def make_synthetic_data(n_train=1600, n_test=400, input_dim=220,
                        n_classes=10, seed=0):
    """
    Structured synthetic dataset used in prior MLP experiments.
    - Class means are orthogonal in first 10 dims
    - Quadratic features: squares of first 20 dims (not random pairs)
    - Remaining dims: noise
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    n_per_class = n_total // n_classes

    X_list, Y_list = [], []
    for c in range(n_classes):
        # Class mean: signal in first 10 dims, one-hot style
        mean = np.zeros(input_dim)
        mean[c] = 3.0   # strong signal

        X_c = rng.normal(mean, 1.0, size=(n_per_class, input_dim))

        # Quadratic features: replace dims 10-29 with squares of dims 0-19
        X_c[:, 10:30] = X_c[:, :20] ** 2

        Y_list.append(np.full(n_per_class, c))
        X_list.append(X_c)

    X = np.vstack(X_list).astype(np.float32)
    Y_labels = np.concatenate(Y_list)

    # Shuffle
    perm = rng.permutation(len(X))
    X, Y_labels = X[perm], Y_labels[perm]

    # One-hot targets for MSE
    Y_oh = np.zeros((len(X), n_classes), dtype=np.float32)
    Y_oh[np.arange(len(X)), Y_labels] = 1.0

    X_tr = torch.tensor(X[:n_train])
    Y_tr = torch.tensor(Y_oh[:n_train])
    Y_tr_labels = torch.tensor(Y_labels[:n_train], dtype=torch.long)

    X_te = torch.tensor(X[n_train:])
    Y_te_labels = torch.tensor(Y_labels[n_train:], dtype=torch.long)

    return X_tr, Y_tr, Y_tr_labels, X_te, Y_te_labels


# ═══════════════════════════════════════════════════════════════════════════
# CIFAR-10 data for CNN (same as P1)
# ═══════════════════════════════════════════════════════════════════════════

def load_cifar10(n_train=5000, n_test=2000, seed=0):
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

    train_idx = rng.choice(len(train_ds), n_train, replace=False)
    X_tr = torch.stack([train_ds[i][0] for i in train_idx]).to(DEVICE)
    Y_tr_labels = torch.tensor([train_ds[i][1] for i in train_idx]).to(DEVICE)
    Y_tr = torch.zeros(n_train, 10, device=DEVICE)
    Y_tr.scatter_(1, Y_tr_labels.unsqueeze(1), 1.0)

    test_idx = rng.choice(len(test_ds), n_test, replace=False)
    X_te = torch.stack([test_ds[i][0] for i in test_idx]).to(DEVICE)
    Y_te = torch.tensor([test_ds[i][1] for i in test_idx], device=DEVICE)

    return X_tr, Y_tr, X_te, Y_te


# ═══════════════════════════════════════════════════════════════════════════
# Sharpness
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_sharpness(model, X, Y, n_iter=50, is_classification=True):
    criterion = nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]

    def hvp(v_flat):
        model.zero_grad()
        loss = criterion(model(X), Y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g_flat = torch.cat([g.reshape(-1) for g in grads])
        gv = (g_flat * v_flat).sum()
        hvp_g = torch.autograd.grad(gv, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hvp_g]).detach()

    n_params = sum(p.numel() for p in params)
    v = torch.randn(n_params, device=X.device)
    v = v / v.norm()
    ev = 0.0
    for _ in range(n_iter):
        hv = hvp(v)
        ev = (hv * v).sum().item()
        v = hv / (hv.norm() + 1e-12)
    return float(ev)


# ═══════════════════════════════════════════════════════════════════════════
# Train + measure test accuracy
# ═══════════════════════════════════════════════════════════════════════════

def train_and_eval(model_class, model_kwargs, lr,
                   X_train, Y_train, X_test, Y_test_labels,
                   n_steps, seed=0):
    """Generic training loop. Returns final test accuracy."""
    torch.manual_seed(seed)
    device = X_train.device
    model = model_class(**model_kwargs).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, Y_train)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"    step={step}/{n_steps} loss={loss.item():.4f}", flush=True)

    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        acc = (preds == Y_test_labels).float().mean().item()

    return acc


def sweep_architecture(name, model_class, model_kwargs,
                       X_tr, Y_tr, X_te, Y_te,
                       eos_threshold, lr_fractions,
                       n_steps, n_seeds=3):
    """Runs the LR sweep for one architecture. Returns list of result dicts."""
    results = []
    for frac in lr_fractions:
        lr = frac * eos_threshold
        accs = []
        for seed in range(n_seeds):
            print(f"\n[{name}] LR={lr:.5f} ({frac*100:.0f}% EoS) seed={seed}",
                  flush=True)
            acc = train_and_eval(model_class, model_kwargs, lr,
                                 X_tr, Y_tr, X_te, Y_te,
                                 n_steps=n_steps, seed=seed)
            accs.append(acc)
            print(f"  test_acc={acc:.4f}", flush=True)

        results.append({
            'arch': name,
            'lr_frac': frac,
            'lr': lr,
            'acc_mean': np.mean(accs),
            'acc_std': np.std(accs),
            'acc_seeds': accs,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EoS threshold estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_eos(model_class, model_kwargs, X_tr, Y_tr,
                 warmup_lr=0.01, warmup_steps=1000, seed=0):
    torch.manual_seed(seed)
    device = X_tr.device
    model = model_class(**model_kwargs).to(device)
    criterion = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=warmup_lr, momentum=0.0)

    for step in range(warmup_steps):
        opt.zero_grad()
        loss = criterion(model(X_tr), Y_tr)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  warmup {step}/{warmup_steps} loss={loss.item():.4f}",
                  flush=True)

    lam = compute_sharpness(model, X_tr, Y_tr)
    eos = 2.0 / lam
    print(f"  λ_max={lam:.4f}  EoS threshold={eos:.5f}", flush=True)
    return lam, eos


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_depth_comparison(mlp_results, cnn_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: generalization curves (normalized LR axis)
    ax = axes[0]
    for results, label, color, marker in [
        (mlp_results, 'MLP (~14K)', 'steelblue', 'o'),
        (cnn_results, 'CNN (~269K)', 'tomato', 's'),
    ]:
        fracs = np.array([r['lr_frac'] * 100 for r in results])
        acc_m = np.array([r['acc_mean'] for r in results])
        acc_s = np.array([r['acc_std'] for r in results])
        ax.errorbar(fracs, acc_m, yerr=acc_s,
                    label=label, color=color, marker=marker, linewidth=2)

    ax.set_xlabel('Learning rate (% of EoS threshold)')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Generalization Curves\n(normalized to EoS threshold)')
    ax.legend()
    ax.axvspan(20, 40, alpha=0.08, color='green', label='Predicted optimal zone')

    # Right: generalization gap
    ax = axes[1]

    def gen_gap(results):
        accs = [r['acc_mean'] for r in results]
        stds = [r['acc_std'] for r in results]
        # Low LR = first point; optimal = max
        opt_idx = np.argmax(accs)
        low_acc = accs[0]
        opt_acc = accs[opt_idx]
        # Error bar via quadrature
        gap_std = np.sqrt(stds[0]**2 + stds[opt_idx]**2)
        return opt_acc - low_acc, gap_std, results[opt_idx]['lr_frac']

    mlp_gap, mlp_gap_std, mlp_opt_frac = gen_gap(mlp_results)
    cnn_gap, cnn_gap_std, cnn_opt_frac = gen_gap(cnn_results)

    bars = ax.bar(['MLP\n(~14K params)', 'CNN\n(~269K params)'],
                  [mlp_gap, cnn_gap],
                  color=['steelblue', 'tomato'],
                  yerr=[mlp_gap_std, cnn_gap_std],
                  capsize=6)

    # Annotate with optimal LR fraction
    for bar, frac, val in zip(bars, [mlp_opt_frac, cnn_opt_frac],
                               [mlp_gap, cnn_gap]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.003,
                f'opt={frac*100:.0f}% EoS',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Generalization gap\n(acc_optimal − acc_low_LR)')
    ax.set_title('Generalization Gap by Architecture\n(Prediction: CNN >> MLP)')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(DRIVE_DIR, 'p3_depth_generalization_gap.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {path}", flush=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=== P3: Depth Comparison — MLP vs CNN Generalization Gap ===\n",
          flush=True)

    LR_FRACTIONS = np.array([0.05, 0.10, 0.15, 0.20, 0.25,
                              0.30, 0.35, 0.40, 0.50, 0.60, 0.70])
    N_SEEDS = 3

    # ── MLP ────────────────────────────────────────────────────────────────
    print("── MLP: preparing data ──", flush=True)
    X_tr_mlp, Y_tr_mlp, _, X_te_mlp, Y_te_mlp = make_synthetic_data()
    X_tr_mlp = X_tr_mlp.to(DEVICE)
    Y_tr_mlp = Y_tr_mlp.to(DEVICE)
    X_te_mlp = X_te_mlp.to(DEVICE)
    Y_te_mlp = Y_te_mlp.to(DEVICE)

    print("── MLP: estimating EoS ──", flush=True)
    _, mlp_eos = estimate_eos(MLP, {}, X_tr_mlp, Y_tr_mlp, warmup_steps=1000)

    print(f"\n── MLP: sweeping LR (EoS={mlp_eos:.5f}) ──", flush=True)
    mlp_results = sweep_architecture(
        'MLP', MLP, {},
        X_tr_mlp, Y_tr_mlp, X_te_mlp, Y_te_mlp,
        eos_threshold=mlp_eos,
        lr_fractions=LR_FRACTIONS,
        n_steps=5000,
        n_seeds=N_SEEDS,
    )
    np.save(os.path.join(DRIVE_DIR, 'p3_mlp_results.npy'), mlp_results)

    # ── CNN ────────────────────────────────────────────────────────────────
    print("\n── CNN: preparing data ──", flush=True)
    X_tr_cnn, Y_tr_cnn, X_te_cnn, Y_te_cnn = load_cifar10(
        n_train=5000, n_test=2000)

    print("── CNN: estimating EoS ──", flush=True)
    _, cnn_eos = estimate_eos(CNN, {}, X_tr_cnn, Y_tr_cnn, warmup_steps=1000)

    # If P1 already ran, load its EoS to save time
    eos_cache = os.path.join(DRIVE_DIR, 'p1_eos_threshold.npy')
    if os.path.exists(eos_cache):
        _, cnn_eos = np.load(eos_cache)
        print(f"Loaded EoS from P1 cache: {cnn_eos:.5f}", flush=True)

    print(f"\n── CNN: sweeping LR (EoS={cnn_eos:.5f}) ──", flush=True)
    cnn_results = sweep_architecture(
        'CNN', CNN, {},
        X_tr_cnn, Y_tr_cnn, X_te_cnn, Y_te_cnn,
        eos_threshold=cnn_eos,
        lr_fractions=LR_FRACTIONS,
        n_steps=3000,
        n_seeds=N_SEEDS,
    )
    np.save(os.path.join(DRIVE_DIR, 'p3_cnn_results.npy'), cnn_results)

    # ── Plot ───────────────────────────────────────────────────────────────
    print("\n── Generating plots ──", flush=True)
    plot_depth_comparison(mlp_results, cnn_results)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'Arch':>6}  {'LR frac':>8}  {'Acc mean':>10}  {'Acc std':>9}")
    print("-" * 38)
    for r in mlp_results:
        print(f"{'MLP':>6}  {r['lr_frac']*100:>7.0f}%  "
              f"{r['acc_mean']:>10.4f}  {r['acc_std']:>9.4f}")
    print()
    for r in cnn_results:
        print(f"{'CNN':>6}  {r['lr_frac']*100:>7.0f}%  "
              f"{r['acc_mean']:>10.4f}  {r['acc_std']:>9.4f}")

    def gap_summary(results, name):
        accs = [r['acc_mean'] for r in results]
        opt_idx = np.argmax(accs)
        gap = accs[opt_idx] - accs[0]
        opt_frac = results[opt_idx]['lr_frac']
        print(f"{name}: opt LR={opt_frac*100:.0f}% EoS  "
              f"gap={gap:.4f}  ({accs[0]:.4f} → {accs[opt_idx]:.4f})")

    print()
    gap_summary(mlp_results, 'MLP')
    gap_summary(cnn_results, 'CNN')
    print("\nPrediction: gap_CNN >> gap_MLP")


if __name__ == '__main__':
    main()
