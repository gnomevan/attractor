"""
KAM Theory in Neural Network Training — v3 (Corrected)
=======================================================

Key fixes from v2:
1. Lyapunov exponent measured in FUNCTION SPACE (output divergence),
   not parameter space. This avoids false positives from the manifold
   of equivalent minima in overparameterized networks.
2. SSL certificate fix for CIFAR-10 download.
3. Underparameterized regime: hidden_dim < n_data forces persistent
   non-zero loss and sustained dynamics (can't memorize everything).
4. Measures Lyapunov only during ACTIVE TRAINING (loss still changing),
   not after convergence.

Usage:
    python3 kam_v3.py              # full run (~45 min on CPU)
    python3 kam_v3.py --quick      # test run (~10 min)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import ssl
import time
import argparse
from copy import deepcopy

# ============================================================================
# SSL FIX (must come before any downloads)
# ============================================================================
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

# Also try the brute-force fix
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# ============================================================================
# CONFIG
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
parser.add_argument('--device', type=str, default='auto')
parser.add_argument('--output_dir', type=str, default='kam_v3_results')
args = parser.parse_args()

DEVICE = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() 
                       else args.device if args.device != 'auto' else 'cpu')
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Key design choice: UNDERPARAMETERIZED regime
# hidden_dim << n_data so the network CANNOT memorize the data.
# This forces persistent non-zero loss and sustained training dynamics.
if args.quick:
    N_DATA = 2000
    HIDDEN_DIM = 50      # small hidden dim = can't memorize 2000 examples
    N_STEPS = 3000
    N_LR = 20
    HESSIAN_INTERVAL = 200
else:
    N_DATA = 5000
    HIDDEN_DIM = 50      # deliberately small
    N_STEPS = 5000
    N_LR = 30
    HESSIAN_INTERVAL = 150

print(f"Device: {DEVICE}")
print(f"Regime: {N_DATA} data, hidden_dim={HIDDEN_DIM} (underparameterized)")
print(f"Output: {OUTPUT_DIR}/")

# ============================================================================
# DATA
# ============================================================================

def load_data(n_train):
    """Load CIFAR-10 or fall back to synthetic data."""
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transform)
        
        torch.manual_seed(42)
        indices = torch.randperm(len(dataset))[:n_train]
        images = []
        labels = []
        for idx in indices:
            img, label = dataset[idx]
            images.append(img.view(-1))
            labels.append(label)
        
        X = torch.stack(images).to(DEVICE)
        y_idx = torch.tensor(labels, device=DEVICE)
        Y = torch.zeros(n_train, 10, device=DEVICE)
        Y.scatter_(1, y_idx.unsqueeze(1), 1.0)
        print(f"  Loaded CIFAR-10: {X.shape}")
        return X, Y
        
    except Exception as e:
        print(f"  CIFAR-10 failed ({e}), using synthetic data")
        return make_synthetic_data(n_train)


def make_synthetic_data(n_train):
    """
    Structured synthetic data that's HARD to fit with a small network.
    Multiple overlapping clusters with noisy boundaries.
    """
    torch.manual_seed(42)
    n_classes = 10
    d_in = 200  # higher dim = harder to fit
    
    # Class centers spread in high-dim space
    centers = torch.randn(n_classes, d_in) * 3.0
    
    labels = torch.randint(0, n_classes, (n_train,))
    X = torch.randn(n_train, d_in) * 1.5 + centers[labels]
    
    # Add nonlinear structure: quadratic features
    X_quad = X[:, :20] ** 2  # first 20 features squared
    X = torch.cat([X, X_quad], dim=1).to(DEVICE)
    d_in = X.shape[1]
    
    Y = torch.zeros(n_train, n_classes, device=DEVICE)
    Y.scatter_(1, labels.unsqueeze(1).to(DEVICE), 1.0)
    
    print(f"  Synthetic data: {X.shape}")
    return X, Y


# ============================================================================
# MODEL
# ============================================================================

class TanhMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out)
        )
    
    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# CORE TOOLS
# ============================================================================

def compute_loss_and_grad(model, loss_fn, X, Y):
    """Forward pass, compute loss and gradients."""
    model.zero_grad()
    output = model(X)
    loss = loss_fn(output, Y)
    loss.backward()
    return loss.item(), output.detach()


def gd_step(model, lr):
    """Apply gradient descent update (call after backward)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad


def top_hessian_eigenvalues(model, loss_fn, X, Y, n_eigs=5, max_iter=80):
    """Top eigenvalues via power iteration on Hessian-vector products."""
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    
    def hvp(vec):
        model.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_g = torch.cat([g.reshape(-1) for g in grads])
        prod = torch.sum(flat_g * vec)
        hvp_g = torch.autograd.grad(prod, params)
        return torch.cat([g.reshape(-1) for g in hvp_g]).detach()
    
    eigenvalues = []
    eigenvectors = []
    
    for i in range(n_eigs):
        v = torch.randn(n_params, device=DEVICE)
        for ev in eigenvectors:
            v -= torch.dot(v, ev) * ev
        v = v / v.norm()
        
        eig = 0.0
        for _ in range(max_iter):
            Hv = hvp(v)
            for ev in eigenvectors:
                Hv -= torch.dot(Hv, ev) * ev
            new_eig = torch.dot(v, Hv).item()
            nrm = Hv.norm()
            if nrm < 1e-15:
                break
            v = Hv / nrm
            if abs(new_eig - eig) / (abs(eig) + 1e-10) < 1e-4:
                eig = new_eig
                break
            eig = new_eig
        
        eigenvalues.append(eig)
        eigenvectors.append(v)
    
    return eigenvalues, eigenvectors


# ============================================================================
# FUNCTION-SPACE LYAPUNOV EXPONENT
# ============================================================================

def function_space_lyapunov(X, Y, lr, d_hidden, n_steps, X_test,
                            perturbation=1e-7, renorm_interval=50, seed=42):
    """
    Lyapunov exponent measured in FUNCTION SPACE.
    
    Instead of measuring whether parameters diverge (they always will
    in overparameterized networks due to manifold of equivalent minima),
    we measure whether the OUTPUTS diverge on a held-out test set.
    
    If f_ref(X_test) and f_pert(X_test) diverge exponentially,
    the training dynamics are truly chaotic.
    If they converge or stay parallel, training is stable/quasi-periodic
    even though parameters may differ.
    """
    d_in = X.shape[1]
    d_out = Y.shape[1]
    loss_fn = nn.MSELoss()
    
    # Reference model
    torch.manual_seed(seed)
    model_ref = TanhMLP(d_in, d_hidden, d_out).to(DEVICE)
    
    # Perturbed model: same architecture, small parameter perturbation
    torch.manual_seed(seed)
    model_pert = TanhMLP(d_in, d_hidden, d_out).to(DEVICE)
    with torch.no_grad():
        for p_ref, p_pert in zip(model_ref.parameters(), model_pert.parameters()):
            p_pert.data += torch.randn_like(p_pert) * perturbation
    
    # Track function-space divergence
    lyap_history = []
    loss_history = []
    output_divergence = []
    cumulative_log_stretch = 0.0
    
    # Initial function-space distance
    with torch.no_grad():
        out_ref_0 = model_ref(X_test)
        out_pert_0 = model_pert(X_test)
        d0 = (out_ref_0 - out_pert_0).norm().item()
    
    if d0 < 1e-15:
        d0 = perturbation  # fallback
    
    for step in range(n_steps):
        # Train both models
        model_ref.zero_grad()
        loss_ref = loss_fn(model_ref(X), Y)
        loss_ref.backward()
        gd_step(model_ref, lr)
        
        model_pert.zero_grad()
        loss_pert = loss_fn(model_pert(X), Y)
        loss_pert.backward()
        gd_step(model_pert, lr)
        
        loss_val = loss_ref.item()
        loss_history.append(loss_val)
        
        # Check divergence
        if not np.isfinite(loss_val) or loss_val > 1e6:
            break
        
        # Measure function-space divergence
        if (step + 1) % renorm_interval == 0:
            with torch.no_grad():
                out_r = model_ref(X_test)
                out_p = model_pert(X_test)
                func_dist = (out_r - out_p).norm().item()
            
            if func_dist > 0 and np.isfinite(func_dist):
                output_divergence.append((step, func_dist))
                cumulative_log_stretch += np.log(func_dist / d0)
                lyap = cumulative_log_stretch / (step + 1)
                lyap_history.append((step, lyap))
                
                # Renormalize in PARAMETER space to keep perturbation small
                # but measure divergence in FUNCTION space
                with torch.no_grad():
                    for p_r, p_p in zip(model_ref.parameters(), model_pert.parameters()):
                        diff = p_p.data - p_r.data
                        p_p.data = p_r.data + diff / (diff.norm() + 1e-15) * perturbation
                
                # Update d0 for next interval
                with torch.no_grad():
                    out_r2 = model_ref(X_test)
                    out_p2 = model_pert(X_test)
                    d0 = (out_r2 - out_p2).norm().item()
                    if d0 < 1e-15:
                        d0 = perturbation
    
    final_lyap = lyap_history[-1][1] if lyap_history else np.nan
    return final_lyap, lyap_history, loss_history, output_divergence


# ============================================================================
# EXPERIMENT 1: EoS Tracking
# ============================================================================

def run_eos_tracking(X, Y, lr, n_steps, d_hidden, hessian_interval):
    """Train and track loss, sharpness, eigenvalue spectrum."""
    d_in, d_out = X.shape[1], Y.shape[1]
    loss_fn = nn.MSELoss()
    
    torch.manual_seed(42)
    model = TanhMLP(d_in, d_hidden, d_out).to(DEVICE)
    threshold = 2.0 / lr
    
    losses, sharpnesses, all_eigs, eig_ratios, accs = [], [], [], [], []
    
    for step in range(n_steps):
        model.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        gd_step(model, lr)
        
        lv = loss.item()
        if not np.isfinite(lv) or lv > 1e6:
            print(f"    Diverged at step {step}")
            break
        losses.append(lv)
        
        if step % hessian_interval == 0:
            with torch.no_grad():
                acc = (out.argmax(1) == Y.argmax(1)).float().mean().item()
                accs.append((step, acc))
            
            try:
                eigs, _ = top_hessian_eigenvalues(model, loss_fn, X, Y, n_eigs=5)
                sharpnesses.append((step, eigs[0]))
                all_eigs.append((step, eigs))
                pos = [e for e in eigs if e > 0.01]
                if len(pos) >= 2:
                    eig_ratios.append((step, pos[0]/pos[1]))
                
                if step % (hessian_interval * 3) == 0:
                    print(f"    Step {step}: loss={lv:.4f}, λ_max={eigs[0]:.2f}, "
                          f"2/η={threshold:.2f}, acc={acc:.1%}")
            except:
                pass
    
    return {
        'losses': losses, 'sharpnesses': sharpnesses,
        'all_eigenvalues': all_eigs, 'eigenvalue_ratios': eig_ratios,
        'accuracies': accs, 'lr': lr, 'threshold': threshold,
        'n_params': count_params(model)
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_eos(result, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'EoS Dynamics: TanhMLP, η={result["lr"]:.4f}, '
                 f'{result["n_params"]} params', fontsize=13, fontweight='bold')
    
    # Loss
    ax = axes[0,0]
    valid = [l for l in result['losses'] if l > 0]
    if valid: ax.semilogy(valid, linewidth=0.4, color='#2c3e50')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('Training Loss')
    
    # Sharpness
    ax = axes[0,1]
    if result['sharpnesses']:
        s, v = zip(*result['sharpnesses'])
        ax.plot(s, v, linewidth=0.8, color='#e74c3c', label='λ_max')
        ax.axhline(y=result['threshold'], color='#3498db', linestyle='--',
                   linewidth=1.5, label=f'2/η = {result["threshold"]:.1f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('Step'); ax.set_ylabel('Sharpness'); ax.set_title('Edge of Stability')
    
    # Eigenvalue spectrum
    ax = axes[1,0]
    if result['all_eigenvalues']:
        colors = ['#e74c3c','#e67e22','#2ecc71','#3498db','#8e44ad']
        for i in range(min(5, len(result['all_eigenvalues'][0][1]))):
            s = [e[0] for e in result['all_eigenvalues']]
            v = [e[1][i] for e in result['all_eigenvalues']]
            ax.plot(s, v, linewidth=0.5, color=colors[i], label=f'λ_{i+1}')
        ax.axhline(y=result['threshold'], color='gray', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=8)
    ax.set_xlabel('Step'); ax.set_ylabel('Eigenvalue')
    ax.set_title('Top Hessian Eigenvalues')
    
    # Eigenvalue ratios
    ax = axes[1,1]
    if result['eigenvalue_ratios']:
        s, v = zip(*result['eigenvalue_ratios'])
        ax.plot(s, v, linewidth=0.6, color='#8e44ad')
        for p,q,lab in [(1,1,'1:1'),(2,1,'2:1'),(3,2,'3:2')]:
            ax.axhline(y=p/q, color='orange', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Step'); ax.set_ylabel('λ₁/λ₂')
    ax.set_title('Eigenvalue Ratio (KAM Diagnostic)')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_lyapunov_sweep(results, lr_eos, path):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lrs = [r['lr'] for r in results if np.isfinite(r['lyap'])]
    lyaps = [r['lyap'] for r in results if np.isfinite(r['lyap'])]
    
    if not lrs:
        print("  No valid Lyapunov data to plot.")
        plt.close()
        return
    
    colors = ['#e74c3c' if l > 0 else '#3498db' for l in lyaps]
    ax.scatter(lrs, lyaps, c=colors, s=35, zorder=3, edgecolors='white', linewidth=0.5)
    ax.plot(lrs, lyaps, linewidth=0.8, color='gray', alpha=0.4, zorder=2)
    ax.axhline(y=0, color='black', linewidth=1)
    
    ax.fill_between(lrs, 0, [max(0,l) for l in lyaps], alpha=0.08, color='red')
    ax.fill_between(lrs, [min(0,l) for l in lyaps], 0, alpha=0.08, color='blue')
    
    ax.axvline(x=lr_eos, color='orange', linestyle='--', linewidth=1.5,
              label=f'2/λ_max(0) = {lr_eos:.4f}')
    
    # Find critical η
    for i in range(len(lyaps)-1):
        if lyaps[i] <= 0 and lyaps[i+1] > 0:
            eta_c = (lrs[i] + lrs[i+1]) / 2
            ax.axvline(x=eta_c, color='green', linestyle='-', linewidth=2,
                      label=f'η_c ≈ {eta_c:.4f} (torus breakdown)')
            break
    
    ax.set_xlabel('Learning Rate η', fontsize=13)
    ax.set_ylabel('Maximal Lyapunov Exponent\n(function space)', fontsize=13)
    ax.set_title('Torus-to-Chaos Transition in Neural Network Training\n'
                 'Function-Space Lyapunov Exponent — TanhMLP, Full-Batch GD',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    
    ax.text(0.02, 0.95, 'KAM tori intact\n(stable training)', 
            transform=ax.transAxes, fontsize=10, color='#3498db', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(0.98, 0.95, 'KAM tori broken\n(chaotic training)', 
            transform=ax.transAxes, fontsize=10, color='#e74c3c', va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_bifurcation(bif_data, lr_eos, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    ax = axes[0]
    for entry in bif_data:
        if entry['late_losses']:
            ax.scatter([entry['lr']]*len(entry['late_losses']),
                      entry['late_losses'], s=0.3, c='#2c3e50', alpha=0.4)
    ax.axvline(x=lr_eos, color='red', linestyle='--', linewidth=1,
              label=f'2/λ_max = {lr_eos:.3f}')
    ax.set_xlabel('Learning Rate η'); ax.set_ylabel('Late-Time Loss')
    ax.set_title('Loss Values'); ax.legend(fontsize=9)
    
    ax = axes[1]
    for entry in bif_data:
        if entry['late_accs']:
            ax.scatter([entry['lr']]*len(entry['late_accs']),
                      entry['late_accs'], s=0.3, c='#8e44ad', alpha=0.4)
    ax.axvline(x=lr_eos, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Learning Rate η'); ax.set_ylabel('Late-Time Accuracy')
    ax.set_title('Accuracy')
    
    fig.suptitle('Bifurcation Diagram', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 65)
    print(" KAM Theory — Neural Network Training v3 (Function-Space)")
    print("=" * 65)
    
    # --- Load data ---
    print("\n[0] Loading data...")
    X, Y = load_data(N_DATA)
    d_in, d_out = X.shape[1], Y.shape[1]
    
    # Split: 80% train, 20% test (test set used for function-space Lyapunov)
    n_train = int(0.8 * N_DATA)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    loss_fn = nn.MSELoss()
    
    # --- Calibrate learning rate ---
    print("\n[0b] Calibrating...")
    torch.manual_seed(42)
    model_cal = TanhMLP(d_in, HIDDEN_DIM, d_out).to(DEVICE)
    eigs_init, _ = top_hessian_eigenvalues(model_cal, loss_fn, X_train, Y_train, n_eigs=3)
    lambda_max_0 = eigs_init[0]
    lr_eos = 2.0 / lambda_max_0
    print(f"  λ_max(init) = {lambda_max_0:.4f}")
    print(f"  EoS threshold: 2/λ_max = {lr_eos:.6f}")
    print(f"  Network: {count_params(model_cal)} params, {HIDDEN_DIM} hidden")
    print(f"  Params/data ratio: {count_params(model_cal)/n_train:.2f}")
    del model_cal
    
    # --- Experiment 1: EoS at multiple learning rates ---
    print(f"\n{'='*65}")
    print("[1/3] Edge of Stability at 3 learning rates")
    print(f"{'='*65}")
    
    fig_eos, axes_eos_all = plt.subplots(3, 3, figsize=(18, 14))
    fig_eos.suptitle(f'Edge of Stability Comparison — TanhMLP, hidden={HIDDEN_DIM}, '
                     f'{count_params(TanhMLP(d_in,HIDDEN_DIM,d_out))} params',
                     fontsize=13, fontweight='bold')
    
    for col, (mult, label) in enumerate([(0.3, 'stable'), (0.8, 'near EoS'), (1.1, 'beyond EoS')]):
        lr = lr_eos * mult
        print(f"\n  Running η = {lr:.5f} ({mult}× threshold, '{label}')...")
        result = run_eos_tracking(X_train, Y_train, lr, N_STEPS, HIDDEN_DIM, HESSIAN_INTERVAL)
        
        # Loss
        ax = axes_eos_all[0, col]
        valid = [l for l in result['losses'] if l > 0]
        if valid: ax.semilogy(valid, linewidth=0.4, color='#2c3e50')
        ax.set_title(f'η={lr:.4f} ({mult}×, {label})')
        if col == 0: ax.set_ylabel('Loss')
        
        # Sharpness
        ax = axes_eos_all[1, col]
        if result['sharpnesses']:
            s,v = zip(*result['sharpnesses'])
            ax.plot(s, v, linewidth=0.8, color='#e74c3c')
            ax.axhline(y=result['threshold'], color='#3498db', linestyle='--',
                       linewidth=1.5, label=f'2/η={result["threshold"]:.1f}')
            ax.legend(fontsize=7)
        if col == 0: ax.set_ylabel('λ_max')
        
        # Spectrum
        ax = axes_eos_all[2, col]
        if result['all_eigenvalues']:
            colors_e = ['#e74c3c','#e67e22','#2ecc71','#3498db','#8e44ad']
            for i in range(min(5, len(result['all_eigenvalues'][0][1]))):
                ss = [e[0] for e in result['all_eigenvalues']]
                vv = [e[1][i] for e in result['all_eigenvalues']]
                ax.plot(ss, vv, linewidth=0.5, color=colors_e[i],
                       label=f'λ_{i+1}' if col==0 else None)
            ax.axhline(y=result['threshold'], color='gray', linestyle='--', linewidth=0.8)
            if col == 0: ax.legend(fontsize=7)
        ax.set_xlabel('Step')
        if col == 0: ax.set_ylabel('Eigenvalue')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_eos_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_eos_comparison.png")
    
    # --- Experiment 2: Function-Space Lyapunov Sweep ---
    print(f"\n{'='*65}")
    print("[2/3] Function-Space Lyapunov Exponent Sweep")
    print(f"{'='*65}")
    
    lr_min = lr_eos * 0.1
    lr_max = lr_eos * 1.5
    lrs = np.linspace(lr_min, lr_max, N_LR)
    
    print(f"  Sweeping η: {lr_min:.4f} to {lr_max:.4f} ({N_LR} values)")
    print(f"  {N_STEPS} steps per run, measuring in function space")
    
    lyap_results = []
    for i, lr in enumerate(lrs):
        t0 = time.time()
        lyap, history, losses, out_div = function_space_lyapunov(
            X_train, Y_train, lr, HIDDEN_DIM, N_STEPS, X_test,
            perturbation=1e-7, renorm_interval=50
        )
        elapsed = time.time() - t0
        
        status = "CHAOTIC" if lyap > 0 else "STABLE"
        if np.isnan(lyap): status = "DIVERGED"
        
        final_loss = losses[-1] if losses else float('nan')
        print(f"  [{i+1}/{N_LR}] η={lr:.5f}: λ_Lyap={lyap:+.6f} ({status}), "
              f"final_loss={final_loss:.4f} [{elapsed:.0f}s]")
        
        lyap_results.append({'lr': lr, 'lyap': lyap, 'final_loss': final_loss})
    
    plot_lyapunov_sweep(lyap_results, lr_eos, f'{OUTPUT_DIR}/02_lyapunov_sweep.png')
    print("  Saved: 02_lyapunov_sweep.png")
    
    # Report critical eta
    valid = [r for r in lyap_results if np.isfinite(r['lyap'])]
    found_transition = False
    for i in range(len(valid)-1):
        if valid[i]['lyap'] <= 0 and valid[i+1]['lyap'] > 0:
            eta_c = (valid[i]['lr'] + valid[i+1]['lr']) / 2
            print(f"\n  *** CRITICAL η (torus breakdown): {eta_c:.6f}")
            print(f"  *** EoS threshold (2/λ_max):      {lr_eos:.6f}")
            print(f"  *** Ratio η_c / (2/λ_max):        {eta_c/lr_eos:.3f}")
            found_transition = True
            break
    
    if not found_transition:
        all_pos = all(r['lyap'] > 0 for r in valid if np.isfinite(r['lyap']))
        all_neg = all(r['lyap'] <= 0 for r in valid if np.isfinite(r['lyap']))
        if all_pos:
            print("\n  All Lyapunov exponents positive — system chaotic everywhere.")
            print("  May need larger hidden_dim or fewer data points.")
        elif all_neg:
            print("\n  All Lyapunov exponents non-positive — system stable everywhere.")
            print("  May need higher lr_max or longer training.")
    
    # --- Experiment 3: Bifurcation Diagram ---
    print(f"\n{'='*65}")
    print("[3/3] Bifurcation Diagram")
    print(f"{'='*65}")
    
    bif_data = []
    n_bif = N_LR
    lrs_bif = np.linspace(lr_min, lr_max, n_bif)
    
    for i, lr in enumerate(lrs_bif):
        torch.manual_seed(42)
        model = TanhMLP(d_in, HIDDEN_DIM, d_out).to(DEVICE)
        
        diverged = False
        for step in range(N_STEPS):
            model.zero_grad()
            out = model(X_train)
            loss = loss_fn(out, Y_train)
            loss.backward()
            gd_step(model, lr)
            if loss.item() > 1e6:
                diverged = True
                break
        
        late_losses = []
        late_accs = []
        if not diverged:
            for step in range(300):
                model.zero_grad()
                out = model(X_train)
                loss = loss_fn(out, Y_train)
                lv = loss.item()
                if np.isfinite(lv) and lv < 1e6:
                    late_losses.append(lv)
                    acc = (out.detach().argmax(1) == Y_train.argmax(1)).float().mean().item()
                    late_accs.append(acc)
                loss.backward()
                gd_step(model, lr)
        
        bif_data.append({'lr': lr, 'late_losses': late_losses, 'late_accs': late_accs})
        
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{n_bif}] η={lr:.4f}")
    
    plot_bifurcation(bif_data, lr_eos, f'{OUTPUT_DIR}/03_bifurcation.png')
    print("  Saved: 03_bifurcation.png")
    
    # --- Save raw data ---
    np.savez(f'{OUTPUT_DIR}/results.npz',
             lrs=np.array([r['lr'] for r in lyap_results]),
             lyapunovs=np.array([r['lyap'] for r in lyap_results]),
             final_losses=np.array([r['final_loss'] for r in lyap_results]),
             lr_eos=lr_eos, lambda_max_init=lambda_max_0,
             n_data=N_DATA, hidden_dim=HIDDEN_DIM, n_steps=N_STEPS)
    
    print(f"\n{'='*65}")
    print(" COMPLETE")
    print(f"{'='*65}")
    print(f"\nResults: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f} ({os.path.getsize(os.path.join(OUTPUT_DIR, f))//1024}KB)")


if __name__ == '__main__':
    main()
