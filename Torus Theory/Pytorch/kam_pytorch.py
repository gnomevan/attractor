"""
KAM Theory in Neural Network Training — PyTorch Extension
==========================================================

Extends the UV model results to nonlinear networks on real data.
Follows Cohen et al. (2021) setup: small MLP with tanh activation,
CIFAR-10 subset, full-batch gradient descent, MSE loss.

Experiments:
1. EoS reproduction with full Hessian spectrum tracking
2. Lyapunov exponent sweep across learning rates (THE KEY RESULT)
3. Bifurcation diagram projected onto top Hessian eigenvector
4. Eigenvalue ratio analysis at the critical transition

Requirements: torch, torchvision, numpy, matplotlib
Run time estimate: 30-90 min depending on GPU (CPU works but slower)

Usage:
    python kam_pytorch.py                  # full experiment suite
    python kam_pytorch.py --quick          # quick test (fewer lr values)
    python kam_pytorch.py --device cuda    # use GPU if available
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import time
import argparse
from copy import deepcopy

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Quick test with fewer samples')
parser.add_argument('--device', type=str, default='auto', help='cpu or cuda')
parser.add_argument('--output_dir', type=str, default='kam_pytorch_results')
parser.add_argument('--n_data', type=int, default=2000, help='Number of training examples')
parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden layer width')
parser.add_argument('--n_steps', type=int, default=5000, help='Training steps per run')
args = parser.parse_args()

if args.device == 'auto':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device(args.device)

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if args.quick:
    args.n_data = 500
    args.n_steps = 2000
    args.hidden_dim = 100

print(f"Device: {DEVICE}")
print(f"Data: {args.n_data} examples, Hidden dim: {args.hidden_dim}")
print(f"Output: {OUTPUT_DIR}/")

# ============================================================================
# DATA: CIFAR-10 subset
# ============================================================================

def load_cifar10_subset(n_train, seed=42):
    """Load a subset of CIFAR-10, flattened, normalized."""
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, 
                                    transform=transform)
    except Exception as e:
        print(f"Could not load CIFAR-10: {e}")
        print("Generating synthetic data instead...")
        return generate_synthetic_data(n_train, seed)
    
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:n_train]
    
    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img.view(-1))  # flatten: 3*32*32 = 3072
        labels.append(label)
    
    X = torch.stack(images).to(DEVICE)
    y_indices = torch.tensor(labels, device=DEVICE)
    
    # One-hot encode for MSE loss (following Cohen et al.)
    Y = torch.zeros(n_train, 10, device=DEVICE)
    Y.scatter_(1, y_indices.unsqueeze(1), 1.0)
    
    return X, Y


def generate_synthetic_data(n_train, seed=42):
    """Fallback: structured synthetic data if CIFAR-10 unavailable."""
    torch.manual_seed(seed)
    d_in = 100
    n_classes = 10
    
    # Create clustered data
    X = torch.randn(n_train, d_in, device=DEVICE)
    labels = torch.randint(0, n_classes, (n_train,), device=DEVICE)
    
    # Add class-dependent structure
    centers = torch.randn(n_classes, d_in, device=DEVICE) * 2
    X = X + centers[labels]
    
    Y = torch.zeros(n_train, n_classes, device=DEVICE)
    Y.scatter_(1, labels.unsqueeze(1), 1.0)
    
    return X, Y


# ============================================================================
# MODEL: MLP with tanh (following Cohen et al. 2021)
# ============================================================================

class TanhMLP(nn.Module):
    """2-hidden-layer MLP with tanh activation, matching Cohen et al."""
    
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
# HESSIAN TOOLS
# ============================================================================

def hessian_top_eigenvalues(model, loss_fn, X, Y, n_eigs=5, max_iter=100, tol=1e-4):
    """
    Compute top-n eigenvalues of the Hessian via power iteration
    using Hessian-vector products (memory efficient).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    def hvp(vec):
        """Hessian-vector product via double backprop."""
        model.zero_grad()
        output = model(X)
        loss = loss_fn(output, Y)
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        
        grad_vec_product = torch.sum(flat_grad * vec)
        hvp_grads = torch.autograd.grad(grad_vec_product, params)
        flat_hvp = torch.cat([g.reshape(-1) for g in hvp_grads])
        
        return flat_hvp.detach()
    
    n_params = sum(p.numel() for p in params)
    eigenvalues = []
    eigenvectors = []
    
    for i in range(n_eigs):
        # Random initial vector
        v = torch.randn(n_params, device=DEVICE)
        
        # Orthogonalize against previous eigenvectors
        for ev in eigenvectors:
            v = v - torch.dot(v, ev) * ev
        v = v / v.norm()
        
        eigenvalue = 0.0
        
        for iteration in range(max_iter):
            Hv = hvp(v)
            
            # Orthogonalize against previous
            for ev in eigenvectors:
                Hv = Hv - torch.dot(Hv, ev) * ev
            
            new_eigenvalue = torch.dot(v, Hv).item()
            
            norm_Hv = Hv.norm()
            if norm_Hv < 1e-15:
                break
            v_new = Hv / norm_Hv
            
            if abs(new_eigenvalue - eigenvalue) / (abs(eigenvalue) + 1e-10) < tol:
                eigenvalue = new_eigenvalue
                break
            
            eigenvalue = new_eigenvalue
            v = v_new
        
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
    
    return eigenvalues, eigenvectors


def compute_gradient(model, loss_fn, X, Y):
    """Compute flat gradient vector."""
    model.zero_grad()
    output = model(X)
    loss = loss_fn(output, Y)
    loss.backward()
    
    flat_grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    return flat_grad.detach(), loss.item()


def get_flat_params(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].reshape(p.shape))
        offset += numel


def gd_step(model, loss_fn, X, Y, lr):
    """One full-batch gradient descent step. Returns loss."""
    grad, loss = compute_gradient(model, loss_fn, X, Y)
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad
    return loss


# ============================================================================
# EXPERIMENT 1: EoS with Hessian Spectrum Tracking
# ============================================================================

def experiment_eos(X, Y, lr, n_steps, d_hidden, hessian_interval=100, n_eigs=5):
    """Run training and track loss, sharpness, and eigenvalue spectrum."""
    d_in = X.shape[1]
    d_out = Y.shape[1]
    
    model = TanhMLP(d_in, d_hidden, d_out).to(DEVICE)
    loss_fn = nn.MSELoss()
    
    results = {
        'losses': [],
        'sharpnesses': [],
        'all_eigenvalues': [],
        'eigenvalue_ratios': [],
        'accuracies': [],
        'lr': lr,
        'threshold': 2.0 / lr,
        'n_params': count_params(model)
    }
    
    for step in range(n_steps):
        # Forward + backward + step
        loss_val = gd_step(model, loss_fn, X, Y, lr)
        results['losses'].append(loss_val)
        
        # Check divergence
        if not np.isfinite(loss_val) or loss_val > 1e6:
            print(f"    Diverged at step {step}, loss={loss_val:.2e}")
            break
        
        # Accuracy
        if step % hessian_interval == 0:
            with torch.no_grad():
                pred = model(X).argmax(dim=1)
                true = Y.argmax(dim=1)
                acc = (pred == true).float().mean().item()
                results['accuracies'].append((step, acc))
        
        # Hessian spectrum
        if step % hessian_interval == 0:
            try:
                eigs, _ = hessian_top_eigenvalues(model, loss_fn, X, Y, n_eigs=n_eigs)
                results['sharpnesses'].append((step, eigs[0]))
                results['all_eigenvalues'].append((step, eigs))
                
                # Eigenvalue ratios
                pos_eigs = [e for e in eigs if e > 0.01]
                if len(pos_eigs) >= 2:
                    results['eigenvalue_ratios'].append((step, pos_eigs[0] / pos_eigs[1]))
                
                if step % (hessian_interval * 5) == 0:
                    print(f"    Step {step}: loss={loss_val:.4f}, λ_max={eigs[0]:.2f}, "
                          f"2/η={results['threshold']:.2f}, acc={acc:.2%}")
            except Exception as e:
                print(f"    Hessian computation failed at step {step}: {e}")
    
    return results, model


# ============================================================================
# EXPERIMENT 2: Lyapunov Exponent Sweep (THE KEY EXPERIMENT)
# ============================================================================

def compute_lyapunov(X, Y, lr, d_hidden, n_steps, perturbation=1e-8, 
                     renorm_interval=50, seed=42):
    """
    Compute maximal Lyapunov exponent for a given learning rate.
    
    Run two copies of GD from nearby initial conditions,
    track divergence rate, periodically renormalize.
    """
    d_in = X.shape[1]
    d_out = Y.shape[1]
    loss_fn = nn.MSELoss()
    
    # Reference model
    torch.manual_seed(seed)
    model_ref = TanhMLP(d_in, d_hidden, d_out).to(DEVICE)
    
    # Perturbed model (same init + small perturbation)
    torch.manual_seed(seed)
    model_pert = TanhMLP(d_in, d_hidden, d_out).to(DEVICE)
    
    with torch.no_grad():
        params_pert = get_flat_params(model_pert)
        direction = torch.randn_like(params_pert)
        direction = direction / direction.norm() * perturbation
        set_flat_params(model_pert, params_pert + direction)
    
    cumulative_log_stretch = 0.0
    lyapunov_history = []
    losses = []
    
    for step in range(n_steps):
        # Step both models
        loss_ref = gd_step(model_ref, loss_fn, X, Y, lr)
        gd_step(model_pert, loss_fn, X, Y, lr)
        losses.append(loss_ref)
        
        # Check divergence
        if not np.isfinite(loss_ref) or loss_ref > 1e6:
            break
        
        # Renormalize and measure stretch
        if (step + 1) % renorm_interval == 0:
            with torch.no_grad():
                params_r = get_flat_params(model_ref)
                params_p = get_flat_params(model_pert)
                diff = params_p - params_r
                dist = diff.norm().item()
                
                if dist > 0 and np.isfinite(dist) and dist < 1e10:
                    cumulative_log_stretch += np.log(dist / perturbation)
                    lyap = cumulative_log_stretch / (step + 1)
                    lyapunov_history.append((step, lyap))
                    
                    # Renormalize
                    diff_normalized = diff / diff.norm() * perturbation
                    set_flat_params(model_pert, params_r + diff_normalized)
                else:
                    break
    
    final_lyap = lyapunov_history[-1][1] if lyapunov_history else np.nan
    return final_lyap, lyapunov_history, losses


def experiment_lyapunov_sweep(X, Y, d_hidden, n_steps, lr_range, n_lr):
    """Sweep learning rate and compute Lyapunov exponent for each."""
    lrs = np.linspace(lr_range[0], lr_range[1], n_lr)
    results = []
    
    for i, lr in enumerate(lrs):
        t0 = time.time()
        lyap, history, losses = compute_lyapunov(X, Y, lr, d_hidden, n_steps)
        elapsed = time.time() - t0
        
        status = "CHAOTIC" if lyap > 0 else "STABLE"
        if np.isnan(lyap):
            status = "DIVERGED"
        
        print(f"  [{i+1}/{n_lr}] η={lr:.5f}: λ_Lyap={lyap:+.6f} ({status}) [{elapsed:.1f}s]")
        results.append({'lr': lr, 'lyapunov': lyap, 'diverged': np.isnan(lyap)})
    
    return results


# ============================================================================
# EXPERIMENT 3: Bifurcation Diagram
# ============================================================================

def experiment_bifurcation(X, Y, d_hidden, lr_range, n_lr, n_train_steps, 
                           n_record=200, seed=42):
    """
    Sweep learning rate, record late-time loss values.
    Project onto top Hessian eigenvector for 1D bifurcation diagram.
    """
    loss_fn = nn.MSELoss()
    lrs = np.linspace(lr_range[0], lr_range[1], n_lr)
    bif_data = []
    
    for i, lr in enumerate(lrs):
        torch.manual_seed(seed)
        d_in = X.shape[1]
        d_out = Y.shape[1]
        model = TanhMLP(d_in, d_hidden, d_out).to(DEVICE)
        
        # Train
        diverged = False
        for step in range(n_train_steps):
            loss_val = gd_step(model, loss_fn, X, Y, lr)
            if not np.isfinite(loss_val) or loss_val > 1e6:
                diverged = True
                break
        
        if diverged:
            bif_data.append({'lr': lr, 'late_losses': [], 'projections': []})
            continue
        
        # Record late-time losses and project onto top eigenvector
        late_losses = []
        projections = []
        
        # Get top eigenvector at this point
        try:
            eigs, evecs = hessian_top_eigenvalues(model, loss_fn, X, Y, n_eigs=1)
            top_evec = evecs[0]
        except:
            top_evec = None
        
        ref_params = get_flat_params(model).clone()
        
        for step in range(n_record):
            loss_val = gd_step(model, loss_fn, X, Y, lr)
            if np.isfinite(loss_val) and loss_val < 1e6:
                late_losses.append(loss_val)
                if top_evec is not None:
                    proj = torch.dot(get_flat_params(model) - ref_params, top_evec).item()
                    projections.append(proj)
            else:
                break
        
        bif_data.append({
            'lr': lr, 
            'late_losses': late_losses, 
            'projections': projections
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_lr}] η={lr:.5f}: {len(late_losses)} late-time samples")
    
    return bif_data


# ============================================================================
# PLOTTING
# ============================================================================

def plot_lyapunov_sweep(results, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    lrs = [r['lr'] for r in results if not r['diverged']]
    lyaps = [r['lyapunov'] for r in results if not r['diverged']]
    
    if not lrs:
        print("  No valid Lyapunov results to plot.")
        plt.close()
        return
    
    colors = ['#e74c3c' if l > 0 else '#3498db' for l in lyaps]
    ax.scatter(lrs, lyaps, c=colors, s=30, zorder=3, edgecolors='white', linewidth=0.5)
    ax.plot(lrs, lyaps, linewidth=0.8, color='gray', alpha=0.4, zorder=2)
    ax.axhline(y=0, color='black', linewidth=1)
    
    ax.fill_between(lrs, 0, [max(0, l) for l in lyaps], alpha=0.08, color='red')
    ax.fill_between(lrs, [min(0, l) for l in lyaps], 0, alpha=0.08, color='blue')
    
    # Find and mark critical η
    for i in range(len(lyaps) - 1):
        if lyaps[i] <= 0 and lyaps[i+1] > 0:
            eta_c = (lrs[i] + lrs[i+1]) / 2
            ax.axvline(x=eta_c, color='orange', linestyle='--', linewidth=1.5,
                      label=f'η_c ≈ {eta_c:.4f} (KAM torus breakdown)')
            break
    
    ax.set_xlabel('Learning Rate η', fontsize=13)
    ax.set_ylabel('Maximal Lyapunov Exponent', fontsize=13)
    ax.set_title('Torus-to-Chaos Transition in Neural Network Training\n'
                 'Nonlinear MLP on CIFAR-10 — Full-Batch Gradient Descent',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    
    # Annotations
    ax.text(0.02, 0.95, 'KAM tori intact\n(quasi-periodic training)', 
            transform=ax.transAxes, fontsize=10, color='#3498db', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(0.98, 0.95, 'KAM tori broken\n(chaotic training)', 
            transform=ax.transAxes, fontsize=10, color='#e74c3c', va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_eos(result, output_path):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    losses = result['losses']
    valid = [l for l in losses if l > 0 and np.isfinite(l)]
    if valid:
        ax1.semilogy(range(len(valid)), valid, linewidth=0.4, color='#2c3e50')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    
    # Sharpness
    ax2 = fig.add_subplot(gs[0, 1])
    if result['sharpnesses']:
        steps_s = [s[0] for s in result['sharpnesses']]
        vals_s = [s[1] for s in result['sharpnesses']]
        ax2.plot(steps_s, vals_s, linewidth=0.8, color='#e74c3c', label='λ_max')
        ax2.axhline(y=result['threshold'], color='#3498db', linestyle='--',
                   linewidth=1.5, label=f'2/η = {result["threshold"]:.1f}')
        ax2.legend(fontsize=9)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Sharpness')
    ax2.set_title('Edge of Stability')
    
    # Eigenvalue spectrum
    ax3 = fig.add_subplot(gs[1, 0])
    if result['all_eigenvalues']:
        colors_eig = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#8e44ad']
        n_show = min(5, len(result['all_eigenvalues'][0][1]))
        for i in range(n_show):
            steps_e = [e[0] for e in result['all_eigenvalues']]
            vals_e = [e[1][i] for e in result['all_eigenvalues']]
            ax3.plot(steps_e, vals_e, linewidth=0.5, color=colors_eig[i], label=f'λ_{i+1}')
        ax3.axhline(y=result['threshold'], color='gray', linestyle='--', linewidth=0.8)
        ax3.legend(fontsize=8)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('Top Hessian Eigenvalues')
    
    # Eigenvalue ratios
    ax4 = fig.add_subplot(gs[1, 1])
    if result['eigenvalue_ratios']:
        steps_r = [r[0] for r in result['eigenvalue_ratios']]
        vals_r = [r[1] for r in result['eigenvalue_ratios']]
        ax4.plot(steps_r, vals_r, linewidth=0.6, color='#8e44ad')
        for p, q, label in [(1,1,'1:1'), (2,1,'2:1'), (3,2,'3:2')]:
            ax4.axhline(y=p/q, color='orange', linestyle=':', linewidth=0.8, alpha=0.4)
            ax4.text(steps_r[0], p/q + 0.03, label, fontsize=8, color='orange')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('λ₁/λ₂')
    ax4.set_title('Top Eigenvalue Ratio (KAM Diagnostic)')
    
    fig.suptitle(f'EoS Dynamics: MLP(tanh), η={result["lr"]:.4f}, '
                 f'{result["n_params"]} params', fontsize=13, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_bifurcation(bif_data, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Loss bifurcation
    ax = axes[0]
    for entry in bif_data:
        if entry['late_losses']:
            ll = entry['late_losses']
            ax.scatter([entry['lr']] * len(ll), ll, s=0.3, c='#2c3e50', alpha=0.3)
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Late-Time Loss', fontsize=12)
    ax.set_title('Bifurcation: Loss Values')
    
    # Projection bifurcation (onto top Hessian eigenvector)
    ax = axes[1]
    for entry in bif_data:
        if entry['projections']:
            pp = entry['projections']
            ax.scatter([entry['lr']] * len(pp), pp, s=0.3, c='#8e44ad', alpha=0.3)
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Projection onto top eigenvector', fontsize=12)
    ax.set_title('Bifurcation: Hessian Eigenvector Projection')
    
    fig.suptitle('Bifurcation Diagram — Period-Doubling Route to Chaos',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print(" KAM Theory in Neural Network Training — PyTorch Experiments")
    print("=" * 65)
    
    # Load data
    print("\n[0] Loading data...")
    X, Y = load_cifar10_subset(args.n_data)
    d_in = X.shape[1]
    d_out = Y.shape[1]
    print(f"  Data: {X.shape[0]} examples, d_in={d_in}, d_out={d_out}")
    
    loss_fn = nn.MSELoss()
    
    # Determine initial sharpness for learning rate calibration
    print("\n[0b] Calibrating learning rate range...")
    torch.manual_seed(42)
    model_test = TanhMLP(d_in, args.hidden_dim, d_out).to(DEVICE)
    eigs_init, _ = hessian_top_eigenvalues(model_test, loss_fn, X, Y, n_eigs=3)
    lambda_max_init = eigs_init[0]
    lr_eos = 2.0 / lambda_max_init
    print(f"  Initial λ_max = {lambda_max_init:.4f}")
    print(f"  EoS threshold learning rate: 2/λ_max = {lr_eos:.6f}")
    del model_test
    
    # --- Experiment 1: EoS tracking at 0.5× threshold ---
    print(f"\n{'='*65}")
    print("[1/3] Edge of Stability tracking")
    print(f"{'='*65}")
    
    lr_eos_run = lr_eos * 0.5
    print(f"  Running at η = {lr_eos_run:.6f} (0.5× EoS threshold)")
    eos_result, _ = experiment_eos(X, Y, lr_eos_run, args.n_steps, args.hidden_dim,
                                   hessian_interval=max(50, args.n_steps // 50))
    plot_eos(eos_result, f'{OUTPUT_DIR}/01_eos_tracking.png')
    print("  Saved: 01_eos_tracking.png")
    
    # --- Experiment 2: Lyapunov sweep (THE KEY RESULT) ---
    print(f"\n{'='*65}")
    print("[2/3] Lyapunov Exponent Sweep — The Torus-to-Chaos Transition")
    print(f"{'='*65}")
    
    n_lr = 15 if args.quick else 30
    lr_min = lr_eos * 0.1
    lr_max = lr_eos * 1.2
    lyap_steps = args.n_steps
    
    print(f"  Sweeping η from {lr_min:.6f} to {lr_max:.6f} ({n_lr} values)")
    print(f"  {lyap_steps} steps per run")
    
    lyap_results = experiment_lyapunov_sweep(X, Y, args.hidden_dim, lyap_steps,
                                             lr_range=(lr_min, lr_max), n_lr=n_lr)
    plot_lyapunov_sweep(lyap_results, f'{OUTPUT_DIR}/02_lyapunov_sweep.png')
    print("  Saved: 02_lyapunov_sweep.png")
    
    # Report critical η
    valid = [r for r in lyap_results if not r['diverged']]
    for i in range(len(valid) - 1):
        if valid[i]['lyapunov'] <= 0 and valid[i+1]['lyapunov'] > 0:
            eta_c = (valid[i]['lr'] + valid[i+1]['lr']) / 2
            print(f"\n  *** CRITICAL LEARNING RATE (KAM torus breakdown): η_c ≈ {eta_c:.6f}")
            print(f"  *** EoS threshold (2/λ_max_init): {lr_eos:.6f}")
            print(f"  *** Ratio η_c / (2/λ_max): {eta_c / lr_eos:.3f}")
            break
    
    # --- Experiment 3: Bifurcation diagram ---
    print(f"\n{'='*65}")
    print("[3/3] Bifurcation Diagram")
    print(f"{'='*65}")
    
    n_lr_bif = 20 if args.quick else 50
    bif_data = experiment_bifurcation(X, Y, args.hidden_dim,
                                      lr_range=(lr_min, lr_max),
                                      n_lr=n_lr_bif,
                                      n_train_steps=args.n_steps,
                                      n_record=200)
    plot_bifurcation(bif_data, f'{OUTPUT_DIR}/03_bifurcation.png')
    print("  Saved: 03_bifurcation.png")
    
    # --- Summary ---
    print(f"\n{'='*65}")
    print(" COMPLETE")
    print(f"{'='*65}")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print(f"\nModel: TanhMLP, hidden_dim={args.hidden_dim}, {count_params(TanhMLP(d_in, args.hidden_dim, d_out))} params")
    print(f"Data: {args.n_data} CIFAR-10 examples")
    print(f"Device: {DEVICE}")
    print(f"\nFiles:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size//1024}KB)")
    
    # Save results as numpy for further analysis
    np.savez(f'{OUTPUT_DIR}/lyapunov_data.npz',
             lrs=np.array([r['lr'] for r in lyap_results]),
             lyapunovs=np.array([r['lyapunov'] for r in lyap_results]),
             diverged=np.array([r['diverged'] for r in lyap_results]),
             lr_eos=lr_eos,
             lambda_max_init=lambda_max_init)
    print(f"\n  Raw data saved to: {OUTPUT_DIR}/lyapunov_data.npz")


if __name__ == '__main__':
    main()
