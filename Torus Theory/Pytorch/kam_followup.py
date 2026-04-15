"""
KAM Theory — Follow-up Experiments
===================================

Two experiments:
1. REPRODUCIBILITY: Run the Lyapunov sweep across 5 random seeds
   to confirm the transition is not a fluke of initialization.
   
2. TRANSITION ZONE: Fine-grained sweep from η=0.005 to η=0.08
   with 30 points to map exactly where and how sharply the
   Lyapunov exponent crosses zero.

Usage:
    python3 kam_followup.py              # both experiments (~2-3 hours)
    python3 kam_followup.py --repro      # reproducibility only (~1.5 hours)
    python3 kam_followup.py --zoom       # transition zone only (~45 min)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import ssl
import time
import argparse

# SSL fix
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
except: pass
try: ssl._create_default_https_context = ssl._create_unverified_context
except: pass

parser = argparse.ArgumentParser()
parser.add_argument('--repro', action='store_true', help='Reproducibility only')
parser.add_argument('--zoom', action='store_true', help='Transition zone only')
parser.add_argument('--output_dir', type=str, default='kam_followup_results')
args = parser.parse_args()

RUN_REPRO = args.repro or (not args.repro and not args.zoom)
RUN_ZOOM = args.zoom or (not args.repro and not args.zoom)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_DATA = 2000
HIDDEN_DIM = 50
N_STEPS = 3000

print(f"Device: {DEVICE}")
print(f"Experiments: {'REPRO ' if RUN_REPRO else ''}{'ZOOM' if RUN_ZOOM else ''}")

# ============================================================================
# MODEL + DATA (same as v3)
# ============================================================================

class TanhMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)


def make_data(n_train, data_seed=42):
    """Structured synthetic data, same as v3."""
    torch.manual_seed(data_seed)
    n_classes = 10
    d_in = 200
    centers = torch.randn(n_classes, d_in) * 3.0
    labels = torch.randint(0, n_classes, (n_train,))
    X = torch.randn(n_train, d_in) * 1.5 + centers[labels]
    X_quad = X[:, :20] ** 2
    X = torch.cat([X, X_quad], dim=1).to(DEVICE)
    Y = torch.zeros(n_train, n_classes, device=DEVICE)
    Y.scatter_(1, labels.unsqueeze(1).to(DEVICE), 1.0)
    return X, Y


def gd_step(model, lr):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad


def get_initial_sharpness(X_train, Y_train, model_seed=0):
    """Get λ_max for a given initialization seed."""
    d_in, d_out = X_train.shape[1], Y_train.shape[1]
    loss_fn = nn.MSELoss()
    torch.manual_seed(model_seed)
    model = TanhMLP(d_in, HIDDEN_DIM, d_out).to(DEVICE)
    
    # Power iteration for top eigenvalue
    params = [p for p in model.parameters()]
    n_params = sum(p.numel() for p in params)
    
    def hvp(vec):
        model.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, Y_train)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_g = torch.cat([g.reshape(-1) for g in grads])
        prod = torch.sum(flat_g * vec)
        hvp_g = torch.autograd.grad(prod, params)
        return torch.cat([g.reshape(-1) for g in hvp_g]).detach()
    
    v = torch.randn(n_params, device=DEVICE)
    v = v / v.norm()
    eig = 0.0
    for _ in range(80):
        Hv = hvp(v)
        new_eig = torch.dot(v, Hv).item()
        nrm = Hv.norm()
        if nrm < 1e-15: break
        v = Hv / nrm
        if abs(new_eig - eig) / (abs(eig) + 1e-10) < 1e-4:
            eig = new_eig
            break
        eig = new_eig
    
    del model
    return eig


# ============================================================================
# FUNCTION-SPACE LYAPUNOV (same as v3)
# ============================================================================

def compute_lyapunov(X_train, Y_train, X_test, lr, model_seed=42,
                     perturbation=1e-7, renorm_interval=50):
    """Function-space Lyapunov exponent."""
    d_in, d_out = X_train.shape[1], Y_train.shape[1]
    loss_fn = nn.MSELoss()
    
    torch.manual_seed(model_seed)
    model_ref = TanhMLP(d_in, HIDDEN_DIM, d_out).to(DEVICE)
    
    torch.manual_seed(model_seed)
    model_pert = TanhMLP(d_in, HIDDEN_DIM, d_out).to(DEVICE)
    with torch.no_grad():
        for p_r, p_p in zip(model_ref.parameters(), model_pert.parameters()):
            p_p.data += torch.randn_like(p_p) * perturbation
    
    # Initial function-space distance
    with torch.no_grad():
        d0 = (model_ref(X_test) - model_pert(X_test)).norm().item()
    if d0 < 1e-15:
        d0 = perturbation
    
    cumulative = 0.0
    lyap_history = []
    
    for step in range(N_STEPS):
        # Train ref
        model_ref.zero_grad()
        loss_r = loss_fn(model_ref(X_train), Y_train)
        loss_r.backward()
        gd_step(model_ref, lr)
        
        # Train pert
        model_pert.zero_grad()
        loss_p = loss_fn(model_pert(X_train), Y_train)
        loss_p.backward()
        gd_step(model_pert, lr)
        
        if not np.isfinite(loss_r.item()) or loss_r.item() > 1e6:
            break
        
        if (step + 1) % renorm_interval == 0:
            with torch.no_grad():
                func_dist = (model_ref(X_test) - model_pert(X_test)).norm().item()
            
            if func_dist > 0 and np.isfinite(func_dist):
                cumulative += np.log(func_dist / d0)
                lyap = cumulative / (step + 1)
                lyap_history.append((step, lyap))
                
                # Renormalize parameters
                with torch.no_grad():
                    for p_r, p_p in zip(model_ref.parameters(), model_pert.parameters()):
                        diff = p_p.data - p_r.data
                        p_p.data = p_r.data + diff / (diff.norm() + 1e-15) * perturbation
                
                # Update d0
                with torch.no_grad():
                    d0 = (model_ref(X_test) - model_pert(X_test)).norm().item()
                    if d0 < 1e-15:
                        d0 = perturbation
            else:
                break
    
    final_lyap = lyap_history[-1][1] if lyap_history else np.nan
    final_loss = loss_r.item() if np.isfinite(loss_r.item()) else np.nan
    
    del model_ref, model_pert
    return final_lyap, final_loss


# ============================================================================
# EXPERIMENT 1: REPRODUCIBILITY (5 seeds)
# ============================================================================

def experiment_reproducibility(X_train, Y_train, X_test):
    print("\n" + "=" * 65)
    print(" EXPERIMENT 1: Reproducibility Across 5 Random Seeds")
    print("=" * 65)
    
    seeds = [0, 1, 2, 3, 4]
    n_lr = 20
    
    # Use same lr range as v3
    lambda_max_0 = get_initial_sharpness(X_train, Y_train, model_seed=0)
    lr_eos = 2.0 / lambda_max_0
    lr_min = lr_eos * 0.05
    lr_max = lr_eos * 1.5
    lrs = np.linspace(lr_min, lr_max, n_lr)
    
    print(f"  λ_max(0) = {lambda_max_0:.4f}, EoS threshold = {lr_eos:.4f}")
    print(f"  η range: {lr_min:.4f} to {lr_max:.4f}, {n_lr} values")
    print(f"  Seeds: {seeds}")
    print(f"  Estimated time: {len(seeds) * n_lr * 75 / 60:.0f} minutes\n")
    
    all_results = {}
    
    for seed in seeds:
        print(f"  --- Seed {seed} ---")
        results = []
        for i, lr in enumerate(lrs):
            t0 = time.time()
            lyap, final_loss = compute_lyapunov(X_train, Y_train, X_test, lr, 
                                                 model_seed=seed)
            elapsed = time.time() - t0
            status = "STABLE" if lyap <= 0 else "CHAOTIC"
            if np.isnan(lyap): status = "DIVERGED"
            
            results.append({'lr': lr, 'lyap': lyap, 'loss': final_loss})
            
            if (i + 1) % 5 == 0 or i == 0:
                print(f"    [{i+1}/{n_lr}] η={lr:.5f}: λ={lyap:+.6f} ({status}) [{elapsed:.0f}s]")
        
        all_results[seed] = results
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: all seeds overlaid
    ax = axes[0]
    colors_seed = ['#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#8e44ad']
    
    for seed, color in zip(seeds, colors_seed):
        res = all_results[seed]
        lr_vals = [r['lr'] for r in res if np.isfinite(r['lyap'])]
        ly_vals = [r['lyap'] for r in res if np.isfinite(r['lyap'])]
        ax.plot(lr_vals, ly_vals, 'o-', color=color, markersize=4, linewidth=0.8,
                alpha=0.7, label=f'Seed {seed}')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=lr_eos, color='orange', linestyle='--', linewidth=1.5,
              label=f'2/λ_max = {lr_eos:.3f}')
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent (function space)', fontsize=12)
    ax.set_title('Reproducibility: 5 Seeds', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Right: mean ± std
    ax = axes[1]
    mean_lyaps = []
    std_lyaps = []
    for i in range(n_lr):
        vals = [all_results[s][i]['lyap'] for s in seeds if np.isfinite(all_results[s][i]['lyap'])]
        if vals:
            mean_lyaps.append(np.mean(vals))
            std_lyaps.append(np.std(vals))
        else:
            mean_lyaps.append(np.nan)
            std_lyaps.append(np.nan)
    
    mean_lyaps = np.array(mean_lyaps)
    std_lyaps = np.array(std_lyaps)
    valid = np.isfinite(mean_lyaps)
    
    ax.plot(lrs[valid], mean_lyaps[valid], 'o-', color='#2c3e50', markersize=5, linewidth=1.2)
    ax.fill_between(lrs[valid], 
                     (mean_lyaps - std_lyaps)[valid],
                     (mean_lyaps + std_lyaps)[valid],
                     alpha=0.2, color='#3498db')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=lr_eos, color='orange', linestyle='--', linewidth=1.5,
              label=f'2/λ_max = {lr_eos:.3f}')
    
    # Find mean transition point
    for i in range(len(mean_lyaps) - 1):
        if valid[i] and valid[i+1] and mean_lyaps[i] <= 0 and mean_lyaps[i+1] > 0:
            eta_c_mean = (lrs[i] + lrs[i+1]) / 2
            ax.axvline(x=eta_c_mean, color='green', linewidth=2,
                      label=f'η_c ≈ {eta_c_mean:.4f}')
            break
    
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent (mean ± std)', fontsize=12)
    ax.set_title('Mean Across Seeds with Error Bands', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_reproducibility.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: 01_reproducibility.png")
    
    # Save raw data
    np.savez(f'{OUTPUT_DIR}/reproducibility_data.npz',
             lrs=lrs, seeds=np.array(seeds), lr_eos=lr_eos,
             **{f'lyap_seed{s}': np.array([r['lyap'] for r in all_results[s]]) for s in seeds},
             **{f'loss_seed{s}': np.array([r['loss'] for r in all_results[s]]) for s in seeds})
    
    # Report per-seed transition points
    print("\n  Transition points per seed:")
    eta_cs = []
    for seed in seeds:
        res = all_results[seed]
        for i in range(len(res) - 1):
            l1, l2 = res[i]['lyap'], res[i+1]['lyap']
            if np.isfinite(l1) and np.isfinite(l2) and l1 <= 0 and l2 > 0:
                ec = (res[i]['lr'] + res[i+1]['lr']) / 2
                eta_cs.append(ec)
                print(f"    Seed {seed}: η_c ≈ {ec:.5f} ({ec/lr_eos:.3f} × EoS)")
                break
        else:
            if all(r['lyap'] > 0 for r in res if np.isfinite(r['lyap'])):
                print(f"    Seed {seed}: all CHAOTIC (η_c < {lrs[0]:.5f})")
            else:
                print(f"    Seed {seed}: transition not found")
    
    if eta_cs:
        print(f"\n  Mean η_c = {np.mean(eta_cs):.5f} ± {np.std(eta_cs):.5f}")
        print(f"  Mean η_c / (2/λ_max) = {np.mean(eta_cs)/lr_eos:.3f} ± {np.std(eta_cs)/lr_eos:.3f}")
    
    return all_results


# ============================================================================
# EXPERIMENT 2: TRANSITION ZONE (fine-grained)
# ============================================================================

def experiment_transition_zone(X_train, Y_train, X_test):
    print("\n" + "=" * 65)
    print(" EXPERIMENT 2: Fine-Grained Transition Zone")
    print("=" * 65)
    
    # From v3 results, transition is near η ≈ 0.038
    # Sweep from 0.005 to 0.08 with high resolution
    n_lr = 30
    lr_min = 0.005
    lr_max = 0.08
    lrs = np.linspace(lr_min, lr_max, n_lr)
    
    # Use 3 seeds for each point
    seeds = [0, 1, 2]
    
    lambda_max_0 = get_initial_sharpness(X_train, Y_train, model_seed=0)
    lr_eos = 2.0 / lambda_max_0
    
    print(f"  η range: {lr_min:.4f} to {lr_max:.4f}, {n_lr} values")
    print(f"  Seeds per point: {seeds}")
    print(f"  Estimated time: {n_lr * len(seeds) * 75 / 60:.0f} minutes\n")
    
    all_lyaps = np.zeros((len(seeds), n_lr))
    
    for si, seed in enumerate(seeds):
        print(f"  --- Seed {seed} ---")
        for i, lr in enumerate(lrs):
            t0 = time.time()
            lyap, _ = compute_lyapunov(X_train, Y_train, X_test, lr, model_seed=seed)
            elapsed = time.time() - t0
            all_lyaps[si, i] = lyap
            
            status = "STABLE" if lyap <= 0 else "CHAOTIC"
            if np.isnan(lyap): status = "?"
            
            if (i + 1) % 5 == 0 or i == 0:
                print(f"    [{i+1}/{n_lr}] η={lr:.5f}: λ={lyap:+.7f} ({status}) [{elapsed:.0f}s]")
    
    # Compute statistics
    mean_lyaps = np.nanmean(all_lyaps, axis=0)
    std_lyaps = np.nanstd(all_lyaps, axis=0)
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: individual seeds + mean
    ax = axes[0]
    colors_s = ['#e74c3c', '#3498db', '#2ecc71']
    for si, (seed, col) in enumerate(zip(seeds, colors_s)):
        valid = np.isfinite(all_lyaps[si])
        ax.plot(lrs[valid], all_lyaps[si][valid], 'o', color=col, markersize=3, 
                alpha=0.5, label=f'Seed {seed}')
    
    valid_m = np.isfinite(mean_lyaps)
    ax.plot(lrs[valid_m], mean_lyaps[valid_m], 's-', color='#2c3e50', markersize=5,
            linewidth=1.5, label='Mean', zorder=5)
    ax.fill_between(lrs[valid_m],
                     (mean_lyaps - std_lyaps)[valid_m],
                     (mean_lyaps + std_lyaps)[valid_m],
                     alpha=0.15, color='gray')
    
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent', fontsize=12)
    ax.set_title('Transition Zone: Individual Seeds', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Color the background
    for i in range(len(lrs) - 1):
        if valid_m[i]:
            color = '#ffcccc' if mean_lyaps[i] > 0 else '#ccccff'
            ax.axvspan(lrs[i], lrs[i+1], alpha=0.15, color=color)
    
    # Right: zoom on zero crossing with interpolation
    ax = axes[1]
    ax.plot(lrs[valid_m], mean_lyaps[valid_m], 'o-', color='#2c3e50', markersize=6,
            linewidth=1.5)
    ax.fill_between(lrs[valid_m],
                     (mean_lyaps - std_lyaps)[valid_m],
                     (mean_lyaps + std_lyaps)[valid_m],
                     alpha=0.2, color='#3498db')
    ax.axhline(y=0, color='black', linewidth=1.5)
    
    # Find zero crossing via linear interpolation
    eta_c_values = []
    for si in range(len(seeds)):
        lyaps_s = all_lyaps[si]
        for i in range(len(lyaps_s) - 1):
            if np.isfinite(lyaps_s[i]) and np.isfinite(lyaps_s[i+1]):
                if lyaps_s[i] <= 0 and lyaps_s[i+1] > 0:
                    # Linear interpolation
                    frac = -lyaps_s[i] / (lyaps_s[i+1] - lyaps_s[i])
                    eta_c = lrs[i] + frac * (lrs[i+1] - lrs[i])
                    eta_c_values.append(eta_c)
                    break
    
    if eta_c_values:
        eta_c_mean = np.mean(eta_c_values)
        eta_c_std = np.std(eta_c_values) if len(eta_c_values) > 1 else 0
        
        ax.axvline(x=eta_c_mean, color='green', linewidth=2.5,
                  label=f'η_c = {eta_c_mean:.5f} ± {eta_c_std:.5f}')
        if eta_c_std > 0:
            ax.axvspan(eta_c_mean - eta_c_std, eta_c_mean + eta_c_std,
                      alpha=0.2, color='green')
        
        print(f"\n  *** CRITICAL η (interpolated): {eta_c_mean:.6f} ± {eta_c_std:.6f}")
        print(f"  *** Ratio η_c / (2/λ_max): {eta_c_mean/lr_eos:.4f} ± {eta_c_std/lr_eos:.4f}")
    
    # Also find from mean
    for i in range(len(mean_lyaps) - 1):
        if valid_m[i] and valid_m[i+1] and mean_lyaps[i] <= 0 and mean_lyaps[i+1] > 0:
            frac = -mean_lyaps[i] / (mean_lyaps[i+1] - mean_lyaps[i])
            eta_c_from_mean = lrs[i] + frac * (lrs[i+1] - lrs[i])
            ax.axvline(x=eta_c_from_mean, color='orange', linewidth=1.5, linestyle='--',
                      label=f'η_c (from mean) = {eta_c_from_mean:.5f}')
            break
    
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent (mean ± std)', fontsize=12)
    ax.set_title('Critical η: Zero Crossing', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_transition_zone.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 02_transition_zone.png")
    
    # Save data
    np.savez(f'{OUTPUT_DIR}/transition_zone_data.npz',
             lrs=lrs, all_lyaps=all_lyaps, mean_lyaps=mean_lyaps,
             std_lyaps=std_lyaps, seeds=np.array(seeds), lr_eos=lr_eos)
    
    # --- Summary plot combining both ---
    if eta_c_values:
        fig_summary, ax_sum = plt.subplots(figsize=(10, 5))
        
        ax_sum.bar(range(len(eta_c_values)), eta_c_values, color=colors_s[:len(eta_c_values)],
                  alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_sum.axhline(y=eta_c_mean, color='green', linewidth=2, 
                      label=f'Mean = {eta_c_mean:.5f}')
        if eta_c_std > 0:
            ax_sum.axhspan(eta_c_mean - eta_c_std, eta_c_mean + eta_c_std,
                          alpha=0.2, color='green')
        ax_sum.axhline(y=lr_eos, color='orange', linestyle='--', linewidth=1.5,
                      label=f'2/λ_max = {lr_eos:.4f}')
        
        ax_sum.set_xlabel('Seed Index', fontsize=12)
        ax_sum.set_ylabel('Critical η_c', fontsize=12)
        ax_sum.set_title(f'Critical Learning Rate Across Seeds\n'
                        f'η_c = {eta_c_mean:.5f} ± {eta_c_std:.5f} '
                        f'({eta_c_mean/lr_eos:.1%} of EoS threshold)',
                        fontsize=13, fontweight='bold')
        ax_sum.legend(fontsize=11)
        ax_sum.set_xticks(range(len(eta_c_values)))
        ax_sum.set_xticklabels([f'Seed {s}' for s in seeds[:len(eta_c_values)]])
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/03_critical_eta_summary.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 03_critical_eta_summary.png")
    
    return all_lyaps, lrs


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 65)
    print(" KAM Theory — Follow-up Experiments")
    print("=" * 65)
    
    # Load data (same for all experiments, same seed)
    print("\n[0] Loading data...")
    X, Y = make_data(N_DATA, data_seed=42)
    n_train = int(0.8 * N_DATA)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    if RUN_REPRO:
        experiment_reproducibility(X_train, Y_train, X_test)
    
    if RUN_ZOOM:
        experiment_transition_zone(X_train, Y_train, X_test)
    
    print(f"\n{'='*65}")
    print(" ALL EXPERIMENTS COMPLETE")
    print(f"{'='*65}")
    print(f"\nResults in: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size//1024}KB)")


if __name__ == '__main__':
    main()
