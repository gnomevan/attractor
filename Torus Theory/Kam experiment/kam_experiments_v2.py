"""
KAM Theory in Neural Network Training — Retuned Experiments
============================================================

Key fix: The target matrix Y must have rank > k (the hidden dim),
so the UV factorization CANNOT fit it perfectly. This creates
persistent non-zero loss and the oscillatory dynamics we need.

Also: larger initial scale to start in the "catapult" regime,
and learning rates tuned to the actual sharpness scale.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

output_dir = '/home/claude/kam_experiments_v2'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# UV MODEL (same as before)
# ============================================================================

class UVModel:
    def __init__(self, d, k, init_scale=1.0, seed=42):
        rng = np.random.RandomState(seed)
        self.d = d
        self.k = k
        self.U = rng.randn(d, k) * init_scale / np.sqrt(k)
        self.V = rng.randn(k, d) * init_scale / np.sqrt(d)
    
    def product(self):
        return self.U @ self.V
    
    def loss(self, Y):
        """L = 0.5 * ||UV - Y||_F^2"""
        diff = self.product() - Y
        return 0.5 * np.sum(diff**2)
    
    def gradients(self, Y):
        diff = self.product() - Y  # (d, d)
        dU = diff @ self.V.T       # (d, k)
        dV = self.U.T @ diff       # (k, d)
        return dU, dV
    
    def hessian_eigenvalues(self, Y):
        """Compute Hessian eigenvalues via finite differences (simpler, more robust)"""
        params = np.concatenate([self.U.ravel(), self.V.ravel()])
        n = len(params)
        eps = 1e-5
        
        # Use the Gauss-Newton approximation for speed on larger models
        # For the UV model: H ≈ J^T J where J is the Jacobian of residuals
        # But let's just compute the top eigenvalues via power iteration
        
        def loss_fn(p):
            U = p[:self.d*self.k].reshape(self.d, self.k)
            V = p[self.d*self.k:].reshape(self.k, self.d)
            diff = U @ V - Y
            return 0.5 * np.sum(diff**2)
        
        def grad_fn(p):
            U = p[:self.d*self.k].reshape(self.d, self.k)
            V = p[self.d*self.k:].reshape(self.k, self.d)
            diff = U @ V - Y
            dU = (diff @ V.T).ravel()
            dV = (U.T @ diff).ravel()
            return np.concatenate([dU, dV])
        
        # Hessian-vector product via finite differences
        g0 = grad_fn(params)
        
        def hvp(v):
            g1 = grad_fn(params + eps * v)
            return (g1 - g0) / eps
        
        # Power iteration for top eigenvalues
        eigenvalues = []
        vectors = []
        
        for _ in range(min(6, n)):
            v = np.random.randn(n)
            # Orthogonalize against previous eigenvectors
            for prev_v in vectors:
                v -= np.dot(v, prev_v) * prev_v
            v /= np.linalg.norm(v)
            
            for iteration in range(100):
                Hv = hvp(v)
                # Orthogonalize
                for prev_v in vectors:
                    Hv -= np.dot(Hv, prev_v) * prev_v
                
                eigenvalue = np.dot(v, Hv)
                v_new = Hv / (np.linalg.norm(Hv) + 1e-15)
                
                if np.linalg.norm(v_new - v) < 1e-6:
                    break
                v = v_new
            
            eigenvalues.append(eigenvalue)
            vectors.append(v)
        
        return np.sort(eigenvalues)[::-1]
    
    def step(self, Y, lr):
        dU, dV = self.gradients(Y)
        self.U -= lr * dU
        self.V -= lr * dV
    
    def get_params(self):
        return np.concatenate([self.U.ravel(), self.V.ravel()])
    
    def set_params(self, params):
        self.U = params[:self.d*self.k].reshape(self.d, self.k).copy()
        self.V = params[self.d*self.k:].reshape(self.k, self.d).copy()


def make_target(d, rank, seed=123):
    """Create a target matrix of specified rank"""
    rng = np.random.RandomState(seed)
    A = rng.randn(d, rank)
    B = rng.randn(rank, d)
    return A @ B / np.sqrt(rank)


# ============================================================================
# EXPERIMENT 1: Edge of Stability with proper parameters
# ============================================================================

print("=" * 60)
print("KAM Theory Experiments v2 — Retuned Parameters")
print("=" * 60)

d = 5       # dimension
k = 2       # hidden dim (rank constraint)
target_rank = 5  # target has full rank > k, so UV can't fit it perfectly

Y = make_target(d, target_rank, seed=123)

# First: find the right learning rate range by checking initial sharpness
test_model = UVModel(d, k, init_scale=1.0, seed=42)
init_eigs = test_model.hessian_eigenvalues(Y)
print(f"\nInitial Hessian top eigenvalues: {init_eigs[:4]}")
print(f"Suggested EoS learning rate: η ≈ {2.0/init_eigs[0]:.4f}")
suggested_lr = 2.0 / init_eigs[0]

# --- Run EoS at different learning rates ---
fig_eos, axes_eos = plt.subplots(3, 3, figsize=(16, 12))
fig_eos.suptitle('Edge of Stability Across Learning Rates\n'
                 f'UV Model: d={d}, k={k}, target rank={target_rank}', 
                 fontsize=14, fontweight='bold')

lr_multipliers = [0.3, 0.7, 1.1]
n_steps = 5000

for col, mult in enumerate(lr_multipliers):
    lr = suggested_lr * mult
    model = UVModel(d, k, init_scale=1.0, seed=42)
    
    losses = []
    sharpnesses = []
    all_eigs = []
    ratios_12 = []
    
    for step in range(n_steps):
        l = model.loss(Y)
        if not np.isfinite(l) or l > 1e8:
            break
        losses.append(l)
        
        if step % 10 == 0:
            try:
                eigs = model.hessian_eigenvalues(Y)
                if np.all(np.isfinite(eigs)):
                    sharpnesses.append((step, eigs[0]))
                    all_eigs.append((step, eigs))
                    pos = eigs[eigs > 0.01]
                    if len(pos) >= 2:
                        ratios_12.append((step, pos[0] / pos[1]))
            except:
                pass
        
        model.step(Y, lr)
    
    threshold = 2.0 / lr
    losses = np.array(losses) if losses else np.array([np.nan])
    
    # Row 0: Loss
    ax = axes_eos[0, col]
    valid = losses[np.isfinite(losses) & (losses > 0)]
    if len(valid) > 0:
        ax.semilogy(range(len(valid)), valid, linewidth=0.4, color='#2c3e50')
    ax.set_title(f'η = {lr:.4f} ({mult}× suggested)')
    ax.set_ylabel('Loss' if col == 0 else '')
    
    # Row 1: Sharpness vs threshold
    ax = axes_eos[1, col]
    steps_s = [s[0] for s in sharpnesses]
    vals_s = [s[1] for s in sharpnesses]
    ax.plot(steps_s, vals_s, linewidth=0.6, color='#e74c3c')
    ax.axhline(y=threshold, color='#3498db', linestyle='--', linewidth=1.5, 
              label=f'2/η={threshold:.1f}')
    ax.legend(fontsize=8)
    ax.set_ylabel('λ_max' if col == 0 else '')
    
    # Row 2: Full spectrum
    ax = axes_eos[2, col]
    n_show = min(4, len(all_eigs[0][1]))
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    for i in range(n_show):
        steps_e = [e[0] for e in all_eigs]
        vals_e = [e[1][i] for e in all_eigs]
        ax.plot(steps_e, vals_e, linewidth=0.5, color=colors[i], label=f'λ_{i+1}')
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Eigenvalue' if col == 0 else '')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_eos_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[1/5] EoS comparison saved.")


# ============================================================================
# EXPERIMENT 2: Bifurcation Diagram
# ============================================================================

print("[2/5] Running bifurcation diagram...")

n_lr = 120
lr_min = suggested_lr * 0.1
lr_max = suggested_lr * 2.0
lrs = np.linspace(lr_min, lr_max, n_lr)

fig_bif, ax_bif = plt.subplots(figsize=(14, 6))

for lr in lrs:
    model = UVModel(d, k, init_scale=1.0, seed=42)
    
    # Train
    n_train = 5000
    for step in range(n_train):
        model.step(Y, lr)
        # Check for divergence
        if model.loss(Y) > 1e10:
            break
    
    # Record late-time losses
    late_losses = []
    for step in range(400):
        l = model.loss(Y)
        if np.isfinite(l) and l < 1e6:
            late_losses.append(l)
        model.step(Y, lr)
    
    if late_losses:
        ax_bif.scatter([lr] * len(late_losses), late_losses, 
                      s=0.2, c='#2c3e50', alpha=0.4)

ax_bif.axvline(x=suggested_lr, color='red', linestyle='--', linewidth=1, 
              label=f'2/λ_max(0) = {suggested_lr:.4f}')
ax_bif.set_xlabel('Learning Rate η', fontsize=12)
ax_bif.set_ylabel('Late-Time Loss', fontsize=12)
ax_bif.set_title('Bifurcation Diagram: Late-Time Loss vs Learning Rate\n'
                 'Period-doubling should appear as splitting of loss bands', fontsize=12)
ax_bif.legend()

plt.savefig(f'{output_dir}/02_bifurcation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Bifurcation diagram saved.")


# ============================================================================
# EXPERIMENT 3: Lyapunov Exponent Sweep
# ============================================================================

print("[3/5] Lyapunov exponent sweep...")

n_lr_lyap = 60
lrs_lyap = np.linspace(suggested_lr * 0.1, suggested_lr * 1.8, n_lr_lyap)
perturbation = 1e-8

lyap_results = []

for lr in lrs_lyap:
    model_ref = UVModel(d, k, init_scale=1.0, seed=42)
    model_pert = UVModel(d, k, init_scale=1.0, seed=42)
    
    params_pert = model_pert.get_params()
    pv = np.random.randn(len(params_pert))
    pv = pv / np.linalg.norm(pv) * perturbation
    model_pert.set_params(params_pert + pv)
    
    n_steps_lyap = 4000
    renorm_interval = 50
    cumulative = 0.0
    final_lyap = np.nan
    diverged = False
    
    for step in range(n_steps_lyap):
        model_ref.step(Y, lr)
        model_pert.step(Y, lr)
        
        # Check divergence
        if model_ref.loss(Y) > 1e10 or model_pert.loss(Y) > 1e10:
            diverged = True
            break
        
        if (step + 1) % renorm_interval == 0:
            pr = model_ref.get_params()
            pp = model_pert.get_params()
            diff = pp - pr
            dist = np.linalg.norm(diff)
            
            if dist > 0 and np.isfinite(dist):
                cumulative += np.log(dist / perturbation)
                final_lyap = cumulative / (step + 1)
                diff_n = diff / dist * perturbation
                model_pert.set_params(pr + diff_n)
    
    lyap_results.append((lr, final_lyap, diverged))

fig_lyap, ax_lyap = plt.subplots(figsize=(12, 5))

lrs_plot = [r[0] for r in lyap_results if np.isfinite(r[1])]
lyaps_plot = [r[1] for r in lyap_results if np.isfinite(r[1])]
colors = ['#e74c3c' if l > 0 else '#3498db' for l in lyaps_plot]

ax_lyap.scatter(lrs_plot, lyaps_plot, c=colors, s=25, zorder=3)
ax_lyap.plot(lrs_plot, lyaps_plot, linewidth=0.8, color='gray', alpha=0.4, zorder=2)
ax_lyap.axhline(y=0, color='black', linewidth=1)
ax_lyap.axvline(x=suggested_lr, color='orange', linestyle='--', linewidth=1, 
               label=f'2/λ_max(0) = {suggested_lr:.4f}')

ax_lyap.fill_between(lrs_plot, 0, [max(0, l) for l in lyaps_plot], alpha=0.08, color='red')
ax_lyap.fill_between(lrs_plot, [min(0, l) for l in lyaps_plot], 0, alpha=0.08, color='blue')

ax_lyap.set_xlabel('Learning Rate η', fontsize=12)
ax_lyap.set_ylabel('Maximal Lyapunov Exponent', fontsize=12)
ax_lyap.set_title('Torus-to-Chaos Transition: Lyapunov Exponent vs Learning Rate\n'
                  'Blue (λ ≤ 0) = KAM tori intact  |  Red (λ > 0) = torus breakdown',
                  fontsize=12)
ax_lyap.legend(fontsize=10)

plt.savefig(f'{output_dir}/03_lyapunov_sweep.png', dpi=150, bbox_inches='tight')
plt.close()

# Find critical η
for i in range(len(lyap_results)-1):
    l1, l2 = lyap_results[i][1], lyap_results[i+1][1]
    if np.isfinite(l1) and np.isfinite(l2) and l1 <= 0 and l2 > 0:
        eta_c = (lyap_results[i][0] + lyap_results[i+1][0]) / 2
        print(f"  Critical η (KAM torus breakdown): ≈ {eta_c:.4f}")
        break

print("  Lyapunov sweep saved.")


# ============================================================================
# EXPERIMENT 4: Eigenvalue Ratio vs Stability (KAM Prediction)
# ============================================================================

print("[4/5] Eigenvalue ratio stability test...")

ratio_data = []
test_lr = suggested_lr * 0.9  # just below EoS

for seed in range(30):
    for init_scale in np.linspace(0.3, 3.0, 12):
        model = UVModel(d, k, init_scale=init_scale, seed=seed)
        
        eigs0 = model.hessian_eigenvalues(Y)
        pos = eigs0[eigs0 > 0.01]
        if len(pos) < 2:
            continue
        ratio0 = pos[0] / pos[1]
        
        # Train
        losses = []
        for step in range(3000):
            l = model.loss(Y)
            if not np.isfinite(l) or l > 1e8:
                break
            losses.append(l)
            model.step(Y, test_lr)
        
        if len(losses) < 2000:
            continue
        
        losses = np.array(losses)
        late = losses[2000:]
        cv = np.std(late) / (np.mean(late) + 1e-15)
        
        # Distance to nearest simple rational
        rationals = sorted(set([p/q for q in range(1,6) for p in range(1, 5*q+1)]))
        dist_rat = min(abs(ratio0 - r) for r in rationals)
        
        ratio_data.append({
            'ratio': ratio0,
            'cv': cv,
            'dist_rational': dist_rat,
            'init_scale': init_scale,
            'seed': seed
        })

fig_kam, axes_kam = plt.subplots(1, 2, figsize=(14, 5))

ratios = [r['ratio'] for r in ratio_data]
cvs = [r['cv'] for r in ratio_data]
dists = [r['dist_rational'] for r in ratio_data]

ax = axes_kam[0]
sc = ax.scatter(ratios, cvs, c=dists, cmap='RdYlBu', s=12, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Distance to nearest\nsimple rational (p/q, q≤5)')
for p, q in [(1,1), (2,1), (3,1), (3,2), (4,3), (5,3)]:
    ax.axvline(x=p/q, color='orange', linestyle=':', linewidth=0.6, alpha=0.4)
ax.set_xlabel('Initial λ₁/λ₂', fontsize=11)
ax.set_ylabel('Late-Time Loss CV (instability)', fontsize=11)
ax.set_title('Eigenvalue Ratio vs Training Instability')

ax = axes_kam[1]
ax.scatter(dists, cvs, s=12, alpha=0.5, color='#8e44ad')
ax.set_xlabel('Distance to Nearest Simple Rational', fontsize=11)
ax.set_ylabel('Late-Time Loss CV', fontsize=11)
ax.set_title('KAM Prediction: Far from rationals = more stable?')
if len(dists) > 10:
    z = np.polyfit(dists, cvs, 1)
    x_line = np.linspace(min(dists), max(dists), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=1.5,
           label=f'slope = {z[0]:.3f}')
    ax.legend()
    # Correlation coefficient
    corr = np.corrcoef(dists, cvs)[0, 1]
    ax.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
           ha='right', va='top', fontsize=11, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig_kam.suptitle('KAM Prediction Test: Do eigenvalue ratios near simple rationals\n'
                 'predict training instability?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_kam_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {len(ratio_data)} conditions tested.")


# ============================================================================
# EXPERIMENT 5: Number of unstable eigenvalues vs chaos (Ruelle-Takens)
# ============================================================================

print("[5/5] Ruelle-Takens test: number of unstable directions vs chaos...")

# Run at a learning rate where EoS is active
eos_lr = suggested_lr * 1.2
model = UVModel(d, k, init_scale=1.0, seed=42)

n_unstable_history = []
lyap_local_history = []
loss_history = []

for step in range(6000):
    l = model.loss(Y)
    if not np.isfinite(l) or l > 1e8:
        break
    loss_history.append(l)
    
    if step % 20 == 0 and step > 100:
        try:
            eigs = model.hessian_eigenvalues(Y)
            threshold = 2.0 / eos_lr
            n_unstable = np.sum(eigs > threshold)
            n_unstable_history.append((step, n_unstable))
        except:
            pass
    
    model.step(Y, eos_lr)

loss_history = np.array(loss_history) if loss_history else np.array([1.0])

fig_rt, axes_rt = plt.subplots(3, 1, figsize=(12, 10))

# Loss
valid_loss = loss_history[loss_history > 0]
if len(valid_loss) > 0:
    axes_rt[0].semilogy(range(len(valid_loss)), valid_loss, linewidth=0.4, color='#2c3e50')
axes_rt[0].set_ylabel('Loss')
axes_rt[0].set_title(f'Ruelle-Takens Test: η = {eos_lr:.4f} (1.2× suggested)')

# Number of unstable eigenvalues
steps_n = [n[0] for n in n_unstable_history]
vals_n = [n[1] for n in n_unstable_history]
axes_rt[1].plot(steps_n, vals_n, linewidth=0.8, color='#e74c3c', marker='.', markersize=2)
axes_rt[1].axhline(y=3, color='orange', linestyle='--', linewidth=1, 
                   label='Ruelle-Takens threshold (3)')
axes_rt[1].set_ylabel('# eigenvalues > 2/η')
axes_rt[1].legend()
axes_rt[1].set_title('Number of Unstable Directions')

# Rolling variance of loss (proxy for chaos)
window = 100
if len(loss_history) > window:
    finite_loss = loss_history[np.isfinite(loss_history)]
    if len(finite_loss) > window:
        rolling_var = np.array([np.var(finite_loss[max(0,i-window):i+1]) 
                               for i in range(len(finite_loss))])
        safe_rv = rolling_var[rolling_var > 0]
        if len(safe_rv) > 0:
            axes_rt[2].semilogy(range(len(rolling_var)), rolling_var + 1e-30, 
                               linewidth=0.4, color='#8e44ad')
axes_rt[2].set_xlabel('Step')
axes_rt[2].set_ylabel('Rolling Variance of Loss')
axes_rt[2].set_title('Local Chaos Indicator (Rolling Loss Variance)')

try:
    plt.tight_layout()
except:
    pass
plt.savefig(f'{output_dir}/05_ruelle_takens.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Ruelle-Takens test saved.")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("All experiments complete.")
print(f"Figures saved to: {output_dir}/")
print("=" * 60)
print(f"\nModel: UV factorization, d={d}, k={k}, target rank={target_rank}")
print(f"Suggested EoS learning rate: {suggested_lr:.4f}")
print(f"\nFiles generated:")
for f in sorted(os.listdir(output_dir)):
    print(f"  {f}")
