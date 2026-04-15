#!/usr/bin/env python3
"""
EXPERIMENT B: Perturbation Sensitivity (ε Sweep)
==================================================

Purpose:
    Validate that Lyapunov exponent measurements are true dynamical invariants,
    not artifacts of the perturbation magnitude.
    
    If λ is a genuine Lyapunov exponent, it should be independent of ε in the
    small-ε limit. If λ shifts systematically with ε, the measurement reflects
    nonlinear effects rather than linearized dynamics and needs reinterpretation.

Protocol:
    - 5 learning rates × 7 epsilon values × 5 seeds = 175 runs
    - LRs chosen to span: well below η_c, near η_c, well above η_c
    - ε from 10⁻¹² (deep linear regime) to 10⁻⁴ (potentially nonlinear)
    
    Diagnostic criteria:
    - PASS: λ is constant (within noise) across ε for each LR
    - CONCERN: λ varies monotonically with ε (nonlinear contamination)
    - FAIL: λ changes sign with ε (measurement is meaningless)

Expected runtime:
    ~175 runs × ~30s each = ~90 minutes on GPU, ~4-5 hours on CPU

Output:
    results/exp_b_epsilon_sweep.json — all results
    figures/exp_b_epsilon_sensitivity.png — λ vs ε for each LR
    figures/exp_b_epsilon_heatmap.png — heatmap view

Usage:
    python exp_b_epsilon_sweep.py                    # full run
    python exp_b_epsilon_sweep.py --n_seeds 3        # quick test
    python exp_b_epsilon_sweep.py --plot_only         # just make figures
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

import torch

from phase1_shared import (
    generate_data, compute_lyapunov, estimate_top_eigenvalue,
    ChaosOnsetMLP, save_results, get_device, format_time
)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Learning rates: span the transition
    # Well below η_c, near η_c, above η_c, well above, near EoS
    'learning_rates': [0.005, 0.010, 0.020, 0.040, 0.080],
    
    # Epsilon values: log-spaced from deep linear to potentially nonlinear
    'epsilons': [1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3],
    
    # Seeds
    'n_seeds': 5,
    
    # Training parameters (match chaos onset report)
    'n_steps': 5000,
    'data_seed': 42,
    'hidden1': 50,
    'hidden2': 50,
    
    # Recording
    'record_every': 10,
    
    # Output
    'results_dir': 'results',
    'figures_dir': 'figures',
}


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────

def run_experiment(config):
    """Run the epsilon sensitivity sweep."""
    
    device = get_device()
    
    # Generate data
    print("Generating data...")
    X, Y, labels = generate_data(data_seed=config['data_seed'], device=device)
    print(f"  Data shape: X={X.shape}, Y={Y.shape}")
    
    # Compute EoS reference
    print("Estimating Hessian top eigenvalue...")
    torch.manual_seed(0)
    ref_model = ChaosOnsetMLP(
        input_dim=X.shape[1], hidden1=config['hidden1'],
        hidden2=config['hidden2'], output_dim=Y.shape[1]
    ).to(device)
    print(f"  Model parameters: {ref_model.count_params()}")
    
    lambda_max = estimate_top_eigenvalue(ref_model, X, Y, device=device)
    eos_threshold = 2.0 / lambda_max
    print(f"  λ_max ≈ {lambda_max:.4f}")
    print(f"  EoS threshold ≈ {eos_threshold:.4f}")
    del ref_model
    
    learning_rates = config['learning_rates']
    epsilons = config['epsilons']
    seeds = list(range(config['n_seeds']))
    
    total_runs = len(learning_rates) * len(epsilons) * len(seeds)
    print(f"\nSweep: {len(learning_rates)} LRs × {len(epsilons)} εs × {len(seeds)} seeds = {total_runs} runs")
    
    all_results = []
    run_count = 0
    start_time = time.time()
    
    for lr in learning_rates:
        for eps in epsilons:
            for seed in seeds:
                run_count += 1
                elapsed = time.time() - start_time
                if run_count > 1:
                    rate = run_count / elapsed
                    remaining = (total_runs - run_count) / rate
                    eta_str = format_time(remaining)
                else:
                    eta_str = "estimating..."
                
                print(f"  [{run_count}/{total_runs}] lr={lr:.4f}, ε={eps:.0e}, seed={seed} "
                      f"(ETA: {eta_str})", end='', flush=True)
                
                t0 = time.time()
                result = compute_lyapunov(
                    lr=lr, X=X, Y=Y,
                    n_steps=config['n_steps'],
                    epsilon=eps,
                    init_seed=seed,
                    device=device,
                    hidden1=config['hidden1'],
                    hidden2=config['hidden2'],
                    record_every=config['record_every'],
                    verbose=False,
                )
                dt = time.time() - t0
                
                result['lr_over_eos'] = lr / eos_threshold
                
                # Don't save full distance trajectories to keep file manageable
                # Keep only first, last, and 10 evenly spaced points
                dists = result['distances']
                if len(dists) > 12:
                    indices = [0] + list(np.linspace(1, len(dists)-2, 10, dtype=int)) + [len(dists)-1]
                    result['distances_sparse'] = [dists[i] for i in indices]
                else:
                    result['distances_sparse'] = dists
                del result['distances']
                
                all_results.append(result)
                
                lyap = result['lyapunov_exponent']
                print(f"  → λ={lyap:+.6f} ({dt:.1f}s)")
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {format_time(total_time)}")
    
    # Save results
    output = {
        'config': config,
        'lambda_max': lambda_max,
        'eos_threshold': eos_threshold,
        'results': all_results,
        'total_time_seconds': total_time,
    }
    
    save_results(output, 'exp_b_epsilon_sweep.json', config['results_dir'])
    
    # Analysis
    analyze_sensitivity(all_results, learning_rates, epsilons, eos_threshold)
    
    return output


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────

def analyze_sensitivity(results, learning_rates, epsilons, eos_threshold):
    """Analyze whether Lyapunov exponents are stable across epsilon."""
    
    print("\n" + "="*70)
    print("EXPERIMENT B: SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Group: by_lr_eps[lr][eps] = [lyap1, lyap2, ...]
    by_lr_eps = {}
    for r in results:
        lr, eps = r['lr'], r['epsilon']
        if lr not in by_lr_eps:
            by_lr_eps[lr] = {}
        if eps not in by_lr_eps[lr]:
            by_lr_eps[lr][eps] = []
        by_lr_eps[lr][eps].append(r['lyapunov_exponent'])
    
    for lr in learning_rates:
        print(f"\n  η = {lr:.4f} (η/EoS = {lr/eos_threshold:.3f})")
        print(f"  {'ε':>10s}  {'mean λ':>10s}  {'std λ':>8s}  {'sign':>5s}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*5}")
        
        means = []
        for eps in epsilons:
            lyaps = np.array(by_lr_eps[lr].get(eps, []))
            if len(lyaps) == 0:
                continue
            m = np.mean(lyaps)
            s = np.std(lyaps)
            sign = '+' if m > 0 else '-'
            means.append(m)
            print(f"  {eps:10.0e}  {m:+10.7f}  {s:8.6f}  {sign:>5s}")
        
        # Diagnostic: correlation between log(ε) and λ
        if len(means) >= 3:
            log_eps = np.log10(epsilons[:len(means)])
            corr = np.corrcoef(log_eps, means)[0, 1]
            
            # Check for sign changes
            signs = [m > 0 for m in means]
            sign_changes = sum(1 for i in range(len(signs)-1) if signs[i] != signs[i+1])
            
            # Coefficient of variation across epsilon
            if np.mean(np.abs(means)) > 1e-8:
                cv = np.std(means) / np.mean(np.abs(means))
            else:
                cv = float('inf')
            
            # Verdict
            if sign_changes > 0:
                verdict = "⚠ SIGN CHANGE — measurement may be near transition"
            elif abs(corr) > 0.8 and cv > 0.3:
                verdict = "⚠ MONOTONIC DRIFT — possible nonlinear contamination"
            elif cv < 0.2:
                verdict = "✓ STABLE — invariant across ε"
            else:
                verdict = "~ MODERATE VARIATION — acceptable"
            
            print(f"  Correlation(log ε, λ): {corr:+.3f}")
            print(f"  CV across ε: {cv:.3f}")
            print(f"  Sign changes: {sign_changes}")
            print(f"  Verdict: {verdict}")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def make_figures(results_file, figures_dir='figures'):
    """Generate figures from results."""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    eos = data['eos_threshold']
    
    # Group results
    by_lr_eps = {}
    for r in results:
        lr, eps = r['lr'], r['epsilon']
        key = (lr, eps)
        if key not in by_lr_eps:
            by_lr_eps[key] = []
        by_lr_eps[key].append(r['lyapunov_exponent'])
    
    learning_rates = sorted(set(r['lr'] for r in results))
    epsilons = sorted(set(r['epsilon'] for r in results))
    
    # ── Figure 1: λ vs ε for each LR ──
    fig, axes = plt.subplots(1, len(learning_rates), figsize=(4*len(learning_rates), 4),
                             sharey=True)
    if len(learning_rates) == 1:
        axes = [axes]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(learning_rates)))
    
    for ax, lr, color in zip(axes, learning_rates, colors):
        means = []
        stds = []
        valid_eps = []
        for eps in epsilons:
            lyaps = by_lr_eps.get((lr, eps), [])
            if lyaps:
                means.append(np.mean(lyaps))
                stds.append(np.std(lyaps))
                valid_eps.append(eps)
        
        if valid_eps:
            ax.errorbar(valid_eps, means, yerr=stds, fmt='o-', color=color,
                       capsize=3, markersize=5, linewidth=1.5)
            
            # Individual points
            for eps in valid_eps:
                lyaps = by_lr_eps.get((lr, eps), [])
                jitter = np.random.RandomState(42).uniform(0.85, 1.15, len(lyaps))
                ax.scatter([eps * j for j in jitter], lyaps, 
                          c=[color], alpha=0.3, s=15, zorder=1)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('ε', fontsize=11)
        ax.set_title(f'η = {lr:.3f}\n(η/EoS = {lr/eos:.2f})', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('Lyapunov Exponent', fontsize=12)
    fig.suptitle('Perturbation Sensitivity: Is λ Independent of ε?', fontsize=14, y=1.02)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'exp_b_epsilon_sensitivity.png'), 
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir}/exp_b_epsilon_sensitivity.png")
    
    # ── Figure 2: Heatmap ──
    fig, ax = plt.subplots(figsize=(8, 5))
    
    heatmap_data = np.full((len(epsilons), len(learning_rates)), np.nan)
    for i, eps in enumerate(epsilons):
        for j, lr in enumerate(learning_rates):
            lyaps = by_lr_eps.get((lr, eps), [])
            if lyaps:
                heatmap_data[i, j] = np.mean(lyaps)
    
    # Symmetric colormap centered on zero
    vmax = np.nanmax(np.abs(heatmap_data))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                   vmin=-vmax, vmax=vmax, origin='lower')
    
    ax.set_xticks(range(len(learning_rates)))
    ax.set_xticklabels([f'{lr:.3f}' for lr in learning_rates])
    ax.set_yticks(range(len(epsilons)))
    ax.set_yticklabels([f'{eps:.0e}' for eps in epsilons])
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Perturbation ε', fontsize=12)
    ax.set_title('Mean Lyapunov Exponent: ε × η\nStable = consistent color down each column', fontsize=13)
    
    # Annotate cells
    for i in range(len(epsilons)):
        for j in range(len(learning_rates)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=8,
                       color='white' if abs(val) > vmax * 0.5 else 'black')
    
    plt.colorbar(im, label='Lyapunov Exponent')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'exp_b_epsilon_heatmap.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir}/exp_b_epsilon_heatmap.png")
    
    # ── Figure 3: Stability diagnostic ──
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for j, lr in enumerate(learning_rates):
        means = []
        valid_eps = []
        for eps in epsilons:
            lyaps = by_lr_eps.get((lr, eps), [])
            if lyaps:
                means.append(np.mean(lyaps))
                valid_eps.append(eps)
        
        if len(means) >= 2:
            # Normalize: subtract mean across epsilon, divide by max variation
            means_arr = np.array(means)
            deviation = means_arr - np.mean(means_arr)
            max_dev = np.max(np.abs(deviation)) if np.max(np.abs(deviation)) > 0 else 1
            normalized = deviation / max_dev
            
            ax.plot(valid_eps, normalized, 'o-', label=f'η={lr:.3f}', linewidth=1.5)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.axhspan(-0.2, 0.2, alpha=0.1, color='green', label='±20% stability band')
    ax.set_xscale('log')
    ax.set_xlabel('Perturbation ε', fontsize=12)
    ax.set_ylabel('Normalized Deviation from Mean λ', fontsize=12)
    ax.set_title('ε-Stability Diagnostic\nFlat lines = λ is a true dynamical invariant', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'exp_b_stability_diagnostic.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir}/exp_b_stability_diagnostic.png")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Experiment B: Perturbation sensitivity (epsilon sweep)'
    )
    parser.add_argument('--n_seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--n_steps', type=int, default=5000, help='Training steps per run')
    parser.add_argument('--plot_only', action='store_true', help='Only generate figures')
    parser.add_argument('--hidden', type=int, default=50, help='Hidden layer size')
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config['n_seeds'] = args.n_seeds
    config['n_steps'] = args.n_steps
    config['hidden1'] = args.hidden
    config['hidden2'] = args.hidden
    
    if args.plot_only:
        results_file = os.path.join(config['results_dir'], 'exp_b_epsilon_sweep.json')
        if not os.path.exists(results_file):
            print(f"Error: {results_file} not found. Run the experiment first.")
            sys.exit(1)
        make_figures(results_file, config['figures_dir'])
    else:
        print("="*70)
        print("EXPERIMENT B: Perturbation Sensitivity (ε Sweep)")
        print("="*70)
        print(f"  Learning rates: {config['learning_rates']}")
        print(f"  Epsilon values: {[f'{e:.0e}' for e in config['epsilons']]}")
        print(f"  Seeds: {config['n_seeds']}")
        print(f"  Steps per run: {config['n_steps']}")
        print(f"  Architecture: 220 → {config['hidden1']} → {config['hidden2']} → 10 (tanh)")
        print()
        
        output = run_experiment(config)
        
        results_file = os.path.join(config['results_dir'], 'exp_b_epsilon_sweep.json')
        make_figures(results_file, config['figures_dir'])


if __name__ == '__main__':
    main()
