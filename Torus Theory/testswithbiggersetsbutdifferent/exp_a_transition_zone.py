#!/usr/bin/env python3
"""
EXPERIMENT A: 20-Seed Transition Zone Sweep
============================================

Purpose:
    Tighten the estimate of η_c (chaos onset learning rate) from 
    η_c ≈ 0.018 ± 0.012 (3 seeds) to η_c ≈ X ± 0.003 (20 seeds).
    
    Also characterize the transition zone: does it show KAM-predicted
    fractal interleaving of ordered and chaotic states?

Protocol:
    - 20 seeds × 30 learning rates (η = 0.005 to 0.08, log-spaced)
    - Same architecture, data, and Lyapunov measurement as original experiments
    - Record: Lyapunov exponent, final loss, per-seed trajectories
    
Expected runtime:
    ~600 runs × ~30s each = ~5 hours on GPU, ~15-20 hours on CPU
    Script includes checkpointing — can be interrupted and resumed.

Output:
    results/exp_a_transition_zone.json — all Lyapunov exponents
    results/exp_a_summary.json — statistics per learning rate
    figures/exp_a_lyapunov_transition.png — main result figure
    figures/exp_a_fraction_chaotic.png — fraction of seeds with λ > 0

Usage:
    python exp_a_transition_zone.py                    # full run
    python exp_a_transition_zone.py --n_seeds 5        # quick test
    python exp_a_transition_zone.py --resume            # resume from checkpoint
    python exp_a_transition_zone.py --plot_only         # just make figures
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
    # Sweep parameters
    'n_seeds': 20,
    'n_learning_rates': 30,
    'lr_min': 0.005,
    'lr_max': 0.08,
    
    # Training parameters (match chaos onset report)
    'n_steps': 5000,
    'epsilon': 1e-8,
    'data_seed': 42,
    'hidden1': 50,
    'hidden2': 50,
    
    # Recording
    'record_every': 10,
    
    # Output
    'results_dir': 'results',
    'figures_dir': 'figures',
    'checkpoint_file': 'results/exp_a_checkpoint.json',
}


def get_learning_rates(config):
    """Generate log-spaced learning rates across transition zone."""
    return np.logspace(
        np.log10(config['lr_min']),
        np.log10(config['lr_max']),
        config['n_learning_rates']
    ).tolist()


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────

def run_experiment(config, resume=False):
    """Run the full 20-seed transition zone sweep."""
    
    device = get_device()
    
    # Generate data (deterministic, same for all runs)
    print("Generating data...")
    X, Y, labels = generate_data(data_seed=config['data_seed'], device=device)
    print(f"  Data shape: X={X.shape}, Y={Y.shape}")
    
    # Compute EoS reference point
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
    print(f"  EoS threshold (2/λ_max) ≈ {eos_threshold:.4f}")
    del ref_model
    
    # Learning rates
    learning_rates = get_learning_rates(config)
    seeds = list(range(config['n_seeds']))
    
    print(f"\nSweep: {len(learning_rates)} LRs × {len(seeds)} seeds = {len(learning_rates)*len(seeds)} runs")
    print(f"  LR range: [{learning_rates[0]:.4f}, {learning_rates[-1]:.4f}]")
    print(f"  η_c/EoS range: [{learning_rates[0]/eos_threshold:.3f}, {learning_rates[-1]/eos_threshold:.3f}]")
    
    # Load checkpoint if resuming
    completed = {}
    if resume and os.path.exists(config['checkpoint_file']):
        with open(config['checkpoint_file'], 'r') as f:
            checkpoint = json.load(f)
        completed = {(r['lr'], r['seed']): r for r in checkpoint.get('results', [])}
        print(f"  Resuming: {len(completed)} runs already completed")
    
    # Run sweep
    all_results = list(completed.values())
    total_runs = len(learning_rates) * len(seeds)
    completed_count = len(completed)
    start_time = time.time()
    
    for lr_idx, lr in enumerate(learning_rates):
        for seed in seeds:
            key = (lr, seed)
            if key in completed:
                continue
            
            completed_count += 1
            elapsed = time.time() - start_time
            if completed_count > len(completed) + 1:
                rate = (completed_count - len(completed)) / elapsed
                remaining = (total_runs - completed_count) / rate
                eta_str = format_time(remaining)
            else:
                eta_str = "estimating..."
            
            print(f"  [{completed_count}/{total_runs}] lr={lr:.5f}, seed={seed} "
                  f"(ETA: {eta_str})", end='', flush=True)
            
            t0 = time.time()
            result = compute_lyapunov(
                lr=lr, X=X, Y=Y,
                n_steps=config['n_steps'],
                epsilon=config['epsilon'],
                init_seed=seed,
                device=device,
                hidden1=config['hidden1'],
                hidden2=config['hidden2'],
                record_every=config['record_every'],
                verbose=False,
            )
            dt = time.time() - t0
            
            # Add metadata
            result['lr_index'] = lr_idx
            result['lr_over_eos'] = lr / eos_threshold
            
            all_results.append(result)
            
            lyap = result['lyapunov_exponent']
            print(f"  → λ={lyap:+.6f} ({dt:.1f}s)")
            
            # Checkpoint every 10 runs
            if completed_count % 10 == 0:
                _save_checkpoint(all_results, config, lambda_max, eos_threshold)
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {format_time(total_time)}")
    
    # Save final results
    output = {
        'config': config,
        'lambda_max': lambda_max,
        'eos_threshold': eos_threshold,
        'learning_rates': learning_rates,
        'seeds': seeds,
        'results': all_results,
        'total_time_seconds': total_time,
        'timestamp': datetime.now().isoformat() if 'datetime' in dir() else str(time.time()),
    }
    
    save_results(output, 'exp_a_transition_zone.json', config['results_dir'])
    
    # Compute and save summary statistics
    summary = compute_summary(all_results, learning_rates, eos_threshold)
    save_results(summary, 'exp_a_summary.json', config['results_dir'])
    
    return output


def _save_checkpoint(results, config, lambda_max, eos_threshold):
    """Save checkpoint for resuming."""
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'results': results,
        'lambda_max': lambda_max,
        'eos_threshold': eos_threshold,
    }
    with open(config['checkpoint_file'], 'w') as f:
        json.dump(checkpoint, f)


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────

def compute_summary(results, learning_rates, eos_threshold):
    """Compute per-LR statistics and estimate η_c."""
    
    from datetime import datetime
    
    # Group by learning rate
    by_lr = {}
    for r in results:
        lr = r['lr']
        if lr not in by_lr:
            by_lr[lr] = []
        by_lr[lr].append(r['lyapunov_exponent'])
    
    # Per-LR statistics
    lr_stats = []
    for lr in sorted(by_lr.keys()):
        lyaps = np.array(by_lr[lr])
        n_chaotic = np.sum(lyaps > 0)
        n_total = len(lyaps)
        
        stat = {
            'lr': lr,
            'lr_over_eos': lr / eos_threshold,
            'mean_lyapunov': float(np.mean(lyaps)),
            'std_lyapunov': float(np.std(lyaps)),
            'median_lyapunov': float(np.median(lyaps)),
            'min_lyapunov': float(np.min(lyaps)),
            'max_lyapunov': float(np.max(lyaps)),
            'n_chaotic': int(n_chaotic),
            'n_total': int(n_total),
            'fraction_chaotic': float(n_chaotic / n_total),
            'sem_lyapunov': float(np.std(lyaps) / np.sqrt(n_total)),
        }
        lr_stats.append(stat)
    
    # Estimate η_c: where median Lyapunov crosses zero
    # Use linear interpolation between last negative and first positive median
    eta_c_estimate = None
    eta_c_ci_low = None
    eta_c_ci_high = None
    
    sorted_stats = sorted(lr_stats, key=lambda s: s['lr'])
    for i in range(len(sorted_stats) - 1):
        if sorted_stats[i]['median_lyapunov'] <= 0 and sorted_stats[i+1]['median_lyapunov'] > 0:
            # Linear interpolation
            lr1, ly1 = sorted_stats[i]['lr'], sorted_stats[i]['median_lyapunov']
            lr2, ly2 = sorted_stats[i+1]['lr'], sorted_stats[i+1]['median_lyapunov']
            eta_c_estimate = lr1 + (0 - ly1) * (lr2 - lr1) / (ly2 - ly1)
            
            # Bootstrap CI: use the range where fraction_chaotic is between 0.25 and 0.75
            transition_lrs = [s['lr'] for s in sorted_stats 
                            if 0.25 <= s['fraction_chaotic'] <= 0.75]
            if transition_lrs:
                eta_c_ci_low = min(transition_lrs)
                eta_c_ci_high = max(transition_lrs)
            break
    
    # Alternative: where mean Lyapunov crosses zero
    eta_c_mean = None
    for i in range(len(sorted_stats) - 1):
        if sorted_stats[i]['mean_lyapunov'] <= 0 and sorted_stats[i+1]['mean_lyapunov'] > 0:
            lr1, ly1 = sorted_stats[i]['lr'], sorted_stats[i]['mean_lyapunov']
            lr2, ly2 = sorted_stats[i+1]['lr'], sorted_stats[i+1]['mean_lyapunov']
            eta_c_mean = lr1 + (0 - ly1) * (lr2 - lr1) / (ly2 - ly1)
            break
    
    # Alternative: where 50% of seeds are chaotic
    eta_c_50pct = None
    for i in range(len(sorted_stats) - 1):
        if sorted_stats[i]['fraction_chaotic'] < 0.5 and sorted_stats[i+1]['fraction_chaotic'] >= 0.5:
            f1, f2 = sorted_stats[i]['fraction_chaotic'], sorted_stats[i+1]['fraction_chaotic']
            lr1, lr2 = sorted_stats[i]['lr'], sorted_stats[i+1]['lr']
            eta_c_50pct = lr1 + (0.5 - f1) * (lr2 - lr1) / (f2 - f1)
            break
    
    summary = {
        'lr_stats': lr_stats,
        'eta_c_median_crossing': eta_c_estimate,
        'eta_c_mean_crossing': eta_c_mean,
        'eta_c_50pct_chaotic': eta_c_50pct,
        'eta_c_ci_low': eta_c_ci_low,
        'eta_c_ci_high': eta_c_ci_high,
        'eos_threshold': eos_threshold,
        'eta_c_over_eos': eta_c_estimate / eos_threshold if eta_c_estimate else None,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT A SUMMARY")
    print("="*70)
    print(f"  EoS threshold: {eos_threshold:.4f}")
    print(f"  η_c (median crossing):  {eta_c_estimate:.5f}" if eta_c_estimate else "  η_c (median): not found")
    print(f"  η_c (mean crossing):    {eta_c_mean:.5f}" if eta_c_mean else "  η_c (mean): not found")
    print(f"  η_c (50% chaotic):      {eta_c_50pct:.5f}" if eta_c_50pct else "  η_c (50%): not found")
    if eta_c_ci_low and eta_c_ci_high:
        print(f"  Transition zone:        [{eta_c_ci_low:.5f}, {eta_c_ci_high:.5f}]")
    if eta_c_estimate:
        print(f"  η_c / EoS:              {eta_c_estimate/eos_threshold:.4f} ({eta_c_estimate/eos_threshold*100:.1f}%)")
    print()
    
    # Print per-LR table
    print(f"  {'η':>8s}  {'η/EoS':>6s}  {'mean λ':>9s}  {'std λ':>8s}  {'chaotic':>8s}  {'fraction':>8s}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8}")
    for s in sorted_stats:
        print(f"  {s['lr']:8.5f}  {s['lr_over_eos']:6.3f}  {s['mean_lyapunov']:+9.6f}  "
              f"{s['std_lyapunov']:8.6f}  {s['n_chaotic']:>3d}/{s['n_total']:<3d}   {s['fraction_chaotic']:8.3f}")
    
    return summary


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def make_figures(results_file, figures_dir='figures'):
    """Generate publication-quality figures from results."""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    eos = data['eos_threshold']
    
    # Group by learning rate
    by_lr = {}
    for r in results:
        lr = r['lr']
        if lr not in by_lr:
            by_lr[lr] = []
        by_lr[lr].append(r['lyapunov_exponent'])
    
    lrs = sorted(by_lr.keys())
    means = [np.mean(by_lr[lr]) for lr in lrs]
    stds = [np.std(by_lr[lr]) for lr in lrs]
    sems = [np.std(by_lr[lr]) / np.sqrt(len(by_lr[lr])) for lr in lrs]
    fracs = [np.mean(np.array(by_lr[lr]) > 0) for lr in lrs]
    
    # ── Figure 1: Lyapunov exponents across transition ──
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Top panel: Lyapunov exponents
    ax = axes[0]
    
    # Individual points (jittered)
    for lr in lrs:
        lyaps = by_lr[lr]
        jitter = np.random.RandomState(42).uniform(-0.003, 0.003, len(lyaps)) * lr
        for lyap, j in zip(lyaps, jitter):
            color = '#d62728' if lyap > 0 else '#1f77b4'
            ax.scatter(lr + j, lyap, c=color, alpha=0.3, s=15, zorder=2)
    
    # Mean ± SEM
    ax.errorbar(lrs, means, yerr=sems, fmt='ko-', markersize=4, linewidth=1.5,
                capsize=3, zorder=3, label='Mean ± SEM')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_ylabel('Lyapunov Exponent (per step)', fontsize=12)
    ax.set_title('Chaos Onset in Gradient Descent — 20-Seed Transition Zone', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    
    # Add EoS reference
    ax.axvline(x=eos, color='green', linestyle=':', alpha=0.5, label=f'EoS = {eos:.3f}')
    ax.legend(loc='upper left')
    
    # Bottom panel: Fraction chaotic
    ax2 = axes[1]
    ax2.bar(range(len(lrs)), fracs, color=['#d62728' if f > 0.5 else '#1f77b4' for f in fracs],
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.set_xticks(range(len(lrs)))
    ax2.set_xticklabels([f'{lr:.4f}' for lr in lrs], rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Fraction Chaotic\n(λ > 0)', fontsize=11)
    ax2.set_xlabel('Learning Rate η', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'exp_a_lyapunov_transition.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir}/exp_a_lyapunov_transition.png")
    
    # ── Figure 2: Detailed transition zone ──
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for lr in lrs:
        lyaps = sorted(by_lr[lr])
        x = [lr] * len(lyaps)
        colors = ['#d62728' if l > 0 else '#1f77b4' for l in lyaps]
        ax.scatter(x, lyaps, c=colors, alpha=0.5, s=20, zorder=2)
    
    ax.fill_between(lrs, 
                    [m - 2*s for m, s in zip(means, stds)],
                    [m + 2*s for m, s in zip(means, stds)],
                    alpha=0.1, color='gray', label='±2σ band')
    ax.plot(lrs, means, 'k-', linewidth=2, label='Mean', zorder=3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent', fontsize=12)
    ax.set_title('Transition Zone Detail — Each Dot is One Seed', fontsize=13)
    ax.legend()
    ax.set_xscale('log')
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'exp_a_transition_detail.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir}/exp_a_transition_detail.png")
    
    # ── Figure 3: KAM interleaving diagnostic ──
    # At each LR, show which seeds are chaotic vs ordered
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for lr_idx, lr in enumerate(lrs):
        lyaps = by_lr[lr]
        for seed_idx, lyap in enumerate(sorted(lyaps)):
            color = '#d62728' if lyap > 0 else '#1f77b4'
            ax.scatter(lr_idx, seed_idx, c=color, s=30, marker='s', edgecolors='none')
    
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f'{lr:.4f}' for lr in lrs], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('Seed (sorted by Lyapunov)', fontsize=12)
    ax.set_title('KAM Interleaving Diagnostic\nRed = chaotic (λ>0), Blue = ordered (λ≤0)', fontsize=13)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'exp_a_kam_interleaving.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir}/exp_a_kam_interleaving.png")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Experiment A: 20-seed transition zone sweep for η_c estimation'
    )
    parser.add_argument('--n_seeds', type=int, default=20, help='Number of random seeds')
    parser.add_argument('--n_lrs', type=int, default=30, help='Number of learning rates')
    parser.add_argument('--lr_min', type=float, default=0.005, help='Min learning rate')
    parser.add_argument('--lr_max', type=float, default=0.08, help='Max learning rate')
    parser.add_argument('--n_steps', type=int, default=5000, help='Training steps per run')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--plot_only', action='store_true', help='Only generate figures')
    parser.add_argument('--hidden', type=int, default=50, help='Hidden layer size')
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config['n_seeds'] = args.n_seeds
    config['n_learning_rates'] = args.n_lrs
    config['lr_min'] = args.lr_min
    config['lr_max'] = args.lr_max
    config['n_steps'] = args.n_steps
    config['hidden1'] = args.hidden
    config['hidden2'] = args.hidden
    
    if args.plot_only:
        results_file = os.path.join(config['results_dir'], 'exp_a_transition_zone.json')
        if not os.path.exists(results_file):
            print(f"Error: {results_file} not found. Run the experiment first.")
            sys.exit(1)
        make_figures(results_file, config['figures_dir'])
    else:
        print("="*70)
        print("EXPERIMENT A: 20-Seed Transition Zone Sweep")
        print("="*70)
        print(f"  Seeds: {config['n_seeds']}")
        print(f"  Learning rates: {config['n_learning_rates']} in [{config['lr_min']}, {config['lr_max']}]")
        print(f"  Steps per run: {config['n_steps']}")
        print(f"  Architecture: 220 → {config['hidden1']} → {config['hidden2']} → 10 (tanh)")
        print()
        
        output = run_experiment(config, resume=args.resume)
        
        results_file = os.path.join(config['results_dir'], 'exp_a_transition_zone.json')
        make_figures(results_file, config['figures_dir'])


if __name__ == '__main__':
    from datetime import datetime
    main()
