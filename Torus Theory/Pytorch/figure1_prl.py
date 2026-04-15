"""
Figure 1 — Publication-quality four-panel figure for PRL
=========================================================

Generates the main figure from the merged 10-seed CNN data:
  (a) Lyapunov exponent vs. fraction of EoS (chaos window)
  (b) Correlation dimension D₂ vs. fraction of EoS
  (c) PC2 variance (off-axis dynamics)
  (d) Sharpness time series at 5% vs 95% EoS

USAGE:
    python figure1_prl.py                           # from merged JSON
    python figure1_prl.py --data path/to/merged.json
    python figure1_prl.py --with-tda tda.json       # add TDA inset/panel
"""

import argparse, json, os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    
    fracs = data['lr_fractions']
    n_lrs = len(fracs)
    
    lyap_mean, lyap_std = [], []
    d2_mean, d2_std = [], []
    pc1_mean, pc2_mean = [], []
    pc2_std = []
    
    for li in range(n_lrs):
        r = data[f'lr_{li}']
        
        lyaps = np.array([v for v in r['lyapunov'] if v is not None])
        d2s = np.array([v for v in r['corr_dim'] if v is not None])
        pc1s = np.array([v for v in r['pc1'] if v is not None])
        pc2s = np.array([v for v in r['pc2'] if v is not None])
        
        lyap_mean.append(lyaps.mean())
        lyap_std.append(lyaps.std())
        d2_mean.append(np.nanmean(d2s))
        d2_std.append(np.nanstd(d2s))
        pc1_mean.append(pc1s.mean())
        pc2_mean.append(pc2s.mean())
        pc2_std.append(pc2s.std())
    
    return {
        'fracs': fracs,
        'lyap_mean': np.array(lyap_mean),
        'lyap_std': np.array(lyap_std),
        'd2_mean': np.array(d2_mean),
        'd2_std': np.array(d2_std),
        'pc1_mean': np.array(pc1_mean),
        'pc2_mean': np.array(pc2_mean),
        'pc2_std': np.array(pc2_std),
        'lam_max': data['lam_max'],
        'lr_eos': data.get('lr_eos', 2.0 / data['lam_max']),
        'n_params': data.get('n_params', 268650),
        'n_seeds': len(data[f'lr_0']['lyapunov']),
        'raw': data,
    }


def plot_figure1(d, tda_data=None, output='figures/figure1_prl.png'):
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fracs = d['fracs']
    
    # Colors
    c_main = '#1a1a1a'
    c_eos = '#E8890C'
    c_zero = '#888888'
    c_d1 = '#888888'
    c_d2 = '#888888'
    c_low = '#2166AC'   # blue for low LR
    c_high = '#E8890C'  # orange for high LR
    c_lmax = '#CC3333'
    
    # ── (a) Lyapunov exponent ──
    ax = axes[0, 0]
    ax.errorbar(fracs, d['lyap_mean'], yerr=d['lyap_std'],
                fmt='o-', color=c_main, ms=5, capsize=3, lw=1.5,
                ecolor='#555555', elinewidth=1, capthick=1)
    ax.axhline(0, color=c_zero, ls='--', lw=0.8)
    ax.axvline(1.0, color=c_eos, ls='--', lw=1.5, label='EoS threshold')
    
    # Shade chaos window
    pos_mask = d['lyap_mean'] > 0
    if pos_mask.any():
        pos_fracs = np.array(fracs)[pos_mask]
        ax.axvspan(pos_fracs.min() - 0.02, pos_fracs.max() + 0.02,
                   alpha=0.06, color='red', zorder=0)
    
    ax.set_xlabel('Fraction of 2/λ$_{max}$')
    ax.set_ylabel('Lyapunov exponent')
    ax.set_title('(a) Chaos vs. distance to EoS')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xlim(-0.02, 1.05)
    
    # ── (b) Correlation dimension ──
    ax = axes[0, 1]
    ax.errorbar(fracs, d['d2_mean'], yerr=d['d2_std'],
                fmt='o-', color=c_main, ms=5, capsize=3, lw=1.5,
                ecolor='#555555', elinewidth=1, capthick=1)
    ax.axhline(1, color=c_d1, ls=':', lw=0.8, label='D = 1')
    ax.axhline(2, color=c_d2, ls='--', lw=0.8, label='D = 2')
    ax.axvline(1.0, color=c_eos, ls='--', lw=1.5)
    
    # Mark the peak
    peak_idx = np.argmax(d['d2_mean'])
    ax.annotate(f'D₂ = {d["d2_mean"][peak_idx]:.2f} ± {d["d2_std"][peak_idx]:.2f}',
                xy=(fracs[peak_idx], d['d2_mean'][peak_idx]),
                xytext=(fracs[peak_idx] + 0.15, d['d2_mean'][peak_idx] + 0.15),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))
    
    ax.set_xlabel('Fraction of 2/λ$_{max}$')
    ax.set_ylabel('Correlation dimension D₂')
    ax.set_title('(b) D₂ crosses 1, peaks at 3.7')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.5, 4.5)
    
    # ── (c) Off-axis dynamics (PC2) ──
    ax = axes[1, 0]
    ax.errorbar(fracs, d['pc2_mean'], yerr=d['pc2_std'],
                fmt='o-', color=c_main, ms=5, capsize=3, lw=1.5,
                ecolor='#555555', elinewidth=1, capthick=1)
    ax.axvline(1.0, color=c_eos, ls='--', lw=1.5)
    
    ax.set_xlabel('Fraction of 2/λ$_{max}$')
    ax.set_ylabel('PC2 variance (%)')
    ax.set_title('(c) Off-axis dynamics growth')
    ax.set_xlim(-0.02, 1.05)
    
    # ── (d) Sharpness dynamics ──
    ax = axes[1, 1]
    
    # Plot sharpness from first seed at lowest and highest LR
    raw = d['raw']
    n_lrs = len(fracs)
    
    # 5% EoS (index 0) — monotonic climb
    sh_low = raw['lr_0']['sharpness_series'][0]  # first seed
    ax.plot(range(len(sh_low)), sh_low, '-', color=c_low, lw=1.5,
            label=f'{fracs[0]:.0%} EoS')
    
    # 95% EoS (last index) — flat/oscillatory
    sh_high = raw[f'lr_{n_lrs-1}']['sharpness_series'][0]
    ax.plot(range(len(sh_high)), sh_high, '-', color=c_high, lw=1.5,
            label=f'{fracs[-1]:.0%} EoS')
    
    ax.axhline(d['lam_max'], color=c_lmax, ls=':', lw=1,
               label=f'λ$_{{max}}$ = {d["lam_max"]:.1f}')
    
    ax.set_xlabel('Measurement index (every 100 steps)')
    ax.set_ylabel('Top Hessian eigenvalue')
    ax.set_title('(d) Sharpness: monotonic vs. oscillatory')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # ── Suptitle ──
    fig.suptitle(
        f'CIFAR-10 CNN ({d["n_params"]:,} params) — '
        f'{d["n_seeds"]} seeds per learning rate',
        fontsize=14, y=1.01, fontweight='bold'
    )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    plt.savefig(output)
    print(f'Saved → {output}')
    
    # Also save PDF for submission
    pdf_path = output.replace('.png', '.pdf')
    plt.savefig(pdf_path)
    print(f'Saved → {pdf_path}')
    plt.close()


def plot_tda_supplement(tda_path, output='figures/tda_supplement.png'):
    """Supplemental TDA figure: H₁ count and gap ratio vs EoS fraction."""
    
    with open(tda_path) as f:
        results = json.load(f)
    
    lr_fractions = results['lr_fractions']
    n_lrs = len(lr_fractions)
    
    h1_counts, h1_tops, h1_gaps, h2_counts = [], [], [], []
    for li in range(n_lrs):
        r = results[f'lr_{li}']
        ps_list = r['persistence_summaries']
        h1c, h1t, h1g, h2c = [], [], [], []
        for ps in ps_list:
            if ps:
                h1 = ps.get('h1', {})
                h1c.append(h1.get('n_features', 0))
                h1t.append(h1.get('max_lifetime', 0))
                h1g.append(h1.get('gap_ratio', 0) if h1.get('gap_ratio') is not None else 0)
                h2 = ps.get('h2', {})
                h2c.append(h2.get('n_features', 0))
        h1_counts.append(np.mean(h1c) if h1c else 0)
        h1_tops.append(np.mean(h1t) if h1t else 0)
        h1_gaps.append(np.mean(h1g) if h1g else 0)
        h2_counts.append(np.mean(h2c) if h2c else 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # H₁ feature count
    ax = axes[0]
    ax.plot(lr_fractions, h1_counts, 'ko-', ms=8, lw=2)
    ax.axhline(2, color='red', ls='--', lw=1, alpha=0.5, label='2 (clean 2-torus)')
    ax.set_xlabel('Fraction of EoS')
    ax.set_ylabel('Number of persistent H₁ features')
    ax.set_title('(a) H₁ feature count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.annotate('No loops\n(convergence path)',
                xy=(lr_fractions[0], h1_counts[0]),
                xytext=(0.15, 50), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#333333'))
    ax.annotate('Hundreds of loops\n(fractal structure)',
                xy=(lr_fractions[4], h1_counts[4]),
                xytext=(0.55, 350), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#333333'))
    
    # H₁ gap ratio
    ax = axes[1]
    ax.plot(lr_fractions, h1_gaps, 'ko-', ms=8, lw=2)
    ax.axhline(3, color='red', ls='--', lw=1, alpha=0.5, label='Gap > 3 (clear torus)')
    ax.set_xlabel('Fraction of EoS')
    ax.set_ylabel('H₁ gap ratio (1st / 2nd lifetime)')
    ax.set_title('(b) Topological signal clarity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 4)
    ax.annotate('Gap ≈ 1 everywhere\n→ no dominant loops\n→ fractal, not torus',
                xy=(0.4, 1.1), xytext=(0.45, 2.5), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#333333'))
    
    # H₂ feature count
    ax = axes[2]
    ax.plot(lr_fractions, h2_counts, 'ko-', ms=8, lw=2)
    ax.axhline(1, color='red', ls='--', lw=1, alpha=0.5, label='1 (clean torus void)')
    ax.set_xlabel('Fraction of EoS')
    ax.set_ylabel('Number of persistent H₂ features')
    ax.set_title('(c) H₂ feature count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    fig.suptitle('Persistent Homology Confirms Fractal Topology (Not Smooth Torus)',
                 fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    plt.savefig(output)
    print(f'Saved → {output}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PRL Figure 1')
    parser.add_argument('--data', type=str, default='cifar10_eos_10seeds.json',
                        help='Path to merged 10-seed JSON')
    parser.add_argument('--tda', type=str, default=None,
                        help='Path to TDA results JSON (for supplemental figure)')
    parser.add_argument('--output', type=str, default='figures/figure1_prl.png')
    args = parser.parse_args()
    
    print('Loading data...')
    d = load_data(args.data)
    print(f'  {d["n_seeds"]} seeds, {len(d["fracs"])} learning rates')
    print(f'  λ_max = {d["lam_max"]:.4f}')
    print(f'  Peak D₂ = {d["d2_mean"].max():.3f} ± {d["d2_std"][d["d2_mean"].argmax()]:.3f} '
          f'at {d["fracs"][d["d2_mean"].argmax()]:.0%} EoS')
    
    print('\nGenerating Figure 1...')
    plot_figure1(d, output=args.output)
    
    if args.tda:
        print('\nGenerating TDA supplemental figure...')
        plot_tda_supplement(args.tda, output='figures/tda_supplement.png')
    
    print('\nDone.')


if __name__ == '__main__':
    main()
