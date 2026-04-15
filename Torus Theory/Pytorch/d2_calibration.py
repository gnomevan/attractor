"""
D₂ Pipeline Calibration — Known Attractors Across Dimensions
==============================================================

Tests our Grassberger-Procaccia correlation dimension pipeline against
systems with KNOWN D₂, spanning the range relevant to the CNN result.

Systems tested:
  Smooth manifolds (exact integer dimension):
    - 1-torus (circle):          D₂ = 1.0
    - 2-torus (quasiperiodic):   D₂ = 2.0
    - 3-torus (quasiperiodic):   D₂ = 3.0
    - 4-torus (quasiperiodic):   D₂ = 4.0

  Fractal / strange attractors:
    - Hénon map:                 D₂ ≈ 1.21
    - Lorenz:                    D₂ ≈ 2.05
    - Rössler:                   D₂ ≈ 1.99
    - Mackey-Glass τ=17:         D₂ ≈ 2.1
    - Mackey-Glass τ=23:         D₂ ≈ 3.6   ← same range as CNN result
    - Mackey-Glass τ=30:         D₂ ≈ 5.0
    - Mackey-Glass τ=50:         D₂ ≈ 7.0

Both fitting methods:
  A) Fixed-index [4:16] out of 20 radii (original Experiment K protocol)
  B) Adaptive C(r) window: fit where 0.01 < C(r) < 0.5

Multiple sample sizes: 400, 800, 1600 (matching CNN trajectory scale)

USAGE:
    python -u d2_calibration.py
    python -u d2_calibration.py --quick        # fewer systems, faster
    python -u d2_calibration.py --plot-only     # from saved results

REQUIREMENTS:
    pip install numpy scipy matplotlib
    (CPU only — no GPU needed)
"""

import argparse, os, time, json
import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp


# ============================================================
# CORRELATION DIMENSION — TWO METHODS
# ============================================================

def corr_dim_fixed(points, seed=42):
    """
    Method A: Fixed-index fitting.
    Matches original Experiment K protocol exactly.
    20 log-spaced radii, percentile(1) to percentile(95), fit [4:16].
    """
    n = len(points)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(points[i] - points[j]))
    dists = np.array(dists)

    if len(dists) < 100:
        return float('nan')

    p1 = np.percentile(dists, 1)
    p95 = np.percentile(dists, 95)
    if p1 <= 0 or p95 <= p1:
        return float('nan')

    log_eps = np.linspace(np.log(p1 + 1e-15), np.log(p95), 20)
    N_pairs = n * (n - 1) / 2
    log_C = np.array([
        np.log(max(np.sum(dists < np.exp(le)) / N_pairs, 1e-30))
        for le in log_eps
    ])

    slope = stats.linregress(log_eps[4:16], log_C[4:16])[0]
    return float(slope)


def corr_dim_adaptive(points, seed=42):
    """
    Method B: Adaptive C(r) window.
    30 log-spaced radii, percentile(1) to percentile(90),
    fit where 0.01 < C(r) < 0.5.
    """
    from scipy.spatial.distance import pdist

    n = len(points)
    dists = pdist(points)
    dists = dists[dists > 0]

    if len(dists) < 100:
        return float('nan')

    r_min = np.percentile(dists, 1)
    r_max = np.percentile(dists, 90)
    if r_min <= 0 or r_max <= r_min:
        return float('nan')

    radii = np.logspace(np.log10(r_min), np.log10(r_max), 30)
    N_pairs = len(dists)
    C_r = np.array([np.sum(dists < r) / N_pairs for r in radii])

    mask = (C_r > 0.01) & (C_r < 0.5)
    if mask.sum() < 5:
        mask = (C_r > 0.005) & (C_r < 0.8)
    if mask.sum() < 4:
        return float('nan')

    log_r = np.log(radii[mask])
    log_C = np.log(C_r[mask])
    slope = np.polyfit(log_r, log_C, 1)[0]
    return float(slope)


# ============================================================
# KNOWN DYNAMICAL SYSTEMS
# ============================================================

def generate_torus(dim, n_points=10000, seed=42):
    """
    Generate quasiperiodic trajectory on a dim-torus.
    Uses mutually incommensurate frequencies.
    Returns points embedded in (2*dim)-dimensional space.
    """
    # Incommensurate frequencies: 1, sqrt(2), sqrt(3), sqrt(5), ...
    primes_sqrt = [1.0, np.sqrt(2), np.sqrt(3), np.sqrt(5),
                   np.sqrt(7), np.sqrt(11), np.sqrt(13)]
    freqs = primes_sqrt[:dim]

    t = np.linspace(0, 200 * np.pi, n_points)
    coords = []
    for i in range(dim):
        coords.append(np.cos(freqs[i] * t))
        coords.append(np.sin(freqs[i] * t))

    return np.column_stack(coords), float(dim)


def generate_henon(n_points=50000, seed=42):
    """
    Hénon map: x_{n+1} = 1 - a*x_n² + y_n, y_{n+1} = b*x_n
    Standard params: a=1.4, b=0.3
    D₂ ≈ 1.21
    """
    a, b = 1.4, 0.3
    x, y = 0.0, 0.0
    transient = 10000
    points = []
    for i in range(n_points + transient):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        if i >= transient:
            points.append([x, y])
        if abs(x) > 1e10:  # diverged
            return np.array(points) if points else np.zeros((0, 2)), 1.21

    return np.array(points), 1.21


def generate_lorenz(n_points=80000, seed=42):
    """
    Lorenz attractor. D₂ ≈ 2.05.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    def rhs(t, s):
        return [sigma * (s[1] - s[0]),
                s[0] * (rho - s[2]) - s[1],
                s[0] * s[1] - beta * s[2]]

    t_span = (0, 200)
    t_eval = np.linspace(0, 200, n_points)
    sol = solve_ivp(rhs, t_span, [1.0, 1.0, 1.0], t_eval=t_eval,
                    method='RK45', rtol=1e-10, atol=1e-12)
    start = len(sol.t) // 5
    return sol.y[:, start:].T, 2.05


def generate_rossler(n_points=80000, seed=42):
    """
    Rössler attractor. Standard params: a=0.2, b=0.2, c=5.7.
    D₂ ≈ 1.99.
    """
    a, b, c = 0.2, 0.2, 5.7
    def rhs(t, s):
        return [-(s[1] + s[2]),
                s[0] + a * s[1],
                b + s[2] * (s[0] - c)]

    t_eval = np.linspace(0, 500, n_points)
    sol = solve_ivp(rhs, (0, 500), [1.0, 1.0, 0.0], t_eval=t_eval,
                    method='RK45', rtol=1e-10, atol=1e-12)
    start = len(sol.t) // 5
    return sol.y[:, start:].T, 1.99


def generate_mackey_glass(tau=17, n_points=50000, dt=0.1, seed=42):
    """
    Mackey-Glass delay differential equation:
      dx/dt = β * x(t-τ) / (1 + x(t-τ)^10) - γ * x(t)

    β=0.2, γ=0.1. The delay τ controls the attractor dimension:
      τ = 17:  D₂ ≈ 2.1
      τ = 23:  D₂ ≈ 3.6
      τ = 30:  D₂ ≈ 5.0
      τ = 50:  D₂ ≈ 7.0

    Reference values from Grassberger & Procaccia (1983),
    Farmer (1982), and Sprott (2003).
    """
    beta_mg = 0.2
    gamma = 0.1
    n_exp = 10

    # Expected D₂ (from literature)
    expected = {17: 2.1, 23: 3.6, 30: 5.0, 50: 7.0}
    d2_expected = expected.get(tau, None)

    # Integrate using Euler with history buffer
    delay_steps = int(tau / dt)
    total_steps = int(n_points * 10)  # oversample then subsample
    transient_steps = int(total_steps * 0.3)

    # Initialize history
    history_len = delay_steps + 1
    x_hist = np.ones(history_len) * 0.9  # constant initial history
    # Add small perturbation to break symmetry
    rng = np.random.RandomState(seed)
    x_hist += rng.randn(history_len) * 0.01

    trajectory = []
    x = x_hist[-1]

    for step in range(total_steps):
        # Get delayed value
        x_delayed = x_hist[0]

        # Mackey-Glass equation
        dxdt = beta_mg * x_delayed / (1 + x_delayed**n_exp) - gamma * x

        # Euler step
        x_new = x + dt * dxdt

        # Update history buffer (shift left, add new value)
        x_hist[:-1] = x_hist[1:]
        x_hist[-1] = x_new
        x = x_new

        if step >= transient_steps:
            trajectory.append(x)

    trajectory = np.array(trajectory)

    # Subsample to target number of points
    if len(trajectory) > n_points:
        idx = np.linspace(0, len(trajectory) - 1, n_points, dtype=int)
        trajectory = trajectory[idx]

    # Delay embedding for dimension estimation
    # Use embedding dimension = 2*ceil(D₂_expected) + 1
    if d2_expected is not None:
        embed_dim = min(2 * int(np.ceil(d2_expected)) + 1, 15)
    else:
        embed_dim = 10

    # Delay for embedding: use tau_embed ≈ tau (the system's own delay)
    tau_embed = max(1, int(tau / dt / 10))  # subsample factor

    # Build delay embedding
    n_embed = len(trajectory) - (embed_dim - 1) * tau_embed
    if n_embed < 500:
        # Reduce embedding dimension
        embed_dim = min(embed_dim, (len(trajectory) - 1) // tau_embed)
        n_embed = len(trajectory) - (embed_dim - 1) * tau_embed

    embedded = np.zeros((n_embed, embed_dim))
    for d in range(embed_dim):
        embedded[:, d] = trajectory[d * tau_embed: d * tau_embed + n_embed]

    return embedded, d2_expected


# ============================================================
# RUN CALIBRATION
# ============================================================

def run_calibration(sample_sizes=[400, 800, 1600], quick=False):
    """Test D₂ pipeline on all known systems."""

    if quick:
        sample_sizes = [800]

    print("=" * 70)
    print("D₂ PIPELINE CALIBRATION")
    print(f"  Sample sizes: {sample_sizes}")
    print("=" * 70)

    # Define all systems
    systems = []

    # Smooth tori
    for dim in [1, 2, 3, 4]:
        systems.append({
            'name': f'{dim}-torus',
            'generator': lambda d=dim: generate_torus(d),
            'expected': float(dim),
            'type': 'smooth',
        })

    # Strange attractors
    systems.extend([
        {'name': 'Hénon',        'generator': generate_henon,    'expected': 1.21, 'type': 'fractal'},
        {'name': 'Rössler',      'generator': generate_rossler,  'expected': 1.99, 'type': 'fractal'},
        {'name': 'Lorenz',       'generator': generate_lorenz,   'expected': 2.05, 'type': 'fractal'},
        {'name': 'MG τ=17',      'generator': lambda: generate_mackey_glass(tau=17), 'expected': 2.1,  'type': 'fractal'},
        {'name': 'MG τ=23',      'generator': lambda: generate_mackey_glass(tau=23), 'expected': 3.6,  'type': 'fractal'},
        {'name': 'MG τ=30',      'generator': lambda: generate_mackey_glass(tau=30), 'expected': 5.0,  'type': 'fractal'},
    ])

    if not quick:
        systems.append(
            {'name': 'MG τ=50', 'generator': lambda: generate_mackey_glass(tau=50), 'expected': 7.0, 'type': 'fractal'},
        )

    results = []
    t0 = time.time()

    for si, sys in enumerate(systems):
        print(f"\n--- {sys['name']} (expected D₂ = {sys['expected']}) ---")
        t1 = time.time()

        # Generate trajectory
        traj, expected = sys['generator']()
        print(f"  Trajectory: {traj.shape[0]} points × {traj.shape[1]} dims")

        for n_pts in sample_sizes:
            if n_pts > len(traj):
                n_pts_actual = len(traj)
            else:
                n_pts_actual = n_pts

            # Subsample
            idx = np.random.RandomState(42).choice(len(traj), n_pts_actual, replace=False)
            pts = traj[idx]

            # Method A: fixed-index
            d2_fixed = corr_dim_fixed(pts)

            # Method B: adaptive
            d2_adaptive = corr_dim_adaptive(pts)

            error_fixed = d2_fixed - expected if not np.isnan(d2_fixed) else float('nan')
            error_adaptive = d2_adaptive - expected if not np.isnan(d2_adaptive) else float('nan')

            results.append({
                'system': sys['name'],
                'type': sys['type'],
                'expected': expected,
                'n_points': n_pts_actual,
                'd2_fixed': d2_fixed,
                'd2_adaptive': d2_adaptive,
                'error_fixed': error_fixed,
                'error_adaptive': error_adaptive,
            })

            print(f"  n={n_pts_actual:5d}: "
                  f"fixed={d2_fixed:6.3f} (err={error_fixed:+.3f})  "
                  f"adaptive={d2_adaptive:6.3f} (err={error_adaptive:+.3f})")

        elapsed = time.time() - t1
        print(f"  ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY (n=800 or closest)")
    print("=" * 70)
    print(f"{'System':<12s} {'Type':<8s} {'Expected':>8s} {'Fixed':>8s} {'Err':>7s} {'Adapt':>8s} {'Err':>7s}")
    print("-" * 70)

    # Filter to n=800 (or closest available)
    for sys in systems:
        matching = [r for r in results
                    if r['system'] == sys['name'] and r['n_points'] >= 400]
        if matching:
            # Prefer 800
            r800 = [r for r in matching if r['n_points'] == 800]
            r = r800[0] if r800 else matching[0]
            print(f"{r['system']:<12s} {r['type']:<8s} {r['expected']:8.2f} "
                  f"{r['d2_fixed']:8.3f} {r['error_fixed']:+7.3f} "
                  f"{r['d2_adaptive']:8.3f} {r['error_adaptive']:+7.3f}")

    # ── Bias analysis ──
    print("\n" + "=" * 70)
    print("BIAS ANALYSIS")
    print("=" * 70)

    # Separate smooth vs fractal
    for atype in ['smooth', 'fractal']:
        subset = [r for r in results if r['type'] == atype and r['n_points'] >= 400]
        if not subset:
            continue

        # Use n=800 where available
        best = {}
        for r in subset:
            key = r['system']
            if key not in best or abs(r['n_points'] - 800) < abs(best[key]['n_points'] - 800):
                best[key] = r
        subset = list(best.values())

        errors_fixed = [r['error_fixed'] for r in subset if not np.isnan(r['error_fixed'])]
        errors_adaptive = [r['error_adaptive'] for r in subset if not np.isnan(r['error_adaptive'])]

        if errors_fixed:
            print(f"\n  {atype.upper()} SYSTEMS:")
            print(f"    Fixed method:    mean error = {np.mean(errors_fixed):+.3f}, "
                  f"std = {np.std(errors_fixed):.3f}")
            print(f"    Adaptive method: mean error = {np.mean(errors_adaptive):+.3f}, "
                  f"std = {np.std(errors_adaptive):.3f}")

    # ── Key question: does bias scale with dimension? ──
    print("\n" + "=" * 70)
    print("DOES BIAS SCALE WITH DIMENSION?")
    print("=" * 70)

    # Get n=800 results for fractal systems, sorted by expected D₂
    fractal_800 = sorted(
        [r for r in results if r['type'] == 'fractal' and r['n_points'] >= 400],
        key=lambda r: r['expected']
    )
    # Deduplicate by system name, preferring n=800
    seen = {}
    for r in fractal_800:
        key = r['system']
        if key not in seen or abs(r['n_points'] - 800) < abs(seen[key]['n_points'] - 800):
            seen[key] = r
    fractal_sorted = sorted(seen.values(), key=lambda r: r['expected'])

    print(f"  {'System':<12s} {'Expected':>8s} {'Fixed':>8s} {'%err':>7s} {'Adapt':>8s} {'%err':>7s}")
    print("  " + "-" * 60)
    for r in fractal_sorted:
        pct_fixed = 100 * r['error_fixed'] / r['expected'] if r['expected'] > 0 else 0
        pct_adapt = 100 * r['error_adaptive'] / r['expected'] if r['expected'] > 0 else 0
        print(f"  {r['system']:<12s} {r['expected']:8.2f} "
              f"{r['d2_fixed']:8.3f} {pct_fixed:+6.1f}% "
              f"{r['d2_adaptive']:8.3f} {pct_adapt:+6.1f}%")

    # Linear regression: measured vs expected
    if len(fractal_sorted) >= 3:
        exp_vals = np.array([r['expected'] for r in fractal_sorted])
        fix_vals = np.array([r['d2_fixed'] for r in fractal_sorted])
        adp_vals = np.array([r['d2_adaptive'] for r in fractal_sorted])

        mask_f = ~np.isnan(fix_vals)
        mask_a = ~np.isnan(adp_vals)

        if mask_f.sum() >= 3:
            slope_f, intercept_f, r_f, _, _ = stats.linregress(exp_vals[mask_f], fix_vals[mask_f])
            print(f"\n  Fixed method:    D₂_measured = {slope_f:.3f} × D₂_true + {intercept_f:+.3f}  (R² = {r_f**2:.4f})")
            # What does this predict for CNN D₂_measured = 3.6?
            if slope_f > 0:
                corrected_fixed = (3.6 - intercept_f) / slope_f
                print(f"    → If CNN measures 3.6, true D₂ ≈ {corrected_fixed:.2f}")

        if mask_a.sum() >= 3:
            slope_a, intercept_a, r_a, _, _ = stats.linregress(exp_vals[mask_a], adp_vals[mask_a])
            print(f"  Adaptive method: D₂_measured = {slope_a:.3f} × D₂_true + {intercept_a:+.3f}  (R² = {r_a**2:.4f})")
            if slope_a > 0:
                corrected_adaptive = (3.6 - intercept_a) / slope_a
                print(f"    → If CNN measures 3.6, true D₂ ≈ {corrected_adaptive:.2f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/d2_calibration.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → results/d2_calibration.json")

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_calibration(json_path="results/d2_calibration.json"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(json_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Deduplicate to n=800 or closest
    by_system = {}
    for r in results:
        key = r['system']
        if key not in by_system or abs(r['n_points'] - 800) < abs(by_system[key]['n_points'] - 800):
            by_system[key] = r
    data = sorted(by_system.values(), key=lambda r: r['expected'])

    smooth = [r for r in data if r['type'] == 'smooth']
    fractal = [r for r in data if r['type'] == 'fractal']

    # Panel 1: Measured vs Expected (both methods)
    ax = axes[0]
    max_d = max(r['expected'] for r in data) + 1
    ax.plot([0, max_d], [0, max_d], 'k--', lw=1, label='Perfect')

    for r in smooth:
        ax.plot(r['expected'], r['d2_fixed'], 'bs', ms=10)
        ax.plot(r['expected'], r['d2_adaptive'], 'bo', ms=10)
    for r in fractal:
        ax.plot(r['expected'], r['d2_fixed'], 'rs', ms=10)
        ax.plot(r['expected'], r['d2_adaptive'], 'ro', ms=10)

    ax.plot([], [], 'bs', ms=10, label='Smooth (fixed)')
    ax.plot([], [], 'bo', ms=10, label='Smooth (adaptive)')
    ax.plot([], [], 'rs', ms=10, label='Fractal (fixed)')
    ax.plot([], [], 'ro', ms=10, label='Fractal (adaptive)')

    # Mark CNN result
    ax.axhline(3.6, color='green', ls=':', lw=1, alpha=0.7)
    ax.annotate('CNN measured', xy=(0.5, 3.6), fontsize=9, color='green')

    ax.set_xlabel('Expected D₂')
    ax.set_ylabel('Measured D₂')
    ax.set_title('Measured vs Expected D₂')
    ax.legend(fontsize=8)
    ax.set_xlim(0, max_d)
    ax.set_ylim(0, max_d)
    ax.grid(True, alpha=0.3)

    # Panel 2: Error vs Expected (both methods)
    ax = axes[1]
    ax.axhline(0, color='k', ls='-', lw=0.5)

    exp_s = [r['expected'] for r in smooth]
    exp_f = [r['expected'] for r in fractal]
    err_s_fix = [r['error_fixed'] for r in smooth]
    err_s_adp = [r['error_adaptive'] for r in smooth]
    err_f_fix = [r['error_fixed'] for r in fractal]
    err_f_adp = [r['error_adaptive'] for r in fractal]

    ax.plot(exp_s, err_s_fix, 'bs-', ms=8, label='Smooth (fixed)')
    ax.plot(exp_s, err_s_adp, 'bo-', ms=8, label='Smooth (adaptive)')
    ax.plot(exp_f, err_f_fix, 'rs-', ms=8, label='Fractal (fixed)')
    ax.plot(exp_f, err_f_adp, 'ro-', ms=8, label='Fractal (adaptive)')

    ax.set_xlabel('Expected D₂')
    ax.set_ylabel('Error (measured − expected)')
    ax.set_title('Systematic Bias vs Dimension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Percentage error vs Expected
    ax = axes[2]
    ax.axhline(0, color='k', ls='-', lw=0.5)

    pct_s_fix = [100 * r['error_fixed'] / r['expected'] for r in smooth if r['expected'] > 0]
    pct_s_adp = [100 * r['error_adaptive'] / r['expected'] for r in smooth if r['expected'] > 0]
    pct_f_fix = [100 * r['error_fixed'] / r['expected'] for r in fractal if r['expected'] > 0]
    pct_f_adp = [100 * r['error_adaptive'] / r['expected'] for r in fractal if r['expected'] > 0]

    exp_s_pos = [r['expected'] for r in smooth if r['expected'] > 0]
    exp_f_pos = [r['expected'] for r in fractal if r['expected'] > 0]

    ax.plot(exp_s_pos, pct_s_fix, 'bs-', ms=8, label='Smooth (fixed)')
    ax.plot(exp_s_pos, pct_s_adp, 'bo-', ms=8, label='Smooth (adaptive)')
    ax.plot(exp_f_pos, pct_f_fix, 'rs-', ms=8, label='Fractal (fixed)')
    ax.plot(exp_f_pos, pct_f_adp, 'ro-', ms=8, label='Fractal (adaptive)')

    ax.set_xlabel('Expected D₂')
    ax.set_ylabel('Error (%)')
    ax.set_title('Percentage Bias vs Dimension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('D₂ Pipeline Calibration Against Known Attractors', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/d2_calibration.png", dpi=200, bbox_inches="tight")
    print(f"  Saved → figures/d2_calibration.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="D₂ Pipeline Calibration")
    parser.add_argument("--quick", action="store_true", help="Fewer systems, one sample size")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--sample-sizes", type=int, nargs='+', default=[400, 800, 1600])
    args = parser.parse_args()

    if args.plot_only:
        plot_calibration()
        return

    results = run_calibration(sample_sizes=args.sample_sizes, quick=args.quick)
    plot_calibration()
    print("\nDone.")


if __name__ == "__main__":
    main()
