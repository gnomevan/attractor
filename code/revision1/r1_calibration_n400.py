#!/usr/bin/env python3
"""
r1_calibration_n400.py — Phase 2A calibration extension
=========================================================

Referee issue #5: calibration table uses n=800, but production neural runs
use n≈400. Calibration must match.

The existing d2_calibration.json already contains data at n={400, 800, 1600}
for all 11 known systems. This script:

1. READS existing calibration data (does NOT rerun — results are deterministic).
2. EXTENDS with n=200 and n=3200 for Lorenz and Mackey-Glass τ=30, producing
   the D₂(N) reference curves needed for Phase 2B comparison.
3. Saves outputs to data/supplemental/revision1/.

Output files:
  data/supplemental/revision1/d2_calibration_n400.json
    → n=400 calibration for all 11 systems (extracted from existing data)
  data/supplemental/revision1/d2_vs_n_reference.json
    → D₂(N) at n={200, 400, 800, 1600, 3200} for Lorenz and MG τ=30

USAGE:
    python -u code/revision1/r1_calibration_n400.py
    python -u code/revision1/r1_calibration_n400.py --dry-run
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

# ── Repo root ──
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXISTING_CALIB = os.path.join(ROOT, "data", "supplemental", "d2_calibration.json")
OUT_DIR = os.path.join(ROOT, "data", "supplemental", "revision1")


# ============================================================
# D₂ PIPELINE (copied from d2_calibration.py for self-containment)
# ============================================================

def corr_dim_fixed(points, seed=42):
    """Fixed-index [4:16] out of 20 radii. Original Experiment K protocol."""
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
    """Adaptive C(r) window: fit where 0.01 < C(r) < 0.5."""
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
# TRAJECTORY GENERATORS (Lorenz and MG τ=30 only)
# ============================================================

def generate_lorenz(n_points=80000, seed=42):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    def rhs(t, s):
        return [sigma * (s[1] - s[0]),
                s[0] * (rho - s[2]) - s[1],
                s[0] * s[1] - beta * s[2]]
    t_eval = np.linspace(0, 200, n_points)
    sol = solve_ivp(rhs, (0, 200), [1.0, 1.0, 1.0], t_eval=t_eval,
                    method='RK45', rtol=1e-10, atol=1e-12)
    start = len(sol.t) // 5
    return sol.y[:, start:].T


def generate_mackey_glass_30(n_points=50000, dt=0.1, seed=42):
    tau = 30
    beta_mg, gamma, n_exp = 0.2, 0.1, 10
    delay_steps = int(tau / dt)
    total_steps = int(n_points * 10)
    transient_steps = int(total_steps * 0.3)
    history_len = delay_steps + 1
    x_hist = np.ones(history_len) * 0.9
    rng = np.random.RandomState(seed)
    x_hist += rng.randn(history_len) * 0.01
    trajectory = []
    x = x_hist[-1]
    for step in range(total_steps):
        x_delayed = x_hist[0]
        dxdt = beta_mg * x_delayed / (1 + x_delayed**n_exp) - gamma * x
        x_new = x + dt * dxdt
        x_hist[:-1] = x_hist[1:]
        x_hist[-1] = x_new
        x = x_new
        if step >= transient_steps:
            trajectory.append(x)
    trajectory = np.array(trajectory)
    if len(trajectory) > n_points:
        idx = np.linspace(0, len(trajectory) - 1, n_points, dtype=int)
        trajectory = trajectory[idx]
    # Delay embedding: embed_dim = 2*ceil(5.0) + 1 = 11
    embed_dim = 11
    tau_embed = max(1, int(tau / dt / 10))
    n_embed = len(trajectory) - (embed_dim - 1) * tau_embed
    embedded = np.zeros((n_embed, embed_dim))
    for d in range(embed_dim):
        embedded[:, d] = trajectory[d * tau_embed: d * tau_embed + n_embed]
    return embedded


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2A: D₂ calibration extension")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Step 1: Extract n=400 calibration from existing data ──
    print("=== Step 1: Extract n=400 calibration from existing data ===")
    with open(EXISTING_CALIB) as f:
        all_calib = json.load(f)

    n400_data = [r for r in all_calib if r['n_points'] == 400]
    print(f"  Found {len(n400_data)} entries at n=400")
    for r in n400_data:
        print(f"    {r['system']:15s}  d2_fixed={r['d2_fixed']:.3f}  expected={r['expected']:.2f}")

    out_n400 = os.path.join(OUT_DIR, "d2_calibration_n400.json")
    if not args.dry_run:
        with open(out_n400, 'w') as f:
            json.dump(n400_data, f, indent=2)
        print(f"  Saved → {out_n400}")

    # ── Step 2: D₂(N) reference curves ──
    sample_sizes = [200, 400, 800, 1600, 3200]
    systems = [
        ("Lorenz", 2.05, generate_lorenz),
        ("MG τ=30", 5.0, generate_mackey_glass_30),
    ]

    print(f"\n=== Step 2: D₂(N) reference curves ===")
    print(f"  Systems: {[s[0] for s in systems]}")
    print(f"  Sample sizes: {sample_sizes}")

    if args.dry_run:
        print("  [DRY RUN — would generate trajectories and compute D₂ at each N]")
        return

    d2_vs_n = {}

    for sys_name, expected, generator in systems:
        print(f"\n--- {sys_name} (expected D₂ = {expected}) ---")
        t0 = time.time()

        traj = generator()
        print(f"  Full trajectory: {traj.shape[0]} points × {traj.shape[1]} dims")

        results = []
        for n_pts in sample_sizes:
            if n_pts > len(traj):
                print(f"  n={n_pts}: SKIP (trajectory too short)")
                continue

            idx = np.random.RandomState(42).choice(len(traj), n_pts, replace=False)
            pts = traj[idx]

            d2_f = corr_dim_fixed(pts)
            d2_a = corr_dim_adaptive(pts)

            results.append({
                "n_points": n_pts,
                "d2_fixed": d2_f,
                "d2_adaptive": d2_a,
                "error_fixed": d2_f - expected,
                "error_adaptive": d2_a - expected,
            })
            print(f"  n={n_pts:5d}: fixed={d2_f:.3f}  adaptive={d2_a:.3f}")

        d2_vs_n[sys_name] = {
            "expected": expected,
            "results": results,
        }

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")

    out_vsn = os.path.join(OUT_DIR, "d2_vs_n_reference.json")
    with open(out_vsn, 'w') as f:
        json.dump(d2_vs_n, f, indent=2)
    print(f"\nSaved → {out_vsn}")

    # ── Summary ──
    print("\n=== D₂(N) Summary ===")
    for sys_name, data in d2_vs_n.items():
        print(f"\n  {sys_name} (expected {data['expected']}):")
        for r in data['results']:
            print(f"    n={r['n_points']:5d}  D₂(fixed)={r['d2_fixed']:.3f}  D₂(adapt)={r['d2_adaptive']:.3f}")


if __name__ == "__main__":
    main()
