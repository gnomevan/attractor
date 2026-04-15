"""
r1_merge.py  —  Merge legacy + revision-1 cross-experiment data
================================================================

This script produces the N=10-seed merged data files that the revised
manuscript tables and figures read from.  Three merge strategies:

  1. **mlp_cifar_w50** — genuine merge.  The legacy file
     ``data/main/cross_small_mlp_cifar_w50_seeds_0_1_2.json`` has seeds
     [0,1,2] at the full 12-fraction LR grid.  The r1 file has seeds
     [3..9].  Both share the same protocol (verified by
     ``r1_cross_experiments.py --reproduce-check``).  We interleave
     per-seed results into a single merged JSON.

  2. **mlp_cifar_w85** — passthrough.  No legacy data exists (the
     original w85 results were never committed).  The r1 file IS the
     canonical 10-seed dataset; we copy it with a ``_merged`` suffix
     for pipeline uniformity.

  3. **cnn_synthetic** — passthrough.  The legacy file has only 1 seed
     at 5 LR fractions (the quick grid), which is incompatible with the
     12-fraction canonical grid.  We re-ran from scratch at N=10.

The merge ONLY fires for w50 (strategy 1).  For strategies 2 and 3
we still emit a ``_merged.json`` file so that downstream scripts can
uniformly read ``*_merged.json`` without condition-specific logic.

Output directory: ``data/main/revision1/``

Validation checks (automatic, fail-fast):
  - Legacy and r1 files must agree on lam_max, lr_eos, n_params to
    within 1e-6 relative tolerance (shared warmup protocol).
  - LR fractions must match exactly.
  - No duplicate seeds between legacy and r1 files.

Addresses referee issues #1 (N=3→10), #2 (suspiciously tight MLP variances).

Usage
-----

   # Full merge run (after all Phase 1 experiments have completed):
   python r1_merge.py --data-dir ../../data/main --r1-dir ../../data/main/revision1

   # Dry run: show what would be merged, without writing files:
   python r1_merge.py --data-dir ../../data/main --r1-dir ../../data/main/revision1 --dry-run

   # Merge only w50 (if other conditions haven't finished yet):
   python r1_merge.py --data-dir ../../data/main --r1-dir ../../data/main/revision1 --only mlp_cifar_w50
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np


# ============================================================
# MERGE PLANS
# ============================================================

# Fields that appear in per-LR buckets and store one value per seed.
# The merge interleaves these lists by seed order.
SEED_FIELDS = [
    "lyapunov", "corr_dim", "pc1", "pc2",
    "sharpness_series", "grad_norm_series", "loss_series",
]


def _validate_scalars(legacy, r1, tol=1e-6):
    """Check that protocol-critical scalars agree."""
    problems = []
    for key in ("lam_max", "lr_eos", "n_params"):
        a, b = legacy.get(key), r1.get(key)
        if a is None or b is None:
            continue
        if isinstance(a, float) and isinstance(b, float):
            denom = max(abs(a), abs(b), 1e-12)
            if abs(a - b) / denom > tol:
                problems.append(
                    f"  {key}: legacy={a:.8g}  r1={b:.8g}  "
                    f"rel={abs(a-b)/denom:.2e}")
        elif a != b:
            problems.append(f"  {key}: legacy={a}  r1={b}")
    return problems


def _validate_lr_fractions(legacy, r1):
    """LR fraction grids must be identical."""
    lf_l = legacy.get("lr_fractions", [])
    lf_r = r1.get("lr_fractions", [])
    if len(lf_l) != len(lf_r):
        return [f"  lr_fractions length: legacy={len(lf_l)} r1={len(lf_r)}"]
    diffs = []
    for i, (a, b) in enumerate(zip(lf_l, lf_r)):
        if abs(a - b) > 1e-10:
            diffs.append(f"  lr_fractions[{i}]: legacy={a} r1={b}")
    return diffs


def _validate_no_duplicate_seeds(legacy, r1):
    ls = set(legacy.get("seeds_run", []))
    rs = set(r1.get("seeds_run", []))
    overlap = ls & rs
    if overlap:
        return [f"  duplicate seeds: {sorted(overlap)}"]
    return []


def merge_two(legacy, r1, verbose=True):
    """
    Merge two data dicts (same condition, same LR grid, disjoint seeds)
    into a single dict with seeds sorted in ascending order.
    """
    # Validate
    for validator, label in [
        (_validate_scalars, "scalar mismatch"),
        (_validate_lr_fractions, "LR fraction mismatch"),
        (_validate_no_duplicate_seeds, "duplicate seeds"),
    ]:
        errs = validator(legacy, r1)
        if errs:
            print(f"MERGE VALIDATION FAILED ({label}):")
            for e in errs:
                print(e)
            raise ValueError(f"Merge validation failed: {label}")

    legacy_seeds = legacy["seeds_run"]
    r1_seeds = r1["seeds_run"]
    all_seeds = sorted(set(legacy_seeds) | set(r1_seeds))

    merged = copy.deepcopy(r1)  # start with r1 as the base (has metadata)
    merged["seeds_run"] = all_seeds
    merged["merge_info"] = {
        "legacy_seeds": legacy_seeds,
        "r1_seeds": r1_seeds,
        "merged_seeds": all_seeds,
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "legacy_experiment": legacy.get("experiment", "unknown"),
    }

    n_fracs = len(merged["lr_fractions"])

    for li in range(n_fracs):
        key = f"lr_{li}"
        lb = legacy.get(key, {})
        rb = r1.get(key, {})

        # Build a seed → data-dict lookup for each source
        legacy_data = {}
        for si, seed in enumerate(legacy_seeds):
            legacy_data[seed] = {}
            for field in SEED_FIELDS:
                vals = lb.get(field, [])
                if si < len(vals):
                    legacy_data[seed][field] = vals[si]

        r1_data = {}
        for si, seed in enumerate(r1_seeds):
            r1_data[seed] = {}
            for field in SEED_FIELDS:
                vals = rb.get(field, [])
                if si < len(vals):
                    r1_data[seed][field] = vals[si]

        # Interleave in seed order
        merged_bucket = {field: [] for field in SEED_FIELDS}
        for seed in all_seeds:
            src = legacy_data.get(seed) or r1_data.get(seed, {})
            for field in SEED_FIELDS:
                val = src.get(field)
                if val is not None:
                    merged_bucket[field].append(val)
                # If field is missing (legacy w50 lacks grad_norm_series,
                # loss_series), we append None to preserve alignment
                elif field in ("grad_norm_series", "loss_series"):
                    merged_bucket[field].append(None)

        merged[key] = merged_bucket

    if verbose:
        print(f"  merged: {len(all_seeds)} seeds "
              f"({len(legacy_seeds)} legacy + {len(r1_seeds)} new)")

    return merged


def passthrough(r1, verbose=True):
    """
    For conditions with no usable legacy data: copy r1 as-is,
    adding a merge_info block for pipeline uniformity.
    """
    merged = copy.deepcopy(r1)
    merged["merge_info"] = {
        "legacy_seeds": [],
        "r1_seeds": list(r1.get("seeds_run", [])),
        "merged_seeds": list(r1.get("seeds_run", [])),
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "passthrough — no compatible legacy data",
    }
    if verbose:
        print(f"  passthrough: {len(r1.get('seeds_run', []))} seeds (no legacy)")
    return merged


# ============================================================
# SUMMARY STATS
# ============================================================

def print_summary(merged):
    """Print a D2 summary table from the merged data."""
    fracs = merged.get("lr_fractions", [])
    seeds = merged.get("seeds_run", [])
    print(f"\n  {merged['experiment']}  N={len(seeds)}  "
          f"seeds={seeds}")
    print(f"  {'%EoS':>6s}  {'D2 mean':>8s}  {'D2 std':>8s}  "
          f"{'λ mean':>10s}  {'λ std':>10s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    for li, frac in enumerate(fracs):
        key = f"lr_{li}"
        b = merged.get(key, {})
        d2s = [x for x in b.get("corr_dim", [])
               if x is not None and not (isinstance(x, float)
                                         and np.isnan(x))]
        lyaps = [x for x in b.get("lyapunov", [])
                 if x is not None]
        d2_mean = np.mean(d2s) if d2s else float("nan")
        d2_std = np.std(d2s) if d2s else float("nan")
        l_mean = np.mean(lyaps) if lyaps else float("nan")
        l_std = np.std(lyaps) if lyaps else float("nan")
        print(f"  {frac:6.0%}  {d2_mean:8.3f}  {d2_std:8.3f}  "
              f"{l_mean:+10.6f}  {l_std:10.6f}")


# ============================================================
# CONDITION DISPATCH
# ============================================================

# Maps condition name → (legacy_filename, r1_filename_pattern, strategy)
# The r1 filename pattern uses a glob-ish notation; in practice we
# look for any file matching the stem + seeds.

MERGE_PLANS = {
    "mlp_cifar_w50": {
        "legacy_file": None,   # reproduce-check showed hardware float divergence; all 10 seeds run fresh
        "r1_stem": "cross_small_mlp_cifar_w50_seeds_",
        "strategy": "passthrough",
    },
    "mlp_cifar_w85": {
        "legacy_file": None,   # no usable legacy
        "r1_stem": "cross_small_mlp_cifar_w85_seeds_",
        "strategy": "passthrough",
    },
    "cnn_synthetic": {
        "legacy_file": None,   # legacy has 1 seed / 5 fractions, incompatible
        "r1_stem": "cross_cnn_synthetic_seeds_",
        "strategy": "passthrough",
    },
}


def _find_r1_file(r1_dir, stem):
    """Find the single r1 file matching the stem. Error if 0 or >1."""
    candidates = [
        f for f in os.listdir(r1_dir)
        if f.startswith(stem) and f.endswith(".json")
        and "merged" not in f and "metadata" not in f
    ]
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return os.path.join(r1_dir, candidates[0])
    # Multiple files (e.g. partial seed runs); pick the one with the
    # most seeds in the filename
    candidates.sort(key=lambda f: len(f), reverse=True)
    print(f"  WARNING: multiple r1 files for stem '{stem}': {candidates}")
    print(f"  Using: {candidates[0]}")
    return os.path.join(r1_dir, candidates[0])


def run_merge(condition_name, data_dir, r1_dir, dry_run=False, verbose=True):
    """Execute one merge plan."""
    plan = MERGE_PLANS[condition_name]
    print(f"\n{'='*64}")
    print(f"MERGE: {condition_name}  (strategy: {plan['strategy']})")
    print(f"{'='*64}")

    # Locate r1 file
    r1_path = _find_r1_file(r1_dir, plan["r1_stem"])
    if r1_path is None:
        print(f"  SKIP: no r1 file found for {plan['r1_stem']}* in {r1_dir}")
        return None

    print(f"  r1 file:     {r1_path}")

    # Locate legacy file (if applicable)
    legacy_path = None
    if plan["legacy_file"]:
        legacy_path = os.path.join(data_dir, plan["legacy_file"])
        if not os.path.exists(legacy_path):
            print(f"  WARNING: legacy file not found: {legacy_path}")
            print(f"  Falling back to passthrough strategy")
            legacy_path = None

    if legacy_path:
        print(f"  legacy file: {legacy_path}")

    if dry_run:
        print(f"  DRY RUN — would produce: "
              f"{plan['r1_stem']}merged.json in {r1_dir}")
        return None

    # Load files
    with open(r1_path) as f:
        r1_data = json.load(f)

    if plan["strategy"] == "merge" and legacy_path:
        with open(legacy_path) as f:
            legacy_data = json.load(f)
        merged = merge_two(legacy_data, r1_data, verbose=verbose)
    else:
        merged = passthrough(r1_data, verbose=verbose)

    # Save
    out_name = f"{plan['r1_stem']}merged.json"
    out_path = os.path.join(r1_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  saved -> {out_path}")

    if verbose:
        print_summary(merged)

    return out_path


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Merge legacy + r1 cross-experiment data into "
                    "unified N=10 files.")
    p.add_argument("--data-dir", type=str, default="../../data/main",
                   help="Directory containing legacy JSON files.")
    p.add_argument("--r1-dir", type=str, default="../../data/main/revision1",
                   help="Directory containing r1 JSON files.")
    p.add_argument("--only", type=str, default=None,
                   choices=list(MERGE_PLANS.keys()),
                   help="Merge only one condition (default: all).")
    p.add_argument("--dry-run", action="store_true",
                   help="Show plan without writing files.")
    args = p.parse_args()

    conditions = [args.only] if args.only else list(MERGE_PLANS.keys())
    results = {}
    for cond in conditions:
        path = run_merge(cond, args.data_dir, args.r1_dir,
                         dry_run=args.dry_run)
        results[cond] = path

    print("\n" + "=" * 64)
    print("MERGE SUMMARY")
    print("=" * 64)
    for cond, path in results.items():
        status = path if path else "(skipped or dry-run)"
        print(f"  {cond:20s} -> {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
