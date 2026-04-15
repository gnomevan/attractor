# Phase 1: Statistical Hardening for Chaos Onset in Gradient Descent

Two experiments to tighten the chaos onset threshold estimate and validate the Lyapunov measurement protocol.

## Setup

```bash
pip install torch numpy matplotlib
```

All three `.py` files must be in the same directory.

## Experiment A: 20-Seed Transition Zone Sweep

**Goal:** Tighten η_c from 0.018 ± 0.012 (3 seeds) to ± 0.003 (20 seeds).

```bash
# Quick test (5 seeds, ~30 min on GPU)
python exp_a_transition_zone.py --n_seeds 5

# Full run (20 seeds, ~5 hrs GPU / ~15 hrs CPU)
python exp_a_transition_zone.py

# Resume after interruption
python exp_a_transition_zone.py --resume

# Regenerate figures from saved data
python exp_a_transition_zone.py --plot_only
```

**Outputs:**
- `results/exp_a_transition_zone.json` — raw Lyapunov exponents for all runs
- `results/exp_a_summary.json` — per-LR statistics, η_c estimates
- `figures/exp_a_lyapunov_transition.png` — main result (λ vs η with error bars)
- `figures/exp_a_transition_detail.png` — all seeds visible, ±2σ band
- `figures/exp_a_kam_interleaving.png` — KAM diagnostic (which seeds chaotic at each η)

## Experiment B: Perturbation Sensitivity (ε Sweep)

**Goal:** Confirm Lyapunov exponents are true dynamical invariants, not ε artifacts.

```bash
# Quick test (3 seeds, ~1 hr GPU)
python exp_b_epsilon_sweep.py --n_seeds 3

# Full run (5 seeds, ~90 min GPU)
python exp_b_epsilon_sweep.py

# Regenerate figures
python exp_b_epsilon_sweep.py --plot_only
```

**Outputs:**
- `results/exp_b_epsilon_sweep.json` — all results
- `figures/exp_b_epsilon_sensitivity.png` — λ vs ε panels for each LR
- `figures/exp_b_epsilon_heatmap.png` — ε × η heatmap
- `figures/exp_b_stability_diagnostic.png` — normalized deviation diagnostic

**Interpretation:**
- **STABLE** (flat across ε): λ is a genuine Lyapunov exponent
- **MONOTONIC DRIFT**: nonlinear contamination, may need smaller ε
- **SIGN CHANGE**: measurement is at the transition boundary, needs more seeds

## Architecture Note

The chaos onset report states the architecture as `Input(220) → Linear(220,50) → Tanh → Linear(50,50) → Tanh → Linear(50,10)` but also says "156,710 parameters." These are inconsistent — the stated architecture has ~14,110 parameters. 

The scripts default to the stated architecture (hidden=50). If your original experiments used larger layers, use `--hidden 297` to get ~157K parameters:

```bash
python exp_a_transition_zone.py --hidden 297
python exp_b_epsilon_sweep.py --hidden 297
```

If you can share one of the original experiment scripts, I can match it exactly.

## Run Order

1. **Experiment B first** (faster, validates the measurement)
2. **Experiment A** (longer, but only meaningful if B passes)

If Experiment B shows ε-sensitivity, we may need to adjust the perturbation protocol before running the full A sweep.
