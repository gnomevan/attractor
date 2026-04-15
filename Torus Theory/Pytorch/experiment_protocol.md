# Experiment Protocol: Rigorous Replication of Chaos Onset in Gradient Descent

## Overview

This protocol specifies the experiments needed to strengthen the paper's claims from preliminary to publication-quality. The experiments are organized in three phases by priority.

---

## Phase 1: Statistical Robustness (Required for Submission)

### Goal
Increase seed count to eliminate the current statistical weakness (η_c = 0.018 ± 0.012 with 3 seeds is not convincing).

### Experiment 1A: Transition Zone — 20 Seeds

**What:** Re-run the transition zone experiment with 20 random seeds instead of 3.

```python
# Key parameters
N_SEEDS = 20
LR_MIN = 0.005
LR_MAX = 0.08
N_LRS = 40  # increase density from 30 to 40
LEARNING_RATES = np.linspace(LR_MIN, LR_MAX, N_LRS)

# Architecture (unchanged)
# Input(220) → Linear(220,50) → Tanh → Linear(50,50) → Tanh → Linear(50,10)

# Data generation (must be identical across all runs)
DATA_SEED = 42  # Fix the data generation seed
# 2000 points, 10 classes, 200 random features + 20 quadratic features

# For each seed s in range(N_SEEDS):
#   For each lr in LEARNING_RATES:
#     1. Generate data with DATA_SEED (same every time)
#     2. Initialize weights with seed s
#     3. Initialize perturbed weights with seed s + perturbation
#     4. Train both for T steps
#     5. Record Lyapunov exponent
```

**Expected result:** η_c with confidence interval roughly ±0.003 (shrinks by √(20/3) ≈ 2.6× from current ±0.012).

**Estimated compute:** 20 seeds × 40 learning rates × 2 networks per run = 1,600 training runs. Each is a small MLP, so this should be feasible on a single GPU in a few hours.

### Experiment 1B: Broad Sweep — 20 Seeds

**What:** Re-run the broad sweep with 20 seeds instead of 5.

```python
N_SEEDS = 20
LR_MIN = 0.01
LR_MAX = 0.42
N_LRS = 25
LEARNING_RATES = np.linspace(LR_MIN, LR_MAX, N_LRS)
```

**Purpose:** Tighten the error bands on the full Lyapunov curve and confirm reproducibility.

### Experiment 1C: Perturbation Sensitivity Analysis

**What:** Verify that the Lyapunov exponent estimates are stable across different perturbation magnitudes.

```python
EPSILONS = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
TEST_LRS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20]  # sample from each regime
N_SEEDS = 5

# For each (epsilon, lr, seed):
#   Compute function-space Lyapunov exponent
#   Record the result

# Expected: Lyapunov exponents should be approximately constant across
# epsilons (confirming we're measuring the true exponential divergence rate,
# not a numerical artifact). If they depend strongly on epsilon, the
# computation method needs revision.
```

**Key diagnostic:** Plot Lyapunov exponent vs. log(ε) for each learning rate. If the exponent is flat (plateau), the measurement is valid. If it trends with ε, the perturbation may be too large (nonlinear effects) or too small (numerical precision floor).

### Deliverables from Phase 1
- [ ] Figure: Transition zone with 20-seed error bands (replaces current Figure 2)
- [ ] Figure: Broad sweep with 20-seed error bands (replaces current Figure 1)
- [ ] Figure: Perturbation sensitivity (new supplementary figure)
- [ ] Table: η_c estimates per seed, mean, std, 95% CI
- [ ] Statistical test: Is η_c significantly less than 2/λ_max? (One-sample t-test, η_c vs. 0.27)

---

## Phase 2: Generalization Across Architectures (Required for Strong Paper)

### Goal
Show the phenomenon is a property of gradient descent, not the specific network.

### Experiment 2A: ReLU MLP

Same architecture dimensions but replace tanh with ReLU:

```python
# Input(220) → Linear(220,50) → ReLU → Linear(50,50) → ReLU → Linear(50,10)
# Same synthetic dataset, same protocol
# N_SEEDS = 10, same LR ranges as Phase 1

# Key question: Does η_c shift? The EoS threshold 2/λ_max will likely
# differ for ReLU (different Hessian spectrum), but the RATIO η_c/(2/λ_max)
# is the key quantity. Is it still ~6-7%?
```

**Why this matters:** ReLU networks have fundamentally different loss landscape geometry (piecewise linear). If the ratio η_c/(2/λ_max) is preserved, the phenomenon is about gradient descent geometry, not activation function specifics.

### Experiment 2B: Deeper Networks

```python
# 3-layer: 220 → 50 → 50 → 50 → 10 (tanh)
# 4-layer: 220 → 50 → 50 → 50 → 50 → 10 (tanh)
# Same dataset, same protocol
# N_SEEDS = 10

# Key question: Does deeper nesting of nonlinear transformations
# lower the chaos threshold? The Ruelle-Takens theory predicts that
# additional "nested cycles" (layers as iterated maps) should make
# chaos easier to reach — so we might expect η_c/(2/λ_max) to decrease
# with depth.
```

**Connection to theory:** Each layer in a deep network applies a nonlinear map, creating nested transformations analogous to nested frequencies in the Ruelle-Takens route. The theorem says 3+ nested frequencies generically produce chaos. Deeper networks have more "nested" structure, so the chaos threshold should be lower if the analogy holds.

### Experiment 2C: CIFAR-10 with CNN

```python
# Small CNN on CIFAR-10 subset (5,000 images, following Cohen et al.)
# Architecture: Conv(3,16,3) → ReLU → MaxPool → Conv(16,32,3) → ReLU → 
#              MaxPool → Flatten → Linear(?, 10)
# Full-batch gradient descent, MSE loss
# N_SEEDS = 10

# This tests whether the phenomenon survives:
# - Convolutional weight sharing
# - Real image data (not synthetic)
# - Different input dimensionality and structure
```

### Deliverables from Phase 2
- [ ] Table: η_c and η_c/(2/λ_max) for each architecture
- [ ] Figure: Lyapunov curves for all architectures on same axes
- [ ] Statistical test: Is η_c/(2/λ_max) constant across architectures?

---

## Phase 3: Dynamical Systems Characterization (Strengthens Paper Significantly)

### Goal
Go beyond Lyapunov exponents to characterize the dynamical regime (toroidal vs. chaotic) directly.

### Experiment 3A: Bifurcation Diagram

```python
# For a single seed, train to convergence at each learning rate
# Record the FINAL training loss (or last N loss values)
# Plot loss values vs. learning rate

# At low η: single converged loss value (fixed point)
# At moderate η: 2 alternating values (period-2 orbit)
# At higher η: 4 values (period-4), then 8, then chaos
# At EoS: self-stabilizing oscillation

# This should produce a diagram resembling the logistic map bifurcation
# diagram, with period-doubling visible below EoS.

N_SEEDS = 5
N_LRS = 200  # very fine spacing
LR_RANGE = (0.005, 0.35)
RECORD_LAST_N = 100  # last 100 loss values at each LR
```

**Why this matters:** If period-doubling is visible below the EoS threshold, it directly connects to the Feigenbaum route and allows testing whether the universal constants apply.

### Experiment 3B: Power Spectrum of Training Loss

```python
# For selected learning rates spanning the transition:
# LRS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30]
# Record loss at every training step (long training: 10,000+ steps)
# Compute FFT of the loss time series

# Expected:
# - Below η_c: discrete peaks (periodic or quasiperiodic)
# - Near η_c: peaks broadening (onset of chaos)
# - Above η_c: broadband spectrum (chaotic)
# - At EoS: mixed (broadband with persistent peaks from self-stabilization)
```

**Why this matters:** The frequency content directly distinguishes quasiperiodic (toroidal) dynamics from chaotic dynamics. Discrete peaks = motion on a torus. Broadband = strange attractor. This is the most direct test of the torus-to-chaos interpretation.

### Experiment 3C: Phase Space Reconstruction (Takens Embedding)

```python
# For selected learning rates:
# Take the training loss time series L(t)
# Construct delay embedding: [L(t), L(t+τ), L(t+2τ)]
# where τ is chosen by mutual information minimum

# Visualize the reconstructed attractor:
# - Below η_c: should look like a torus or limit cycle
# - Above η_c: should look like a strange attractor
# - Compute correlation dimension to quantify

# This is exploratory but would be visually compelling and
# theoretically meaningful if toroidal structure is visible.
```

### Experiment 3D: Fractal Dimension of Training Trajectory

```python
# In function space, record the network output f_θ(X_eval) at each step
# This gives a trajectory in R^(N_eval × 10)
# Compute the correlation dimension using Grassberger-Procaccia algorithm

# Expected:
# - Below η_c: low integer dimension (consistent with torus)
# - Above η_c: non-integer dimension (consistent with strange attractor)
# - Increasing dimension with learning rate (more complex attractor)
```

### Deliverables from Phase 3
- [ ] Figure: Bifurcation diagram (loss vs. learning rate)
- [ ] Figure: Power spectra at selected learning rates
- [ ] Figure: Phase space reconstructions (2D or 3D projections)
- [ ] Table: Estimated attractor dimensions at different learning rates

---

## Code Architecture

### Recommended Structure

```
chaos_onset/
├── config.py              # All hyperparameters, architecture specs
├── data.py                # Synthetic data generation (deterministic)
├── model.py               # MLP definition (supports tanh/ReLU, variable depth)
├── train.py               # Training loop with Lyapunov computation
├── lyapunov.py            # Function-space Lyapunov exponent estimation
├── analysis/
│   ├── transition_zone.py # Phase 1A: fine-grained η_c estimation
│   ├── broad_sweep.py     # Phase 1B: full Lyapunov curve
│   ├── sensitivity.py     # Phase 1C: perturbation sensitivity
│   ├── bifurcation.py     # Phase 3A: bifurcation diagram
│   ├── power_spectrum.py  # Phase 3B: FFT analysis
│   └── phase_space.py     # Phase 3C: Takens embedding
├── plotting/
│   ├── figures.py         # Publication-quality figures
│   └── style.py           # Matplotlib style settings
└── run_all.py             # Master script for full experiment suite
```

### Key Implementation Details

**Data generation must be deterministic and separate from model seeding:**

```python
def generate_data(data_seed=42, n_samples=2000, n_classes=10, 
                  n_random_features=200, n_quadratic_features=20):
    rng = np.random.RandomState(data_seed)
    # ... generate class centers, sample points, add quadratic features
    # This function must produce identical output every time
    return X, y
```

**Lyapunov computation must be careful about numerical precision:**

```python
def compute_function_lyapunov(model, perturbed_model, X_eval, n_steps, lr):
    """
    Train both models identically, measuring function-space divergence.
    
    Returns estimated Lyapunov exponent (slope of log-divergence vs. time).
    """
    distances = []
    for t in range(n_steps):
        # Forward pass on evaluation set
        f1 = model(X_eval)
        f2 = perturbed_model(X_eval)
        
        # Function-space distance
        d = torch.norm(f1 - f2).item()
        distances.append(d)
        
        # Identical training step for both
        loss1 = criterion(model(X_train), y_train)
        loss2 = criterion(perturbed_model(X_train), y_train)
        
        loss1.backward()
        loss2.backward()
        
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
            for p in perturbed_model.parameters():
                p -= lr * p.grad
            model.zero_grad()
            perturbed_model.zero_grad()
    
    # Fit exponential to distance time series
    log_d = np.log(np.array(distances) + 1e-30)  # avoid log(0)
    # Use robust fitting (ignore early transient, late saturation)
    # Fit over middle 60% of the time series
    start = len(log_d) // 5
    end = 4 * len(log_d) // 5
    t_range = np.arange(start, end)
    slope, _, _, _, _ = scipy.stats.linregress(t_range, log_d[start:end])
    
    return slope  # This is the Lyapunov exponent
```

**Sharpness computation:**

```python
def compute_sharpness(model, X, y, criterion, n_iter=50):
    """Largest eigenvalue of the Hessian via power iteration (Lanczos)."""
    # Use torch.autograd.functional.hvp for Hessian-vector products
    # Power iteration: v → Hv / ‖Hv‖, converges to top eigenvector
    # λ_max = v^T H v after convergence
    pass
```

---

## Compute Estimates

| Experiment | Runs | Est. Time (single GPU) |
|-----------|------|----------------------|
| 1A: Transition zone (20 seeds × 40 LRs) | 1,600 | 2–4 hours |
| 1B: Broad sweep (20 seeds × 25 LRs) | 1,000 | 1–3 hours |
| 1C: Sensitivity (6 ε × 6 LRs × 5 seeds) | 360 | 30–60 min |
| 2A: ReLU MLP (10 seeds × 40 LRs) | 800 | 1–2 hours |
| 2B: Deeper nets (10 seeds × 40 LRs × 2 depths) | 1,600 | 3–6 hours |
| 2C: CIFAR-10 CNN (10 seeds × 40 LRs) | 800 | 4–8 hours |
| 3A: Bifurcation (5 seeds × 200 LRs) | 1,000 | 2–4 hours |
| 3B: Power spectrum (7 LRs × 5 seeds, long runs) | 70 | 1–2 hours |
| 3C–D: Phase space (selected LRs) | ~50 | 1–2 hours |
| **Total** | **~7,300** | **~15–30 hours** |

All estimates assume the small MLP from the original experiment. The CNN on CIFAR-10 will be slower per run but the architecture is still small.

---

## Reporting Standards

For each key result, report:

- **η_c:** Mean ± standard deviation, 95% confidence interval, and individual per-seed values
- **Ratio η_c/(2/λ_max):** With propagated uncertainty
- **Statistical tests:** One-sample t-test (η_c < 2/λ_max), bootstrapped confidence intervals
- **Effect sizes:** Not just p-values
- **All hyperparameters:** Exact values, not ranges
- **Code availability:** Full code in supplementary or public repository

### Figure Standards

- All error bands should show ±1 std (shaded) AND individual seed traces (semi-transparent lines)
- Mark the EoS threshold (2/λ_max) and critical learning rate (η_c) on every relevant plot
- Use colorblind-friendly palettes
- Include sample size (n = X seeds) in every caption

---

## Pre-Submission Checklist

### Phase 1 (minimum for preprint)
- [ ] 20-seed transition zone with tight η_c estimate
- [ ] 20-seed broad sweep
- [ ] Perturbation sensitivity analysis
- [ ] All figures regenerated with publication-quality formatting
- [ ] Paper draft updated with new results

### Phase 2 (minimum for conference/journal submission)
- [ ] ReLU MLP results
- [ ] Deeper network results
- [ ] CIFAR-10 CNN results (or at minimum, CIFAR-10 MLP)
- [ ] Cross-architecture comparison table and figure
- [ ] Discussion updated with generalization evidence

### Phase 3 (recommended for strong journal submission)
- [ ] Bifurcation diagram
- [ ] Power spectrum analysis
- [ ] At least one phase-space reconstruction figure
- [ ] Discussion of toroidal vs. strange attractor interpretation backed by data

---

## Target Venues

### Option A: ML Theory Workshop (fastest to publication)
- **HiLD (High-dimensional Learning Dynamics) at ICML** — directly relevant, accepts dynamical systems perspectives on training
- **SciForDL (Science for Deep Learning) at NeurIPS** — theoretical foundations
- **Deadline advantage:** Workshop papers are shorter (4-6 pages), Phase 1 may suffice

### Option B: ML Conference (higher impact)
- **ICML or NeurIPS** main conference — requires Phase 1 + Phase 2, strong framing
- **ICLR** — where Cohen et al. (2021) and Damian et al. (2023) were published

### Option C: Interdisciplinary Journal (best for torus framework connection)
- **Physical Review Letters** — precedent: Züchner et al. (2024) published Lyapunov exponents of DNNs here
- **Chaos** (AIP) — directly about nonlinear dynamics
- **Frontiers in Complex Systems** — precedent: Morales et al. (2024) published here

### Option D: Complexity Science (if combined with core theory paper)
- **Complexity** (Wiley) — good home for the cross-disciplinary framing
- **Entropy** (MDPI) — information-theoretic and dynamical systems perspectives
