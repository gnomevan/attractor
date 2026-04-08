# Chaos Onset in Gradient Descent — Experimental Findings Report

**Date:** March 31, 2026
**Version:** 5.0 — Phase 1, 2, and 3 complete
**Status:** Core result confirmed. Paper drafts need rewriting.

---

## 1. Summary

Training dynamics of neural networks undergo a transition from one-dimensional convergence to multi-dimensional chaos as architecture complexity and learning rate increase. The transition is quantified by the correlation dimension D₂ of the function-space trajectory:

| Architecture | Params | D₂ | PC1 | Key observation |
|---|---|---|---|---|
| MLP (2 hidden, tanh) | 14K | 0.9 | 96% | One-dimensional throughout |
| MLP (4 hidden, tanh) | 19K | 0.9 | 86% | Wider chaos window, still 1D |
| ReLU MLP (4 hidden, wide) | 167K | 0.97 | 90% | Approaching threshold |
| **CNN on CIFAR-10** | **269K** | **3.6** | **52%** | **Multi-dimensional chaos confirmed** |

The CIFAR-10 CNN at 30% of the Edge of Stability threshold produces training dynamics with D₂ ≈ 3.6 — a genuinely multi-dimensional attractor in function space. This is the regime where toroidal and strange attractor geometry become applicable descriptions of gradient descent.

---

## 2. Experimental setup

All experiments use full-batch gradient descent, MSE loss, no momentum, no weight decay. Lyapunov exponents measured in function space with ε = 10⁻⁵ (validated by sensitivity analysis; ε = 10⁻⁸ produces artifacts). Perturbation along unit-norm random direction, seeded at init_seed + 999999.

**MLP experiments:** Synthetic data, 2,000 samples, 10 classes, 220 features (200 Gaussian + 20 quadratic). 5,000 training steps.

**CNN experiment:** CIFAR-10, 2,000-image subset. Architecture: Conv(3→16, 3×3) → ReLU → MaxPool → Conv(16→32, 3×3) → ReLU → MaxPool → FC(2048→128) → ReLU → FC(128→10). 268,650 parameters. 5,000 training steps. Learning rates expressed as fractions of 2/λ_max (EoS threshold).

---

## 3. Phase 1: Lyapunov characterization (MLP, 14K params)

### 3.1 Methodological correction

The perturbation sensitivity sweep revealed that ε = 10⁻⁸ produces uniform λ ≈ +0.00004 across all learning rates — a numerical artifact, not genuine dynamics. Converged measurements require ε ≥ 10⁻⁶. All results use ε = 10⁻⁵.

**Implication for the field:** Prior work computing Lyapunov exponents of neural network training at ε = 10⁻⁸ should be interpreted with caution.

### 3.2 The chaos window (20 seeds × 50 LRs)

The mean Lyapunov exponent is non-monotonic with learning rate:
- Mean λ slightly positive for η < 0.048 (60–80% of seeds chaotic)
- Mean λ strongly negative above η ≈ 0.05 (basin convergence dominates)
- Peak chaos at η ≈ 0.035 (14/20 seeds positive)
- Deepest contraction at η ≈ 0.12

The system is near-critical at all tested learning rates. No sharp η_c — the fraction of chaotic seeds declines smoothly from ~75% to ~0%, consistent with KAM-type fractal interleaving of ordered and chaotic orbits.

### 3.3 Seed comparison and Lyapunov vectors

All seeds converge to functionally identical solutions. Correlations increase with η (0.987 → 0.9999). The chaos is in the journey, not the destination.

The Lyapunov vector direction wanders isotropically — no period-doubling signature. Consistent with multi-directional KAM torus destruction rather than Feigenbaum cascade.

---

## 4. Phase 2: Dynamical characterization (MLP, 14K params)

### 4.1 Attractor dimension: D₂ ≈ 0.9 everywhere

Correlation dimension stays below 1.0 at all learning rates, all embedding dimensions. PC1 captures 96–99% of variance. The MLP training dynamics are fundamentally one-dimensional — a convergence path, not a torus or strange attractor.

### 4.2 Power spectra: the sharpness is the right observable

Loss and gradient norm spectra show featureless 1/f^α power laws (α = 2.2–3.7) at all learning rates. No discrete peaks, negligible spectral entropy variation.

The Hessian sharpness spectrum shows a measurable transition: spectral flatness drops from 0.83 (structured, η = 0.005) to 0.30 (broadband, η = 0.030) in the 2-layer baseline. The transition occurs at the chaos window boundary. This is the only spectral evidence of a qualitative dynamical change in the MLP.

### 4.3 Bifurcation diagram: no period-doubling

Both loss and gradient norm decrease smoothly and monotonically with η. No Feigenbaum cascade, no period-2 or period-4 windows. Consistent with isotropic chaos.

---

## 5. Phase 3: Architecture scaling

### 5.1 Depth scaling (tanh MLP, h=50)

**The chaos window widens with depth, exactly as Ruelle-Takens predicts.**

| Depth | Params | Window width | Peak λ | Max PC2 |
|---|---|---|---|---|
| 2 (baseline) | 14K | 0.038 | +0.000135 | 5.2% |
| 3 | 17K | 0.046 | +0.000236 | 8.7% |
| 4 | 19K | 0.145 | +0.000312 | 12.6% |
| 5 | 22K | 0.137 | +0.000236 | 12.9% |

At depth=4 and depth=5, chaos persists across the entire tested learning rate range — the "basin wins" regime from the 2-layer MLP disappears. Off-axis dynamics grow to 13% of variance, but D₂ remains below 1.

The sharpness spectral transition also shifts: at depth=2, flatness drops sharply from 0.83 to 0.30 between η = 0.005 and η = 0.030. At depth=4, flatness is already 0.80→0.72 — the dynamics are born broadband. Deeper networks start chaotic.

### 5.2 ReLU comparison

| Architecture | Params | Peak η | Peak λ | Window | PC2 range |
|---|---|---|---|---|---|
| tanh h=50 d=2 | 14K | 0.020 | +0.000135 | 0.005–0.043 | 1.7–5.2% |
| ReLU h=50 d=2 | 14K | 0.119 | +0.000814 | 0.005–0.15+ | 4.9–6.9% |
| ReLU h=200 d=2 | 86K | 0.043 | +0.000560 | 0.005–0.15+ | 2.6–4.1% |
| ReLU h=200 d=4 | 167K | 0.142 | +0.000718 | 0.005–0.15+ | 2.9–8.0% |

ReLU shifts chaos to higher learning rates (peak at η = 0.12 vs η = 0.02 for tanh) and produces chaos that never closes within the tested range. The piecewise-linear geometry of ReLU sustains chaos more broadly than tanh's smooth saturation. All ReLU architectures maintain PC2 in the 3–8% range — higher baseline off-axis dynamics than tanh — but D₂ stays below 1.

### 5.3 CIFAR-10 CNN: the breakthrough

**D₂ crosses 1 and reaches 3.6.** Multi-dimensional chaos confirmed.

| Fraction of EoS | η | Mean λ | D₂ | PC1% | PC2% |
|---|---|---|---|---|---|
| 5% | 0.015 | +0.000233 | 0.98 | 94.8 | 4.6 |
| 10% | 0.030 | +0.000464 | 1.02 | 84.7 | 8.4 |
| 15% | 0.044 | +0.001866 | 1.19 | 57.5 | 10.6 |
| 20% | 0.059 | +0.001418 | 2.06 | 51.8 | 9.1 |
| 30% | 0.089 | +0.000320 | 3.64 | 52.0 | 7.8 |
| 40% | 0.118 | +0.000138 | 3.47 | 53.4 | 7.3 |
| 50% | 0.148 | −0.000006 | 3.07 | 52.8 | 7.5 |
| 60% | 0.178 | −0.000093 | 2.73 | 53.0 | 7.5 |
| 70% | 0.207 | −0.000192 | 2.39 | 52.2 | 7.6 |
| 80% | 0.237 | −0.000256 | 2.14 | 51.2 | 7.9 |
| 90% | 0.266 | −0.000342 | 1.97 | 50.1 | 8.5 |
| 95% | 0.281 | −0.000359 | 1.91 | 50.2 | 8.4 |

**Four findings from the CNN experiment:**

**1. The Lyapunov curve has the same shape — chaos window then basin convergence.** Peak at 15% EoS, crossing zero at ~45% EoS. The window extends much further toward EoS (45% vs 8% in the MLP), but the qualitative shape is identical. The basin-convergence mechanism is universal; the chaos window width scales with architecture complexity.

**2. D₂ peaks at 3.64 (30% EoS) and remains above 2.0 out to 80% EoS.** The function-space trajectory is not a line — it's a three-to-four dimensional object. Even at 95% EoS (deeply in the contraction regime), D₂ = 1.91, still well above 1. The multi-dimensionality persists even where the Lyapunov exponent is negative.

**3. PC1 drops to ~52% above 20% EoS.** Half the trajectory variance is off the primary convergence axis. This is qualitatively different from the MLP's 96–99%. The CNN trajectory genuinely explores a multi-dimensional function space during training.

**4. The sharpness dynamics reveal the mechanism.** At 5% EoS: sharpness climbs from 4.5 to 30.6 over training — progressive sharpening, monotonic, no oscillation. At 95% EoS: sharpness stays flat between 7.0 and 8.9 — self-stabilization, the hallmark of Edge of Stability. The D₂ transition from ~1 to ~3.6 corresponds exactly to the shift from progressive sharpening (one mode, one direction) to oscillatory self-stabilization (multiple coupled modes, multiple directions).

**Why D₂ > 1 in the CNN but not the MLP:** The MLP on synthetic data converges along a single direction because the loss landscape is simple — one basin, one path. The CNN on CIFAR-10 faces a complex landscape with convolutional weight-sharing constraints, real image structure, and 269K parameters interacting through nonlinear feature hierarchies. The Edge of Stability oscillations in sharpness represent at least one additional oscillatory mode coupling to the convergence dynamics. The presence of D₂ ≈ 3.6 implies three to four coupled modes — consistent with the Ruelle-Takens prediction that three or more coupled frequencies can produce strange attractor behavior.

---

## 6. The complete picture

### The scaling story

The transition from D₂ < 1 to D₂ > 1 depends on three factors, each of which we tested independently:

**Depth** (more nested nonlinear maps): Widens the chaos window from 0.038 to 0.145 in LR range. Increases off-axis dynamics from 5% to 13%. Confirms the Ruelle-Takens prediction: more nested maps → more robust chaos. But insufficient alone for D₂ > 1 at this parameter count.

**Width and activation** (more degrees of freedom): Shifts the chaos window to higher learning rates and raises the floor of off-axis dynamics. ReLU produces more persistent chaos than tanh. But insufficient alone for D₂ > 1.

**Task complexity + architecture** (the missing ingredient): The CIFAR-10 CNN combines convolutional structure, real data complexity, and 269K parameters. At learning rates between 15% and 80% of EoS, the trajectory fills a multi-dimensional space (D₂ = 2–3.6). The sharpness oscillations at near-EoS learning rates provide the coupled oscillatory modes that the MLP on synthetic data lacks.

### Connection to the torus framework

The torus framework predicts that nested periodic processes generate toroidal geometry, and that perturbation of toroidal systems produces fractal (strange attractor) geometry through the Ruelle-Takens route. The experimental findings map onto this prediction:

**D₂ ≈ 1 (MLP regime):** One effective degree of freedom. The training trajectory is a convergence path — analogous to a limit cycle (1-torus). Chaos is present (positive Lyapunov exponents) but one-dimensional.

**D₂ ≈ 2–3.6 (CNN regime):** Multiple coupled oscillatory modes. The function-space trajectory fills a multi-dimensional space. D₂ ≈ 2 at the edges of the chaos window corresponds to the dimensionality of a 2-torus surface. D₂ ≈ 3.6 at peak (non-integer, between 3 and 4) is consistent with a strange attractor — a fractalized torus with dimension higher than 3 but not filling 4-dimensional space.

**The sharpness dynamics provide the mechanism.** At low learning rates: one mode (progressive sharpening), one dimension (D₂ ≈ 1). As learning rate approaches EoS: sharpness oscillations couple to the convergence dynamics, adding independent frequencies. At 30% EoS: three to four effective degrees of freedom (D₂ ≈ 3.6). This is the Ruelle-Takens route playing out in gradient descent: limit cycle → torus → strange attractor, driven by the coupling of sharpness oscillations to training convergence.

### What the experiments support and what they don't

**Supported:**
- Multi-dimensional chaotic dynamics exist in CNN training (D₂ = 3.6)
- The dimensionality increases are consistent with the Ruelle-Takens route (D₂ increases through 1, 2, 3+)
- The chaos window is bounded by basin convergence at high η (universal across architectures)
- Depth widens the chaos window (Ruelle-Takens prediction confirmed)
- The Hessian sharpness is the right observable for detecting the transition

**Not yet demonstrated:**
- Direct detection of toroidal topology (persistent homology on the CNN trajectory)
- Clean discrete spectral peaks corresponding to torus frequencies
- Identification of specific coupled oscillatory modes by frequency
- Whether D₂ continues to increase with larger architectures (ResNets, transformers)

---

## 7. Open questions and next experiments

### Immediate priorities
- **Persistent homology of CNN trajectory:** Apply the Takens-embedding-plus-TDA pipeline (validated in the companion paper's Experiment 2) to the CNN function-space trajectory at 20–40% EoS. Does the point cloud have toroidal topology (two H₁ generators)?
- **Spectral analysis of CNN sharpness:** The MLP sharpness showed a structured→broadband transition. Does the CNN sharpness show discrete peaks at specific learning rates (evidence of quasiperiodic motion on a torus)?
- **Larger architectures:** ResNet-18 on CIFAR-10, transformer on language data. Does D₂ continue to grow?

### Theoretical questions
- Why does D₂ peak at 30% EoS and not at the chaos peak (15% EoS)? The peak Lyapunov exponent and peak D₂ are at different learning rates — the attractor is most complex where the chaos is weakening, not where it's strongest.
- What sets the maximum D₂? Is 3.6 a property of this architecture, or does it increase with network size?
- Can the specific coupled modes be identified? The sharpness oscillation is one; what are the others?

---

## 8. Literature

| Paper | Relevance |
|---|---|
| Cohen et al. (2021), ICLR | Edge of Stability phenomenon |
| Damian, Nichani & Lee (2023), ICLR | Self-stabilization at EoS |
| Kalra, He & Barkeshli (2023) | Period-doubling route at EoS |
| arXiv:2502.20531 (2025) | Period-doubling in deep linear networks |
| Morales et al. (2024), Frontiers | Lyapunov exponents during training |
| Züchner et al. (2024), PRL 132 | Finite-time Lyapunov exponents of DNNs |
| Ruelle & Takens (1971) | Torus-to-chaos transition |
| KAM: Kolmogorov, Arnold, Moser | KAM theory |

---

## 9. Paper strategy

### The paper's story (revised)

The paper now has a complete arc: from methodological correction (ε sensitivity) through one-dimensional chaos (MLP) to multi-dimensional chaos (CNN) to the Ruelle-Takens interpretation.

**Title candidates:**
- "Multi-Dimensional Chaos in Neural Network Training: From Edge of Stability to Strange Attractors"
- "The Chaos Window in Gradient Descent: Dimension, Depth, and the Ruelle-Takens Route"
- "D₂ = 3.6: Evidence for Strange Attractor Dynamics in Convolutional Network Training"

**Structure:**
1. Introduction: gradient descent as a dynamical system
2. Methodological correction: ε sensitivity, sharpness as the right observable
3. The chaos window: non-monotonic Lyapunov curve, basin convergence
4. Scaling: depth widens the window, width shifts it, task complexity enables multi-dimensional dynamics
5. The CIFAR-10 result: D₂ = 3.6, sharpness oscillations as coupled modes
6. Connection to Ruelle-Takens: the route from limit cycle to strange attractor in gradient descent
7. Open directions: persistent homology, larger architectures, transformer training dynamics

**Target venue:** Physical Review Letters or Chaos (the dynamical systems story is stronger than the ML story at this point). Secondary submission to ICML/NeurIPS if framed for the ML audience.
