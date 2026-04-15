# Chaos Onset in Gradient Descent — Experimental Findings Report

**Date:** March 31, 2026
**Version:** 4.0 — Phase 1 complete, Phase 2 complete, Phase 3 designed
**Status:** Phase 3 (architecture scaling) ready to run

---

## 1. Summary of findings

Five key results from two phases of experiments:

1. **Chaos is non-monotonic.** The mean Lyapunov exponent peaks in a window (η ≈ 0.01–0.05), then goes strongly negative as basin convergence overwhelms trajectory divergence. Higher learning rates are *less* chaotic in function space, not more.

2. **The system is near-critical throughout.** At every tested learning rate, 20–80% of seeds show positive Lyapunov exponents. There is no sharp onset — initialization determines which side of the order/chaos boundary each trajectory falls on, consistent with KAM theory.

3. **The dynamics are fundamentally one-dimensional.** Despite positive Lyapunov exponents, the attractor dimension is D₂ ≈ 0.9 everywhere. PC1 captures 96–99% of variance. The training trajectory is a convergence path, not a torus or strange attractor.

4. **The spectral transition lives in the sharpness.** Loss and gradient norm spectra show featureless power laws at all learning rates. The Hessian sharpness spectrum transitions from structured (flatness ≈ 0.87) to broadband (flatness ≈ 0.96) at η ≈ 0.02 — the only spectral evidence of a qualitative dynamical change.

5. **ε = 10⁻⁸ was in the noise floor.** Prior measurements at this perturbation scale were numerical artifacts. Converged measurements require ε ≥ 10⁻⁶.

---

## 2. Experimental setup

**Architecture:** MLP: Input(220) → Linear(220,50) → Tanh → Linear(50,50) → Tanh → Linear(50,10). 14,110 parameters.

**Data:** 2,000 synthetic samples, 10 classes, 220 features (200 Gaussian + 20 quadratic). Deterministic (seed 42).

**Training:** Full-batch gradient descent, MSE loss, no momentum, no weight decay.

**Perturbation:** ε = 10⁻⁵ (corrected from 10⁻⁸). Unit-norm random direction, seeded at init_seed + 999999.

**Sharpness:** λ_max ≈ 3.18 (post-warmup). EoS threshold: 2/λ_max ≈ 0.628.

---

## 3. Phase 1 results: Lyapunov characterization

### 3.1 Perturbation sensitivity (Experiment 4)

7 ε values × 5 LRs × 3 seeds. At ε ≤ 10⁻⁸, all LRs return uniform ~+0.00004 (numerical artifact). At ε ≥ 10⁻⁶, exponents plateau and differentiate across LRs. Default corrected to ε = 10⁻⁵.

### 3.2 Transition zone (Experiment 5): 20 seeds × 50 LRs

The mean Lyapunov curve shows three regimes:

| Regime | η range | Mean λ | Fraction chaotic | Character |
|---|---|---|---|---|
| Marginal chaos | 0.005–0.048 | +0.0001 to +0.0002 | 60–80% | Near-critical |
| Reconvergence | 0.048–0.120 | −0.0003 to −0.0006 | 5–35% | Basin dominates |
| Deep contraction | >0.120 | −0.0006 plateau | 0–5% | Complete basin control |

Peak chaos: η ≈ 0.035, λ ≈ +0.00016 (14/20 seeds positive, p = 0.096).

### 3.3 Broad sweep (Experiment 6): 20 seeds × 40 LRs (η = 0.01–0.42)

Confirms three-regime picture. Trough at η ≈ 0.13 (λ ≈ −0.0006). Plateau at η > 0.15 (λ ≈ −0.0004). No second chaos window — outlier seeds (7/20) spike positive at scattered high LRs but not systematically. EoS (0.628) entirely in contraction regime.

### 3.4 Seed comparison (Experiment 2) and Lyapunov vectors (Experiment 3)

Experiments at ε = 10⁻⁸ — Lyapunov magnitudes unreliable, but relative findings hold:

- All seeds converge to same function. Correlations increase with η (0.987 → 0.9999).
- Chaos is isotropic (KAM-type), not period-doubling. Directional persistence dissolves with η.

---

## 4. Phase 2 results: Dynamical characterization

### 4.1 Power spectrum (Experiment C): loss, gradient norm, and sharpness

**Loss and gradient norm:** Featureless 1/f^α power laws at all learning rates. No discrete peaks. Spectral entropy varies by <1%. The convergence trend dominates both observables, even after log-transformation and linear detrending.

| Observable | Spectral slope range | Entropy range | Verdict |
|---|---|---|---|
| Loss | −3.2 to −2.2 | 0.037–0.038 | No oscillatory structure |
| Gradient norm | −3.7 to −2.8 | 0.037–0.038 | No oscillatory structure |

**Sharpness (top Hessian eigenvalue):** Shows a measurable spectral transition:

| η range | Spectral flatness | Low-freq power | Interpretation |
|---|---|---|---|
| 0.005–0.010 | 0.87 | 31% | Structured, concentrated |
| ≥ 0.020 | 0.94–0.96 | 13–18% | Broadband, near-uniform |

Spectral entropy jumps from 0.94 to 0.99 between η = 0.010 and η = 0.020. This is the only spectral evidence of a qualitative change in the training dynamics, and it occurs at the boundary of the chaos window.

**Conclusion:** The dynamical transition is visible in the Hessian curvature but not in first-order quantities (loss, gradient). The sharpness is the right observable; loss and gradient norm are dominated by the convergence envelope.

### 4.2 Bifurcation diagram (Experiment E): 5 seeds × 200 LRs

**No period-doubling.** Both loss and gradient norm decrease smoothly and monotonically with η. Oscillation amplitude also decreases monotonically:

| η | Loss oscillation | Grad norm oscillation |
|---|---|---|
| 0.005 | 1.5 × 10⁻⁵ | 6.2 × 10⁻⁵ |
| 0.020 | 3.5 × 10⁻⁶ | 2.1 × 10⁻⁵ |
| 0.050 | 1.2 × 10⁻⁶ | 7.4 × 10⁻⁶ |
| 0.100 | 5.5 × 10⁻⁷ | 3.4 × 10⁻⁶ |
| 0.200 | 2.7 × 10⁻⁷ | 1.7 × 10⁻⁶ |

No Feigenbaum cascade, no period-2 or period-4 windows. Consistent with Experiment 3's finding of isotropic (not period-doubling) chaos.

### 4.3 Takens embedding (Experiment D): gradient norm, 5 seeds × 8 LRs

**All embeddings are lines.** At every learning rate and every embedding dimension (d = 3, 5, 7, 9), the delay-embedded gradient norm signal is one-dimensional. Correlation dimension D₂ ≈ 0.9–1.0 throughout. No toroidal or strange attractor geometry visible.

| η | D₂ (d=3) | D₂ (d=9) | τ |
|---|---|---|---|
| 0.005 | 0.92 | 1.01 | 125 |
| 0.020 | 0.91 | 1.01 | 125 |
| 0.080 | 0.90 | 0.98 | 125 |
| 0.200 | 0.90 | 0.99 | 125 |

### 4.4 Function-space trajectory dimension (Experiment F): 5 seeds × 8 LRs

**One-dimensional at all learning rates.** Correlation dimension D₂ ≈ 0.9 throughout. The interesting signal is in the PCA:

| η | D₂ | PC1 (%) | PC2 (%) |
|---|---|---|---|
| 0.005 | 0.93 | 96.5 | 3.2 |
| 0.010 | 0.90 | 96.6 | 3.0 |
| 0.020 | 0.89 | 97.7 | 2.1 |
| 0.050 | 0.90 | 99.0 | 0.9 |
| 0.080 | 0.90 | 99.6 | 0.4 |
| 0.200 | 0.91 | 99.9 | 0.1 |

PC1 captures 96.5% at η = 0.005 and 99.9% at η = 0.200. The residual variance (PC2) decreases from 3.2% to 0.1% — the transverse dynamics shrink as the basin tightens. The chaos window corresponds to the learning rates where the off-axis dynamics are largest, even though they never push the attractor dimension above 1.

---

## 5. Synthesis

### What the experiments show

The 14,110-parameter MLP on this synthetic task has training dynamics that are:

- **Near-critical but one-dimensional.** Positive Lyapunov exponents coexist with D₂ ≈ 0.9. This is stretching and folding along a single convergence direction — analogous to the logistic map, which is one-dimensional but chaotic.

- **Bounded by a single deep basin.** Higher learning rates compress the dynamics further into one dimension (PC2 drops from 3.2% to 0.1%), producing tighter functional convergence across seeds.

- **Spectrally featureless in first-order quantities.** The dynamical transition from structured to broadband is visible only in the Hessian sharpness, not in loss or gradient norm.

### What the experiments don't show

- No toroidal geometry in any observable at any learning rate.
- No period-doubling or bifurcation structure.
- No attractor dimension above 1.
- No discrete spectral peaks in loss or gradient dynamics.

### Honest assessment for the torus framework

This architecture is too simple for the torus-to-chaos transition to manifest geometrically. The training dynamics have one effective degree of freedom — convergence along a single direction. Multi-dimensional oscillatory structure (the prerequisite for toroidal geometry) requires more complex dynamics than this network produces.

This is consistent with the companion paper's Experiment 5 (83-parameter network, D ≈ 1) and suggests a **scaling threshold**: toroidal training dynamics likely require architectures where the EoS oscillations documented by Cohen et al. (2021) involve multiple coupled oscillatory modes. This means deeper networks, wider layers, or convolutional architectures on more complex tasks.

The Phase 1 findings (non-monotonic Lyapunov curve, near-criticality, basin-dominated convergence) remain novel and publishable. The Phase 2 findings constrain the interpretation: the chaos is real but one-dimensional, and the torus interpretation requires larger models to test.

---

## 6. What's still open

### Phase 3: Architecture scaling (designed, ready to run)
- **Wider networks** (hidden=100, 200, 400): Does D₂ increase above 1?
- **Deeper networks** (3, 4, 5 layers): Does the chaos window widen?
- **ReLU activation**: Does the transition shift?
- **Critical test**: Does PC2 grow large enough at any architecture to produce D₂ > 1?

### Phase 1 extensions
- Extend transition zone below η = 0.005 to find where fraction-chaotic drops to zero
- Sharpness-based Lyapunov exponent (perturb and track sharpness divergence, not function output)

### Longer-term
- CNN on CIFAR-10 (real data, real architecture)
- Transformer on language modeling (the architecture where EoS is best documented)

---

## 7. Literature

| Paper | Relevance |
|---|---|
| Cohen et al. (2021), ICLR | Edge of Stability |
| Damian, Nichani & Lee (2023), ICLR | Self-stabilization at EoS |
| Kalra, He & Barkeshli (2023), arXiv:2311.02076 | Period-doubling route at EoS |
| arXiv:2502.20531 (2025) | Period-doubling in deep linear networks |
| Morales et al. (2024), Frontiers | Lyapunov exponents during training |
| Züchner et al. (2024), PRL 132 | Finite-time Lyapunov exponents of DNNs |
| Ruelle & Takens (1971) | Torus-to-chaos transition theory |
| KAM: Kolmogorov (1954), Arnold (1963), Moser (1962) | KAM theory |

---

## 8. Paper strategy — revised after Phase 2

### Central contributions

1. **Methodological:** ε = 10⁻⁸ is in the noise floor. Lyapunov measurements of neural network training require ε ≥ 10⁻⁶ for convergence. This affects interpretation of prior work.

2. **The chaos window:** Non-monotonic Lyapunov curve — chaos peaks at low η and gives way to basin-dominated convergence. Novel finding, no prior report in EoS literature.

3. **Near-criticality:** KAM-type interleaving of chaotic and ordered seeds at every learning rate. The transition is gradual, not sharp.

4. **Observable hierarchy:** The dynamical transition is visible in sharpness but not in loss or gradient norm. This tells the field which quantity to measure.

5. **Dimensionality constraint:** The chaos is one-dimensional at this architecture scale. This sets up the scaling question: how large must a network be for multi-dimensional chaotic structure to emerge?

### The torus claim is deferred, not abandoned

The paper should not claim toroidal structure in training dynamics — Phase 2 ruled that out for this architecture. Instead, the paper establishes the *prerequisites* for testing the torus framework in gradient descent: the chaos is real, the transition is measurable, the right observable is identified, and the next step (architecture scaling) is specified. The torus test requires finding an architecture where D₂ > 1.

### Recommended framing

"We characterize the chaos window in gradient descent training of small MLPs. Training dynamics exhibit positive Lyapunov exponents in a bounded learning rate range, with basin-dominated convergence above. The chaos is one-dimensional (D₂ ≈ 0.9) and spectrally featureless in loss and gradient observables, but a spectral transition is detectable in the Hessian sharpness. We identify the perturbation scale (ε ≥ 10⁻⁶) required for reliable Lyapunov measurements and show that the common ε = 10⁻⁸ produces artifacts. Architecture scaling experiments to determine the threshold for multi-dimensional chaotic structure are underway."
