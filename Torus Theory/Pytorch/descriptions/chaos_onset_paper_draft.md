# Chaos Onset in Gradient Descent: Lyapunov Evidence for a Torus-to-Strange-Attractor Transition Below the Edge of Stability

## Abstract

The Edge of Stability (EoS) phenomenon, in which the largest Hessian eigenvalue during gradient descent training stabilizes near 2/η, has reshaped understanding of neural network optimization (Cohen et al., 2021). Recent theoretical work has identified period-doubling routes to chaos as learning rate increases beyond this threshold (Kalra et al., 2023). We present empirical evidence that the transition to chaotic dynamics—measured via function-space Lyapunov exponents—occurs at a critical learning rate η_c far below the EoS threshold. In a two-hidden-layer tanh MLP trained on structured synthetic data, function-space Lyapunov exponents cross from negative (convergent) to positive (divergent) at η_c ≈ 0.02, approximately 6.6% of the classical instability boundary 2/λ_max ≈ 0.27. This finding suggests that sensitive dependence on initial conditions—a hallmark of chaotic dynamics—emerges well before the loss landscape's curvature reaches its stability limit. We frame this result within the broader theory of torus-to-chaos transitions in dynamical systems, connecting the Ruelle-Takens route, KAM theory, and the specific phenomenology of gradient descent. The pre-EoS chaos regime may correspond to a transition from quasiperiodic to strange-attractor dynamics in parameter-update space, with implications for training reproducibility, hyperparameter selection, and the dynamical systems interpretation of deep learning.

**Keywords:** edge of stability, Lyapunov exponents, chaos, gradient descent, dynamical systems, torus, strange attractor, neural network training dynamics

---

## 1. Introduction

Training a neural network by gradient descent is, at its core, an iterated dynamical system. The parameter vector θ ∈ ℝ^d evolves under a deterministic map θ_{t+1} = θ_t − η∇L(θ_t), and the trajectory through parameter space is shaped by the geometry of the loss landscape. Recent work has revealed that this dynamical system exhibits behavior far more complex than simple convergence to a minimum.

Cohen et al. (2021) documented the Edge of Stability (EoS) phenomenon: during full-batch gradient descent, the largest eigenvalue of the Hessian (the "sharpness" S(θ) = λ_max(∇²L(θ))) steadily increases until it reaches the classical instability threshold 2/η, then hovers there while the loss continues to decrease non-monotonically. This challenged the standard optimization-theoretic picture, in which training is "stable" precisely when S(θ) < 2/η.

Subsequent work has deepened the picture. Damian, Nichani, and Lee (2023) showed that a cubic Taylor expansion captures the self-stabilization mechanism at the EoS—a negative feedback loop that prevents divergence. Kalra, He, and Barkeshli (2023) identified a period-doubling route to chaos on the EoS manifold as learning rate increases, directly echoing the Feigenbaum scenario from classical nonlinear dynamics. Most recently, analyses of deep linear networks beyond the EoS have confirmed period-doubling cascades with loss oscillations confined to learning-rate-dependent subspaces (arXiv:2502.20531, 2025).

Meanwhile, Morales et al. (2024) pioneered the direct measurement of Lyapunov exponents along training trajectories, finding positive exponents in the EoS regime and interpreting this as an exploitation-to-exploration transition. Their work established the methodological precedent for treating gradient descent as a dynamical system amenable to chaos-theoretic analysis.

### 1.1 Contribution

We contribute three findings:

**1. Chaos onset occurs far below the EoS threshold.** By computing function-space Lyapunov exponents—measuring the divergence of network outputs under perturbed initial conditions—we identify a critical learning rate η_c ≈ 0.02 at which the maximal Lyapunov exponent crosses zero. This is approximately 6.6% of the EoS threshold 2/λ_max ≈ 0.27, indicating that sensitive dependence on initial conditions emerges well before the loss landscape reaches its curvature stability limit.

**2. The transition is reproducible but noisy.** Across multiple random seeds, the zero-crossing of the Lyapunov exponent is consistent in location (η_c ≈ 0.018 ± 0.012) but exhibits substantial variance, consistent with the theory of stochastic transitions in systems approaching chaos.

**3. Connection to the Ruelle-Takens route.** We frame the pre-EoS chaos onset within the broader theory of torus-to-strange-attractor transitions (Ruelle & Takens, 1971). In this interpretation, the stable training regime corresponds to quasiperiodic dynamics on invariant tori in parameter space; the chaos onset at η_c marks the destruction of these tori; and the EoS regime corresponds to dynamics on or near a strange attractor. The period-doubling route identified by Kalra et al. (2023) and the Lyapunov transition we report are complementary signatures of the same underlying bifurcation sequence.

### 1.2 Significance

If chaos in gradient descent begins at ~7% of the classical stability threshold rather than at or near 100%, this has practical implications for training reproducibility, hyperparameter tuning, and the interpretation of training loss curves. It also connects neural network optimization to a body of mathematical theory—KAM theorem, Arnold tongues, Ruelle-Takens bifurcation—that describes precisely how ordered periodic dynamics transition to chaos through intermediate stages of toroidal and quasiperiodic motion.

---

## 2. Background

### 2.1 The Edge of Stability

Classical optimization theory predicts that gradient descent with step size η converges monotonically when the sharpness S(θ) = λ_max(∇²L(θ)) satisfies S(θ) < 2/η throughout training. Cohen et al. (2021) demonstrated empirically that this condition is routinely violated during neural network training. In a two-phase process:

1. **Progressive sharpening:** S(θ) increases during early training until it reaches 2/η.
2. **Edge of stability:** S(θ) hovers near 2/η while the loss continues to decrease, albeit non-monotonically.

The EoS phenomenon has been observed across architectures (MLPs, CNNs, transformers), loss functions (MSE, cross-entropy), and datasets (CIFAR-10, SST-2), suggesting it is a generic feature of gradient descent in high-dimensional non-convex landscapes.

### 2.2 Routes to Chaos in Dynamical Systems

The theory of dynamical systems identifies several canonical routes from ordered to chaotic dynamics. The most relevant to our context are:

**The Ruelle-Takens route** (Ruelle & Takens, 1971): Beginning from a fixed point, a system undergoes successive Hopf bifurcations producing limit cycles (1-tori), then quasiperiodic motion on 2-tori, then 3-tori, and then—crucially—strange attractors. Ruelle and Takens proved that 3-tori are structurally unstable: arbitrarily small perturbations can replace them with fractal strange attractors.

**The Feigenbaum (period-doubling) route** (Feigenbaum, 1978): A system undergoes a cascade of period-doubling bifurcations at parameter values converging geometrically, with universal scaling constants (δ ≈ 4.669, α ≈ 2.503), culminating in chaos. This route has been explicitly identified in neural network training dynamics (Kalra et al., 2023).

**KAM theory** (Kolmogorov, 1954; Arnold, 1963; Moser, 1962): In Hamiltonian systems, invariant tori in phase space survive small perturbations if their frequency ratios are sufficiently irrational. Resonant tori are destroyed, shattering into fractal Cantori. The resulting phase space is a nested mixture of toroidal islands and chaotic seas.

### 2.3 Lyapunov Exponents

The maximal Lyapunov exponent (MLE) quantifies the asymptotic rate of divergence of initially nearby trajectories. For a discrete dynamical system x_{t+1} = F(x_t):

$$\lambda = \lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \ln \|DF(x_t)\|$$

A positive MLE indicates sensitive dependence on initial conditions—the operational definition of chaos. The zero-crossing of the MLE is a standard diagnostic for the onset of chaos.

In the context of neural network training, we compute function-space Lyapunov exponents: rather than tracking parameter-space divergence (which can reflect gauge symmetries and irrelevant reparametrizations), we measure the divergence of network outputs f_θ(x) for fixed inputs x under perturbed initial weight conditions. This captures the dynamically meaningful divergence—whether different training runs produce functionally different networks.

---

## 3. Methods

### 3.1 Architecture and Data

Following Cohen et al. (2021), we use a two-hidden-layer MLP with tanh activation:

- Input: 220 dimensions
- Hidden layers: 50 units each
- Output: 10 classes
- Total parameters: 156,710

The dataset consists of 2,000 synthetic data points with 10 classes. Each point has 200 random features drawn from class-specific Gaussian clusters, plus 20 quadratic features (the squared values of the first 20 input dimensions). The quadratic features introduce nonlinear structure that the small network cannot trivially memorize, ensuring persistent and nontrivial training dynamics.

**Rationale for synthetic data:** Using synthetic data with known structure enables exact reproducibility and isolates the dynamics of gradient descent from dataset-specific confounds. The 220-dimensional input with 2,000 samples creates a regime where the network has sufficient capacity to learn meaningful structure but insufficient capacity for memorization, producing sustained optimization dynamics across a range of learning rates.

### 3.2 Training Protocol

Full-batch gradient descent with MSE loss. No momentum, weight decay, or learning rate scheduling. Each experiment trains for a fixed number of steps (sufficient for loss convergence at moderate learning rates). Sharpness λ_max is computed via the Lanczos algorithm (power iteration) at convergence for the lowest learning rate to establish the EoS threshold 2/λ_max.

### 3.3 Function-Space Lyapunov Exponent Computation

For each learning rate η and random seed s:

1. Initialize network weights θ_0 from a fixed seed s.
2. Initialize a perturbed copy θ_0' = θ_0 + εδ, where δ is drawn from a unit-norm random vector and ε is small (typically 10^{-8} to 10^{-6}).
3. Train both networks under identical conditions (same data, same learning rate, same batch ordering) for T steps.
4. At each step t, compute the L2 distance in function space: d(t) = ‖f_{θ_t}(X) − f_{θ_t'}(X)‖ for a fixed evaluation set X.
5. Estimate the Lyapunov exponent as the slope of log(d(t)/d(0)) vs. t, fitted over the interval where exponential growth or decay is observed.

A positive Lyapunov exponent indicates that initially nearby networks (in weight space) diverge in function space—they learn meaningfully different functions despite identical training data and hyperparameters.

### 3.4 Experimental Design

**Experiment 1: Broad sweep (reproducibility).** 20 learning rates logarithmically spaced from η ≈ 0.013 to η ≈ 0.404, spanning from well below to well above the EoS threshold. 5 random seeds per learning rate. Purpose: establish the overall shape of the Lyapunov curve and verify reproducibility across seeds.

**Experiment 2: Transition zone.** 30 learning rates linearly spaced from η = 0.005 to η = 0.08, densely sampling the region where the Lyapunov exponent crosses zero. 3 random seeds per learning rate. Purpose: precisely locate the critical learning rate η_c.

---

## 4. Results

### 4.1 Broad Sweep

[Figure 1: Reproducibility across 5 seeds]

Across 5 random seeds and 20 learning rates, the function-space Lyapunov exponent increases monotonically with learning rate. At the lowest learning rates (η ≈ 0.013), Lyapunov exponents are near zero or slightly negative, indicating convergent or neutrally stable dynamics. At higher learning rates, exponents are consistently positive and increase roughly linearly with η.

The EoS threshold 2/λ_max ≈ 0.270 falls in the upper third of the tested range. Crucially, positive Lyapunov exponents appear at learning rates far below this threshold—some seeds show positive exponents as low as η ≈ 0.03, well within what classical optimization theory considers the "stable" regime.

Seed-to-seed variability is moderate: individual seeds trace similar curves but with fluctuations, particularly in the mid-range of learning rates (η ≈ 0.15–0.30). The mean across seeds shows a smooth monotonic increase, with error bands widening at higher learning rates.

### 4.2 Transition Zone

[Figure 2: Transition zone detail with zero-crossing]

In the fine-grained transition zone analysis, the mean Lyapunov exponent crosses zero at η_c ≈ 0.019 (computed from the mean curve) with individual seed zero-crossings at η_c = 0.018 ± 0.012.

The transition is not a sharp discontinuity but a gradual passage through a region where the Lyapunov exponent fluctuates near zero—consistent with the theory of intermittency near chaos onset, where trajectories alternate between nearly-periodic and chaotic episodes.

### 4.3 Critical Learning Rate

[Figure 3: Critical η_c relative to EoS threshold]

The critical learning rate η_c = 0.018 ± 0.012 represents approximately 6.6% of the EoS threshold 2/λ_max ≈ 0.270. This is the central empirical finding: chaos, as measured by positive function-space Lyapunov exponents, begins at a learning rate more than an order of magnitude below the classical instability boundary.

---

## 5. Discussion

### 5.1 Interpretation: A Torus-to-Chaos Transition

The Ruelle-Takens route provides a natural framework for interpreting the observed transition. In this interpretation:

**Below η_c (stable regime):** Training dynamics are convergent. Different initializations produce trajectories that converge to the same or similar functions. The dynamics in parameter space may be quasiperiodic, with trajectories confined to invariant tori—the parameter-space analogue of KAM tori in Hamiltonian systems. The negative Lyapunov exponent indicates that the training map is locally contracting.

**Near η_c (transition zone):** The system passes through the torus-destruction regime. Some invariant structures survive (analogous to KAM tori with sufficiently irrational frequency ratios), while others are destroyed (analogous to resonant tori shattering into Cantori). The Lyapunov exponent fluctuates near zero, with intermittent bursts of divergence interspersed with convergent episodes.

**Above η_c but below 2/λ_max (pre-EoS chaos):** Dynamics are chaotic in function space (positive Lyapunov exponent) but the loss still decreases on average. This is the regime identified by Morales et al. (2024) as an exploitation-exploration balance. The system operates on or near a strange attractor in parameter space, but the attractor is still contained within a region of decreasing loss.

**At and above 2/λ_max (EoS regime):** The sharpness saturates at 2/η. The self-stabilization mechanism identified by Damian et al. (2023) prevents divergence. Period-doubling cascades occur (Kalra et al., 2023). The dynamics are chaotic but self-regulating—a regime with no simple analogue in classical dynamical systems theory, where the nonlinearity of the loss landscape creates an endogenous stability mechanism absent from typical Hamiltonian or dissipative systems.

### 5.2 Relationship to Existing Work

**Kalra et al. (2023)** identify a period-doubling route to chaos on the EoS manifold. Our finding is complementary: we observe chaos onset (positive Lyapunov exponents) below the EoS manifold, suggesting the period-doubling route begins before the system reaches EoS. The period-doubling structure may be present in the pre-EoS regime but require finer temporal resolution to detect.

**Damian et al. (2023)** argue that EoS dynamics are "far from chaotic." Our results suggest this characterization applies at the EoS itself—where self-stabilization constrains dynamics—but that the sub-EoS regime harbors genuine chaos as measured by Lyapunov exponents. The self-stabilization mechanism may actively suppress pre-existing chaotic tendencies once sharpness reaches 2/η.

**Morales et al. (2024)** measure Lyapunov exponents in parameter space and find positive values in the EoS regime. Our approach differs by measuring in function space (capturing dynamically meaningful divergence) and by focusing on the sub-EoS regime where chaos emerges.

### 5.3 Implications for Practice

If the chaos boundary lies at ~7% of the classical stability threshold, then:

- **Reproducibility:** Training runs with learning rates above η_c will produce functionally different networks from different initializations, even with identical data and hyperparameters. This has implications for scientific reproducibility and model auditing.

- **Hyperparameter selection:** The "safe" learning rate range for deterministic convergence is much narrower than classical theory suggests. Most practical learning rates likely operate in the chaotic regime.

- **Ensemble diversity:** The pre-EoS chaos regime naturally produces diverse models from different seeds, which may partly explain why deep ensembles work well at standard learning rates.

### 5.4 Limitations

**Sample size.** The transition zone analysis uses only 3 random seeds, yielding a confidence interval on η_c (±0.012) nearly as large as the estimate itself. The broad sweep uses 5 seeds. Both are insufficient for definitive claims and should be expanded to ≥20 seeds.

**Single architecture and dataset.** Results are from one architecture (2-layer tanh MLP) on synthetic data. Generalization to ReLU networks, deeper architectures, CNNs, transformers, and standard benchmarks (CIFAR-10, ImageNet subsets) is necessary before the finding can be considered robust.

**Lyapunov estimation method.** The perturbation-based function-space Lyapunov estimate depends on the perturbation magnitude ε, the evaluation set, and the fitting interval. Sensitivity analysis across these parameters is needed.

**Causal interpretation.** We observe that Lyapunov exponents become positive below the EoS threshold, but the causal mechanism is unclear. Is the chaos driven by the same curvature dynamics that produce EoS, just at a lower threshold in function space? Or is it a separate phenomenon?

---

## 6. Future Directions

### 6.1 Immediate Extensions

- Increase seed count to ≥20 for robust statistical inference on η_c
- Test additional architectures: ReLU MLPs, 3–4 layer networks, small CNNs and transformers on CIFAR-10
- Construct bifurcation diagrams of training loss at successive learning rates to detect period-doubling cascades below EoS
- Compute power spectra of training loss time series to identify the frequency structure (quasiperiodic vs. broadband) at different learning rates

### 6.2 Connecting to Dynamical Systems Theory

- Apply Takens embedding to training loss time series and attempt phase-space reconstruction, looking for toroidal structure below η_c and strange-attractor structure above
- Compute the fractal dimension of the training trajectory in function space as a function of learning rate
- Test whether the Feigenbaum constants (δ ≈ 4.669) govern the period-doubling cascade observed by Kalra et al. (2023) and whether this extends to the sub-EoS regime
- Investigate whether Arnold tongue structures appear in the frequency-locking behavior of multi-frequency oscillations in the training loss

### 6.3 Broader Framework

This work connects to a broader perspectival framework that positions toroidal geometry as a lens for understanding nested periodic systems across disciplines. The torus—the geometry of one cycle within another—is a natural description of training dynamics in the stable regime, where parameter updates cycle through recurring loss landscape features. The transition to chaos at η_c corresponds to the destruction of these toroidal structures, with fractal strange attractors emerging as predicted by the Ruelle-Takens theorem. The EoS phenomenon, in this framing, is a self-stabilization mechanism that prevents the system from fully entering the chaotic regime—a dynamical structure unique to the gradient descent setting.

---

## 7. Conclusion

We have presented evidence that the onset of chaos in gradient descent—measured by the zero-crossing of function-space Lyapunov exponents—occurs at a critical learning rate approximately 6.6% of the Edge of Stability threshold. This finding places the transition to chaotic dynamics far earlier in the learning rate schedule than previously recognized, with implications for training reproducibility and hyperparameter selection.

The result connects neural network optimization to the rich mathematical theory of torus-to-chaos transitions in dynamical systems. The Ruelle-Takens route, KAM theory, and the Feigenbaum period-doubling scenario provide a unified theoretical framework within which the progressive sharpening, edge of stability, and chaos-onset phenomena can be understood as stages of a single bifurcation sequence—from fixed point through quasiperiodic motion on tori to strange attractors.

Further work with larger sample sizes, diverse architectures, and explicit detection of toroidal and strange-attractor structure in training dynamics will determine whether this connection is merely suggestive or reflects a deep structural relationship between optimization and dynamical systems theory.

---

## References

Arnold, V. I. (1963). Small denominators and problems of stability of motion in classical and celestial mechanics. *Russian Mathematical Surveys*, 18(6), 85–191.

Cohen, J., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2021). Gradient descent on neural networks typically occurs at the edge of stability. In *International Conference on Learning Representations*.

Damian, A., Nichani, E., & Lee, J. D. (2023). Self-stabilization: The implicit bias of gradient descent at the edge of stability. In *International Conference on Learning Representations*.

Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. *Journal of Statistical Physics*, 19(1), 25–52.

Kalra, D. S., He, T., & Barkeshli, M. (2023). Universal sharpness dynamics in neural network training: Fixed point analysis, edge of stability, and route to chaos. *arXiv preprint arXiv:2311.02076*.

Kolmogorov, A. N. (1954). On the conservation of conditionally periodic motions under small perturbation of the Hamiltonian. *Dokl. Akad. Nauk SSSR*, 98, 527–530.

Morales, G. B., Muñoz, M. A., & various (2024). Dynamical stability and chaos in artificial neural network trajectories along training. *Frontiers in Complex Systems*, 2, 1367957.

Moser, J. (1962). On invariant curves of area-preserving mappings of an annulus. *Nachr. Akad. Wiss. Göttingen*, II, 1–20.

Ruelle, D., & Takens, F. (1971). On the nature of turbulence. *Communications in Mathematical Physics*, 20, 167–192.

Various authors (2025). Learning dynamics of deep linear networks beyond the edge of stability. *arXiv preprint arXiv:2502.20531*.

Züchner, T., et al. (2024). Finite-time Lyapunov exponents of deep neural networks. *Physical Review Letters*, 132, 057301.

---

## Appendix A: Experimental Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | MLP: 220 → 50 (tanh) → 50 (tanh) → 10 |
| Parameters | 156,710 |
| Dataset | Synthetic: 2,000 points, 10 classes, 220 dims (200 random + 20 quadratic) |
| Loss function | Mean squared error |
| Optimizer | Full-batch gradient descent (no momentum) |
| λ_max | ~7.42 (computed via Lanczos) |
| 2/λ_max (EoS threshold) | ~0.270 |
| η_c (critical LR) | 0.018 ± 0.012 |
| η_c / (2/λ_max) | ~6.6% |
| Broad sweep LRs | 20 values, ~0.013 to ~0.404 |
| Transition zone LRs | 30 values, 0.005 to 0.08 |
| Seeds (broad) | 5 |
| Seeds (transition) | 3 |
| Perturbation ε | [to be specified] |
