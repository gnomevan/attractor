# The Torus-Fractal Transition in Neural Network Training Dynamics
## A KAM Theory Perspective on the Edge of Stability

### Paper Sketch — Draft v0.1

---

## Abstract (Draft)

Neural network training via gradient descent exhibits a well-documented but poorly understood phenomenon: the Edge of Stability (EoS), where the largest Hessian eigenvalue oscillates around the critical threshold 2/η while the loss continues to decrease non-monotonically. Recent work has shown that the loss landscape itself has toroidal topology (when symmetries are removed) and multifractal geometry. We propose that these observations are unified by a framework from dynamical systems theory: the torus-to-fractal transition described by KAM theory, Arnold tongues, and the Ruelle-Takens route to chaos. In this framework, stable training corresponds to quasi-periodic orbits on KAM tori in function space; the Edge of Stability corresponds to the critical perturbation threshold where these tori begin to break down; and the period-doubling route to chaos already observed in training dynamics corresponds to the Ruelle-Takens route. The learning rate η acts as the perturbation parameter analogous to ε in classical KAM theory. This perspective generates specific, testable predictions about training dynamics and suggests geometric principles for optimizer design.

---

## 1. Introduction: The Problem

Neural network training is a dynamical system. Each gradient descent step moves the network's parameters through a high-dimensional space, tracing a trajectory through a loss landscape. Yet the dominant frameworks for understanding this process — convex optimization theory, statistical learning theory — were not designed for the dynamics actually observed.

The central puzzle is the **Edge of Stability** (Cohen et al., 2021): when training with gradient descent at fixed learning rate η, the sharpness (largest Hessian eigenvalue λ_max) does not remain below the classical stability threshold 2/η. Instead:

1. **Progressive sharpening**: λ_max steadily increases until it reaches 2/η
2. **Edge of Stability**: λ_max hovers just above 2/η, oscillating around it
3. **Non-monotonic loss**: the loss oscillates on short timescales but decreases on long timescales
4. **Period-doubling route to chaos**: as η increases, the oscillations undergo period-doubling bifurcations (Kalra et al., 2023; Ghosh et al., 2025)

Separately, Pittorino et al. (2022) showed that when symmetries are removed from the parameter space, the resulting topology is **toroidal**. And a 2025 Nature Communications paper (optimization on multifractal loss landscapes) demonstrated that loss landscapes exhibit **multifractal** geometry with clustered degenerate minima and multiscale structure.

**These three findings — toroidal topology, edge-of-stability oscillations, and multifractal geometry — are not independent phenomena.** They are stages of a single geometric transition well-characterized in classical dynamical systems theory: the breakdown of invariant tori into fractal structures under perturbation.

This paper proposes that the mathematics of this transition — KAM theory, Arnold tongues, and the Ruelle-Takens route — provides a unified framework for understanding neural network training dynamics.

---

## 2. Background: The Torus-Fractal Transition in Classical Dynamics

### 2.1 KAM Theory

The Kolmogorov-Arnold-Moser (KAM) theorem (Kolmogorov, 1954; Arnold, 1963; Moser, 1962) addresses the stability of quasi-periodic motion under perturbation. For a Hamiltonian system with n degrees of freedom:

- The unperturbed system has motion confined to n-dimensional invariant tori in 2n-dimensional phase space
- Under small perturbation ε, "most" tori survive (those with sufficiently irrational frequency ratios)
- Tori with rational or near-rational frequency ratios break down
- There exists a critical perturbation ε_c above which widespread torus destruction occurs

The key parameters governing torus survival are:
- The **perturbation strength** ε
- The **frequency ratio** of the quasi-periodic motion (rational ratios are unstable)
- The **Diophantine condition** — how "irrational" the frequency ratio is

### 2.2 Arnold Tongues

For coupled oscillators, **Arnold tongues** describe the regions in parameter space where frequency-locking (mode-locking) occurs. In the parameter plane of coupling strength vs. frequency ratio:

- At zero coupling, only exact resonances produce locking
- As coupling increases, locking regions widen into tongue-shaped zones
- Between the tongues, quasi-periodic motion persists on invariant tori
- At sufficient coupling, tongues overlap, producing chaotic dynamics

### 2.3 Ruelle-Takens Route to Chaos

The Ruelle-Takens theorem (1971) describes how chaos emerges from quasi-periodic motion:

1. A fixed point loses stability → periodic orbit (1-torus)
2. The periodic orbit loses stability → quasi-periodic motion on a 2-torus
3. The 2-torus loses stability → **chaos** (not a 3-torus, as might be expected)

This route requires only three independent frequencies before the invariant torus generically breaks down into a strange attractor. The resulting strange attractor has fractal geometry.

### 2.4 Summary: The Transition

**Torus → perturbation → torus breakdown → fractal (strange attractor)**

This sequence is universal — it occurs in fluid dynamics (Bénard convection → turbulence), celestial mechanics (planetary orbits → asteroid belt gaps), and coupled oscillator networks. We propose it also occurs in neural network training.

---

## 3. The Mapping: Training Dynamics as Perturbed Torus

### 3.1 The Loss Landscape Has Toroidal Topology

Pittorino et al. (2022) demonstrated that when the symmetries of neural network parameter spaces are factored out (permutation symmetry of neurons within a layer, rescaling symmetry between layers), the resulting quotient space has **toroidal topology**. This is not metaphorical — the space of distinct functions implemented by a neural network, modulo symmetry, is literally a torus.

Training dynamics thus take place on or near this toroidal manifold.

### 3.2 The Learning Rate as Perturbation Parameter

In KAM theory, the perturbation parameter ε controls the transition from regular to chaotic dynamics. In neural network training:

**The learning rate η plays the role of ε.**

This mapping is precise:

| KAM Theory | Neural Network Training |
|---|---|
| Perturbation parameter ε | Learning rate η |
| Critical threshold ε_c | Edge of Stability threshold 2/λ_max |
| Invariant torus | Quasi-periodic training orbit in function space |
| Torus breakdown | Progressive sharpening → EoS transition |
| Strange attractor (fractal) | Multifractal loss landscape structure |
| Frequency ratio | Ratio of eigenvalues of the Hessian |
| Mode-locking (Arnold tongue) | Feature learning phase transitions |

### 3.3 Progressive Sharpening = Approach to Critical Perturbation

In KAM theory, as ε approaches ε_c, the invariant tori begin to deform. The quasi-periodic orbits stretch and thin. This is progressive sharpening: the training trajectory approaches the boundary of its stable toroidal region in function space.

The sharpness λ_max increasing toward 2/η is the system's way of detecting that the "torus" it is orbiting on is about to break down — the curvature in the most unstable direction is approaching the threshold where the orbit can no longer be contained on a smooth surface.

### 3.4 Edge of Stability = Torus Breakdown Regime

At the Edge of Stability, λ_max hovers around 2/η. The training dynamics oscillate: the loss is non-monotonic on short timescales but decreasing on long timescales.

In KAM terms, this is the regime where:
- The invariant torus has broken down in some directions (those corresponding to eigenvalues > 2/η)
- The orbit is no longer confined to a smooth surface
- But the dynamics have not fully transitioned to chaos — residual KAM tori still provide partial barriers
- The orbit threads through a mixture of regular and chaotic regions (a "stochastic web")

The self-stabilization mechanism observed at EoS (Damian et al., 2022) — where divergence along the top Hessian eigenvector triggers a cubic correction that reduces sharpness — is precisely the mechanism by which orbits near broken tori are reflected back toward surviving tori. The system bounces between the "last KAM torus" and the chaotic layer beyond it.

### 3.5 Period-Doubling = Ruelle-Takens Route

Kalra et al. (2023) and Ghosh et al. (2025) demonstrated that loss oscillations beyond EoS follow a **period-doubling route to chaos** as the learning rate increases. This is a classic signature of the Ruelle-Takens transition:

- At η just above the EoS threshold: period-2 oscillations (the orbit visits two alternating function-space locations)
- At higher η: period-4, period-8, ... (successive bifurcations)
- At sufficiently high η: chaotic training dynamics

The period-doubling cascade is one of the canonical routes to chaos in dynamical systems, and its presence in training dynamics confirms that the torus-to-fractal transition is occurring.

### 3.6 Multifractal Loss Landscape = The Fractal After Torus Breakdown

The 2025 Nature Communications result — that loss landscapes are multifractal — completes the picture. The multifractal structure is not a static property of the loss landscape; it is the **geometric residue of torus breakdown**. Just as the fractal gaps in the asteroid belt are the geometric trace of destroyed KAM tori in the solar system's phase space, the multifractal structure of neural network loss landscapes is the trace of broken invariant tori in function space.

---

## 4. Predictions

### 4.1 KAM-Based Predictions

If the torus-fractal framework is correct, it generates specific predictions:

**Prediction 1: Frequency ratio governs stability.** The ratio of Hessian eigenvalues (not just the largest eigenvalue) should determine training stability. Eigenvalue ratios close to simple rational numbers (1:1, 1:2, 2:3) should correspond to mode-locked training dynamics, while sufficiently irrational ratios should be more stable. This is testable by tracking the full Hessian spectrum during training.

**Prediction 2: Cantori and partial barriers.** After torus breakdown in KAM theory, the destroyed torus leaves behind a fractal remnant called a "cantorus" — a Cantor-set-like object that acts as a partial barrier to diffusion. In training dynamics, this predicts that after EoS onset, the training trajectory should exhibit intermittent trapping: periods of slow movement near cantori alternating with rapid jumps across broken barriers. This intermittency should have characteristic power-law statistics.

**Prediction 3: Arnold tongue structure in multi-objective training.** When a network is trained on multiple objectives (multi-task learning, auxiliary losses), each objective introduces an independent oscillatory component. Arnold tongue theory predicts that mode-locking between objectives will produce plateaus in training (the flat regions of loss curves) and that the boundaries of these tongues will correspond to transitions between qualitatively different training regimes.

**Prediction 4: Three independent frequencies suffice for chaos.** The Ruelle-Takens theorem states that quasi-periodic motion with three independent frequencies is generically unstable. If the Hessian has three or more eigenvalues exceeding the stability threshold 2/η, training should transition to chaos regardless of other parameters. This is testable.

**Prediction 5: Optimal training operates at the last KAM torus.** The best generalization should occur when the training dynamics are near but not beyond the critical perturbation threshold — at the boundary between the last surviving KAM torus and the chaotic sea. This is consistent with the empirical observation that larger learning rates (closer to instability) produce flatter minima and better generalization.

### 4.2 Arnold Tongue Predictions for Learning Rate Schedules

The Arnold tongue framework predicts that learning rate schedules (warmup, cosine annealing, cyclical learning rates) work because they navigate the parameter space in a way that:

- Enters mode-locked regions (Arnold tongues) for efficient feature learning
- Exits mode-locked regions to avoid over-specialization
- Approaches the boundary of torus stability to find flatter minima

This suggests a principled approach to learning rate scheduling based on monitoring the Hessian eigenvalue ratios and deliberately steering toward or away from specific resonance conditions.

---

## 5. Relation to Existing Work

### 5.1 What This Framework Adds

Several lines of research have independently touched pieces of this picture:

- **Toroidal function space topology** (Pittorino et al., 2022) — established the torus but did not connect it to KAM theory or torus breakdown
- **Multifractal loss landscapes** (Nature Comms, 2025) — established the fractal but did not connect it to the torus as its precursor
- **Period-doubling in training** (Kalra et al., 2023; Ghosh et al., 2025) — observed the route to chaos but did not identify it as the Ruelle-Takens route from toroidal dynamics
- **KAM theory for Hamiltonian neural networks** (Offen & Ober-Blöbaum, 2022) — applied KAM theory to the *target* dynamics (the physical system being modeled) but not to the *training* dynamics
- **Edge of Stability** (Cohen et al., 2021) — characterized the phenomenon empirically but the theoretical accounts remain model-specific

What has been missing is the synthesis: **these are all manifestations of the same geometric transition**. The torus comes first. The fractal comes after. The Edge of Stability is the boundary between them. The learning rate is the perturbation parameter. KAM theory is the unified mathematical framework.

### 5.2 Relationship to the Cycles-Within-Cycles Framework

This paper builds on the framework articulated in [core theory paper], which argues that:

1. Nested periodicity generates toroidal geometry as mathematical necessity
2. Toroidal structures under perturbation produce fractal geometry via KAM breakdown
3. The torus occupies a "Goldilocks" position between frozen snapshots and fractal chaos — the scale at which dynamics become legible

Neural network training provides a clean, computationally accessible test case for this framework. The toroidal topology of function space is established. The fractal geometry of the loss landscape is documented. The transition between them is observable in real time through training curves. And the perturbation parameter (learning rate) is under direct experimental control.

---

## 6. Proposed Research Program

### Phase 1: Empirical Validation (Computational)

- Reproduce the EoS phenomenon in controlled settings (small networks, known loss landscapes)
- Track the full Hessian eigenvalue spectrum (not just λ_max) through training
- Compute eigenvalue ratios and test Prediction 1 (irrational ratios more stable)
- Compute Lyapunov exponents along training trajectories and identify the torus-to-chaos transition point
- Test Prediction 4 (three eigenvalues above 2/η → chaos)

### Phase 2: Arnold Tongue Mapping

- Systematically vary learning rate and a second parameter (e.g., batch size, momentum coefficient) 
- Map the regions of mode-locked vs. quasi-periodic training dynamics
- Compare the resulting structure to Arnold tongue geometry
- Test whether tongue boundaries correspond to transitions in generalization performance

### Phase 3: Optimizer Design

- Design a "KAM-aware" optimizer that:
  - Monitors eigenvalue ratios (not just λ_max) during training
  - Adjusts η to maintain the system near the last KAM torus boundary
  - Deliberately navigates Arnold tongue structure for multi-task learning
- Compare performance against standard optimizers (SGD, Adam, SAM) on benchmark tasks

### Phase 4: Theoretical Analysis

- Formal mapping between the gradient descent iteration map and the standard map (a canonical model for KAM torus breakdown)
- Derivation of the critical perturbation threshold ε_c in terms of network architecture and data distribution
- Extension of KAM stability bounds to stochastic gradient descent (where mini-batch noise acts as additional perturbation)

---

## 7. Collaborator Profiles Needed

This research sits at the intersection of:

- **Dynamical systems theory** — someone with KAM theory expertise who can rigorously map the GD iteration to standard form
- **Deep learning theory** — someone with access to Hessian computation pipelines and experience with loss landscape analysis
- **Computational physics** — someone experienced with Lyapunov exponent computation and phase space reconstruction

The theoretical framework and geometric perspective come from the cycles-within-cycles work. The computational infrastructure exists in ML research groups working on loss landscape analysis. The connection has not been made because these communities rarely interact.

---

## 8. Why This Matters

### For AI/ML:
- A unified geometric theory of training dynamics would replace the current patchwork of phenomenon-specific explanations
- KAM-aware optimizers could improve training stability and generalization by operating at the geometrically optimal perturbation threshold
- Arnold tongue theory could provide principled multi-task learning strategies

### For the Torus Framework:
- Neural network training provides a **clean, controllable, computationally reproducible** test case for the torus-to-fractal transition
- All parameters are observable and adjustable — unlike biological or astronomical systems
- If the framework explains training dynamics, it demonstrates the cross-disciplinary universality claimed in the core theory paper

### For Science More Broadly:
- This would be a concrete demonstration that the same geometric transition (torus → perturbation → fractal) operates in computational systems as it does in physical ones
- It connects 20th-century dynamical systems theory to 21st-century machine learning in a way that benefits both fields

---

## References

Arnold, V. I. (1963). Small denominators and problems of stability of motion in classical and celestial mechanics. *Russian Mathematical Surveys*, 18(6), 85–191.

Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2021). Gradient descent on neural networks typically occurs at the edge of stability. *ICLR 2021*.

Damian, A., Ma, T., & Lee, J. D. (2022). Self-stabilization: The implicit bias of gradient descent at the edge of stability. *arXiv:2209.15594*.

Ghosh, A., Kwon, S. M., Wang, R., Ravishankar, S., & Qu, Q. (2025). Learning dynamics of deep matrix factorization beyond the edge of stability. *ICLR 2025*.

Kalra, D. S., et al. (2023). Universal sharpness dynamics in neural network training: Fixed point analysis, edge of stability, and route to chaos. *arXiv:2311.02076*.

Kolmogorov, A. N. (1954). On the conservation of conditionally periodic motions under small perturbation of the Hamiltonian. *Dokl. Akad. Nauk SSSR*, 98, 527–530.

Moser, J. (1962). On invariant curves of area-preserving mappings of an annulus. *Nachr. Akad. Wiss. Göttingen Math.-Phys. Kl. II*, 1, 1–20.

Offen, C., & Ober-Blöbaum, S. (2022). KAM theory meets statistical learning theory: Hamiltonian neural networks with non-zero training loss. *Proceedings of the AAAI Conference on AI*, 36(6), 6322–6330.

Pittorino, F., et al. (2022). Deep networks on toroids: Removing symmetries reveals the structure of flat regions in the landscape geometry. *arXiv:2202.03038*.

Ruelle, D., & Takens, F. (1971). On the nature of turbulence. *Communications in Mathematical Physics*, 20(3), 167–192.

[Nature Communications 2025]. Optimization on multifractal loss landscapes explains a diverse range of geometrical and dynamical properties of deep learning. *Nature Communications*, 2025.

---

## Notes for Development

**Strongest claims (mathematically grounded)**:
- The toroidal topology of function space is established (Pittorino et al.)
- The period-doubling route to chaos at EoS is documented (Kalra, Ghosh)
- KAM theory has been applied to Hamiltonian neural networks (Offen & Ober-Blöbaum)
- The multifractal structure of loss landscapes is documented (Nat. Comms.)

**Claims that need formal proof**:
- That the GD iteration map, restricted to the toroidal quotient space, satisfies the conditions for KAM theory to apply
- That eigenvalue ratios (not just λ_max) govern stability in the way predicted
- That the Arnold tongue structure exists in the (η, other hyperparameter) plane

**Claims that are predictive / speculative but testable**:
- Three eigenvalues above 2/η → chaos (Ruelle-Takens)
- Cantorus-like partial barriers producing intermittent trapping
- KAM-aware optimizers outperforming standard optimizers

**Key risk**: The GD iteration map may not satisfy the Hamiltonian structure required for classical KAM theory. The "KAM for dissipative systems" literature (e.g., Broer et al.) may be needed instead, which is a more complex mathematical landscape. This should be acknowledged upfront and explored carefully.

---

*This sketch is intended as a working document. The next step is to identify specific collaborators in the ML theory community and begin the Phase 1 computational work: reproducing EoS and tracking full Hessian spectra through training.*
