# Strange Attractors in Gradient Descent: The Ruelle-Takens Route to Chaos in Neural Network Training

**Authors:** [Plant, Collaborators TBD]

**Target:** Physical Review Letters

**Status:** DRAFT v0.2 — 10-seed data and TDA results incorporated

---

## Abstract

We show that the training dynamics of neural networks undergo a transition from one-dimensional convergence to multi-dimensional chaos as architecture complexity increases, following the Ruelle-Takens route. Using full-batch gradient descent on a convolutional network (CIFAR-10, 269K parameters), we measure the correlation dimension D₂ of the function-space trajectory and find D₂ = 3.67 ± 0.08 at 30% of the Edge of Stability threshold — a strange attractor in the space of learned functions. Persistent homology confirms fractal topology: hundreds of topological features at all scales with no dominant loops, ruling out smooth toroidal geometry. Shallow networks (14K parameters) produce only one-dimensional chaos (D₂ ≈ 0.9). The transition is mediated by sharpness oscillations: at low learning rates, the top Hessian eigenvalue climbs monotonically (one mode, D₂ ≈ 1); near the Edge of Stability, it oscillates (coupled modes, D₂ > 3). The chaos is bounded — a "chaos window" where Lyapunov exponents peak mid-range and go negative at high learning rates as basin convergence dominates. These results establish gradient descent as a concrete high-dimensional instantiation of the Ruelle-Takens route to chaos.

---

## I. Introduction

Gradient descent is a dynamical system. Each training step maps the current parameters to new ones through a deterministic update rule, and the resulting trajectory through parameter space — or equivalently through function space — exhibits the full range of dynamical behaviors: convergence, oscillation, and chaos [1–4].

Recent work has revealed rich dynamical structure near a critical learning rate threshold. Cohen et al. [1] discovered the Edge of Stability (EoS): the top eigenvalue of the Hessian (the "sharpness") rises to 2/η and then oscillates, a phenomenon absent from classical optimization theory. Damian et al. [2] characterized the self-stabilization mechanism. Kalra et al. [3] observed period-doubling cascades near the EoS in simplified settings. Morales et al. [4] computed Lyapunov exponents during training.

What has been missing is a characterization of the *geometry* of the resulting dynamics. Period-doubling and Lyapunov exponents describe one-dimensional aspects of the transition. But the Ruelle-Takens theorem [5] predicts a different route: when three or more oscillatory modes couple, the resulting torus is structurally unstable and generically produces a strange attractor with fractal dimension. If training dynamics exhibit multiple coupled oscillatory modes — and the sharpness oscillations at EoS suggest they might — the attractor should be multi-dimensional.

We test this prediction by measuring the correlation dimension D₂ [6] of neural network training trajectories in function space across a systematic sweep of architectures and learning rates. Our central finding: a convolutional network trained on CIFAR-10 produces training dynamics with D₂ = 3.6 — a genuinely multi-dimensional strange attractor — while a multilayer perceptron on synthetic data remains one-dimensional (D₂ ≈ 0.9) across all learning rates. The transition is mediated by the coupling of sharpness oscillations to convergence dynamics, exactly as the Ruelle-Takens framework predicts.

## II. Setup

**Architecture and data.** We study two architectures under full-batch gradient descent with MSE loss, no momentum, and no weight decay. The MLP has 2 hidden layers (width 50, tanh activation, 14,110 parameters) trained on 1,600 synthetic structured examples (220 dimensions, 10 classes). The CNN has two convolutional layers (3→16→32 channels, 3×3 kernels) followed by two fully connected layers (2048→128→10), totaling 268,650 parameters, trained on a 2,000-image subset of CIFAR-10. Both are trained for 5,000 steps. Learning rates are expressed as fractions of the Edge of Stability threshold 2/λ_max, where λ_max is the top Hessian eigenvalue after 1,000 warmup steps at η = 0.01.

**Lyapunov exponents.** We measure chaos in function space rather than parameter space to avoid overparameterization artifacts. Two copies of the network are initialized identically, then one is perturbed by ε = 10⁻⁵ along a random unit-norm direction in parameter space. Both are trained with identical gradient descent updates. At each training step t, we record the function-space divergence δ(t) = ‖f(x) − f̃(x)‖ on a held-out evaluation set of 100 inputs. The Lyapunov exponent is extracted as the slope of log δ(t) vs. t via linear regression over the interval [0.2T, 0.8T], discarding the initial transient and the late-time saturation regime. The perturbation scale ε = 10⁻⁵ was determined by sensitivity analysis: ε = 10⁻⁸ produces uniform artifactual positive exponents across all learning rates, indicating numerical noise dominance. Results are averaged over multiple random seeds (10 for the CNN, 20 for the MLP).

**Correlation dimension.** We compute D₂ via the Grassberger-Procaccia algorithm [6] on the function-space trajectory. The trajectory consists of network outputs on the evaluation set recorded every 10 steps, with the first 20% discarded as transient, yielding ~400 points. We compute all pairwise distances and evaluate the correlation integral C(r) at 20 logarithmically spaced radii between the 1st and 95th percentiles of the distance distribution. D₂ is extracted as the slope of log C(r) vs. log r in the intermediate scaling regime. As a robustness check, an independent implementation using an adaptive fitting window (0.01 < C(r) < 0.5) produces consistent D₂ values at low learning rates (Supplemental Material). We validate the pipeline on two known systems: a quasiperiodic 2-torus (recovered D₂ = 2.06, expected 2.0) and the Lorenz attractor (recovered D₂ = 1.68, expected 2.05). Extended calibration against attractors spanning D₂ = 1.2–7.0 (Supplemental Material) reveals a systematic underestimate of 15–35% for fractal attractors, implying our CNN D₂ values are conservative lower bounds on the true attractor dimension.

**Sharpness.** The top Hessian eigenvalue is estimated by power iteration (15 iterations) every 100 training steps.

## III. Results

**The chaos window.** The Lyapunov exponent is non-monotonic with learning rate in both architectures (Fig. 1a). In the MLP, λ is slightly positive for η < 0.048 (the "chaos window"), peaks at η ≈ 0.035, and goes strongly negative above η ≈ 0.05. In the CNN, the same qualitative shape appears on a different scale: λ peaks at 15% of EoS and crosses zero at ~45% of EoS. The chaos window — the range of learning rates producing positive Lyapunov exponents — widens dramatically from the MLP to the CNN. Above the window, basin convergence dominates: trajectories from different initial perturbations collapse to the same function, producing negative λ. This "basin wins" mechanism is universal across all architectures tested.

**From D₂ < 1 to D₂ > 3.** The correlation dimension reveals a qualitative transition between the two architectures (Fig. 1b). The MLP maintains D₂ < 1.0 across all learning rates and embedding dimensions, with PC1 capturing 96–99% of trajectory variance. Training dynamics are fundamentally one-dimensional — convergence along a single direction, with chaos manifesting only as irregular speed along that path.

The CNN is qualitatively different. D₂ crosses 1.0 at 10% of EoS (1.03 ± 0.02), reaches 2.0 at 20% of EoS (2.63 ± 0.62), peaks at D₂ = 3.67 ± 0.08 at 30% of EoS, and remains above 2.0 out to 80% of EoS (Fig. 1b). Even in the negative-Lyapunov regime (above 45% EoS), the trajectory fills a space of dimension ~2, far from one-dimensional. PC1 drops to ~52% above 20% EoS — half the trajectory variance is off the primary convergence axis. The large variance at 20% EoS (± 0.62) reflects the transition zone: some seeds have crossed into multi-dimensional dynamics while others have not.

**Depth widens the window.** To isolate the effect of architecture depth, we trained tanh MLPs of increasing depth (2–5 hidden layers) on the same synthetic data (Table I). The chaos window width increases from 0.038 (depth 2) to 0.145 (depth 4), and off-axis dynamics (PC2 variance) grow from 5% to 13%. But D₂ remains below 1.0 at all depths tested. Depth alone widens the chaos window but does not produce the coupled modes necessary for multi-dimensional dynamics at this parameter count. The transition to D₂ > 1 requires the combination of convolutional structure, real data complexity, and learning rates near the Edge of Stability — the regime where sharpness oscillations provide additional coupled modes.

**Table I.** Architecture scaling summary. Window = range of learning rates with mean λ > 0.

| Architecture | Params | D₂ | Peak λ | Window width | Max PC2 |
|---|---|---|---|---|---|
| MLP, depth 2 | 14K | 0.9 | +0.000135 | 0.038 | 5.2% |
| MLP, depth 3 | 17K | 0.9 | +0.000236 | 0.046 | 8.7% |
| MLP, depth 4 | 19K | 0.9 | +0.000312 | 0.145 | 12.6% |
| MLP, depth 5 | 22K | 0.9 | +0.000236 | 0.137 | 12.9% |
| CNN, CIFAR-10 | 269K | **3.6** | +0.001866 | ~0.13 (45% EoS) | 10.6% |

**Sharpness oscillations as coupled modes.** The mechanism behind the D₂ transition is visible in the sharpness dynamics (Fig. 1d). At 5% of EoS, the top Hessian eigenvalue climbs monotonically from 4.5 to 30.6 over training — progressive sharpening with one degree of freedom. At 95% of EoS, sharpness stays flat between 7.0 and 8.9 — the Edge of Stability self-stabilization mechanism [1,2]. The D₂ transition from ~1 to ~3.7 corresponds to the shift from monotonic sharpening (one mode) to oscillatory self-stabilization (multiple coupled modes). Calibration against known attractors (Supplemental Material) shows that our pipeline underestimates fractal dimensions by 15–35% in this range, so the measured D₂ ≈ 3.7 represents a lower bound — the true attractor dimension is likely higher. The non-integer value and its magnitude are consistent with a strange attractor produced by multiple coupled oscillatory modes via the Ruelle-Takens instability [5].

**Persistent homology confirms fractal topology.** To distinguish between a smooth torus and a strange attractor — which D₂ alone cannot do — we computed the persistent homology of the PCA-reduced function-space trajectory using ripser [12]. A smooth n-torus produces exactly n persistent H₁ generators (independent loops) with a large gap separating them from topological noise. A strange attractor produces many short-lived features at all scales with no gap. The CNN trajectory shows the latter pattern unambiguously: at 5% EoS, zero H₁ features (a convergence path with no loops); at 30% EoS, 386 H₁ features with a gap ratio of 1.2 (no dominant loops — topological complexity at every scale); at 40% EoS, 421 H₁ features with a gap ratio of 1.1. H₂ features (enclosed voids) follow the same pattern, peaking at 401 features at 40% EoS with no persistent dominant void. This is the topological fingerprint of fractal structure, not smooth toroidal geometry, and confirms that the fractal calibration applies: our measured D₂ = 3.67 substantially underestimates the true attractor dimension.

**An unexpected dissociation.** The peak Lyapunov exponent (15% EoS) and the peak correlation dimension (30% EoS) occur at different learning rates (Fig. 1a,b). The attractor is most geometrically complex not where sensitivity to perturbation is greatest, but in the transition zone between the chaos-dominated and basin-dominated regimes.

This dissociation has a natural interpretation within KAM theory. At 15% EoS, the dynamics are strongly chaotic — most invariant tori have been destroyed and the trajectory wanders freely through the chaotic sea. The Lyapunov exponent is large but the attractor, being an essentially connected chaotic region, need not have the highest dimension. At 30% EoS, the system sits in the transition zone where the chaos is weakening but basin convergence has not yet taken over. In this regime, surviving ordered structures (remnants of invariant tori) coexist with chaotic regions. The trajectory must navigate around these ordered islands, and the fractal interleaving of ordered and chaotic regions — the hallmark of the KAM transition — produces an attractor of higher effective dimension than either pure chaos or pure order.

Calibration of our D₂ pipeline against known dynamical systems supports this interpretation. Smooth manifolds (quasiperiodic tori) are systematically overestimated at higher dimensions, while fractal strange attractors are systematically underestimated (see Supplemental Material). The persistent homology results confirm fractal topology, placing the CNN attractor in the underestimated category. Combined with the D₂ calibration, this suggests the true attractor dimension is substantially higher than our measured 3.67 — consistent with the KAM transition picture, in which the fractal interleaving of ordered and chaotic regions produces high-dimensional structure.

## IV. Discussion

These results establish gradient descent training dynamics as an instantiation of the Ruelle-Takens route to chaos in a high-dimensional computational system. Two complementary theoretical frameworks illuminate the transition.

From the perspective of Ruelle-Takens [5], the route is one of successive coupling. At low learning rates, training converges along a single direction — a limit cycle with D₂ ≈ 1. As the learning rate increases toward the Edge of Stability, sharpness oscillations introduce additional oscillatory modes that couple to the convergence dynamics. Each new coupled mode adds structure: two modes produce a 2-torus, three or more produce a system that is structurally unstable, and the coupled dynamics generate a strange attractor. In this framing, the fractal geometry is not the destruction of prior structure but the emergence of new structure from mode interaction — the detailed geometry of how the oscillatory modes relate to one another.

From the perspective of KAM theory [8], the route is one of selective survival. An integrable system's phase space is foliated by invariant tori. Under perturbation — here, increasing the learning rate — tori with sufficiently irrational frequency ratios survive, while resonant tori are destroyed and replaced by Cantori and island chains at finer scales. The fractal dimension of the resulting attractor reflects the proportion of tori destroyed: at 30% EoS, enough structure has been broken to produce D₂ > 3, but enough survives to keep the dynamics bounded. The persistent homology results confirm that the transition has proceeded far enough to produce genuinely fractal topology — hundreds of topological features at all scales rather than a few dominant loops. The dissociation between peak chaos (15% EoS) and peak dimension (30% EoS) arises naturally: maximum geometric complexity occurs where ordered and chaotic regions are most deeply interleaved, not where chaos is most intense.

These are not competing explanations but complementary descriptions. Ruelle-Takens describes what is being built — the coupling of oscillatory modes. KAM describes what is simultaneously being broken — the invariant tori of the uncoupled system. The strange attractor is the geometry of their coexistence.

The Lyapunov vector direction wanders isotropically during training, with no period-doubling signature in the bifurcation diagram. This rules out the Feigenbaum cascade as the route to chaos in this system and is consistent with multi-directional instability — the hallmark of coupled-mode dynamics rather than sequential bifurcation.

A methodological contribution: Lyapunov exponents of neural network training computed at perturbation scale ε = 10⁻⁸ are dominated by numerical noise, producing uniform artifactual positive exponents. Reliable measurements require ε ≥ 10⁻⁶ in our architecture. Prior work computing Lyapunov exponents at smaller perturbation scales [4,7] should be interpreted with caution. The function-space measurement protocol (output divergence on held-out data rather than parameter-space distance) also avoids the false positives from gauge symmetries in overparameterized networks.

Several questions remain open. Does D₂ continue to increase with larger architectures (ResNets, transformers)? Can the specific coupled oscillatory modes beyond sharpness be identified? At lower learning rates (10–20% EoS), where D₂ passes through 2.0, does the trajectory exhibit residual toroidal topology — a partially surviving torus coexisting with the emerging fractal structure? And does the strange attractor geometry during training contribute to the well-known generalization benefits of higher learning rates? What the present results establish is that the geometry of gradient descent training is richer than previously recognized: not merely noisy convergence, but a dynamical system whose attractor structure reflects the interplay between coupled oscillatory modes and the invariant structures they generate and destroy.

## References

[1] J. Cohen, S. Kaur, Y. Li, J. Z. Kolter, and A. Talwalkar, "Gradient descent on neural networks typically occurs at the edge of stability," ICLR (2021).

[2] A. Damian, E. Nichani, and J. D. Lee, "Self-stabilization: the implicit bias of gradient descent at the edge of stability," ICLR (2023).

[3] S. Kalra, X. He, and M. Barkeshli, "Period doubling and the route to chaos at the edge of stability," arXiv (2023).

[4] A. Morales, S. Rosas-Guevara, and J. C. Toledo-Roy, "Lyapunov exponents during training of neural networks," Frontiers in Physics 12 (2024).

[5] D. Ruelle and F. Takens, "On the nature of turbulence," Communications in Mathematical Physics 20, 167–192 (1971).

[6] P. Grassberger and I. Procaccia, "Characterization of strange attractors," Physical Review Letters 50, 346–349 (1983).

[7] T. Züchner, H. Kantz, and others, "Finite-time Lyapunov exponents of deep neural networks," Physical Review Letters 132 (2024).

[8] A. N. Kolmogorov, "On the conservation of conditionally periodic motions under small perturbation of the Hamiltonian," Dokl. Akad. Nauk SSSR 98, 527–530 (1954); V. I. Arnold, Russ. Math. Surv. 18, 85 (1963); J. Moser, Nachr. Akad. Wiss. Göttingen II, 1 (1962).

[9] I. C. Percival, "Variational principles for invariant tori and cantori," AIP Conference Proceedings 57, 302–310 (1979).

[10] F. Pittorino, A. Ferraro, G. Perugini, C. Feinauer, and R. Zecchina, "Deep networks on toroids: removing symmetries reveals the structure of flat regions in the loss landscape," arXiv:2202.02038 (2022).

[11] arXiv:2502.20531, "Period-doubling in deep linear networks" (2025).

[12] C. Tralie, N. Saul, and R. Bar-On, "Ripser.py: A lean persistent homology library for Python," Journal of Open Source Software 3, 925 (2018).

---

## Figure Plan

**Figure 1** (four panels, matching existing cifar10_eos.png layout):

**(a) Chaos vs. Distance to EoS.** Lyapunov exponent vs. fraction of 2/λ_max. Shows the chaos window: positive λ at low fractions, crossing zero at ~45% EoS. Error bars from 10 seeds.

**(b) Correlation Dimension D₂ vs. Distance to EoS.** D₂ rises from ~1 through 2 to peak at 3.64 at 30% EoS. Horizontal reference lines at D = 1 (limit cycle) and D = 2 (torus). Error bars from 10 seeds.

**(c) Off-Axis Dynamics.** PC2 variance percentage, showing multi-dimensional trajectory structure peaking at 15% EoS.

**(d) Sharpness Dynamics.** Top Hessian eigenvalue time series at 5% EoS (monotonic climb) vs. 95% EoS (flat/oscillatory). Illustrates the mechanism: monotonic → coupled oscillatory modes.

**Supplemental figure:** MLP D₂ (flat at ~0.9) and depth scaling of chaos window width (Table I in text). Lorenz/torus D₂ validation. TDA persistence diagram examples (5% vs 30% EoS).

---

## Notes for revision

- [x] Insert 10-seed error bars (D₂ = 3.670 ± 0.083 at 30% EoS)
- [x] Insert Lorenz/torus validation numbers (2-torus: 2.06, Lorenz: 1.68)
- [x] Confirm λ_max value matches across seeds (6.7622 confirmed)
- [x] Correct Lyapunov method description (linregress, not single-point)
- [x] Correct correlation dimension description (fixed-index fitting, not adaptive)
- [x] Correct sharpness iterations (15, not 20)
- [x] Frame D₂ underestimate honestly — "conservative lower bound" (Methods, Results, Discussion)
- [x] Table I with architecture scaling results added
- [x] Robustness check mentioned (alternative D₂ pipeline, partial Colab results)
- [x] TDA results incorporated (fractal topology confirmed)
- [x] PhySH: Nonlinear Dynamics / Chaotic dynamics, Dynamical systems, Bifurcations
- [ ] Word count check and tighten
- [ ] Decide on author list and affiliations
- [ ] Generate updated Figure 1 with 10-seed error bars
- [ ] Convert to RevTeX format for submission

## Supplemental Material outline

- D₂ calibration against known attractors (Fig. S1 = d2_calibration.png, Table S1 = full calibration data)
- TDA persistence diagrams at 5%, 30%, 90% EoS (Fig. S2)
- H₁ and H₂ feature counts vs. EoS fraction (Fig. S3 = tda_cifar10.png)
- Alternative D₂ pipeline robustness check (partial results from adaptive fitting method)
- MLP depth scaling details (full Lyapunov curves, spectral flatness data)
- ReLU comparison results
- Perturbation sensitivity analysis (ε sweep)
- Full 10-seed data table
