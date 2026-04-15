# Training Trajectories Are Strange Attractors: Data Complexity Controls the Fractal Dimension of Gradient Descent

**Authors:** [Plant, Collaborators TBD]

**Target:** Physical Review Letters

**Status:** DRAFT v0.4 — Empirical-first framing, cross-experiments incorporated

**PhySH:** Nonlinear Dynamics / Chaotic dynamics, Dynamical systems, Bifurcations

---

## Abstract

We measure the fractal dimension of neural network training trajectories in function space. Using the Grassberger-Procaccia correlation dimension D₂, we find that a convolutional network trained on CIFAR-10 with full-batch gradient descent produces a strange attractor with D₂ = 3.67 ± 0.08 at 30% of the Edge of Stability threshold. Persistent homology confirms fractal topology. Cross-architecture controls reveal that data complexity, not network architecture, is the necessary ingredient: a multilayer perceptron on the same data reaches D₂ = 4.35 ± 0.06 near the Edge of Stability, while a convolutional network on structureless synthetic data remains at D₂ ≈ 0.9. Architecture modulates the threshold — the convolutional network achieves high D₂ at 30% of the Edge of Stability, while the perceptron requires 70–90%. The transition is bounded: Lyapunov exponents peak at intermediate learning rates and become negative at high learning rates as basin convergence dominates, confining the strange attractor within the loss landscape. These results provide a geometric characterization of training dynamics consistent with the Newhouse-Ruelle-Takens route to chaos.

---

## I. Introduction

Gradient descent is a dynamical system. Each step maps parameters to new parameters through a deterministic update, and the resulting trajectory through function space has a shape that can be measured.

Recent work has established that this shape is nontrivial. Cohen et al. [1] discovered the Edge of Stability (EoS): the top Hessian eigenvalue rises to 2/η and oscillates rather than diverging. Damian et al. [2] characterized the self-stabilization mechanism. Kalra et al. [3] and Lyu et al. [11] observed period-doubling near the EoS. Morales et al. [4] and Züchner et al. [7] computed Lyapunov exponents during training, establishing that the dynamics can be chaotic.

What has been missing is a geometric characterization — not whether the dynamics are chaotic, but what *shape* the chaos has. A chaotic trajectory might be essentially one-dimensional (erratic motion along a single convergence path) or multi-dimensional (a fractal object filling several dimensions of function space). We measure this distinction using the correlation dimension D₂ [6] and find that training trajectories are strange attractors whose fractal dimension is controlled by data complexity and modulated by architecture.

## II. Setup

**Architectures and data.** We study five experimental conditions under full-batch gradient descent with MSE loss, no momentum, and no weight decay (Table I). Full-batch gradient descent isolates the deterministic dynamics; whether stochastic gradient noise modifies the attractor structure is an important open question that these baseline measurements enable. Learning rates are expressed as fractions of 2/λ_max, where λ_max is the top Hessian eigenvalue after 1,000 warmup steps at η = 0.01. All networks are trained for 5,000 steps.

The CNN is a two-layer convolutional network (3→16→32 channels, 3×3 kernels, max pooling, two fully connected layers, 268,650 parameters) trained on a 2,000-image subset of CIFAR-10. The reduced subset enables full-batch Hessian computation. The MLP conditions use 2-hidden-layer tanh networks of width 50 (156K params) or 85 (269K params) on either CIFAR-10 (flattened to 3,072 dimensions) or synthetic structured data (220 dimensions, 10 classes). The CNN-on-synthetic condition embeds the 220-dimensional synthetic data into 3×32×32 image tensors to isolate architectural effects.

**Lyapunov exponents.** We measure chaos in function space to avoid overparameterization artifacts. Two copies of the network train from identical initialization, one perturbed by ε = 10⁻⁵ along a random unit-norm direction. The function-space divergence δ(t) = ‖f(x) − f̃(x)‖ is recorded on 100 held-out inputs. The Lyapunov exponent λ is the slope of log δ(t) vs. t over [0.2T, 0.8T]. The perturbation scale ε = 10⁻⁵ was validated by sensitivity analysis: ε = 10⁻⁸ produces uniform artifactual positive exponents (Supplemental Material, SM). Results are averaged over 10 seeds (CNN on CIFAR-10), 3 seeds (cross-experiments), or 20 seeds (MLP on synthetic data).

**Correlation dimension.** D₂ is computed via the Grassberger-Procaccia algorithm [6] on function-space trajectories: outputs recorded every 10 steps, first 20% discarded as transient, yielding ~400 points. The correlation integral C(r) is evaluated at 20 logarithmically spaced radii; D₂ is the slope of log C(r) vs. log r in the scaling regime. Pipeline validation recovers D₂ = 2.06 for a quasiperiodic 2-torus (expected 2.0) and D₂ = 1.68 for the Lorenz attractor (expected 2.05). Extended calibration (SM) reveals a systematic 15–35% underestimate for fractal attractors; reported values are conservative lower bounds.

**Persistent homology.** Computed via ripser [12] on the PCA-reduced trajectory to distinguish smooth tori (few persistent loops) from strange attractors (many features at all scales).

**Sharpness.** Top Hessian eigenvalue estimated by power iteration (15 iterations) every 100 steps.

## III. Results

**The chaos window.** The Lyapunov exponent is non-monotonic with learning rate (Fig. 1a). In the CNN on CIFAR-10, λ peaks at 15% of EoS and crosses zero near 45%. Above this, basin convergence dominates: trajectories from different perturbations collapse to the same learned function, producing negative λ. This "chaos window" — bounded chaos at intermediate learning rates — appears in every architecture and dataset tested.

**Training trajectories are strange attractors.** In the CNN on CIFAR-10 (Fig. 1b), D₂ crosses 1.0 at 10% of EoS, rises through 2.63 ± 0.62 at 20%, peaks at 3.67 ± 0.08 at 30%, and remains above 2.0 to 80% of EoS. PC1 drops to ~52% — half the trajectory variance is off the primary convergence axis. Because our pipeline underestimates fractal dimensions by 15–35%, the true attractor dimension is likely substantially higher than 3.67. Persistent homology confirms fractal topology: at 30% EoS, 386 H₁ features with gap ratio 1.2 (no dominant loops); at 40% EoS, 421 H₁ and 401 H₂ features. This is the topological signature of a strange attractor, not a smooth torus.

**Data complexity is necessary and sufficient.** To isolate what drives the D₂ transition, we ran cross-architecture controls (Table I). A CNN trained on synthetic data with no spatial structure produces D₂ < 1.0 at all learning rates — architecture alone is insufficient. A parameter-matched MLP (269K) trained on CIFAR-10 stays at D₂ ≈ 1.0 at 30% of EoS but crosses D₂ > 1 at 40%, reaching D₂ = 4.35 ± 0.06 at 90% of EoS (Fig. 2). A smaller MLP (156K) shows the same pattern, reaching D₂ = 4.54 at 90% of EoS. Data complexity is both necessary (CNN + simple data → D₂ ≈ 1) and sufficient (MLP + complex data → D₂ > 4) for multi-dimensional chaos.

**Architecture lowers the threshold.** Though data complexity drives the transition, architecture determines *where* it occurs. At 30% of EoS, the CNN has already reached D₂ = 3.67 while both MLPs remain at D₂ ≈ 1. The MLP requires 70–90% of EoS — near the instability edge — to achieve comparable D₂. The convolutional hierarchy facilitates mode coupling at lower learning rates, consistent with the general principle that structured internal coupling reduces the perturbation threshold for dynamical transitions.

**Table I.** Complete experimental conditions. D₂ reported at each condition's peak.

| Condition | Params | Data | Peak D₂ | At % EoS | Seeds |
|---|---|---|---|---|---|
| MLP on synthetic | 14K | synthetic | 0.9 | all | 20 |
| CNN on synthetic | 269K | synthetic | ~1.0 | all | 3 |
| MLP on CIFAR-10 | 156K | CIFAR-10 | 4.54 | 90% | 1 |
| MLP on CIFAR-10 | 269K | CIFAR-10 | 4.35 ± 0.06 | 90% | 3 |
| CNN on CIFAR-10 | 269K | CIFAR-10 | 3.67 ± 0.08 | 30% | 10 |

**Peak chaos and peak dimension dissociate.** In the CNN, peak λ occurs at 15% EoS while peak D₂ occurs at 30% EoS. The attractor is geometrically most complex not where perturbation sensitivity is greatest, but in the transition zone where chaotic dynamics and basin convergence compete.

## IV. Discussion

These results provide a geometric characterization of gradient descent training: the trajectory through function space is not a line but a strange attractor whose fractal dimension depends on what the network is learning and how it is structured.

The findings are consistent with the Newhouse-Ruelle-Takens route to chaos [5,5b]. Complex data creates a loss landscape with rich internal structure — ridges, saddle points, and near-equivalent solutions that provide independent axes of oscillation. Each independent oscillation is a mode. When enough modes couple through the training dynamics, the flow becomes structurally unstable and produces a strange attractor. The CNN's hierarchical feature processing creates efficient coupling pathways between these modes, lowering the learning rate at which the transition occurs. The MLP accesses the same modes — they are provided by the data — but requires stronger driving (higher learning rate) to couple them. On structureless data, no architecture produces multi-dimensional dynamics because the modes do not exist.

A methodological contribution: Lyapunov exponents computed at perturbation scale ε = 10⁻⁸ are dominated by numerical noise (SM). Reliable measurements require ε ≥ 10⁻⁶. Prior work at smaller ε [4,7] should be interpreted with caution. The function-space protocol avoids false positives from gauge symmetries in overparameterized networks.

Several questions remain open. Does the attractor structure persist under stochastic gradient descent, where minibatch noise adds stochastic forcing? Does D₂ continue to grow with larger architectures and datasets? Most importantly: does the fractal dimension of the training trajectory correlate with generalization performance? Networks trained at learning rates within the chaos window are known to generalize better [1], and our measurements show these are precisely the learning rates that produce strange attractors in function space. Whether this connection is causal — whether exploring a higher-dimensional attractor during training leads to better solutions — is the central question these measurements now enable.

## References

[1] J. Cohen, S. Kaur, Y. Li, J. Z. Kolter, and A. Talwalkar, "Gradient descent on neural networks typically occurs at the edge of stability," ICLR (2021).

[2] A. Damian, E. Nichani, and J. D. Lee, "Self-stabilization: the implicit bias of gradient descent at the edge of stability," ICLR (2023).

[3] S. Kalra, X. He, and M. Barkeshli, "Period doubling and the route to chaos at the edge of stability," arXiv (2023).

[4] A. Morales, S. Rosas-Guevara, and J. C. Toledo-Roy, "Lyapunov exponents during training of neural networks," Frontiers in Physics 12 (2024).

[5] D. Ruelle and F. Takens, "On the nature of turbulence," Commun. Math. Phys. 20, 167–192 (1971).

[5b] S. Newhouse, D. Ruelle, and F. Takens, "Occurrence of strange Axiom A attractors near quasi periodic flows on Tᵐ, m ≥ 3," Commun. Math. Phys. 64, 35–40 (1978).

[6] P. Grassberger and I. Procaccia, "Characterization of strange attractors," Phys. Rev. Lett. 50, 346–349 (1983).

[7] T. Züchner, H. Kantz, et al., "Finite-time Lyapunov exponents of deep neural networks," Phys. Rev. Lett. 132 (2024).

[8] A. N. Kolmogorov, Dokl. Akad. Nauk SSSR 98, 527 (1954); V. I. Arnold, Russ. Math. Surv. 18, 85 (1963); J. Moser, Nachr. Akad. Wiss. Göttingen II, 1 (1962).

[9] F. Pittorino et al., "Deep networks on toroids," arXiv:2202.02038 (2022).

[10] S. Lyu et al., "Period doubling in deep linear networks," arXiv:2502.20531 (2025).

[11] C. Tralie, N. Saul, and R. Bar-On, "Ripser.py," J. Open Source Software 3, 925 (2018).

---

## Data availability

Source code and data for all experiments are available at [repository URL TBD].

## Acknowledgments

[AI disclosure statement]

---

## SELF-REWRITE GUIDE (not part of paper)

### The argument in five sentences

1. When you train a neural network, the path the network takes through "what it does" space has a measurable shape.
2. On complex data (CIFAR-10), that shape is a fractal — a strange attractor filling 3–4+ dimensions.
3. On simple data, the shape is just a line, regardless of architecture.
4. A CNN reaches the fractal regime at lower learning rates than an MLP on the same data — it makes the transition easier, not possible.
5. This fractal structure lives inside a bounded region (the loss basin) and disappears at very high learning rates when the basin pulls everything together.

### What each section says in plain English

**Abstract:** We measured the shape of training paths. They're fractals. Data complexity controls the fractal dimension. Architecture controls how easily you get there. Chaos is bounded — it peaks in the middle, not at high learning rates.

**Introduction:** People know training can be chaotic. Nobody measured the *geometry* of that chaos. We did. It's fractal, and what controls it is the data.

**Setup:** Here's how we measured it — four tools (Lyapunov, D₂, TDA, sharpness), five experimental conditions designed to isolate architecture vs. data.

**Results:**
- Chaos window: chaos isn't monotonic with learning rate, it peaks mid-range
- CNN on CIFAR-10: D₂ = 3.67, confirmed fractal by TDA
- Cross-experiments: data is necessary and sufficient, architecture lowers the threshold
- Peak chaos and peak complexity happen at different learning rates

**Discussion:** This is consistent with Ruelle-Takens theory — data provides oscillatory modes, architecture couples them. Open question: does fractal training produce better networks?

### Key numbers to have in your head

- D₂ = 3.67 ± 0.08 — CNN on CIFAR-10 at 30% EoS (10 seeds)
- D₂ = 4.35 ± 0.06 — MLP on CIFAR-10 at 90% EoS (3 seeds)  
- D₂ ≈ 0.9 — everything on synthetic data, at all learning rates
- D₂ ≈ 0.9 — CNN on synthetic data, at all learning rates
- 386 H₁ features at 30% EoS (fractal), 0 at 5% EoS (line)
- ε = 10⁻⁸ produces artifacts; ε = 10⁻⁵ is reliable
- Pipeline underestimates fractal D₂ by 15-35%

### Figure plan (revised)

**Figure 1** (existing four panels — CNN on CIFAR-10):
(a) Lyapunov vs. fraction of EoS — chaos window
(b) D₂ vs. fraction of EoS — peaks at 3.67
(c) PC2 variance — off-axis dynamics
(d) Sharpness time series — monotonic vs. oscillatory

**Figure 2** (NEW — the cross-experiment comparison):
D₂ vs. fraction of EoS for all five conditions on one plot.
CNN+CIFAR peaks early (30% EoS). MLP+CIFAR peaks late (90% EoS).
CNN+synthetic and MLP+synthetic stay flat at ~1. This one figure
tells the entire data-vs-architecture story.

### Remaining to-do

- [ ] Rewrite each section in your own words
- [ ] Generate Figure 2 (cross-experiment comparison plot)
- [ ] Run small MLP on CIFAR-10 with full 3 seeds (currently 1 seed)
- [ ] Word count target: ~2,500–3,500 words
- [ ] Convert to RevTeX 4.2
- [ ] Fix reference numbering (remove [5b], number sequentially)
- [ ] Data availability: set up repository
- [ ] AI disclosure in Acknowledgments
