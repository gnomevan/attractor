# Strange Attractors in Gradient Descent: Data Structure and Loss Geometry Control Fractal Dimension

**Evan Paul**
Independent Researcher — evan@evanpaul.us

---

## Abstract

We measure the fractal dimension of neural network training trajectories — not in the space of weights, but in the space of the functions the network actually computes. A convolutional network trained on CIFAR-10 under full-batch gradient descent produces a *strange attractor*: a fractal geometric object with correlation dimension D₂ = 3.67 ± 0.08 (a conservative lower bound; convergence analysis yields D₂ ≈ 5). Cross-architecture controls show that structured data is necessary for this multi-dimensional chaos — either architecture on synthetic data gives D₂ ≈ 1 — while architecture modulates the learning-rate threshold at which the transition occurs. A label-noise sweep reveals two distinct mechanisms: at moderate learning rates, degrading label structure reduces D₂ from 3.6 to 2.3 as data-created oscillatory modes are weakened; near the Edge of Stability, loss-surface roughness from random labels sustains high-dimensional dynamics independently. Peak perturbation sensitivity and peak geometric complexity occur at different learning rates across all architectures, with the richest structure emerging where chaotic and convergent dynamics compete. The attractor survives stochastic gradient descent, retaining >82% of its fractal dimension at 20× gradient noise.

---

## Introduction

Gradient descent is a deterministic dynamical system. Each training step takes the network's current input-output function and maps it to a new one. This produces a *trajectory* — a path through the space of all functions the network could represent — whose geometry we can measure directly.

Recent work has established that this geometry is far from trivial. Cohen et al. [1] discovered the **Edge of Stability** (EoS): as training progresses, the sharpness of the loss landscape (technically, the top eigenvalue of the Hessian matrix) rises until it hits a critical threshold of 2/η (where η is the learning rate), then oscillates at that boundary rather than diverging. Damian et al. [2] characterized the self-stabilization mechanism behind this. Kalra et al. [3] and Ghosh et al. [4] observed period-doubling cascades near the EoS — the same route to chaos seen in logistic maps and dripping faucets. Morales et al. [5] computed Lyapunov exponents during training, establishing that the dynamics can be genuinely chaotic (nearby trajectories diverge exponentially). Storm et al. [6] showed that finite-time Lyapunov exponents through the depth of a network form coherent structures across input space.

These results tell us training dynamics are rich, but they leave open a geometric question: **what *shape* does the chaos have?** A chaotic trajectory might be essentially one-dimensional — erratic but confined to a single path through function space — or it might fill a higher-dimensional fractal volume the way the Lorenz attractor fills a butterfly-shaped region in three dimensions. We measure this distinction using the **correlation dimension** D₂ [7], the standard tool from nonlinear dynamics for quantifying the fractal structure of attractors.

We find that training trajectories settle onto strange attractors whose fractal dimension is shaped by two mechanisms: structured data creates oscillatory modes at moderate learning rates, and loss-surface roughness sustains high-dimensional chaos near the Edge of Stability. By *data complexity* we mean the degree to which the input-label mapping requires the network to maintain multiple competing internal representations — operationally, the mutual information between inputs and labels modulated by the geometric structure of the input distribution.

---

## Experimental Framework

### Architectures and data

We study five experimental conditions under full-batch gradient descent (every training step uses all 2,000 data points) with MSE loss, no momentum, and no weight decay. Full-batch training is essential because it makes the dynamics purely deterministic — there is no randomness from mini-batch sampling, so any chaos we observe is a property of gradient descent itself, not of stochastic noise.

Learning rates are expressed as fractions of the Edge of Stability threshold 2/λ_max, where λ_max is the top Hessian eigenvalue measured after a 1,000-step warmup at η = 0.01. This normalization lets us compare across architectures on a common scale: 10% of EoS is "mildly driven," 90% is "near the stability boundary." All networks are trained for 5,000 steps.

The five conditions:

| Condition | Architecture | Parameters | Data | Purpose |
|-----------|-------------|-----------|------|---------|
| CNN/CIFAR | 2-layer CNN (3→16→32 channels, 3×3 kernels, max pool, ReLU, 2 FC layers) | 268,650 | CIFAR-10 (2,000 images) | Flagship: real architecture, real data |
| MLP/CIFAR (small) | 2-layer tanh MLP, width 50 | 156,660 | CIFAR-10 (flattened to 3,072-d) | Different architecture, same data |
| MLP/CIFAR (large) | 2-layer tanh MLP, width 85 | 269,195 | CIFAR-10 (flattened) | Parameter-matched to CNN |
| CNN/synthetic | Same CNN | 268,650 | Synthetic (220-d, 10 classes, zero-padded to 3×32×32) | Same architecture, structureless data |
| MLP/synthetic | 2-layer tanh MLP, width 50 | 14,060 | Synthetic (220-d, 10 classes) | Baseline control |

The synthetic data has no internal structure — no spatial correlations, no edges, no textures. CIFAR-10 has rich structure: objects, backgrounds, spatial hierarchies. By crossing architectures with data types, we can separate data-driven effects from architecture-driven effects.

### Measuring chaos: Lyapunov exponents in function space

To detect chaos, we measure whether tiny perturbations grow or shrink during training. We train two copies of the network from the same initialization, except one is perturbed by ε = 10⁻⁵ along a random direction in weight space. Then we track how their *outputs* on 100 held-out test inputs diverge over training. If the outputs diverge exponentially (the Lyapunov exponent λ > 0), the dynamics are chaotic. If they converge (λ < 0), the dynamics are stable.

**Why function space matters:** Neural networks are massively overparameterized — many different weight configurations produce the same input-output function. If we measured divergence in weight space, we'd see apparent chaos that's really just the two copies shuffling among equivalent weight configurations (gauge symmetries). By measuring in function space — the network's actual outputs — we avoid these artifacts and capture only genuine dynamical sensitivity.

**A methodological caution:** We found that at perturbation scales ε = 10⁻⁸ and below, the measured Lyapunov exponent is uniformly positive regardless of the true dynamics — a numerical artifact from finite-precision arithmetic. Reliable measurements require ε ≥ 10⁻⁶. All our results use ε = 10⁻⁵. This procedure measures a finite-scale, finite-time divergence rate rather than a true Lyapunov exponent (which requires ε → 0 and T → ∞); the ε-dependence documented in our supplemental material implies systematic uncertainty of approximately a factor of 3 beyond the seed-to-seed variability we report. Results are averaged over 10 seeds (CNN/CIFAR-10, MLP/CIFAR-10, CNN/synthetic) or 20 seeds (MLP/synthetic).

### Measuring fractal dimension: the Grassberger-Procaccia algorithm

The correlation dimension D₂ quantifies how a trajectory fills space. We record the network's outputs on 100 test inputs every 10 training steps, discard the first 20% as transient, and compute D₂ using the standard Grassberger-Procaccia algorithm [7]. This gives approximately 400 trajectory points. The algorithm works by counting how many pairs of points lie within distance r of each other, then examining how that count scales as r changes — the scaling exponent is D₂.

**Calibration:** We validated the pipeline on systems with known dimensions. It recovers D₂ = 2.18 for a quasiperiodic 2-torus (expected: 2.0) and D₂ = 1.73 for the Lorenz attractor (expected: 2.05). Extended calibration reveals a systematic 15–35% underestimate for fractal attractors, growing with the true dimension. A convergence analysis extending trajectories to N = 3,200 points confirms that our reported values at N ≈ 400 are conservative lower bounds that have not yet saturated.

At high learning rates where the loss-landscape curvature may not have fully equilibrated within 5,000 steps, the measurements characterize the quasi-stationary dynamics of the approach to equilibrium rather than an asymptotic attractor.

### Measuring topology: persistent homology

We also compute persistent homology (using ripser [8]) on PCA-reduced trajectories. This counts topological features — loops (H₁) and voids (H₂) — at all scales. A smooth torus has a few dominant loops with a large gap between the most persistent and the rest (high "gap ratio"). A strange attractor has many features at all scales with no single dominant one (gap ratio ≈ 1.0). This provides a complementary diagnostic: D₂ measures geometry, persistent homology measures topology.

---

## Results

### The chaos window

The Lyapunov exponent is non-monotonic with learning rate. In the CNN on CIFAR-10, λ peaks at 15% of the EoS threshold and crosses zero near 45%. Below this range, perturbations decay — the network converges smoothly. Above this range, basin convergence dominates — trajectories from different perturbations collapse to the same learned function. This **bounded chaos window** appears in every architecture tested; its width scales with architecture complexity.

*This means chaos in training is not a pathology of extreme learning rates. It occupies a specific, bounded range — and that range overlaps with learning rates commonly used in practice.*

### Training trajectories are strange attractors

Within the chaos window, the CNN/CIFAR-10 trajectory acquires fractal structure. D₂ crosses 1.0 at 10% of EoS, rises through 2.63 ± 0.62 at 20%, peaks at **3.67 ± 0.08 at 30%**, and remains above 2.0 out to 80% of EoS. The first principal component captures only ~52% of trajectory variance — half the dynamics are off the primary convergence axis.

A convergence analysis extending trajectories to N = 3,200 points shows that D₂ has not yet saturated at our production length of N ≈ 400: both MLPs stabilize near **D₂ ≈ 5.1**, confirming that our reported values are conservative lower bounds. Accounting for the 15–35% systematic underestimate from calibration, the true attractor dimensions for CIFAR-10 conditions likely lie in the range **D₂ ≈ 4–7**.

Persistent homology confirms fractal topology. For the CNN at 30% of EoS, 386 H₁ features (loops) appear with a gap ratio of 1.2 — no single dominant topological structure. At 5% of EoS: zero features, a simple convergence path. Both MLPs on CIFAR-10 show the same transition: zero features below 30% EoS, rising to comparable counts at 90% EoS. The gap ratio remains near 1.0 in every condition — the topological signature of a strange attractor, not a smooth torus.

*In plain terms: the network's function isn't converging to a single solution. It's perpetually exploring a fractal neighborhood of solutions — a roughly five-dimensional volume in function space — in a structured, chaotic pattern that never exactly repeats.*

### Structured data is necessary; architecture modulates the threshold

The cross-architecture controls reveal a clean separation:

**A CNN on structureless synthetic data produces D₂ < 1.0 at all learning rates.** The convolutional architecture alone is insufficient. No matter how high the learning rate, without structured data the dynamics remain essentially one-dimensional.

**Both MLPs on CIFAR-10 reach D₂ > 4 at 90% of EoS**, converging to D₂ ≈ 5.1 at longer trajectory lengths. A simple fully-connected network with no special structure produces a high-dimensional strange attractor — as long as it's trained on structured data.

The small difference between the 156K and 269K MLPs at N ≈ 400 (4.61 ± 0.10 vs 4.37 ± 0.05) has not converged and should not be over-interpreted; both reach ~5.1 at longer trajectory lengths.

**Structured data is both necessary** (CNN + synthetic → D₂ ≈ 1) **and sufficient** (MLP + CIFAR-10 → D₂ > 4) **for multi-dimensional chaos.**

Architecture modulates the *threshold*: at 30% of EoS the CNN has already reached D₂ = 3.67 while both MLPs remain at D₂ ≈ 1. The MLPs require 70–90% of EoS to achieve comparable fractal dimension. The convolutional hierarchy facilitates mode coupling at lower learning rates — it's better at extracting the structure that creates the chaos — but it doesn't create that structure.

### Two mechanisms control attractor dimension

A label-noise sweep from p = 0 (clean CIFAR-10 labels) to p = 1 (fully random labels) at each architecture's peak-D₂ learning rate reveals that fractal dimension is shaped by two distinct mechanisms depending on the dynamical regime.

**Mechanism 1 — Data structure creates modes (moderate learning rates):** For the CNN at 30% EoS, D₂ falls monotonically from 3.64 ± 0.08 to 2.34 ± 0.75 as label noise increases. Degrading the relationship between images and labels weakens the oscillatory modes that the attractor is built from. At fully random labels, D₂ is reduced by a factor of 1.6 toward — but not to — the synthetic-data baseline, suggesting that the geometric structure of the inputs contributes to the dynamics even when labels are random.

**Mechanism 2 — Loss roughness sustains modes (near the Edge of Stability):** For the MLP at 90% EoS, the opposite happens: D₂ *rises* modestly from 4.38 ± 0.07 to 4.70 ± 0.09 as label noise increases. Random labels roughen the loss surface — they create more competing local minima and steeper curvature ridges — and this roughness sustains high-dimensional dynamics even without meaningful label structure.

**The sign of D₂'s response to label noise reverses across regimes.** This is the paper's most distinctive finding: two independent sources of dynamical complexity operating in different parts of the learning-rate landscape. Any theoretical account of training dynamics must explain both mechanisms.

### The attractor survives stochastic gradient descent

All preceding measurements use full-batch gradient descent. We test robustness by sweeping batch size B from 2,000 (full batch) down to 100 (5% of the training set per step) for both the CNN at 30% EoS and the MLP at 90% EoS.

Both the original and perturbed model copies see identical mini-batches at each step (a "paired-noise" protocol), so the Lyapunov measurement isolates deterministic sensitivity from stochastic forcing.

The attractor dimension degrades gracefully: the CNN retains 87% of its full-batch D₂ at B = 100 (3.20 ± 0.05 vs 3.67 ± 0.10); the MLP retains 82% (3.59 ± 0.08 vs 4.37 ± 0.01). Both remain well above D₂ = 3 at all batch sizes tested.

Meanwhile, the Lyapunov exponent *increases* by 10× under SGD — stochastic forcing amplifies perturbation sensitivity along the attractor's existing modes without creating new dynamical dimensions. The attractor is a deterministic skeleton; SGD shakes it harder but doesn't reshape it.

### Peak chaos and peak complexity occur at different learning rates

For the CNN on CIFAR-10, peak λ (maximum perturbation sensitivity) occurs at 15% of EoS, while peak D₂ (maximum geometric complexity) appears at 30%. The same dissociation holds across all architectures: MLP 156K peaks at λ = 40% vs D₂ = 90%; MLP 269K at λ = 50% vs D₂ = 90%.

In every case, the attractor is geometrically most complex not where perturbation sensitivity is greatest, but in the **transition zone where chaotic dynamics and basin convergence compete**. This phenomenology is reminiscent of transitions in near-integrable systems [9], where ordered and chaotic regions coexist and the most intricate fractal boundaries form at their interface — though we note that gradient descent is not Hamiltonian and the analogy is phenomenological rather than rigorous.

---

## Summary of experimental conditions

| Condition | Params | Data | D₂ (N≈400) | D₂ (N=3200) | Peak at | Seeds |
|-----------|--------|------|------------|-------------|---------|-------|
| MLP/synthetic | 14K | synthetic | 0.9 | — | all | 20 |
| CNN/synthetic | 269K | synthetic | 0.98 ± 0.03 | — | 90% EoS | 10 |
| MLP/CIFAR (156K) | 156K | CIFAR-10 | 4.61 ± 0.10 | ~5.1 | 90% EoS | 10 |
| MLP/CIFAR (269K) | 269K | CIFAR-10 | 4.37 ± 0.05 | ~5.1 | 90% EoS | 10 |
| CNN/CIFAR | 269K | CIFAR-10 | 3.67 ± 0.08 | ~5* | 30% EoS | 10 |

*CNN convergence from single seed; noisier than MLP conditions.

---

## Discussion

These results provide a geometric characterization of gradient descent dynamics. Training trajectories settle onto strange attractors whose fractal dimension is shaped by two mechanisms: at moderate learning rates, structured data creates oscillatory modes that fill a multi-dimensional volume; near the Edge of Stability, loss-surface roughness — whether from structured or random labels — sustains high-dimensional chaos through competing minima. Architecture modulates the learning-rate threshold at which the transition occurs but does not determine the attractor's destination.

The label-noise sweep provides the clearest evidence for these two mechanisms. The CNN at 30% of EoS shows the data-structure mechanism in isolation: degrading labels smoothly weakens the oscillatory modes, reducing D₂ by a factor of 1.6 toward — but not to — the synthetic baseline. The MLP at 90% of EoS reveals the loss-geometry mechanism: even random labels sustain high-dimensional dynamics because the rougher loss surface provides its own competing modes. The sign reversal of D₂'s response to label noise across regimes is a distinctive signature that any theoretical account of these dynamics must explain.

The findings are suggestive of classical routes to chaos such as Newhouse-Ruelle-Takens [10, 11]: complex data or rough landscapes create independent axes of oscillation, and when enough modes couple through the training dynamics, the flow produces a strange attractor. The bounded chaos window and the universality of the λ–D₂ dissociation across architectures — each showing peak geometric complexity offset from peak sensitivity — are reminiscent of the phenomenology of transitions in near-integrable systems [9], where ordered and chaotic regions coexist. However, gradient descent is not Hamiltonian, and these analogies are phenomenological rather than rigorous identifications. The CNN's hierarchical processing creates efficient coupling pathways, lowering the learning rate at which the transition occurs. The MLP accesses the same modes — provided by the data — but requires stronger driving to couple them. On structureless data, no architecture produces multi-dimensional dynamics because the modes do not exist.

These dynamical findings complement Pittorino et al. [12], who showed that the loss landscape has toroidal topology after symmetry removal — a static geometric property of the parameter space on which these dynamical transitions unfold.

The batch-size sweep establishes that the strange attractor is not an artifact of full-batch training. Reducing B from 2,000 to 100 introduces substantial stochastic forcing, yet both architectures retain >82% of their fractal dimension. The order-of-magnitude increase in λ under SGD while D₂ changes modestly reveals a separation of scales: gradient noise amplifies sensitivity along the attractor's existing modes without generating new dynamical dimensions. This is consistent with the stochastic forcing acting as a perturbation to the deterministic skeleton rather than restructuring the dynamics.

Several questions remain open. Does D₂ continue to grow with larger architectures and datasets? The learning rates that produce strange attractors overlap with those commonly used in practice [1]. Whether this connection is causal — whether exploring a higher-dimensional attractor during training produces better solutions — is the central question these measurements now enable.

---

## Acknowledgments

AI language models (Claude, Anthropic) were used as research tools throughout this work, including assistance with experimental design, code development, data analysis, and manuscript preparation. All research directions, experimental choices, and scientific conclusions are the sole responsibility of the author. Source code, raw data, and analysis scripts are available at https://github.com/gnomevan/attractor.

---

## References

[1] Cohen, J., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2021). Gradient descent on neural networks typically occurs at the edge of stability. *ICLR 2021*.

[2] Damian, A., Nichani, E., & Lee, J. D. (2023). Self-stabilization: The implicit bias of gradient descent at the edge of stability. *ICLR 2023*.

[3] Kalra, D. S., He, T., & Barkeshli, M. (2025). Universal sharpness dynamics in neural network training: Fixed point analysis, edge of stability, and route to chaos. *ICLR 2025*. arXiv:2311.02076.

[4] Ghosh, A., Kwon, S. M., Wang, R., Ravishankar, S., & Qu, Q. (2025). Learning dynamics of deep linear networks beyond the edge of stability. *ICLR 2025*. arXiv:2502.20531.

[5] Morales, A., Rosas-Guevara, Y., & Toledo-Roy, J. C. (2024). Lyapunov exponents during training of neural networks. *Frontiers in Physics*, 12.

[6] Storm, L., Linander, H., Bec, J., Gustavsson, K., & Mehlig, B. (2024). Finite-time Lyapunov exponents of deep neural networks. *Phys. Rev. Lett.*, 132, 057301.

[7] Grassberger, P. & Procaccia, I. (1983). Characterization of strange attractors. *Phys. Rev. Lett.*, 50, 346–349.

[8] Tralie, C., Saul, N., & Bar-On, R. (2018). Ripser.py: A lean persistent homology library for Python. *J. Open Source Softw.*, 3, 925.

[9] Kolmogorov, A. N. (1954). On the conservation of conditionally periodic motions under small perturbation of the Hamiltonian. *Dokl. Akad. Nauk SSSR*, 98, 527–530. See also Arnold (1963) and Moser (1962).

[10] Ruelle, D. & Takens, F. (1971). On the nature of turbulence. *Commun. Math. Phys.*, 20, 167–192.

[11] Newhouse, S., Ruelle, D., & Takens, F. (1978). Occurrence of strange Axiom A attractors near quasi periodic flows on T^m, m ≥ 3. *Commun. Math. Phys.*, 64, 35–40.

[12] Pittorino, F., Ferraro, A., Perugini, G., Feinauer, C., Baldassi, C., & Zecchina, R. (2022). Deep networks on toroids: Removing symmetries reveals the structure of flat regions in the landscape geometry. *ICML 2022*, PMLR 162, 17759–17781.
