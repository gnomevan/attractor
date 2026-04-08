# Strange Attractors and Generalization: How Chaos in Training Finds Good Solutions

## Context — bring this to any new conversation

This document summarizes an active research program and a specific open question for further development. The empirical results described below are from completed experiments.

---

## What we've established

We have experimental evidence that neural network training dynamics follow the Ruelle-Takens route to chaos. The key findings:

**The measurement.** We computed the correlation dimension D₂ of the function-space trajectory during full-batch gradient descent training. Function space means: we track what the network *does* (its outputs on held-out data), not what its parameters are. This avoids artifacts from overparameterization.

**The result.** A CNN (268,650 parameters) trained on CIFAR-10 produces a training trajectory with D₂ = 3.6 at 30% of the Edge of Stability threshold — a strange attractor in function space. A smaller MLP (14,110 parameters) on synthetic data stays at D₂ ≈ 0.9 (one-dimensional) across all learning rates.

**The transition.** D₂ increases through 1, 2, 3+ as learning rate increases, mirroring the Ruelle-Takens sequence: limit cycle → torus → strange attractor. The mechanism is the coupling of sharpness oscillations (oscillations in the top Hessian eigenvalue) to the convergence dynamics. At low learning rates, sharpness climbs monotonically — one mode, one dimension. Near the Edge of Stability, sharpness oscillates — multiple coupled modes, multi-dimensional dynamics.

**The chaos window.** Chaos (positive Lyapunov exponents) is non-monotonic with learning rate. It peaks mid-range and goes strongly negative at high learning rates, where basin convergence dominates. All seeds converge to functionally identical solutions despite chaotically different trajectories. Higher learning rates produce *more* similar final functions (seed-to-seed correlation increases from 0.987 to 0.9999).

**The dissociation.** Peak chaos (strongest Lyapunov exponent) occurs at 15% of EoS. Peak geometric complexity (highest D₂) occurs at 30% of EoS. The attractor is most complex not where chaos is strongest, but at the boundary between the chaotic and basin-convergent regimes — where ordered (toroidal) and chaotic regions coexist, fractally interleaved. This is the KAM transition regime.

---

## The open question: what does the strange attractor have to do with finding good solutions?

### The empirical puzzle

It is well established in deep learning practice that:

1. **Higher learning rates produce better generalization.** Networks trained with larger learning rates (within the trainable range) perform better on unseen data. This is widely observed but not fully explained.

2. **The Edge of Stability produces flat minima.** Cohen et al. (2021) showed that near the EoS threshold, sharpness self-stabilizes — the top Hessian eigenvalue hovers around 2/η rather than growing without bound. Damian et al. (2023) characterized this as an implicit bias toward flat minima (low curvature). Flat minima are associated with better generalization (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017).

3. **Despite chaotic trajectories, all seeds converge to the same function.** Our experiments show that the chaos doesn't prevent convergence — it's a transient phenomenon. The basin of attraction is strong enough to pull all trajectories to the same final function.

### The proposed connection

The strange attractor during training is not an obstacle to finding good solutions — it may be part of *how* good solutions are found. The argument:

**Exploration.** A one-dimensional trajectory (D₂ ≈ 1) converges along a single path. It finds *a* minimum but explores nothing off-axis. A multi-dimensional trajectory (D₂ ≈ 3.6) explores a higher-dimensional region of function space during training. The strange attractor geometry means the trajectory visits a richer set of candidate functions before settling.

**The sharpness oscillation mechanism.** The EoS self-stabilization mechanism works by intermittently destabilizing the trajectory when sharpness gets too high, then reconverging when it drops. Each destabilization kicks the trajectory off the current convergence path. Each reconvergence pulls it back toward a minimum — but potentially a *different* one, or the same one approached from a different direction. This is a natural exploration-exploitation mechanism, and it's the same mechanism that produces the multi-dimensional dynamics we measured.

**Flat minima selection.** The EoS mechanism specifically penalizes sharp minima (high curvature). A trajectory that enters a sharp minimum has its sharpness driven up, which triggers the self-stabilization mechanism, which kicks it out. A trajectory that enters a flat minimum doesn't trigger this mechanism and stays. The strange attractor dynamics are the geometry of this selection process: the trajectory bounces between candidate solutions, preferentially escaping sharp ones and settling into flat ones.

**The chaos window as an exploration budget.** The non-monotonic Lyapunov curve (chaos peaks then declines) means there's an optimal learning rate range for exploration. Too low: the trajectory converges along a single path to the nearest minimum (possibly sharp). Too high: basin convergence dominates immediately and the trajectory collapses to the global structure of the loss landscape without exploring local structure. In between (the chaos window): the trajectory explores multi-dimensionally before converging. The width of this window increases with architecture depth — deeper networks get a larger exploration budget.

### What this would predict

If the strange attractor dynamics are functionally connected to generalization:

1. **Test accuracy should correlate with D₂ during training.** Learning rates that produce higher transient D₂ should produce better-generalizing final networks. Testable with our existing setup by adding test-set evaluation at convergence.

2. **The optimal learning rate for generalization should fall within the chaos window.** Not at the peak of chaos (where the trajectory is too unstable) and not above the window (where exploration is suppressed), but in the transition zone — around 20-40% of EoS where D₂ is highest.

3. **Depth should improve generalization partly through wider chaos windows.** Our experiments show that deeper networks have wider chaos windows (more LR range with positive Lyapunov exponents). If the chaos window enables exploration, deeper networks benefit from a larger exploration budget, independent of their representational capacity.

4. **SGD noise and the strange attractor should interact.** Our experiments use full-batch gradient descent (no stochasticity). Real training uses mini-batch SGD, which adds noise. The question: does SGD noise substitute for the deterministic chaos, supplement it, or interfere with it? If the strange attractor geometry is important, full-batch training at EoS-regime learning rates might generalize comparably to SGD — a testable and surprising prediction.

### Relationship to existing theory

**Implicit bias literature.** There's extensive work on the implicit biases of gradient descent — how the optimizer's dynamics favor certain solutions over others (Neyshabur et al., 2017; Gunasekar et al., 2018). The strange attractor connection would add a geometric dimension to this: the implicit bias isn't just about which minimum is reached, but about the geometry of the exploration process that selects it.

**Loss landscape topology.** Pittorino et al. (2022) found toroidal topology in the loss landscape itself (parameter space, not function space). Our finding of toroidal/strange attractor dynamics in function space is complementary — the trajectory's geometry reflects the landscape's topology.

**Sharpness-aware minimization (SAM).** Foret et al. (2021) introduced SAM, which explicitly seeks flat minima by penalizing sharpness. The EoS mechanism achieves something similar implicitly. Our framework suggests that SAM works because it mimics the natural geometry of the chaos window — forcing multi-dimensional exploration that preferentially settles in flat regions.

---

## Connection to the torus framework

This connects to the broader theoretical project arguing that nested periodic processes generate toroidal geometry, and that toruses naturally transition to fractals under stress, coupling, or nesting.

In this context: the training dynamics are a concrete example of the torus-to-fractal transition playing out in a computational system. The "answer" (the learned function, the minimum) is the *center of the torus* — the attractor that the dynamics orbit around without occupying. The strange attractor geometry during training is the system's way of exploring the relationship to that center from multiple directions before converging.

The uninhabitable center principle from the core theory paper applies directly: the optimal solution is the axis that the training dynamics orbit around. The orbit is not failed arrival — it is the process by which the solution is found. The higher the D₂, the more dimensions of approach are explored, and the more robust the final convergence.

---

## Key references

- Cohen, Kaur, Li, Kolter, Talwalkar (2021). "Gradient descent on neural networks typically occurs at the edge of stability." ICLR.
- Damian, Nichani, Lee (2023). "Self-stabilization: the implicit bias of gradient descent at the edge of stability." ICLR.
- Hochreiter & Schmidhuber (1997). "Flat minima." Neural Computation.
- Keskar, Mudigere, Nocedal, Smelyanskiy, Tang (2017). "On large-batch training for deep learning: Generalization gap and sharp minima." ICLR.
- Foret, Kleiner, Mobahi, Neyshabur (2021). "Sharpness-aware minimization for efficiently improving generalization." ICLR.
- Pittorino, Ferraro, Perugini, Feinauer, Zecchina (2022). "Deep networks on toroids." arXiv:2202.02038.
- Ruelle & Takens (1971). "On the nature of turbulence." Commun. Math. Phys.

## Experimental status

- PRL paper on the D₂ measurement is in draft, awaiting 10-seed error bars from ongoing GPU runs
- D₂ pipeline calibrated against known attractors (Hénon, Lorenz, Rössler, Mackey-Glass τ=17–50)
- TDA experiment (persistent homology of the CNN trajectory) is next, to distinguish smooth torus from fractal attractor directly
- The generalization connection described above is untested — experiments needed
