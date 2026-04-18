# Strange Attractors in Gradient Descent — What This Paper Says and Why It Matters

## The Core Discovery

When you train a neural network, you're running a dynamical system. At each step, gradient descent takes the network's current function — the mapping from inputs to outputs — and moves it to a slightly different function. This creates a trajectory: a path through the space of all possible functions the network could compute.

This paper measures the *shape* of that trajectory. The central finding is that the trajectory doesn't just wander randomly or converge smoothly to a solution. Under the right conditions, it settles onto a **strange attractor** — a fractal geometric object with a measurable, non-integer dimension.

A strange attractor is the hallmark of deterministic chaos. The Lorenz attractor (the famous "butterfly" shape from weather modeling) is a strange attractor with fractal dimension ≈ 2.05. The paper finds that a convolutional network trained on CIFAR-10 produces a strange attractor with fractal dimension ≈ 3.7 at the measurement resolution used, converging to roughly D₂ ≈ 5 when measured with longer trajectories. This means the training dynamics are exploring a five-dimensional fractal volume in function space — far richer than a simple path toward a solution.

## What Is Being Measured, Concretely

The paper works in **function space**, not parameter space. This distinction matters enormously. Neural networks are overparameterized — many different settings of the weights produce the same input-output function. If you measured trajectories in weight space, you'd see apparent chaos that's actually just the network shuffling equivalent weight configurations (gauge symmetries). By recording the network's *outputs* on a fixed set of 100 test inputs at each training step, the paper tracks the trajectory of the actual learned function, avoiding these artifacts.

Three measurements characterize the dynamics:

**Correlation dimension (D₂):** This is the paper's primary metric. It measures how the trajectory fills space. A smooth curve has D₂ = 1. A surface has D₂ = 2. A strange attractor has a non-integer D₂ — the trajectory fills more than a curve but less than a surface (or more than a surface but less than a volume, etc.). The paper uses the Grassberger-Procaccia algorithm, the standard tool from nonlinear dynamics for estimating D₂ from time series data.

**Lyapunov exponent (λ):** This measures sensitivity to initial conditions — the defining feature of chaos. Two copies of the network are initialized identically except for a tiny perturbation (ε = 10⁻⁵). If the outputs of the two copies diverge exponentially over training, λ > 0 and the system is chaotic. If they converge, λ < 0 and the system is stable.

**Persistent homology:** This is a topological tool that counts loops, voids, and higher-dimensional holes in the trajectory. A smooth torus (like the surface of a donut) has a few dominant topological features. A strange attractor has many features at all scales with no single dominant one — the gap ratio (largest feature lifetime / second largest) stays near 1.0.

## The Experimental Design

Five conditions, carefully chosen to isolate what matters:

1. **CNN on CIFAR-10** (268K params) — the flagship condition. Real architecture, real data.
2. **MLP on CIFAR-10** (156K params, width 50) — different architecture, same data.
3. **MLP on CIFAR-10** (269K params, width 85) — same architecture as #2 but matching the CNN's parameter count.
4. **CNN on synthetic data** (268K params) — same architecture as #1, but structureless data.
5. **MLP on synthetic data** (14K params) — baseline control.

The synthetic data is 220-dimensional with 10 classes but has no internal structure — no spatial correlations, no hierarchical features. CIFAR-10 has rich structure: edges, textures, object parts, spatial relationships.

All experiments use **full-batch gradient descent** — every training step uses all 2,000 data points. This makes the dynamics purely deterministic: no randomness from mini-batch sampling. The learning rate is swept from 5% to 95% of the Edge of Stability threshold (the learning rate at which the top Hessian eigenvalue hits 2/η).

## The Five Key Findings

### 1. Training trajectories are strange attractors

On CIFAR-10, the correlation dimension crosses 1.0 at about 10% of the Edge of Stability threshold and peaks at D₂ = 3.67 ± 0.08 at 30% EoS for the CNN. This is a conservative lower bound from ~400 trajectory points; convergence analysis with 3,200 points shows the true dimension is approximately 5. The persistent homology confirms fractal topology: hundreds of loops and voids at all scales, with gap ratios near 1.0 — no dominant topological features, exactly the signature of a strange attractor.

The trajectory is not just noisy or erratic. It has a specific geometric structure — a fractal with a definite, measurable dimension. This is the same kind of object that appears in weather dynamics (Lorenz), fluid turbulence (Ruelle-Takens), and electronic circuits (Chua). Finding one in neural network training is new.

### 2. Data structure is necessary; architecture modulates the threshold

This is the cleanest result in the paper, and it comes from the cross-architecture controls:

- **CNN + synthetic data → D₂ ≈ 1** at all learning rates. The convolutional architecture by itself cannot create a multi-dimensional attractor. No matter how hard you drive it (high learning rate), the dynamics remain essentially one-dimensional.

- **MLP + CIFAR-10 → D₂ > 4** at high learning rates. A simple fully-connected network with no convolutional structure still produces a high-dimensional strange attractor, as long as it's trained on structured data.

- **CNN + CIFAR-10 → D₂ ≈ 3.7** at moderate learning rates, peaking much earlier than the MLPs.

The conclusion: structured data is both necessary and sufficient for multi-dimensional chaos. Architecture doesn't create the chaos — the data does. What architecture controls is the *threshold*: the CNN's hierarchical processing creates efficient coupling pathways, so the transition to chaos happens at a lower learning rate (30% of EoS vs 90% for the MLPs). But all architectures access the same ultimate attractor dimension when driven hard enough. The converged D₂(N=3200) values support this — both MLPs land near 5.1, and the CNN appears headed to the same neighborhood.

### 3. Two distinct mechanisms control attractor dimension

The label-noise sweep is the paper's most revealing experiment. Starting from clean CIFAR-10 labels, a fraction p of labels are randomized (p = 0 is clean, p = 1 is fully random). This is done at each architecture's peak-D₂ learning rate.

**At moderate learning rates (CNN at 30% EoS):** D₂ falls monotonically from 3.64 ± 0.08 to 2.34 ± 0.75 as noise increases. Degrading label structure weakens the oscillatory modes that the attractor is built from. At p = 1.0 (random labels), D₂ is reduced by a factor of 1.6 toward — but not to — the synthetic-data baseline, suggesting that input geometry contributes to the dynamics even when labels are random. This is the **data-structure mechanism**: structured input-label relationships create independent axes of oscillation in the network's function, and the attractor lives in the volume they span.

**Near the Edge of Stability (MLP at 90% EoS):** D₂ *rises* from 4.38 ± 0.07 to 4.70 ± 0.09 as noise increases. Random labels roughen the loss surface, creating competing minima and steeper curvature ridges. This roughness sustains high-dimensional dynamics even without label structure. This is the **loss-geometry mechanism**: the shape of the loss landscape itself provides the modes, regardless of whether the labels are meaningful.

The sign reversal across regimes is the paper's distinctive signature. It means there are two independent sources of dynamical complexity in neural network training, and they dominate in different regimes. Any theory of training dynamics must account for both.

### 4. Peak chaos and peak complexity occur at different learning rates

In every architecture tested, the Lyapunov exponent λ peaks at a *lower* learning rate than the correlation dimension D₂. For the CNN: peak λ at 15% EoS, peak D₂ at 30%. For the MLPs: peak λ at 40-50% EoS, peak D₂ at 90%.

This means the point of maximum sensitivity to perturbations is not the same as the point of maximum geometric complexity. The richest attractor structure appears in a transition zone where chaotic dynamics and basin convergence compete — where the system is being pulled toward a solution but the chaos hasn't been fully tamed. This dissociation is reminiscent of what happens in classical dynamical systems at the boundary between ordered and chaotic phases, where ordered and chaotic regions coexist and create complex fractal boundaries.

### 5. The attractor survives SGD

All the above uses full-batch gradient descent — no stochastic noise. The obvious question: does this matter in practice, where mini-batch SGD is universal?

The batch-size sweep from B = 2000 (full batch) down to B = 100 (5% of the data per step) shows the attractor is robust. Both the CNN and MLP retain more than 82% of their fractal dimension at the smallest batch size. Meanwhile, the Lyapunov exponent increases by 10× — stochastic noise amplifies sensitivity along the attractor's existing modes without creating new dynamical dimensions. The attractor is a deterministic skeleton that the stochastic dynamics ride on top of.

## The Geometry of It All

Picture the space of all possible functions a neural network could compute. This is an enormous space — each point represents a complete input-output mapping. When you train the network, the function moves through this space along a path dictated by gradient descent.

At low learning rates, this path is simple: a smooth curve converging toward a minimum. The function changes monotonically, loss goes down, and the trajectory is essentially one-dimensional (D₂ ≈ 1).

As the learning rate increases, something changes. The network starts to oscillate — not because it's failing to converge, but because the landscape geometry forces it to bounce between competing configurations. These oscillations aren't random; they're organized along specific directions in function space that correspond to different "modes" of the learned representation.

With structured data like CIFAR-10, these modes are created by the data itself. The network must simultaneously represent edges, textures, shapes, and their relationships across 10 different classes. Each of these representational demands creates an axis along which the training dynamics can oscillate. When enough modes are present and the learning rate is high enough to couple them, the trajectory becomes chaotic — it never exactly repeats, but it's confined to a bounded region of function space. The region it fills is the strange attractor.

The fractal dimension tells you how many independent directions the dynamics explore. D₂ ≈ 5 means the trajectory fills a roughly five-dimensional fractal volume. Half the variance in the trajectory comes from off-axis dynamics — components orthogonal to the primary convergence direction. The network isn't just finding a solution; it's perpetually exploring a neighborhood of solutions, jumping between slightly different ways of classifying the data.

On structureless synthetic data, these modes don't exist. The data provides no competing representational demands, so even at high learning rates, the dynamics remain one-dimensional. The network oscillates, but along a single axis. Architecture can't substitute for this — the CNN's convolutional hierarchy is optimized for extracting spatial features, but without spatial features in the data, it has nothing to extract.

## Significance and Implications

**For understanding training dynamics:** This is the first measurement of the fractal dimension of training trajectories. Previous work established that training can be chaotic (positive Lyapunov exponents) and that the Edge of Stability constrains sharpness. This paper adds a geometric dimension to that picture — literally. Chaos is not just "instability"; it has a specific shape, and that shape is controlled by the data.

**For the Edge of Stability:** The EoS phenomenon (the top Hessian eigenvalue rising to 2/η and oscillating) has been studied as a stability phenomenon. This paper shows it's also a *geometric* phenomenon — the bounded oscillation is the EoS regime is the mechanism that sustains the strange attractor. The attractor exists in the dynamical space that the EoS creates.

**For the role of data:** The clean separation between data-driven and architecture-driven effects is new. It suggests that the informational content of the dataset imprints itself on the geometry of the optimization process in a measurable way. This could eventually connect to questions about generalization: does exploring a higher-dimensional attractor during training lead to better solutions?

**For connections to physics:** The paper connects neural network training to the mathematical framework of nonlinear dynamics and chaos theory. Strange attractors, Lyapunov exponents, correlation dimensions, and routes to chaos are well-studied objects in physics. Finding them in gradient descent opens the door to applying decades of dynamical systems theory to understanding how neural networks learn.

**The open question:** The learning rates that produce strange attractors (roughly 10-90% of the EoS threshold) overlap with learning rates commonly used in practice. The paper measures the geometry but doesn't yet establish whether it matters functionally. Whether exploring a higher-dimensional attractor during training produces networks that generalize better — whether the chaos is useful — is the central question these measurements now make it possible to ask.

## What It Relates To

- **Cohen et al. (2021), Edge of Stability:** Showed that the top Hessian eigenvalue rises to 2/η during training. This paper characterizes the dynamics that occur in the EoS regime geometrically.

- **Ruelle & Takens (1971), Newhouse-Ruelle-Takens route to chaos:** The classical theory of how independent oscillatory modes can couple to produce strange attractors. The paper's finding — that data structure creates the modes and learning rate controls the coupling — is suggestive of this route, though the analogy is phenomenological (gradient descent isn't Hamiltonian).

- **Morales et al. (2024), Lyapunov exponents in training:** Measured positive Lyapunov exponents during training. This paper extends the characterization from "chaos exists" to "chaos has this specific fractal shape."

- **Storm et al. (2024), finite-time Lyapunov exponents:** Showed that Lyapunov exponents through network depth form coherent structures. This paper measures Lyapunov exponents through training *time* rather than network depth.

- **Pittorino et al. (2022), toroidal loss landscape:** Showed that after removing symmetries, the loss landscape has toroidal topology. The strange attractors in this paper are the dynamical objects that live on this topological structure.

- **KAM theory (Kolmogorov-Arnold-Moser):** The classical result about persistence of quasi-periodic orbits under perturbation. The dissociation between peak λ and peak D₂ is reminiscent of the KAM transition where ordered and chaotic regions coexist, though the analogy is not rigorous.

## Methodological Contributions

Beyond the scientific findings, the paper makes two methodological contributions:

1. **The ε-sensitivity finding:** Lyapunov exponent measurements at perturbation scales ε ≤ 10⁻⁸ are dominated by numerical noise and produce uniformly positive exponents regardless of the true dynamics. This has implications for prior work that used small perturbation scales.

2. **Function-space measurement protocol:** By measuring divergence in function space (network outputs on held-out inputs) rather than parameter space, the protocol avoids gauge symmetry artifacts from overparameterization. This makes the measurements genuinely about the learned function, not about incidental properties of the weight representation.
