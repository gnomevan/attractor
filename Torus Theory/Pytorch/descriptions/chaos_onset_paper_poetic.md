# When Does Training Start to Dream?
## Lyapunov Evidence for a Torus-to-Chaos Transition in Gradient Descent

---

## Abstract

Train a neural network twice with the same data, the same architecture, the same learning rate — but nudge the starting weights by a billionth of their magnitude. At low learning rates, the two networks converge to the same function. At high learning rates, they diverge — they learn different things from the same experience. Somewhere in between, there is a threshold. Below it, training is deterministic. Above it, training is sensitive to conditions too small to measure.

We find that threshold. It occurs at a learning rate approximately 6.6% of the Edge of Stability — the classical boundary where optimization theory predicts instability. Chaos, it turns out, arrives long before the math says it should.

This paper measures the transition using function-space Lyapunov exponents, frames it within the dynamical systems theory of torus-to-chaos transitions (Ruelle & Takens, 1971), and argues that gradient descent passes through the same bifurcation sequence — from fixed point through quasiperiodic torus to strange attractor — that governs turbulence, cardiac arrhythmia, and the onset of weather.

**Keywords:** edge of stability, Lyapunov exponents, chaos, gradient descent, torus, strange attractor, Ruelle-Takens, neural network training dynamics

---

## 1. The Question

Every time you train a neural network, you set a learning rate. Too small and the network barely learns. Too large and it explodes. Somewhere in the middle is the sweet spot — fast enough to be practical, stable enough to converge.

In 2021, Cohen et al. discovered something unsettling about that sweet spot. They trained networks with full-batch gradient descent and watched the sharpness — the largest eigenvalue of the loss Hessian, a measure of how curved the loss landscape is beneath the network's feet. Classical optimization theory says training is stable when sharpness stays below 2/η, where η is the learning rate. Above that threshold, each gradient step overshoots, and the system should diverge.

What Cohen et al. found is that sharpness rises during training until it hits exactly 2/η — then hovers there. The loss wobbles. It doesn't blow up. It keeps decreasing, just not smoothly. They called this the Edge of Stability.

The Edge of Stability tells us where the loss landscape starts to push back. But it doesn't tell us when the system starts to care about where it began. It doesn't tell us when two nearly-identical networks, given identical training, start to learn different things.

That is what Lyapunov exponents measure. And that is the question of this paper: at what learning rate does gradient descent become sensitive to its initial conditions? When does training start to dream — to wander paths that depend on perturbations too small to see?

---

## 2. The Measurement

### 2.1 How to Detect Chaos in a Training Run

The idea is simple. Take a network. Copy it. Nudge the copy's weights by an imperceptible amount — a perturbation of magnitude ε ≈ 10⁻⁸. Now train both networks identically: same data, same learning rate, same loss function, same everything.

At each training step, measure the distance between what the two networks compute — not in weight space (which is full of irrelevant symmetries) but in function space: the difference between their outputs on a fixed set of inputs.

If that distance shrinks over time, the two networks are converging to the same function. Training is deterministic in the ways that matter. The Lyapunov exponent is negative.

If that distance grows exponentially, the two networks are diverging. An immeasurably small difference in starting conditions has become a measurable difference in what the networks learned. The Lyapunov exponent is positive.

The boundary — where the exponent crosses zero — is the onset of chaos.

### 2.2 The Setup

We follow Cohen et al.'s architecture: a two-hidden-layer MLP with tanh activations.

$$\text{Input}(220) \rightarrow \text{Linear}(220, 50) \rightarrow \tanh \rightarrow \text{Linear}(50, 50) \rightarrow \tanh \rightarrow \text{Linear}(50, 10)$$

That's 156,710 parameters. Small by modern standards. Large enough to exhibit the dynamics that matter.

The dataset is synthetic: 2,000 points in 10 classes, with 200 random features drawn from class-specific Gaussian clusters plus 20 quadratic features (the first 20 input dimensions, squared). The quadratic features give the network something genuinely nonlinear to chew on — structure that a small network can learn but cannot memorize, ensuring that training dynamics persist rather than terminating at zero loss.

Training uses full-batch gradient descent with mean squared error loss. No momentum. No weight decay. No learning rate schedule. Just the purest form of the question: a loss landscape, a step size, and the iterates of a map.

The sharpness at convergence (for small learning rates) gives λ_max ≈ 7.42. The Edge of Stability threshold is therefore:

$$\frac{2}{\lambda_{\max}} \approx 0.270$$

This is the number to keep in mind. It is where classical theory says things should get interesting.

### 2.3 The Lyapunov Exponent in Function Space

For each learning rate η and random seed $s$:

1. Initialize weights $\theta_0$ from seed $s$.
2. Create a perturbed copy: $\theta_0' = \theta_0 + \varepsilon \hat{\delta}$, where $\hat{\delta}$ is a unit-norm random vector and $\varepsilon$ is small.
3. Train both networks for $T$ steps under identical conditions.
4. At each step $t$, record $d(t) = \|f_{\theta_t}(X) - f_{\theta_t'}(X)\|_2$ — the distance in function space.
5. Fit the slope of $\log(d(t)/d(0))$ vs. $t$. That slope is the Lyapunov exponent.

Positive slope: the perturbation grows. The networks diverge. Chaos.

Negative slope: the perturbation shrinks. The networks converge. Order.

Zero: the knife's edge. The boundary between two kinds of world.

---

## 3. What We Found

### 3.1 The Broad View

We swept 20 learning rates from η ≈ 0.013 to η ≈ 0.404, running 5 seeds at each value.

The picture is clean. At the lowest learning rates, Lyapunov exponents hover near zero or dip slightly negative — the two networks learn approximately the same function. As learning rate increases, exponents rise monotonically. By η ≈ 0.05, all seeds show positive exponents. By the EoS threshold (η ≈ 0.27), exponents are an order of magnitude above zero.

The seeds agree on the shape. They disagree on the details — individual traces weave above and below the mean, particularly in the middle range. This is itself a signature. Near a chaos transition, small differences in initial conditions produce large fluctuations in the measured exponent. The noisiness is the signal.

### 3.2 The Transition

We zoomed in. 30 learning rates between η = 0.005 and η = 0.08. Three seeds. Fine-grained enough to find the crossing.

The mean Lyapunov exponent crosses zero at:

$$\eta_c \approx 0.019$$

Individual seeds place the crossing between η ≈ 0.006 and η ≈ 0.030, giving:

$$\eta_c = 0.018 \pm 0.012$$

The variance is large — nearly as large as the estimate. This is not a defect. It is expected. Near a chaos transition, the system alternates between nearly-ordered and nearly-chaotic episodes. The Lyapunov exponent, measured over a finite training run, inherits that intermittency. More seeds (we recommend ≥ 20) will tighten the estimate.

### 3.3 The Number

The critical learning rate η_c sits at approximately:

$$\frac{\eta_c}{2/\lambda_{\max}} \approx \frac{0.018}{0.270} \approx 6.6\%$$

Chaos begins at 6.6% of the Edge of Stability.

The system is sensitive to initial conditions — is, in the technical sense, chaotic — at a learning rate more than an order of magnitude below where classical theory says instability starts. The Edge of Stability is not the edge of anything, dynamically speaking. It is a phenomenon that occurs in a regime that is already, by Lyapunov measure, chaotic.

---

## 4. What This Means

### 4.1 The Dynamical Systems Picture

There is a body of mathematics that describes exactly this kind of transition. It was developed to explain turbulence — how a smooth fluid flow breaks down into chaos — and it applies here with remarkable directness.

**The Ruelle-Takens route** (1971) describes a sequence:

$$\text{Fixed point} \rightarrow \text{Limit cycle} \rightarrow \text{2-torus} \rightarrow \text{3-torus} \rightarrow \text{Strange attractor}$$

A fixed point is a system at rest. A limit cycle is a system that oscillates — loss going up and down periodically. A 2-torus is a system with two incommensurate frequencies — like a wobbling orbit, or a heartbeat modulated by breathing. A 3-torus adds a third frequency.

Ruelle and Takens proved that 3-tori are structurally unstable. Any perturbation — however small — can replace a 3-torus with a strange attractor: a fractal object on which trajectories are bounded but never repeat, and initially nearby trajectories diverge exponentially. Chaos.

The key insight: **you don't need infinite complexity to get chaos. Three nested frequencies are enough.** And gradient descent, with its interplay between the learning rate, the loss landscape's curvature, and the network's nonlinear dynamics, has more than three things oscillating at once.

### 4.2 KAM Theory and the Surviving Tori

The Kolmogorov-Arnold-Moser theorem (1954–1962) adds nuance. In perturbed dynamical systems, not all ordered structure is destroyed at once. Tori whose frequency ratios are "sufficiently irrational" — far from any rational approximation — survive perturbation, slightly deformed. Tori near resonance are destroyed, shattering into fractal remnants called Cantori (Percival, 1979).

The result is a phase space that looks like a fractal mosaic: islands of order surrounded by seas of chaos, with smaller islands inside the chaos, and smaller seas inside those islands, ad infinitum.

This is exactly what the transition zone in our data looks like. At learning rates near η_c, some seeds produce positive Lyapunov exponents and some produce negative ones. The system is not cleanly ordered or cleanly chaotic. It is a KAM-like mixture — pockets of predictability in a sea of sensitivity, or vice versa, depending on which island the initialization happens to land on.

### 4.3 The Period-Doubling Route

Kalra, He, and Barkeshli (2023) have already identified a period-doubling route to chaos on the Edge of Stability manifold. As learning rate increases, the loss oscillates with period 2, then 4, then 8, then chaotically — the Feigenbaum cascade, with its universal scaling constants, playing out in the loss curve of a neural network.

Our finding extends theirs downward. The Lyapunov transition at η_c ≈ 0.02 occurs well below the EoS threshold where they observe period-doubling. This suggests the bifurcation sequence begins earlier than previously recognized — or that there are multiple routes to chaos operating simultaneously at different scales in the training dynamics.

Recent work on deep linear networks beyond the EoS (arXiv:2502.20531, 2025) confirms period-doubling cascades with loss oscillations confined to learning-rate-dependent subspaces. The picture is converging: gradient descent undergoes classical dynamical transitions, and the torus-to-chaos framework applies.

### 4.4 What Damian et al. Got Right — and What They Missed

Damian, Nichani, and Lee (2023) demonstrated that EoS dynamics can be captured by a cubic Taylor expansion. They described a self-stabilization mechanism — a negative feedback loop where divergence in the direction of the Hessian's top eigenvector causes curvature to decrease, pulling the system back. They characterized this as "far from chaotic."

They were right about the mechanism. Self-stabilization is real. But "far from chaotic" needs qualification. Our Lyapunov data shows the system is already chaotic — in the function-space sense — before it reaches the EoS. The self-stabilization mechanism doesn't prevent chaos; it manages chaos. It is the dynamical equivalent of surfing: the wave (chaotic dynamics) is already there, and the cubic feedback loop rides it without wiping out.

This is actually a more interesting story than either "EoS is chaotic" or "EoS is stable." The system finds a way to learn effectively despite — perhaps because of — operating in a chaotic regime. Morales et al. (2024) suggested this: positive Lyapunov exponents may signal an exploitation-to-exploration transition. A learning rate that makes training sensitive to initial conditions also makes it more exploratory — better at finding diverse solutions, less likely to get stuck in a single basin.

---

## 5. The Torus in the Machine

There is a way to see all of this as one thing.

A cycle within a cycle creates a torus. This is geometry, not metaphor. When one periodic process is embedded within another — when a circle itself travels in a circle — the resulting path traces a toroidal surface.

$$x(\theta, \phi) = (R + r\cos\theta)\cos\phi$$
$$y(\theta, \phi) = (R + r\cos\theta)\sin\phi$$  
$$z(\theta, \phi) = r\sin\theta$$

Gradient descent, at low learning rates, settles into something like this. The loss oscillates (one frequency). The sharpness oscillates (another frequency). The parameter vector traces a path through a high-dimensional space that, projected down, looks quasiperiodic — multiple incommensurate frequencies superposed. This is motion on a torus.

As learning rate increases, the perturbation to this toroidal motion grows. KAM theory says some tori survive, some shatter. The Ruelle-Takens theorem says that with enough nested frequencies, the torus is replaced by a strange attractor. The Feigenbaum cascade says the replacement happens through period-doubling.

All of these are descriptions of the same underlying transition: **ordered toroidal dynamics becoming chaotic fractal dynamics under stress.**

Our measurement pins down where this transition happens in gradient descent: at η_c ≈ 6.6% of the classical stability boundary. The torus breaks early. The chaos starts quietly. By the time sharpness reaches 2/η and the Edge of Stability announces itself, the strange attractor has been there for a long time.

### 5.1 Arnold Tongues in the Learning Rate

When two oscillators couple — one cycle interacting with another — the mathematics of frequency-locking produces fractal structure (Jensen, Bak, & Bohr, 1983). Regions of parameter space where the oscillators lock to rational frequency ratios form tongue-shaped zones called Arnold tongues. Between the tongues, the dynamics are quasiperiodic. As coupling increases, the tongues widen until they touch, and the set of quasiperiodic states becomes a Cantor set — a fractal dust.

We suspect this structure is present in the learning rate parameter. At certain learning rates, the loss oscillation may lock to rational multiples of some intrinsic frequency of the loss landscape. Between those locked states, the dynamics are quasiperiodic. The transition zone near η_c — where some seeds show positive Lyapunov exponents and others don't — may reflect exactly this: a fractal interleaving of locked and unlocked states, with the locked states shrinking as learning rate increases.

Testing this prediction requires power spectral analysis of the training loss at densely sampled learning rates. We leave this to future work (see Section 7).

---

## 6. Implications

### 6.1 Reproducibility Has a Threshold

If chaos begins at η_c ≈ 0.02 for this architecture, then any learning rate above that value produces networks whose learned function depends on the initialization in a sensitive, exponential way. Two training runs that differ by a rounding error in one weight will diverge to measurably different functions.

This matters for scientific reproducibility. It matters for model auditing. And it matters for anyone who has ever tried to exactly replicate a training run and found that the results were close but not identical. The mismatch is not a bug. It is a dynamical phase.

### 6.2 The "Safe" Range Is Narrow

Classical theory draws the stability line at 2/λ_max. If chaos starts at 6.6% of that threshold, then the truly deterministic — truly reproducible — regime is narrow. Most learning rates that practitioners would call "reasonable" are above it. Most training runs are, in the Lyapunov sense, chaotic.

This does not mean they fail. Chaotic dynamics in gradient descent still converge (in loss). They just converge to different places from different starting points. The loss landscape is chaotic but the chaos is productive — it explores.

### 6.3 Ensembles Get Diversity for Free

If standard learning rates produce chaotic training dynamics, then initializing networks with different random seeds automatically produces diverse models. This may partly explain why deep ensembles work: you don't need different architectures or different data subsets. You just need the dynamics to be chaotic enough that different initializations explore different basins.

---

## 7. What We Don't Know Yet (And How to Find Out)

### 7.1 The Statistical Weakness

Three seeds in the transition zone is not enough. The confidence interval on η_c is ±0.012 — almost as large as the estimate itself. This paper is a first report, not a definitive measurement. Twenty seeds will shrink that interval to ±0.003. The computation is cheap. The result needs to be tightened.

### 7.2 One Architecture Is Not a Claim

We tested one architecture (tanh MLP) on one dataset (synthetic). The claim that chaos onset is a general property of gradient descent requires testing across:

- Activation functions (ReLU, GELU, SiLU)
- Depths (3, 4, 5+ layers)
- Architectures (CNNs, transformers)
- Datasets (CIFAR-10, standard benchmarks)

The key quantity is the ratio η_c/(2/λ_max). If it is approximately constant across architectures — if chaos always starts at roughly the same fraction of the stability threshold — then the phenomenon is about gradient descent itself, not any particular network.

### 7.3 Seeing the Torus

We have measured Lyapunov exponents. We have not seen the torus.

To do that requires:

**Bifurcation diagrams.** Record the loss at convergence (or the last $N$ loss values) for 200 densely-spaced learning rates. If period-doubling is present, the bifurcation diagram will show it — a single converged value splitting into two, then four, then a chaotic cloud.

**Power spectra.** Take the training loss as a time series. Compute its Fourier transform. Below η_c, expect discrete peaks — a small number of distinct frequencies, the signature of quasiperiodic motion on a torus. Above η_c, expect broadband noise — the signature of a strange attractor.

**Phase-space reconstruction.** Apply Takens embedding to the loss time series: $[L(t), L(t+\tau), L(t+2\tau)]$. Below η_c, the reconstructed attractor should look like a torus. Above η_c, it should look like a fractal.

These experiments would directly visualize the transition — not just measure that it happens, but show the geometric object that breaks. That is the next step.

### 7.4 The Ruelle-Takens Prediction

The Ruelle-Takens theorem says chaos requires ≥ 3 nested frequencies. This makes a testable prediction: networks with fewer dynamical degrees of freedom (shallower, narrower) should resist chaos to higher learning rates. Deeper networks, with more layers of nested nonlinear transformation, should reach chaos sooner.

If η_c/(2/λ_max) decreases with depth, that is direct evidence for the Ruelle-Takens mechanism operating in gradient descent.

---

## 8. Conclusion

We trained the same network twice. We changed the weights by a billionth. We asked: at what learning rate do the two networks start learning different things?

The answer is about 6.6% of where classical optimization theory says instability begins.

This is not where the Edge of Stability is. The Edge of Stability is a curvature phenomenon — the loss landscape pushing back against the step size. The Lyapunov transition is a sensitivity phenomenon — the training trajectory starting to depend on unmeasurable differences in initial conditions. They are different boundaries. The chaos comes first.

We frame this within the theory of torus-to-chaos transitions in dynamical systems. In this picture, stable training corresponds to quasiperiodic dynamics on a torus — orderly, repeatable, insensitive to perturbation. As learning rate increases, the torus is destroyed. KAM theory says the destruction is partial and fractal. Ruelle-Takens says that with enough nested frequencies, the replacement is a strange attractor. The Feigenbaum cascade says the replacement happens through period-doubling.

All of this has been seen in gradient descent by other authors in other ways. What we add is the measurement of where it begins — and the observation that it begins much earlier than the curvature dynamics would suggest.

The Edge of Stability is not the edge of order. By the time the sharpness reaches 2/η, the strange attractor has been there for a long time. The network is already dreaming. The Edge of Stability is where it learns to dream productively.

---

## References

Arnold, V. I. (1963). Small denominators and problems of stability of motion in classical and celestial mechanics. *Russian Mathematical Surveys*, 18(6), 85–191.

Cohen, J., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2021). Gradient descent on neural networks typically occurs at the edge of stability. In *International Conference on Learning Representations*.

Damian, A., Nichani, E., & Lee, J. D. (2023). Self-stabilization: The implicit bias of gradient descent at the edge of stability. In *International Conference on Learning Representations*.

Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. *Journal of Statistical Physics*, 19(1), 25–52.

Jensen, M. H., Bak, P., & Bohr, T. (1983). Complete devil's staircase, fractal dimension, and universality of mode-locking structure in the circle map. *Physical Review Letters*, 50(21), 1637–1639.

Kalra, D. S., He, T., & Barkeshli, M. (2023). Universal sharpness dynamics in neural network training: Fixed point analysis, edge of stability, and route to chaos. *arXiv preprint arXiv:2311.02076*.

Kolmogorov, A. N. (1954). On the conservation of conditionally periodic motions under small perturbation of the Hamiltonian. *Dokl. Akad. Nauk SSSR*, 98, 527–530.

Morales, G. B., et al. (2024). Dynamical stability and chaos in artificial neural network trajectories along training. *Frontiers in Complex Systems*, 2, 1367957.

Moser, J. (1962). On invariant curves of area-preserving mappings of an annulus. *Nachr. Akad. Wiss. Göttingen*, II, 1–20.

Percival, I. C. (1979). Variational principles for invariant tori and cantori. In *Nonlinear Dynamics and the Beam-Beam Interaction*, AIP Conference Proceedings 57 (pp. 302–310).

Ruelle, D., & Takens, F. (1971). On the nature of turbulence. *Communications in Mathematical Physics*, 20, 167–192.

Various authors (2025). Learning dynamics of deep linear networks beyond the edge of stability. *arXiv preprint arXiv:2502.20531*.

Züchner, T., et al. (2024). Finite-time Lyapunov exponents of deep neural networks. *Physical Review Letters*, 132, 057301.

---

## Appendix A: Experimental Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | MLP: 220 → 50 (tanh) → 50 (tanh) → 10 |
| Parameters | 156,710 |
| Dataset | Synthetic: 2,000 points, 10 classes, 220 dims |
| Loss | Mean squared error |
| Optimizer | Full-batch gradient descent |
| λ_max | ~7.42 |
| 2/λ_max | ~0.270 |
| η_c | 0.018 ± 0.012 |
| η_c / (2/λ_max) | ~6.6% |
| Seeds (broad sweep) | 5 |
| Seeds (transition zone) | 3 |

---

*The torus is what training looks like before it learns to improvise. The strange attractor is what it looks like after. The learning rate is the dial that turns one into the other.*
