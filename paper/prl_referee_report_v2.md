# Referee Report (Independent Review)

**Manuscript:** "Strange Attractors in Gradient Descent: Data Structure and Loss Geometry Control Fractal Dimension"  
**Journal:** Physical Review Letters  
**Author:** Evan Paul

---

## Summary

This Letter reports measurements of the correlation dimension $D_2$ of neural network training trajectories in function space. The central claim is that full-batch gradient descent on structured data (CIFAR-10) produces strange attractors with $D_2$ significantly above 1 ($D_2 = 3.67 \pm 0.08$ for a CNN, converging to ~5 at longer trajectory lengths), while synthetic (structureless) data yields $D_2 \approx 1$ regardless of architecture. A five-condition experimental design (two architectures crossed with two data types, plus width variations) is used to disentangle the roles of data and architecture. A label-noise sweep reveals that two distinct mechanisms control $D_2$ depending on dynamical regime, and a batch-size sweep shows the attractor survives SGD noise. Supplemental material includes extensive calibration, convergence analysis, persistent homology, and robustness checks.

## Overall Assessment

This is a genuinely interesting and well-executed piece of work. The central question — what is the geometric character of the dynamical object that gradient descent traces out? — is natural and important, yet surprisingly underexplored. The experimental design is clean: the data-vs-architecture factorization, the function-space measurement protocol (which sidesteps gauge ambiguities from weight-space symmetries), and the calibration against 11 known dynamical systems all reflect careful thinking. The two-mechanism finding from the label-noise sweep is the paper's most striking result and elevates it beyond a straightforward measurement paper.

The manuscript is clearly a product of substantial revision effort — the supplemental material is thorough, the error bars are honest, and the caveats are mostly in the right places. That said, I have concerns in several areas: the strength of some quantitative claims relative to the evidence, the theoretical framing, a few gaps in the experimental program, and some presentation issues. I organize these below.

**Recommendation: Publish with minor revision.**

The core result is novel and sound. The issues below are addressable without new experiments.

---

## Major Issues

### 1. The abstract overpromises relative to the converged measurements

The abstract leads with $D_2 = 3.67 \pm 0.08$ as though this were the headline number, then mentions in a parenthetical that convergence analysis yields $D_2 \approx 5$. But the supplemental convergence analysis (Fig. 5) shows clearly that $N \approx 400$ sits on the rising portion of the $D_2(N)$ curve for all neural conditions. The $N = 400$ values are not wrong — they are valid lower bounds — but foregrounding them with tight error bars creates a misleading impression of precision. The ±0.08 reflects seed-to-seed variability at a fixed (unconverged) trajectory length, not the total uncertainty on the attractor dimension.

The paper already handles this well in the body text ("conservative lower bounds") and in Table 1 (which includes both the $N \approx 400$ and $N = 3200$ columns). The abstract should match this honesty. I'd suggest leading with the converged estimate and presenting the lower bound as supporting evidence rather than the reverse.

### 2. The Ruelle-Takens / classical routes to chaos framing needs more care

The Discussion states that "the findings echo classical routes to chaos [Ruelle & Takens 1971, Newhouse et al. 1978]: structured data creates independent oscillation axes; coupling through training dynamics produces a strange attractor." This is evocative but potentially misleading. The Newhouse-Ruelle-Takens (NRT) route involves the sequential bifurcation of quasi-periodic modes on a torus that eventually destabilizes into a strange attractor. The paper does not demonstrate:

- The existence of quasi-periodic dynamics or invariant tori at any learning rate
- The sequential appearance of independent oscillatory modes as the learning rate increases
- That the transition from $D_2 \approx 1$ to $D_2 > 3$ proceeds through intermediate quasi-periodic states

The PC2 variance in Fig. 1(c) shows off-axis dynamics growing, which is suggestive, but PC2 variance is not equivalent to an independent oscillatory mode. Similarly, the $\lambda$–$D_2$ dissociation is presented as "characteristic of dynamical transitions where ordered and chaotic regions coexist," which is accurate, but this occurs in many types of transitions (intermittency, crisis, period-doubling), not specifically NRT.

The paper already includes a good sentence: "These dynamical findings complement the static toroidal topology identified by Pittorino et al." — this is the right level of connection. I'd recommend the rest of the Discussion adopt a similarly cautious tone: "suggestive of" or "reminiscent of" rather than "echo."

### 3. The two-mechanism finding deserves more mechanistic investigation

The reversal of $D_2$'s response to label noise between the CNN at 30% EoS and the MLP at 90% EoS is the paper's most surprising and important result. However, the mechanistic explanation is somewhat hand-wavy. The paper says "random labels roughen the loss surface, sustaining chaos through competing minima" for the MLP regime — but this is asserted rather than demonstrated. A few questions that would strengthen the interpretation:

- Does the Hessian spectral density change character between the two regimes as labels are randomized? Even a single measurement of $\lambda_{\max}$ vs. label noise at each condition would help.
- Is the MLP effect specific to 90% EoS, or does it appear at any high learning rate? A label-noise sweep at, say, 70% EoS for the MLP would disambiguate whether this is an EoS-proximity effect or a more general high-$\eta$ phenomenon.
- The CNN at $p = 1.0$ still has $D_2 = 2.34$, well above the CNN/synthetic baseline ($D_2 \approx 1$). The paper correctly notes this means "the geometric structure of CIFAR-10 inputs contributes to multi-dimensional dynamics even when labels carry no information." This is an important observation that complicates the clean "two mechanisms" narrative — it suggests a third factor (input geometry) that operates independently of both label structure and loss-surface roughness.

I don't require new experiments for acceptance, but acknowledging this complexity in the Discussion (rather than resolving it into a tidy two-mechanism story) would be more honest and more interesting.

### 4. Statistical power for the label-noise and batch-size sweeps

The label-noise sweep uses 7 seeds per condition, which is adequate. However, the CNN curve at high noise levels shows very large variance ($\sigma = 0.75$ at $p = 1.0$, or ~32% of the mean). This doesn't invalidate the monotone trend, but it does mean the specific shape of the $D_2(p)$ curve — whether it's linear, has a threshold, or saturates — is essentially undetermined for the CNN above $p = 0.5$. The batch-size sweep uses only 3 seeds, and the MLP $\lambda$ values have enormous scatter ($\sigma = 13.4$ at $B = 500$ on a mean of 27.6). These sweeps are valuable as existence proofs (the attractor survives SGD; label noise modulates $D_2$) but overinterpreting their quantitative details is risky.

I'd suggest adding a brief caveat to the caption of Fig. 3 and the relevant supplement tables noting that the CNN error bars at high noise are large enough to accommodate a range of functional forms.

---

## Minor Issues

### 5. Definition of "structured data"

The paper's argument hinges on "structured data" as a necessary condition for multi-dimensional chaos, but this term is never precisely defined. The synthetic control is described as "structureless" (220 dimensions, 10 classes), but its construction is not specified in the main text. Is it Gaussian? Uniform? Are the classes linearly separable? The degree of "structurelessness" matters: if the synthetic data has some geometric structure (e.g., class centroids with unequal spacing), the clean $D_2 \approx 1$ result is less surprising. The supplemental material should specify the data-generating process for the synthetic control.

### 6. Persistent homology: feature counts vs. significance

The supplement reports up to 392 $H_1$ features at 400 points. The text acknowledges that "the high feature counts likely include noise-generated features" and that "the key diagnostic is the gap ratio near 1.0." This is fair, but a quantitative threshold for significance — even a rough one, such as comparing to a null model (e.g., persistence computed on a random point cloud of the same size and ambient dimension) — would substantially strengthen the TDA argument. Without it, the reader cannot assess whether 386 $H_1$ features is 10× or 1.2× what noise alone would produce.

### 7. Stationarity assumption

The $D_2$ measurement discards the first 20% of the trajectory as transient, but Fig. 1(d) shows that at 95% EoS, sharpness (the top Hessian eigenvalue) is still rising at step 5,000. If the dynamics have not reached a stationary regime, the measured $D_2$ characterizes a transient, not an attractor. This is most concerning at high EoS fractions. A windowed analysis — computing $D_2$ over, say, the first half and second half of the post-transient trajectory and comparing — would address this. If such data exists, please include it; if not, a sentence acknowledging the limitation would suffice.

### 8. Embedding dimension and Takens' theorem

The Grassberger-Procaccia algorithm requires choosing an embedding dimension $m$ large enough that the correlation integral converges. The main text and supplement do not discuss how $m$ was selected, whether a $D_2(m)$ convergence plot was checked, or how the scaling region in $\ln C(r)$ vs. $\ln r$ was identified. For a measurement paper, these algorithmic details are important. At minimum, the supplement should state the embedding dimension used and show that $D_2$ is stable for $m$ above some threshold.

### 9. Finite-time Lyapunov exponents and prior work

The paper includes a valuable methodological caution about $\varepsilon = 10^{-8}$ producing numerical artifacts, and correctly notes that "prior work at smaller $\varepsilon$ [Morales et al. 2024] should be interpreted carefully." This is a strong contribution. However, the relationship to Storm et al. (2024), who also measured finite-time Lyapunov exponents in neural networks and published in PRL, deserves more discussion. How do the present measurements compare in magnitude and methodology? Storm et al. worked in weight space; do their reported exponents suffer from the gauge-symmetry issues the present paper's function-space protocol avoids?

### 10. Presentation and PRL formatting

Several presentation issues:

- **Abstract length.** At ~140 words the abstract is within PRL guidelines, but the final sentence about the dissociation feels tacked on. Consider integrating it or cutting it to make room for the converged $D_2$ estimate to appear more prominently.
- **Figure 1 caption.** The caption says "10 seeds" for everything, but the convergence analysis in the supplement notes that the CNN $N = 3200$ value is from a single seed. This potential confusion should be clarified.
- **Table 1 footnote.** The asterisk note "$^*$Single seed; noisier than MLP conditions" is important information that could easily be missed. Consider moving this to the caption proper.
- **The Acknowledgments** disclose AI assistance transparently, which is commendable. However, "including experimental design, code development, data analysis, and manuscript preparation" is quite broad — this is essentially the entire research process. Given APS's evolving policies on AI use, the author may wish to be more specific about which aspects of the experimental design were AI-assisted vs. author-originated.

---

## Strengths

To be clear about what this paper does well:

- **The question is excellent.** Measuring the fractal dimension of training dynamics is a natural thing to do that, remarkably, nobody seems to have done carefully before.
- **The function-space protocol is the right choice.** Working in function space eliminates the gauge ambiguity problem that would plague any weight-space analysis in overparameterized networks. This alone is a methodological contribution.
- **The five-condition experimental design** cleanly separates data effects from architecture effects. The synthetic controls are convincing.
- **The calibration is thorough.** Testing the pipeline against 11 known systems at the production trajectory length, and showing the systematic bias, is exactly what a measurement paper should do.
- **The label-noise sweep** transforms what could be a two-point comparison (CIFAR vs. synthetic) into a continuous control experiment. The sign reversal between regimes is genuinely surprising.
- **The honest error reporting** — including the $\varepsilon$-dependence caveat, the "conservative lower bound" framing, and the convergence analysis — reflects scientific maturity.

---

## Summary of Requested Changes

| # | Priority | Request |
|---|----------|---------|
| 1 | Major | Restructure abstract to lead with converged estimates; present $N = 400$ values as lower bounds |
| 2 | Major | Soften NRT/classical-chaos framing to "suggestive" rather than "echoes" |
| 3 | Major | Acknowledge the input-geometry confound in the two-mechanism narrative |
| 4 | Major | Add brief caveats about statistical power in label-noise/batch-size captions |
| 5 | Minor | Define "structured" / specify synthetic data generation |
| 6 | Minor | Add null-model comparison for persistent homology feature counts |
| 7 | Minor | Address stationarity at high EoS fractions |
| 8 | Minor | State embedding dimension and $D_2(m)$ convergence in supplement |
| 9 | Minor | Discuss relationship to Storm et al. more substantively |
| 10 | Minor | Presentation fixes (abstract, Table 1 footnote, AI disclosure specificity) |
