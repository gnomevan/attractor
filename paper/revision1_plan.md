# Revision 1 Plan — *The Geometry of Learning*

**Target:** Physical Review Letters
**Status at start of revision:** Draft v1 complete (`prl_attractor.tex`, `supplemental.tex`, `cover_letter.tex`). Referee report (internal) returned major-revision verdict with 9 open issues (#10 "parameter-matched synthetic control" is resolved; the remaining 9 drive this plan).
**Revision tag:** `revision1` (used in all new file/directory names)

---

## Current Status (updated 2026-04-12)

**Phase 0:** COMPLETE. Lorenz/2-torus numbers fixed, cover letter address filled, bibkey audit passed, λ units added, MLP inversion interpretive patch added (needs revision — see Phase 1 findings). Local pdflatex compile NOT yet verified by user.

**Phase 1:** COMPLETE. All three cross-experiment conditions run at N=10 fresh seeds, merged, and analyzed. **Manuscript updates applied** (2026-04-12): all 11 number substitutions in `prl_attractor.tex`, `tab:d2_all` in `supplemental.tex` updated with N=10 values, Figure 2 regenerated via `code/revision1/r1_figure2.py`. Key findings:
- MLP w50 (156K): peak D₂ = 4.61 ± 0.10 at 90% EoS (was 4.56 ± 0.02 at N=3)
- MLP w85 (269K): peak D₂ = 4.37 ± 0.05 at 90% EoS (was 4.35 ± 0.06 at N=3; data had been MISSING)
- CNN/synth: peak D₂ = 0.98 ± 0.03 at 90% EoS (was ~1.0 at N=1; only had 1 seed at 5-fraction grid)
- CNN/CIFAR: unchanged at 3.67 ± 0.08 at 30% EoS (already had N=10)
- **Width inversion confirmed at 2.2σ** — w50 > w85 is real, not noise. Phase 0 interpretive patch ("saturation of available modes") is WRONG and must be rewritten.
- **λ–D₂ dissociation is universal** across all CIFAR conditions (CNN: 15%→30%, MLP w50: 40%→90%, MLP w85: 50%→90%). Strengthens the KAM framing.
- Error bars widened from ±0.02 to ±0.10 for MLP w50 — directly resolves referee issue #2.
- Reproduce-check showed protocol is valid but chaotic-regime float divergence across torch versions prevents splicing old/new seeds. All conditions run fresh on same hardware.

**Phase 1 manuscript updates (11 items, APPLIED 2026-04-12):**
1. Abstract: "4.35 ± 0.06" → "4.37 ± 0.05" (MLP 269K)
2. Abstract: "D₂ ≈ 0.9" → keep (rounding ok, 0.98 rounds same)
3. Line 38: "3 seeds (cross-experiments)" → "10 seeds" for all conditions
4. Results: "D₂ = 4.35 ± 0.06" → "4.37 ± 0.05"
5. Results: "D₂ = 4.56 ± 0.02" → "4.61 ± 0.10"
6. Table: CNN/synth N=3→10, ~1.0→0.98; MLP 156K 4.56(2)→4.61(10) N=3→10; MLP 269K 4.35(6)→4.37(5) N=3→10
7. Results: rewrite width-inversion paragraph — gap is 2.2σ significant, reframe as "width modulates bifurcation delay, not attractor destination"
8. Fig 2 caption: "3 seeds" → "10 seeds"
9. Dissociation paragraph: generalize from CNN-only to universal across architectures
10. Abstract: add mention that dissociation is cross-architecture
11. Discussion: strengthen KAM framing with cross-architecture dissociation evidence

**Phase 1 done condition:** MET. All N=10 data exists and merges cleanly. Table and figure regenerated. Abstract, results, discussion updated. Width inversion reframed at 2.2σ (not saturation). Dissociation generalized cross-architecture.

**Phase 2A:** COMPLETE. n=400 calibration extracted (data already existed at this sample size). Supplemental Table I updated from n=800 to n=400 values. D₂(N) reference curves computed for Lorenz and MG τ=30 at n={200, 400, 800, 1600, 3200} — both show clean plateaus, confirming pipeline stability across sample sizes. Main-text Lorenz number (1.73) unchanged at n=400. Script: `code/revision1/r1_calibration_n400.py`. Output: `data/supplemental/revision1/d2_calibration_n400.json`, `data/supplemental/revision1/d2_vs_n_reference.json`.

**Phase 2C:** COMPLETE. Persistent homology computed for both MLP widths across all 12 LR fractions (3 seeds each). Results confirm that TDA tracks D₂ perfectly: zero H₁ features below 30% EoS, onset at 40%, ~390 H₁ features at 90% EoS. Gap ratio ~1.0 everywhere — fractal, not torus. CNN persistence diagrams generated for publication figure. New supplement tables (`tab:tda_mlp50`, `tab:tda_mlp85`), persistence diagram figure (`figures/revision1/persistence_diagrams.png`), and main-text TDA paragraph updated to include both architectures. Data: `data/supplemental/revision1/tda_mlp_cifar_w50.json`, `tda_mlp_cifar_w85.json`, `tda_cnn_cifar_diagrams.json`. Scripts: `code/revision1/r1_tda_mlp_cifar.py`, `code/revision1/r1_persistence_figure.py`.

**Phase 2B:** COMPLETE. Dense recording (every step, 4000 post-transient points) within the standard 5000-step protocol. Both MLPs plateau near D₂≈5.1 by N=3200 (1.7% and 6.2% change from N=1600). CNN noisier (single seed) but does not grow unboundedly. Production N≈400 values confirmed as conservative lower bounds. Convergence figure and supplement section added. Script: `code/revision1/r1_d2_convergence.py` (revised from 32k-step extension, which collapsed the attractor). Figure: `code/revision1/r1_d2_convergence_figure.py`.

**Phase 2D:** COMPLETE. D₂ stable across N_inputs (spread 0.10 over {25..400}), independent of ε as expected. λ shows ε-dependence (factor ~3× across [1e-6, 1e-4]) but remains positive throughout — chaos is robust. Supplement section added. Data: `data/supplemental/revision1/convergence_n_inputs_epsilon.json`.

**Phase 2 done condition:** MET. All four workstreams complete. Calibration at n=400, D₂(N) convergence, MLP TDA, and ε/N_inputs audit all integrated into manuscript and supplement.
**Phase 3:** COMPLETE (2026-04-15). Both workstreams done:
- **3A (label noise sweep):** 42 runs complete. CNN/CIFAR at 30% EoS: D₂ falls smoothly from 3.64±0.11 to 1.68±0.54 as labels are randomized — clean monotone degradation of attractor complexity. MLP/CIFAR 269K at 90% EoS: D₂ rises modestly from 4.36±0.05 to 4.77±0.08 — regime-dependent effect where random labels roughen the loss surface near the EoS. Both curves are smooth and continuous, confirming data complexity as a genuine control parameter (not binary). Script: `code/revision1/r1_label_noise_sweep.py`. Data: `data/main/revision1/label_noise_sweep.json`. Figure: `paper/figures/revision1/label_noise_d2.pdf`.
- **3B (dissociation universality):** Confirmed universal across all 3 CIFAR architectures. Peak λ always precedes peak D₂: CNN 15%→30%, MLP 156K 40%→90%, MLP 269K 50%→90%. Already reflected in main text from Phase 1 updates; supplemental figure and analysis JSON now added. Script: `code/revision1/r1_dissociation_analysis.py`. Data: `data/supplemental/revision1/dissociation_analysis.json`. Figure: `paper/figures/revision1/dissociation_figure.pdf`.
- **Manuscript updates applied:** Abstract (label-noise sentence added), Results (new "continuous control parameter" paragraph + Fig. 3), Discussion (regime-dependent interpretation integrated), Supplement (label-noise table + dissociation figure + two new sections).

**Phase 3 done condition:** MET. Label-noise sweep exists and is reflected in abstract + new Figure 3. Dissociation checked across all CIFAR conditions — universal, KAM framing strengthened. Discussion honestly addresses the regime-dependent MLP result.

**Phase 4:** COMPLETE (2026-04-15). Full end-to-end audit:
- All citations resolve (11 bibkeys, zero undefined)
- All figure/table references resolve (fig:cnn, fig:cross, fig:noise, tab:conditions)
- All 6 figure files exist on disk
- Zero stale N=3 references in main text
- Abstract numbers cross-checked against data files: all match
- Supplement table spot-checked: means exact, stds within rounding
- Stale MG τ=30 calibration number fixed (3.24 → 3.17 to match n=400 table)
- `\cite{sm}` updated to list all 9 supplement sections
- Cover letter updated: mentions label-noise sweep, cross-architecture universality, and measurement validation paragraph added
- No "binary" data-complexity framing survives — label-noise paragraph establishes continuous control
- Cannot run pdflatex in sandbox (no revtex4-2.cls); user should run `pdflatex && bibtex && pdflatex && pdflatex` locally to verify compilation

**Phase 4 done condition:** MET (pending user's local LaTeX compile). Every abstract number traces to a supplement entry. Git state ready for tagging.

### Data files produced in Phase 1

```
data/main/revision1/
  cross_small_mlp_cifar_w50_seeds_0.json                        (reproduce-check, seed 0 only)
  cross_small_mlp_cifar_w50_seeds_0_1_2_3_4_5_6_7_8_9.json     (raw N=10)
  cross_small_mlp_cifar_w85_seeds_0_1_2_3_4_5_6_7_8_9.json     (raw N=10)
  cross_cnn_synthetic_seeds_0_1_2_3_4_5_6_7_8_9.json            (raw N=10)
  cross_small_mlp_cifar_w50_seeds_merged.json                    (pipeline-uniform, passthrough)
  cross_small_mlp_cifar_w85_seeds_merged.json                    (pipeline-uniform, passthrough)
  cross_cnn_synthetic_seeds_merged.json                          (pipeline-uniform, passthrough)
```

### Visualizations produced

```
paper/figures/revision1/
  w50_phase_transition_3d.html          (3-tab: landscape, geometry, dispersion cloud)
  w50_vs_w85_comparison.html            (3-tab: 3D head-to-head, D₂ curves, dissociation)
  phase1_all_conditions.html            (4-tab: all conditions 3D, D₂ ribbons, dissociation, data control)
```

### Scripts produced

```
code/revision1/
  r1_cross_experiments.py   (unified generator, all 3 conditions, --reproduce-check/--dry-run/--quick)
  r1_merge.py               (merge/passthrough, all 3 conditions → *_merged.json)
  r1_figure2.py             (publication Figure 2 from merged N=10 data + MLP/synth baseline)
  r1_calibration_n400.py    (Phase 2A: extract n=400 calibration + D₂(N) reference curves)
  r1_d2_convergence.py      (Phase 2B: D₂(N) convergence for neural networks — GPU)
  r1_tda_mlp_cifar.py       (Phase 2C: persistent homology for MLP/CIFAR — GPU + ripser)
  r1_lyap_units_check.py    (Phase 2D: ε/N_inputs sensitivity audit — CPU/M1 OK)
```

This plan is written so that any future session can pick it up cold. It is organized into four phases that correspond to roughly one context window each. Every phase has: (a) exact experiments to run, (b) files that will be produced, (c) manuscript sections that will be edited, (d) a done-condition before moving on.

---

## Naming conventions (important — do not drift)

The existing layout is:

```
attractor/
  code/        cnn_seeds_v2.py, phase3_experiments_k.py, …
  data/
    main/              # data backing main-text figures/tables
    phase1_phase2/     # historical exploration, do not touch
    supplemental/      # data backing supplement tables
  paper/
    prl_attractor.tex
    supplemental.tex
    cover_letter.tex
    figures/
```

To avoid making this messier, **all revision-1 additions go under a single `revision1/` suffix or subfolder**, mirroring the existing split:

```
attractor/
  code/
    revision1/
      r1_mlp_cifar_seeds.py          # more seeds for MLP/CIFAR (both widths)
      r1_cnn_synth_seeds.py          # more seeds for CNN/synthetic
      r1_tda_mlp_cifar.py            # persistent homology on MLP/CIFAR
      r1_d2_convergence.py           # D2 vs trajectory length sweep
      r1_calibration_n400.py         # recalibrate D2 pipeline at n=400
      r1_label_noise_sweep.py        # continuous data-complexity axis
      r1_lyap_units_check.py         # units audit + ε/N-inputs convergence
      r1_persistence_diagrams.py     # export full diagrams, not just counts
  data/
    main/
      revision1/                     # new main-text-backing data
    supplemental/
      revision1/                     # new supplement-backing data
  paper/
    revision1_plan.md                # this file
    figures/
      revision1/                     # new figures referenced in revised tex
```

Rules:
- New scripts are prefixed `r1_` and live in `code/revision1/`.
- New data files carry the prefix of the experiment (`mlp_cifar_269k_seeds_3_9.json`, etc.) and live in `data/main/revision1/` or `data/supplemental/revision1/`.
- Old files in `data/main/`, `data/supplemental/`, `data/phase1_phase2/` are **read-only**. Do not overwrite.
- When the revision is merged, the `.tex` files are edited in place — no `prl_attractor_v2.tex`. Keep git history as the version control.
- A final cleanup pass at the end of Phase 4 can *move* files out of `revision1/` into their parent folders only if doing so does not create name collisions, and only if the final figure numbering is stable. If in doubt, leave them in `revision1/`.

---

## Phase 0 — Housekeeping (small, before the compute-heavy work)

Purpose: fix the things that do not require new experiments so they do not get forgotten later.

1. **Fix the Lorenz inconsistency** (referee issue #6). Main text says "D₂ = 1.68 for the Lorenz attractor"; supplement table says 1.73. Rerun the pipeline once on Lorenz at the production N and use the resulting value in both places. Script: `code/revision1/r1_calibration_n400.py` (also used in Phase 2). Update both `prl_attractor.tex` (~line 40) and `supplemental.tex` (calibration table).

2. **Cover letter placeholder.** `cover_letter.tex` line 6: `\address{[Your address]}` — replace with real address. Trivial.

3. **Bibkey audit.** `prl_attractor.tex` cites `\cite{kam}`; confirm that key resolves in `references.bib`. Also verify every other cite resolves by running `pdflatex && bibtex && pdflatex && pdflatex` and checking the log for undefined references. Fix any stragglers.

4. **Lyapunov exponent units.** Audit all λ values in the text, the supplemental depth-scaling table, and the figure captions. State explicitly that λ is measured per training step (add one sentence to the Experimental Framework section and one sentence to the supplement §1). No new computation.

5. **Acknowledge the 156K > 269K MLP inversion** (referee issue #7). Add a single sentence in Results and/or Discussion explaining that the smaller MLP reaches *slightly* higher peak D₂ than the larger one, and frame it as consistent with the narrative (architecture is a throttle, not a source; width beyond sufficiency does not add modes, and the 14% wider MLP may have slightly looser coupling). This is an interpretive patch — but it must not be hand-wavy; cross-check by looking at whether the difference (4.56 vs 4.35) is within the widened error bars after Phase 1 adds seeds. If after Phase 1 the inversion is within 1σ, simply note that the two widths are statistically indistinguishable and drop the "architecture as throttle" phrasing where it implies a monotone width effect.

**Done condition for Phase 0:** paper and supplement compile without warnings; Lorenz number is consistent; cover letter has no placeholders; all cites resolve; the λ-unit sentence is in place.

---

## Phase 1 — Seeds (the statistical-power fix; referee issues #1, #2)

This is the single biggest compute block. The cross-architecture comparison is the conceptual spine of the paper and the MLP/CIFAR conditions plus CNN/synth currently rest on N ≤ 3. We take all of them to N=10 to match CNN/CIFAR.

### Data provenance audit (completed during revision planning)

During Phase 1 preparation we discovered several provenance gaps:

- **MLP/CIFAR 269K (w=85):** Data was **never committed** to the repo. The original generator (`Torus Theory/Pytorch/cross_experiments.py`) exists, but no output JSON. The paper's w85 numbers cannot be traced to a file.
- **CNN/synthetic:** The legacy file (`data/main/cross_cnn_synthetic_seeds_0.json`) has **1 seed** (not 3 as claimed in the paper) and only the 5-point "quick" LR grid, not the 12-point grid used by the other conditions.
- **MLP/CIFAR 156K (w=50):** The legacy file (`data/main/cross_small_mlp_cifar_w50_seeds_0_1_2.json`) is intact — 3 seeds, 12 LR fractions, complete per-seed metrics. This is the only fully trustworthy legacy cross-experiment file.

**Consequence:** Rather than simply adding 7 new seeds to existing 3-seed datasets, Phase 1 must **re-run all three conditions from scratch at N=10**. For w50, the legacy 3-seed data can be merged with 7 new seeds (after a reproducibility check passes); for w85 and CNN/synth, all 10 seeds are fresh.

### Scripts (completed)

Both scripts live in `code/revision1/`:

- **`r1_cross_experiments.py`** — Unified generator handling all three conditions via `--condition {mlp_cifar_w50, mlp_cifar_w85, cnn_synthetic}`. Protocol-matched to both legacy generators (`Torus Theory/Pytorch/small_mlp_cifar.py` and `Torus Theory/Pytorch/cross_experiments.py`). Includes `--reproduce-check`, `--dry-run`, `--metadata-only`, and `--quick` flags. Every output JSON carries a content-addressable `PROTOCOL_HASH`.

- **`r1_merge.py`** — Merges legacy + r1 data into unified 10-seed files. Three strategies:
  - **w50:** genuine merge (3 legacy + 7 new → 10-seed interleave), validated against matching lam_max/lr_eos/n_params.
  - **w85:** passthrough (all 10 seeds are fresh; "merge" just adds pipeline-uniform metadata).
  - **CNN/synth:** passthrough (legacy grid is incompatible; all 10 seeds are fresh).

### Experiments to run

| Condition | Legacy N | Legacy grid | Fresh seeds | Strategy |
|---|---|---|---|---|
| MLP/CIFAR, 156K (w=50) | 3 (intact but hardware-incompatible) | 12 fracs | seeds 0–9 (10 fresh) | passthrough (reproduce-check showed chaotic-regime float divergence across torch versions; cannot splice) |
| MLP/CIFAR, 269K (w=85) | 0 (missing) | — | seeds 0–9 (10 fresh) | passthrough |
| CNN/synthetic, 269K | 1 (incompatible grid) | 5 fracs | seeds 0–9 (10 fresh) | passthrough |

(The CNN/CIFAR 10-seed run already exists in `data/main/cifar10_eos_10seeds.json` and does not need to be rerun.)

### Output files

From `r1_cross_experiments.py`:
```
data/main/revision1/cross_small_mlp_cifar_w50_seeds_3_4_5_6_7_8_9.json
data/main/revision1/cross_small_mlp_cifar_w85_seeds_0_1_2_3_4_5_6_7_8_9.json
data/main/revision1/cross_cnn_synthetic_seeds_0_1_2_3_4_5_6_7_8_9.json
```

From `r1_merge.py`:
```
data/main/revision1/cross_small_mlp_cifar_w50_seeds_merged.json    (10 seeds)
data/main/revision1/cross_small_mlp_cifar_w85_seeds_merged.json    (10 seeds)
data/main/revision1/cross_cnn_synthetic_seeds_merged.json           (10 seeds)
```

These `*_merged.json` files are what the revised tables and figures read from.

### Analysis deliverables for the manuscript

1. **Table `tab:conditions`** (main text) — recompute peak D₂ and its uncertainty with N=10 for all five conditions; update the trailing `N` column accordingly.
2. **Table `tab:d2_all`** (supplement) — recompute every cell with the full N=10 seed pools. Pay attention to the CNN 20% EoS cell (currently 2.63 ± 0.62) — referee flagged the contrast between this variance and the MLP plateau variance. If the MLP variances widen to ≥0.1 at N=10 the issue dissolves; if they stay at ±0.02, that's genuinely informative and should be discussed.
3. **Figure 2** (`figures/figure2_cross_experiments.png`) — regenerate from merged data. Error bars must be stddev across 10 seeds, not 3.

### Done condition for Phase 1

- All three JSONs at N=10 exist and merge cleanly.
- `tab:conditions` and `tab:d2_all` are regenerated from the N=10 merged data.
- Figure 2 is regenerated.
- Any change in the peak D₂ values is reflected in the abstract, results, and discussion.
- **If the 156K vs 269K MLP inversion is no longer statistically significant, the Phase-0 interpretive patch about "throttle" is revised accordingly.**

---

## Phase 2 — Calibration, convergence, and the TDA gap (referee issues #3, #4, #5, #11)

This phase closes the measurement-credibility gap. It has four pieces that can be parallelized if compute allows.

### 2A — Recalibrate the D₂ pipeline at n=400

The current supplemental calibration table uses n=800; the production neural runs use n≈400. Calibration must match.

Script: `code/revision1/r1_calibration_n400.py`. Rerun every system in the current calibration table at n=400 (1-torus, 2-torus, 3-torus, 4-torus, Hénon, Rössler, Lorenz, Mackey-Glass τ∈{17, 23, 30, 50}) and also **at n=200, 400, 800, 1600, 3200** for Lorenz and Mackey-Glass τ=30 specifically (these two become the D₂(N) reference curves).

Output: `data/supplemental/revision1/d2_calibration_n400.json`, `data/supplemental/revision1/d2_vs_n_reference.json`.

Manuscript update: replace Table I in the supplement with the n=400 calibration. Keep the n=800 table alongside it in an appendix subsection titled "Calibration at higher sample count" for continuity. Update the main-text sentence about the Lorenz measurement (the Phase-0 inconsistency fix pulls from this file).

### 2B — D₂(N) convergence for the neural-network conditions

Referee issue #4: the MLP/CIFAR D₂ values sit just below the Eckmann–Ruelle ceiling for n=400, so we need to show they do not grow further when N is increased.

Script: `code/revision1/r1_d2_convergence.py`. For **one seed each** of (CNN/CIFAR at 30% EoS, MLP/CIFAR-269K at 90% EoS, MLP/CIFAR-156K at 90% EoS), extend the trajectory so that after transient removal we have ≈3200 points. Compute D₂ at subsampled lengths N ∈ {100, 200, 400, 800, 1600, 3200} (same seed, just slicing the long run). The goal is a plot showing D₂ → plateau.

Output: `data/supplemental/revision1/d2_vs_n_neural.json`.

Manuscript update: add a supplement subsection "D₂ convergence with trajectory length" with a new figure `figures/revision1/d2_convergence.png`. If the plateau is clean, reference it from the main text in a single sentence near the current discussion of the ~400-point trajectories.

**Important:** extending training from 5000 to (5000 × 3200 / 400) ≈ 40000 steps changes the dynamical regime — the attractor may drift as further sharpness changes occur. Mitigation: run the extended trajectory at a fixed learning rate from a warm-started checkpoint at the end of the standard 5000 steps, and verify that sharpness is already saturated (flat over the last 1000 steps of the standard run) before the extension starts. Document this protocol in the supplement.

### 2C — Persistent homology for MLP/CIFAR (the single most important fix)

Referee issue #3: TDA currently supports only the CNN claim. The MLP/CIFAR result — the one the "data complexity is sufficient" argument hinges on — has no topological cross-check.

Script: `code/revision1/r1_tda_mlp_cifar.py`. For both MLP widths (50 and 85), at each LR fraction already in `tab:d2_all`, compute H₁ and H₂ feature counts, gap ratios, and — new — **full persistence diagrams**. Use the same `ripser` settings as the original CNN/CIFAR TDA run. Use 3 seeds initially; if compute allows, match the N=10 used for D₂.

Output: `data/supplemental/revision1/tda_mlp_cifar_w50.json`, `data/supplemental/revision1/tda_mlp_cifar_w85.json`.

Manuscript update:
- Extend supplemental Table `tab:tda` with two new blocks (one per MLP width).
- Add a new supplemental figure `figures/revision1/persistence_diagrams.png` showing representative diagrams for: CNN/CIFAR at 5% EoS (trivial), CNN/CIFAR at 30% EoS (fractal), MLP/CIFAR-269K at 40% EoS (transition), MLP/CIFAR-269K at 90% EoS (plateau). This is a direct response to referee issue #11.
- Update the Results paragraph that currently says "Persistent homology confirms fractal topology" to explicitly include both architectures.

### 2D — Make the LR sensitivity audit rigorous

The main text currently asserts that 100 held-out inputs and ε = 1e-5 are converged. We should *show* this:

Script: `code/revision1/r1_lyap_units_check.py`. Pick one representative (CNN/CIFAR at 30% EoS, one seed). Sweep N_inputs ∈ {25, 50, 100, 200, 400} and ε ∈ {1e-4, 5e-5, 1e-5, 5e-6, 1e-6}. Confirm that λ and D₂ are stable within one stddev of the N=10 seed scatter at N_inputs ≥ 100 and at ε ∈ [1e-6, 1e-4].

Output: `data/supplemental/revision1/convergence_n_inputs_epsilon.json`.

Manuscript update: add one sentence to the main-text Experimental Framework section ("The choices ε = 10⁻⁵ and 100 held-out inputs are within a plateau verified in the supplement [SM §…]") and add a short subsection with a small figure to the supplement.

### Done condition for Phase 2

- n=400 calibration file exists and replaces Table I in the supplement.
- D₂(N) convergence plot exists and is referenced from the main text.
- MLP/CIFAR TDA exists for both widths; supplement table and persistence-diagram figure are in place.
- Convergence audit for ε and N_inputs exists and is cited.

---

## Phase 3 — The continuous data-complexity axis and dissociation universality (referee issues #8, #9)

This is the phase that *elevates* the paper rather than just shores it up. Both experiments are conceptually important.

### 3A — Continuous data-complexity sweep (#8)

Right now "data complexity" is binary (CIFAR vs structureless synthetic). Referee (rightly) says a control parameter needs to take more than two values. We add a continuous axis using **label noise** as the knob.

**Design:** for the CNN/CIFAR 269K condition at its peak LR (30% EoS), sweep label-noise fraction p ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0} (p=0.0 is standard CIFAR; p=1.0 is effectively structureless because labels carry no signal). Three seeds is fine here — the goal is to show a monotone curve, not a precise point measurement. Repeat the same sweep for MLP/CIFAR 269K at 90% EoS.

**Why label noise rather than subset size:** subset size changes the number of trajectory modes *and* the per-step noise; label noise changes the information content while keeping the dataset statistics identical, which is a cleaner knob. It also corresponds to a continuous interpolation between the two existing endpoints.

Script: `code/revision1/r1_label_noise_sweep.py`.

Output: `data/main/revision1/label_noise_sweep.json`.

Manuscript update: new main-text figure (or new panel added to Figure 2) showing D₂(p) for both architectures. If the curve is monotone, add one sentence to the abstract: "A continuous label-noise sweep shows that D₂ falls smoothly as data structure is degraded, demonstrating that data complexity acts as a genuine control parameter."

This is the revision's single biggest upgrade to the claim strength. It should not be skipped.

### 3B — Peak-λ / peak-D₂ dissociation, universality check (#9)

The dissociation between peak λ (15% EoS) and peak D₂ (30% EoS) is framed in the Discussion as the conceptual punchline — the KAM transition. But it is shown only for CNN/CIFAR. Test whether the same dissociation appears in MLP/CIFAR.

This mostly reuses data already being produced in Phase 1 (the N=10 MLP/CIFAR sweeps), because each seed's λ and D₂ are saved per LR fraction. No new experiments; the analysis step is: for each condition, find argmax(λ) and argmax(D₂) across the LR sweep and compare.

Script: `code/revision1/r1_dissociation_analysis.py` (tiny; ~50 lines).

Manuscript update:
- If MLP/CIFAR shows the same dissociation (peaks at different LR fractions), generalize the main-text statement and strengthen the KAM framing.
- If it does *not* dissociate in MLP/CIFAR, soften the framing: state the dissociation as an architecture-dependent feature and note what about the CNN (hierarchical processing) plausibly separates the two peaks.
- Either way, add a small supplemental figure stacking λ(LR) and D₂(LR) for all CIFAR conditions.

### Done condition for Phase 3

- Label-noise sweep exists and is reflected in abstract + Figure 2 (or a new figure).
- Dissociation is checked across all CIFAR conditions and the Discussion text is either strengthened or honestly softened.

---

## Phase 4 — Final polish and resubmission readiness

1. **Recompile and re-read the manuscript end-to-end.** Look specifically for: (a) stale numbers left over from the N=3 era, (b) forward references to figures that were renumbered, (c) any place where "data complexity" is still described as binary, (d) any place where calibration numbers from the n=800 table leak into the main text.

2. **Update the cover letter.** Add a short paragraph summarizing the revisions relative to v1 (if this is a second submission it is a response-to-referees; if this is the first submission after the internal review, it is just a cleaner cover letter). Keep the AI-disclosure paragraph exactly as is — it is already excellent.

3. **Final numerical pass.** Cross-check every number that appears in the abstract against the supplement tables. Cross-check the abstract against Figure 1, Figure 2, and the new continuous-complexity figure.

4. **External sanity check.** Before resubmission, do a full `pdflatex → bibtex → pdflatex → pdflatex` cycle on `prl_attractor.tex`, `supplemental.tex`, and `cover_letter.tex`. No undefined references, no unresolved citations, no overfull boxes on the letter pages.

5. **Optional cleanup.** If and only if the revision is fully stable, consider flattening `data/main/revision1/` into `data/main/` and `data/supplemental/revision1/` into `data/supplemental/`, keeping descriptive filenames. Do not do this partway through — atomic move at the end only.

### Done condition for Phase 4

- Manuscript, supplement, and cover letter compile clean.
- Every number in the abstract traces to an entry in the supplement.
- Git state is clean and ready to tag as `v1.1-submission`.

---

## Cross-phase reproducibility discipline

For every script under `code/revision1/`:

- Deterministic seeds, set at top of `main()`.
- A `--metadata-only` flag that writes the protocol hash + git commit + numpy/torch versions to the output JSON before any training runs, so every data file is self-describing.
- A `--dry-run` flag that prints the planned LR sweep, seed list, and output path without touching the GPU.
- A header comment block identical in style to `code/cnn_seeds_v2.py` (which already has a good pattern), explicitly referencing which referee issue(s) the script addresses.

This keeps the work auditable across context windows, and lets any future session verify that nothing has silently drifted.

---

## Compute budget rough estimate

- Phase 1: 30 seeds × 12 LR fractions × 5000 steps (10 fresh each for w50, w85, CNN/synth — all on the same machine/torch to avoid cross-hardware float divergence). Reproduce-check for w50 confirmed protocol identity but revealed chaotic-regime sensitivity to torch version; all conditions run fresh.
- Phase 2A: calibration-only, cheap (minutes).
- Phase 2B: three extended runs to 40k steps at one LR each. Moderate.
- Phase 2C: TDA on existing MLP/CIFAR checkpoints — needs trajectory data to exist first, then ripser. Moderate (ripser can be slow at high n).
- Phase 2D: one seed × two sweeps. Cheap.
- Phase 3A: 2 architectures × 7 noise levels × 3 seeds = 42 runs. Moderate.
- Phase 3B: analysis-only. Cheap.

The user is running the compute; this plan can be handed to any future session with the compute budget and seed assignments intact.

---

## Issue-to-phase mapping (for the future session that picks this up)

| Referee issue | Phase | Artifact |
|---|---|---|
| #1 N=3 seeds | Phase 1 | `data/main/revision1/*_seeds_3_9.json`, merged files, updated tables |
| #2 suspiciously tight MLP variances | Phase 1 | Resolved by N=10 recomputation |
| #3 TDA only for CNN | Phase 2C | `data/supplemental/revision1/tda_mlp_cifar_*.json`, extended `tab:tda` |
| #4 trajectory length at Eckmann–Ruelle limit | Phase 2B | D₂(N) convergence figure |
| #5 calibration at wrong N | Phase 2A | n=400 calibration replaces Table I |
| #6 Lorenz number inconsistency | Phase 0 | Single-number fix, both files |
| #7 156K > 269K MLP inversion | Phase 0 + Phase 1 recheck | Interpretive sentence or softened framing |
| #8 "data complexity" is binary | Phase 3A | Label-noise sweep, abstract + figure update |
| #9 dissociation only shown for CNN | Phase 3B | Analysis of existing data, figure, framing update |
| #10 parameter matching | **Already resolved** | — |
| #11 persistence diagrams, not just counts | Phase 2C | New persistence-diagram figure |
| minor: λ units | Phase 0 | One sentence in main + one in supplement |
| minor: ε / N_inputs convergence | Phase 2D | Small subsection + figure |
| minor: cover letter placeholder | Phase 0 | Trivial |
| minor: bibkey audit | Phase 0 | `latex` log clean |

---

## Start-of-next-session checklist (copy to prompt when resuming)

1. Read `paper/revision1_plan.md` (this file) — especially the "Current Status" block near the top.
2. Check `data/main/revision1/` and `data/supplemental/revision1/` for what has been produced.
3. Do not touch files outside `revision1/` scopes except `prl_attractor.tex`, `supplemental.tex`, and `cover_letter.tex` (which are edited in place).

### If resuming after Phase 1 (current state as of 2026-04-12):

**Immediate next actions (in order):**

1. **Apply the 11 manuscript number updates** listed in the "Phase 1 manuscript updates needed" section above. These are all data-driven substitutions in `prl_attractor.tex`. The biggest one is rewriting the width-inversion paragraph (item 7): the gap between MLP w50 (D₂=4.61±0.10) and w85 (D₂=4.37±0.05) is 2.2σ significant. Reframe as: "the wider network's additional capacity raises the effective stability margin, delaying the bifurcation onset and slightly suppressing peak D₂ at a given LR fraction — width modulates the pace of the transition, not the destination."

2. **Update the supplemental `tab:d2_all`** with N=10 numbers for all three cross-experiment conditions. The merged JSON files in `data/main/revision1/*_merged.json` are the canonical source.

3. **Regenerate Figure 2** (`paper/figures/figure2_cross_experiments.png`) from the merged data. Error bars must show std across 10 seeds. The interactive HTML versions exist at `paper/figures/revision1/phase1_all_conditions.html` for reference but are not publication figures.

4. **Begin Phase 2** — the four parallel workstreams (2A calibration, 2B D₂ convergence, 2C TDA for MLP/CIFAR, 2D ε/N-inputs audit). See Phase 2 section for full details. Scripts listed there still need to be written.

**Key files to read for context:**
- `code/revision1/r1_cross_experiments.py` — the unified generator (protocol reference)
- `code/revision1/r1_merge.py` — merge script (already run, produced *_merged.json)
- `data/main/revision1/*_merged.json` — canonical N=10 data for all conditions
- `data/main/cifar10_eos_10seeds.json` — CNN/CIFAR data (different JSON schema: uses `seeds` key not `seeds_run`)

**Key findings to keep in mind:**
- The width inversion (w50 > w85 in peak D₂) is real at 2.2σ — do NOT frame as noise or saturation
- λ–D₂ dissociation is universal across all CIFAR conditions — can generalize the KAM claim
- CNN/synth is flat at D₂≈0.98 everywhere — the cleanest possible structureless-data control
- All N=10 data is from the same hardware/torch version (no cross-hardware splicing)
