# PRL Submission Checklist — 2026-04-18

## Compilation
- [x] `prl_attractor.tex` compiles without errors (tested via article-class substitution; revtex4-2 not available in sandbox)
- [x] `supplemental.tex` compiles without errors
- [x] `cover_letter.tex` compiles without errors (2 pages)
- [ ] **ACTION REQUIRED:** Run full `pdflatex → bibtex → pdflatex → pdflatex` locally with revtex4-2.cls to confirm clean compile and check for overfull hboxes in the two-column layout

## Figures (7 total)
- [x] `figures/cifar10_eos.png` — Fig. 1 (CNN on CIFAR-10)
- [x] `figures/figure2_cross_experiments.png` — Fig. 2 (cross-experiment D₂)
- [x] `figures/revision1/label_noise_d2.pdf` — Fig. 3 (label-noise sweep)
- [x] `figures/revision1/persistence_diagrams.png` — SM persistence diagrams
- [x] `figures/revision1/d2_convergence.png` — SM D₂(N) convergence
- [x] `figures/revision1/dissociation_figure.pdf` — SM λ–D₂ dissociation
- [x] `figures/revision1/batch_size_d2.pdf` — SM batch-size sweep

## References
- [x] All 13 bibkeys in `references.bib` resolve
- [x] All `\cite{}` keys in main text match bibkeys
- [x] No undefined citations
- [x] `\cite{sm}` note field lists all 11 supplement sections (including 2 new ones: stationarity, embedding dimension)

## Internal references
- [x] Main text: 4 labels, 4 refs — perfect match
- [x] Supplement: 13 labels, 10 refs — all refs have matching labels (3 unreferenced labels are standalone tables/figures, fine)

## Numerical audit (abstract ↔ tables)
- [x] D₂ ≈ 5 (converged) ↔ SM §D₂ Convergence: "both MLPs stabilize near 5.1"
- [x] 3.67 ± 0.08 (CNN) ↔ Table 1 CNN/CIFAR row; SM tab:d2_all CNN+C-10 at 30%
- [x] 4.4–4.6 (MLPs) ↔ Table 1: 4.61(10) and 4.37(5)
- [x] D₂ ≈ 1 (synthetic) ↔ Table 1: 0.9 and 0.98(3)
- [x] D₂ 3.6 → 2.3 (label noise) ↔ SM tab:label_noise: 3.64 → 2.34
- [x] >82% retention ↔ SM tab:batch_sweep_mlp: 82% at B=100
- [x] 20× gradient noise ↔ B=2000/B=100 = 20

## Referee report v2 changes implemented
- [x] Major 1: Abstract leads with converged D₂ ≈ 5; N=400 values as lower bounds
- [x] Major 2: NRT/KAM framing softened to "suggestive of" / "phenomenological analogies"
- [x] Major 3: Input-geometry third factor acknowledged in Results and Discussion
- [x] Major 4: Statistical power caveats in Fig. 3 caption and SM batch-size figure caption
- [x] Minor 5: "Structured data" defined; synthetic data generation specified (i.i.d. Gaussian)
- [x] Minor 6: Null-model note added to SM persistent homology section
- [x] Minor 7: Stationarity section added to SM
- [x] Minor 8: Embedding dimension section added to SM
- [x] Minor 9: Storm et al. discussed substantively (weight-space gauge artifacts)
- [x] Minor 10: AI disclosure sharpened; Table 1 footnote moved to caption

## Cover letter
- [x] Title matches manuscript
- [x] Summary consistent with revised abstract
- [x] Referee suggestions populated (Cohen, Mehlig, Zecchina)
- [x] AI disclosure detailed and honest

## Remaining items for the author
- [ ] Run local LaTeX compile with revtex4-2.cls (`pdflatex && bibtex && pdflatex && pdflatex` for all three .tex files)
- [ ] Check page count against PRL limit (4 pages for Letters)
- [ ] Visually inspect all figures in the compiled PDF
- [ ] Write response-to-referee letter if this is a resubmission (outline in `revision2_plan.md`)
- [ ] Upload to PRL submission system (main text, supplement, cover letter, source files)
