# Repository Manifest v2: Strange Attractors in Gradient Descent
## Updated inventory after file collection

---

## Status Summary

| Category | Have | Missing |
|---|---|---|
| Paper draft | ✓ v0.5 | — |
| Figure 1 (CNN CIFAR-10) | ✓ image + 10-seed data | May need regeneration for publication quality |
| Figure 2 (cross-experiments) | ✓ image | — |
| Supplemental figures | ✓ TDA + calibration | — |
| CNN 10-seed data | ✓ | — |
| Cross: MLP 156K + CIFAR | ✓ 3 seeds | — |
| Cross: CNN + synthetic | ✓ 1 seed | ⚠️ Paper says 3 seeds, file has 1 |
| Cross: MLP 269K + CIFAR | ✗ | **MISSING — needed for Table I** |
| MLP baseline (Phase 1) | ✓ NPZ files | — |
| D₂ calibration | ✓ JSON + PNG | — |
| Depth scaling | ✓ JSON | — |
| ReLU comparison | ✓ JSON | — |
| ε sensitivity | ✓ NPZ | — |
| All experiment code | ✓ | — |

---

## Complete File Inventory

### DATA — In Project (accessible in repository)

| File | Experiment | Seeds | Status |
|---|---|---|---|
| `cifar10_eos_10seeds.json` | CNN CIFAR-10, 12 LR fractions | 10 | ✓ Paper's headline data |
| `cifar10_eos.json` | CNN CIFAR-10 (older, 3-seed version) | 3 | Superseded by 10-seed |
| `cross_small_mlp_cifar_w50_seeds_0_1_2.json` | MLP 156K on CIFAR-10, 12 LR fracs | 3 | ✓ |
| `cross_small_mlp_cifar_w50_seeds_0.json` | MLP 156K on CIFAR-10 (single seed) | 1 | Superseded by 3-seed |
| `cross_cnn_synthetic_seeds_0.json` | CNN on synthetic data, 5 LR fracs | 1 | ⚠️ Table I says 3 seeds |
| `tda_cifar10_reconstructed.json` | TDA persistence homology, 7 LR fracs | 3 | ✓ |
| `d2_calibration.json` | D₂ pipeline calibration (torus, Lorenz, Hénon, etc.) | — | ✓ |
| `depth_scaling.json` | MLP depth 2/3/4/5 Lyapunov + D₂ | varies | ✓ |
| `relu_comparison.json` | tanh vs ReLU architectures | varies | ✓ |

### DATA — Uploaded NPZ Files (need to be added to repository)

| File | Experiment (Phase) | Seeds × LRs | Maps to |
|---|---|---|---|
| `broad_sweep.npz` | Phase 1: Lyapunov sweep | 20 × 40 | Findings §3.1–3.2 |
| `transition_zone.npz` | Phase 1: Fine chaos window | 20 × 50 | Findings §3.2 (η_c) |
| `sensitivity.npz` | Phase 1: ε perturbation sweep | 5 × 5 LRs × 7 ε | SM: ε validation |
| `seeds_comparison.npz` | Phase 1: Seed convergence | 10 × 10 | Findings §3.3 |
| `bifurcation.npz` | Phase 2: Bifurcation diagram | 5 × 200 | Findings §4.3 |
| `dimension.npz` | Phase 2: MLP D₂ | 5 × 8 | Findings §4.1 |
| `power_spectrum.npz` | Phase 2: Loss spectra | 2 × 12 | Findings §4.2 |
| `power_spectrum_v2.npz` | Phase 2: Sharpness + loss + grad spectra | 2 × 12 | Findings §4.2 |
| `geometric_sweep.npz` | Phase 2: Function-space geometry | 3 × 12 | Findings §3.3 |
| `lyapunov_vectors.npz` | Phase 2: Lyapunov vector directions | 5 × 7 | Findings §3.3 (isotropy) |
| `takens_v2.npz` | Phase 2: Takens embedding D₂ | 5 × 8 × 4 embed dims | SM: embedding robustness |

### FIGURES — In Project

| File | Description | Used in |
|---|---|---|
| `cifar10_eos.png` | Figure 1: 4-panel CNN CIFAR-10 (10-seed, with error bars) | Main paper Fig. 1 |
| `figure2_cross_experiments.png` | Figure 2: D₂ vs EoS for all 5 conditions | Main paper Fig. 2 |
| `tda_supplement.png` | H₁/H₂ feature counts + gap ratio vs EoS | Supplemental |
| `d2_calibration.png` | D₂ measured vs expected for known attractors | Supplemental |

### CODE — In Project

| File | Description |
|---|---|
| `cnn_seeds_v2.py` | Main CNN experiment (CIFAR-10, multi-seed) |
| `cnn_seeds_extension_fixed.py` | CNN extension (cross-experiments) |
| `phase3_experiments_k.py` | Phase 3: architecture scaling (depth, width, ReLU) |
| `experiment_L_tda_fixed.py` | TDA / persistent homology experiment |

### PAPER — In Project

| File | Description | Status |
|---|---|---|
| `prl_draft_v05.md` | Current PRL draft | Active |
| `chaos_onset_findings_report_v5.md` | Complete findings narrative | Reference |
| `strange_attractors_and_generalization.md` | Generalization speculation | Internal |

---

## What's Still Missing

### 1. MLP 269K (w=85) on CIFAR-10 — cross-experiment data
- **Priority: HIGH** — needed for Table I and Figure 2
- Table I claims: D₂ = 4.35 ± 0.06 at 90% EoS, 3 seeds
- This data IS in Figure 2 (blue squares), so it was generated
- Look for: `cross_large_mlp_cifar_w85_seeds_*.json` or similar
- Could also be in a Colab notebook output or Google Drive
- **If you can't find it**: the run can be reproduced using `cnn_seeds_extension_fixed.py` with config for MLP w=85 on CIFAR-10

### 2. CNN on synthetic — needs 2 more seeds
- **Priority: MEDIUM** — current file has 1 seed, Table I claims 3
- Options: (a) find the 3-seed version, (b) rerun 2 more seeds, (c) update Table I to say 1 seed
- The single-seed result (D₂ ≈ 1.0) is unambiguous — more seeds won't change the conclusion

### 3. Figure 1 may need regeneration
- **Priority: LOW** — current `cifar10_eos.png` looks correct with error bars
- But for PRL submission, may want to regenerate with publication-quality formatting
- Need a `generate_figures.py` script for reproducibility

---

## Proposed Repository Structure

```
strange-attractors-gradient-descent/
│
├── README.md                          ← TO WRITE
├── LICENSE                            ← TO ADD (e.g., MIT or CC-BY)
│
├── paper/
│   ├── main_text.tex                  ← TO CONVERT from prl_draft_v05.md
│   ├── supplemental.tex               ← TO WRITE
│   └── figures/
│       ├── fig1_cnn_cifar10.pdf       ← convert from cifar10_eos.png
│       ├── fig2_cross_experiments.pdf  ← convert from figure2_cross_experiments.png
│       ├── figS1_d2_calibration.pdf   ← convert from d2_calibration.png
│       └── figS2_tda_topology.pdf     ← convert from tda_supplement.png
│
├── data/
│   ├── main/                          ← data directly used in paper figures
│   │   ├── cifar10_eos_10seeds.json
│   │   ├── cross_small_mlp_cifar_w50_seeds_0_1_2.json
│   │   ├── cross_cnn_synthetic_seeds_0.json
│   │   ├── cross_large_mlp_cifar_w85.json    ← MISSING
│   │   └── tda_cifar10.json
│   │
│   ├── supplemental/                  ← data for supplemental material
│   │   ├── d2_calibration.json
│   │   ├── depth_scaling.json
│   │   └── relu_comparison.json
│   │
│   └── phase1_phase2/                 ← MLP baseline experiments
│       ├── broad_sweep.npz
│       ├── transition_zone.npz
│       ├── sensitivity.npz
│       ├── seeds_comparison.npz
│       ├── bifurcation.npz
│       ├── dimension.npz
│       ├── power_spectrum_v2.npz
│       ├── geometric_sweep.npz
│       ├── lyapunov_vectors.npz
│       └── takens_v2.npz
│
├── code/
│   ├── experiments/
│   │   ├── cnn_seeds_v2.py
│   │   ├── cnn_seeds_extension_fixed.py
│   │   ├── phase3_experiments_k.py
│   │   └── experiment_L_tda_fixed.py
│   │
│   └── analysis/
│       ├── generate_figures.py        ← TO WRITE
│       └── d2_pipeline.py             ← TO EXTRACT from experiment code
│
└── docs/
    ├── findings_report_v5.md          ← complete narrative
    └── data_dictionary.md             ← TO WRITE (describes each data file)
```

---

## Action Items (Priority Order)

1. **Find MLP 269K cross-experiment data** — search Google Drive or Colab history for the w=85 CIFAR-10 run. If not found, rerun with `cnn_seeds_extension_fixed.py`.

2. **Decide on CNN-synthetic seeds** — either find the 3-seed version, rerun, or update Table I.

3. **Write `generate_figures.py`** — script that reads all JSON/NPZ files and produces publication-quality figures. This is needed for reproducibility and for regenerating figures in vector format (PDF) for PRL.

4. **Write `README.md`** — repository description, requirements, how to reproduce.

5. **Write supplemental material** — compile SM from the calibration, depth scaling, ReLU, ε sensitivity, and TDA data.

6. **Convert to RevTeX 4.2** — translate prl_draft_v05.md to LaTeX.

7. **Write data dictionary** — document what each file contains, how it was generated.

---

## Files NOT needed in repository (internal/superseded)

| File | Reason |
|---|---|
| `prl_draft_v02.md`, `prl_draft_v04.md` | Superseded by v05 |
| `cifar10_eos.json` (3-seed) | Superseded by 10-seed version |
| `cross_small_mlp_cifar_w50_seeds_0.json` | Superseded by 3-seed version |
| `experiment_L_tda.py` | Superseded by fixed version |
| `power_spectrum.npz` | Superseded by v2 |
| `core_theory_paper*.md` | Different paper |
| `Cycles_Within_Cycles_*.docx` | Different paper |
| `torus_framework_paper*.md` | Different paper |
| `donuts_all_the_way_down*` | Different paper |
| `torus_perspective_5min_talk.md` | Different project |
