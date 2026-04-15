# Strange Attractors of Gradient Descent

**Paper:** *The Geometry of Learning: Data Complexity Controls the Fractal Dimension of Gradient Descent*

**Target:** Physical Review Letters

**Author:** Evan Paul (evan@evanpaul.us)

---

## Key Result

Neural network training trajectories in function space are strange attractors. A CNN trained on CIFAR-10 produces a fractal attractor with correlation dimension D₂ = 3.67 ± 0.08 at 30% of the Edge of Stability threshold. Data complexity — not architecture — is the necessary ingredient for multi-dimensional chaos. A continuous label-noise sweep confirms that data complexity acts as a genuine control parameter: degrading label structure smoothly reduces D₂ from 3.6 to 1.7 at moderate learning rates. Peak chaos (Lyapunov exponent) and peak geometric complexity (D₂) dissociate universally across architectures, consistent with the KAM transition.

---

## Repository Layout

```
attractor/
├── README.md                      this file
├── .gitignore
│
├── paper/                         manuscript, supplemental, cover letter
│   ├── prl_attractor.tex          main manuscript (RevTeX 4.2, PRL format)
│   ├── prl_attractor.pdf          compiled manuscript
│   ├── supplemental.tex           supplemental material
│   ├── supplemental.pdf
│   ├── cover_letter.tex           cover letter for PRL submission
│   ├── cover_letter.pdf
│   ├── references.bib             BibTeX bibliography
│   ├── revision1_plan.md          revision plan and status tracker
│   └── figures/
│       ├── cifar10_eos.png        Fig. 1: CNN/CIFAR-10 four-panel
│       ├── figure2_cross_experiments.png    Fig. 2: D₂ vs EoS across conditions
│       └── revision1/
│           ├── label_noise_d2.pdf         Fig. 3: D₂ vs label-noise fraction
│           ├── persistence_diagrams.png   SM: persistence diagrams
│           ├── d2_convergence.png         SM: D₂(N) convergence
│           └── dissociation_figure.pdf    SM: λ–D₂ dissociation
│
├── data/
│   ├── main/                      data backing main-text figures/tables
│   │   ├── cifar10_eos_10seeds.json           CNN/CIFAR-10, 10 seeds
│   │   ├── cifar10_eos.json                   CNN/CIFAR-10, 3 seeds (legacy)
│   │   ├── cross_small_mlp_cifar_w50_seeds_0_1_2.json    MLP 156K legacy
│   │   ├── cross_cnn_synthetic_seeds_0.json   CNN/synthetic legacy
│   │   ├── tda_cifar10_reconstructed.json     persistent homology features
│   │   └── revision1/
│   │       ├── cross_small_mlp_cifar_w50_seeds_merged.json   MLP 156K, 10 seeds
│   │       ├── cross_small_mlp_cifar_w85_seeds_merged.json   MLP 269K, 10 seeds
│   │       ├── cross_cnn_synthetic_seeds_merged.json          CNN/synth, 10 seeds
│   │       └── label_noise_sweep.json         label-noise D₂(p), 2 archs × 7 levels × 3 seeds
│   │
│   ├── supplemental/              data backing supplement tables
│   │   ├── d2_calibration.json    calibration (legacy, n=800)
│   │   ├── depth_scaling.json     MLP depth 2/3/4/5
│   │   ├── relu_comparison.json   tanh vs ReLU
│   │   └── revision1/
│   │       ├── d2_calibration_n400.json       calibration at n=400
│   │       ├── d2_vs_n_reference.json         D₂(N) for Lorenz + MG τ=30
│   │       ├── d2_vs_n_neural.json            D₂(N) for neural conditions
│   │       ├── tda_mlp_cifar_w50.json         persistent homology, MLP 156K
│   │       ├── tda_mlp_cifar_w85.json         persistent homology, MLP 269K
│   │       ├── tda_cnn_cifar_diagrams.json    CNN persistence diagrams
│   │       ├── convergence_n_inputs_epsilon.json   ε/N_inputs audit
│   │       └── dissociation_analysis.json     λ–D₂ dissociation analysis
│   │
│   └── phase1_phase2/             MLP baseline experiments (NPZ arrays, read-only)
│
├── code/                          experiment scripts
│   ├── phase3_experiments_k.py    CNN/CIFAR-10 + architecture scaling (original)
│   ├── cnn_seeds_v2.py            CNN multi-seed, seeds 0–2 (original)
│   ├── cnn_seeds_extension_fixed.py   CNN extension, seeds 3–9 (original)
│   ├── experiment_L_tda_fixed.py  persistent homology (original)
│   └── revision1/
│       ├── r1_cross_experiments.py        unified cross-experiment generator (N=10)
│       ├── r1_merge.py                    merge legacy + new seeds
│       ├── r1_figure2.py                  publication Figure 2
│       ├── r1_calibration_n400.py         D₂ calibration at n=400
│       ├── r1_d2_convergence.py           D₂(N) convergence (GPU)
│       ├── r1_d2_convergence_figure.py    convergence figure
│       ├── r1_tda_mlp_cifar.py            persistent homology for MLPs (GPU + ripser)
│       ├── r1_persistence_figure.py       persistence diagram figure
│       ├── r1_lyap_units_check.py         ε/N_inputs sensitivity audit
│       ├── r1_label_noise_sweep.py        label-noise D₂(p) sweep (GPU)
│       ├── r1_label_noise_figure.py       label-noise figure
│       └── r1_dissociation_analysis.py    λ–D₂ dissociation analysis + figure
│
└── docs/                          internal / reference documents
    ├── chaos_onset_findings_report_v5.md
    ├── prl_draft_v05.md
    ├── strange_attractors_and_generalization.md
    └── repository_manifest_v2.md
```

---

## Building the Paper

The manuscript uses RevTeX 4.2 for PRL format. From the repository root:

```bash
cd paper
pdflatex prl_attractor
bibtex   prl_attractor
pdflatex prl_attractor
pdflatex prl_attractor

pdflatex supplemental
pdflatex cover_letter
```

Or with `latexmk`:

```bash
cd paper
latexmk -pdf prl_attractor.tex
latexmk -pdf supplemental.tex
latexmk -pdf cover_letter.tex
```

Requires a TeX Live distribution with the `revtex` package installed (included in `texlive-publishers` on Debian/Ubuntu, or MacTeX by default).

---

## Reproducing the Experiments

All experiments use full-batch gradient descent with MSE loss, no momentum, no weight decay. CIFAR-10 experiments use a 2,000-image subset.

### Original experiments

```bash
# CNN on CIFAR-10 (Phase 3), seeds 0–2
python code/phase3_experiments_k.py

# CNN extension, seeds 3–9
python code/cnn_seeds_extension_fixed.py --seeds 3 4 5 6 7 8 9

# Persistent homology (TDA) analysis
python code/experiment_L_tda_fixed.py
```

### Revision 1 experiments

```bash
# Cross-architecture, N=10 seeds (MLP w50, MLP w85, CNN/synthetic)
python code/revision1/r1_cross_experiments.py --condition mlp_cifar_w50 --seeds 0 1 2 3 4 5 6 7 8 9
python code/revision1/r1_cross_experiments.py --condition mlp_cifar_w85 --seeds 0 1 2 3 4 5 6 7 8 9
python code/revision1/r1_cross_experiments.py --condition cnn_synthetic --seeds 0 1 2 3 4 5 6 7 8 9

# Merge seeds into unified files
python code/revision1/r1_merge.py

# D₂ calibration at n=400
python code/revision1/r1_calibration_n400.py

# D₂(N) convergence (GPU)
python code/revision1/r1_d2_convergence.py

# Persistent homology for MLPs (GPU + ripser)
python code/revision1/r1_tda_mlp_cifar.py

# ε/N_inputs sensitivity audit
python code/revision1/r1_lyap_units_check.py

# Label-noise sweep (GPU, 42 runs)
python code/revision1/r1_label_noise_sweep.py

# λ–D₂ dissociation analysis (no GPU needed)
python code/revision1/r1_dissociation_analysis.py
```

### Figure generation (no GPU needed)

```bash
python code/revision1/r1_figure2.py
python code/revision1/r1_label_noise_figure.py
python code/revision1/r1_d2_convergence_figure.py
python code/revision1/r1_persistence_figure.py
python code/revision1/r1_dissociation_analysis.py
```

Requirements: PyTorch, NumPy, SciPy, torchvision, matplotlib, `ripser` (for TDA).

---

## License

- Code: MIT
- Data and paper: CC-BY 4.0
