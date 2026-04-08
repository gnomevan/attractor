# Strange Attractors of Gradient Descent

**Paper:** *The Geometry of Learning: Data Complexity Controls the Fractal Dimension of Gradient Descent*

**Target:** Physical Review Letters

**Author:** Evan Paul (evan@evanpaul.us)

---

## Key Result

Neural network training trajectories in function space are strange attractors. A CNN trained on CIFAR-10 produces a fractal attractor with correlation dimension D₂ = 3.67 ± 0.08 at 30% of the Edge of Stability threshold. Data complexity—not architecture—is the necessary ingredient for multi-dimensional chaos.

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
│   └── figures/
│       ├── cifar10_eos.png        Fig. 1: CNN/CIFAR-10 four-panel
│       ├── figure2_cross_experiments.png    Fig. 2: D₂ vs EoS across conditions
│       ├── d2_calibration.png     SM: D₂ pipeline calibration
│       └── tda_supplement.png     SM: persistent homology
│
├── data/
│   ├── main/                      data used directly in the main paper figures
│   │   ├── cifar10_eos_10seeds.json           CNN/CIFAR-10, 10 seeds
│   │   ├── cifar10_eos.json                   CNN/CIFAR-10, 3 seeds (older)
│   │   ├── cross_small_mlp_cifar_w50_seeds_0_1_2.json    MLP 156K/CIFAR-10
│   │   ├── cross_small_mlp_cifar_w50_seeds_0.json        MLP 156K (single seed)
│   │   ├── cross_cnn_synthetic_seeds_0.json   CNN/synthetic
│   │   └── tda_cifar10_reconstructed.json     persistent homology features
│   │
│   ├── supplemental/              data used in the supplemental material
│   │   ├── d2_calibration.json    calibration against known attractors
│   │   ├── depth_scaling.json     MLP depth 2/3/4/5
│   │   └── relu_comparison.json   tanh vs ReLU
│   │
│   └── phase1_phase2/             MLP baseline experiments (NPZ arrays)
│       ├── broad_sweep.npz
│       ├── transition_zone.npz
│       ├── sensitivity.npz
│       ├── seeds_comparison.npz
│       ├── bifurcation.npz
│       ├── dimension.npz
│       ├── power_spectrum.npz
│       ├── power_spectrum_v2.npz
│       ├── geometric_sweep.npz
│       ├── lyapunov_vectors.npz
│       └── takens_v2.npz
│
├── code/                          experiment scripts
│   ├── phase3_experiments_k.py    CNN/CIFAR-10 + architecture scaling
│   ├── cnn_seeds_v2.py            CNN multi-seed (seeds 0–2)
│   ├── cnn_seeds_extension_fixed.py   CNN extension (seeds 3–9) + Lorenz validation
│   └── experiment_L_tda_fixed.py  persistent homology experiments
│
└── docs/                          internal / reference documents
    ├── chaos_onset_findings_report_v5.md      full experimental findings narrative
    ├── prl_draft_v05.md                       earlier markdown draft of the paper
    ├── strange_attractors_and_generalization.md   generalization hypothesis (future work)
    └── repository_manifest_v2.md              detailed data manifest
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

```bash
# CNN on CIFAR-10 (Phase 3), seeds 0–2
python code/phase3_experiments_k.py

# CNN extension, seeds 3–9
python code/cnn_seeds_extension_fixed.py --seeds 3 4 5 6 7 8 9

# Persistent homology (TDA) analysis
python code/experiment_L_tda_fixed.py
```

Requirements: PyTorch, NumPy, SciPy, torchvision, `ripser` (for TDA).

---

## License

- Code: MIT
- Data and paper: CC-BY 4.0
