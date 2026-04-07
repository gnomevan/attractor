# Strange Attractors of Gradient Descent

**Paper:** *The Geometry of Learning: Data Complexity Controls the Fractal Dimension of Gradient Descent*

**Target:** Physical Review Letters

**Author:** Evan Paul

---

## Key Result

Neural network training trajectories in function space are strange attractors. A CNN trained on CIFAR-10 produces a fractal attractor with correlation dimension D₂ = 3.67 ± 0.08 at 30% of the Edge of Stability threshold. Data complexity—not architecture—is the necessary ingredient for multi-dimensional chaos.

## Repository Contents

### Paper
| File | Description |
|---|---|
| `prl_attractor.tex` | Main manuscript (RevTeX 4.2, PRL format) |
| `references.bib` | BibTeX references |
| `supplemental.tex` | Supplemental Material |

### Figures
| File | Description |
|---|---|
| `cifar10_eos.png` | Figure 1: CNN/CIFAR-10 four-panel (Lyapunov, D₂, PC2, sharpness) |
| `figure2_cross_experiments_v2.png` | Figure 2: D₂ vs EoS for all five conditions |
| `d2_calibration.png` | Supplemental: pipeline calibration |
| `tda_supplement.png` | Supplemental: persistent homology |

### Experiment Code
| File | Description |
|---|---|
| `phase3_experiments_k.py` | Phase 3: CNN CIFAR-10 experiments + architecture scaling |
| `cnn_seeds_v2.py` | CNN multi-seed extension (seeds 0–2) |
| `cnn_seeds_extension_fixed.py` | CNN extension (seeds 3–9) + Lorenz validation |
| `experiment_L_tda_fixed.py` | Persistent homology (TDA) experiments |

### Data
| File | Description |
|---|---|
| `cifar10_eos_10seeds.json` | CNN/CIFAR-10 results, 10 seeds, 12 learning rates |
| `cifar10_eos.json` | CNN/CIFAR-10 results, 3 seeds |
| `cross_small_mlp_cifar_w50_seeds_0_1_2.json` | MLP (156K)/CIFAR-10, 3 seeds |
| `cross_cnn_synthetic_seeds_0.json` | CNN/synthetic, 1 seed |
| `d2_calibration.json` | D₂ pipeline calibration (Hénon, Lorenz, Rössler, Mackey-Glass, tori) |
| `tda_cifar10_reconstructed.json` | Persistent homology feature counts |
| `depth_scaling.json` | MLP depth scaling results |
| `relu_comparison.json` | tanh vs ReLU comparison |
| `*.npz` | Raw experimental data (Phase 1–2) |

### Internal Documents
| File | Description |
|---|---|
| `chaos_onset_findings_report_v5.md` | Complete experimental findings narrative |
| `strange_attractors_and_generalization.md` | Generalization hypothesis (future work) |
| `repository_manifest_v2.md` | Detailed data manifest |

## Reproducing Results

All experiments use full-batch gradient descent with MSE loss, no momentum, no weight decay. CIFAR-10 experiments use a 2,000-image subset.

```bash
# CNN on CIFAR-10, seeds 0-2
python phase3_experiments_k.py

# CNN extension, seeds 3-9
python cnn_seeds_extension_fixed.py --seeds 3 4 5 6 7 8 9

# TDA analysis
python experiment_L_tda_fixed.py
```

Requires: PyTorch, NumPy, SciPy, torchvision, ripser (for TDA).

## License

Code: MIT. Data and paper: CC-BY 4.0.
