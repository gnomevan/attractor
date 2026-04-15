"""
Colab Launcher — Generalization Experiments
============================================
Paste each cell into a Colab notebook in order.
All results save to Google Drive at /content/drive/MyDrive/chaos_generalization/

Prerequisites: T4 GPU runtime, Drive mounted.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — Mount Drive + verify GPU
# ═══════════════════════════════════════════════════════════════════════════
CELL_1 = """
from google.colab import drive
drive.mount('/content/drive')

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — switch to T4'}")
print(f"CUDA: {torch.version.cuda}")

import os
os.makedirs('/content/drive/MyDrive/chaos_generalization', exist_ok=True)
print("Drive ready.")
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — Upload scripts (or paste them directly)
# ═══════════════════════════════════════════════════════════════════════════
CELL_2 = """
# Option A: upload via Files panel (left sidebar → upload)
# Option B: paste script content directly into a cell and %%writefile

# Verify files present
import os
for fname in ['p1_cnn_generalization_vs_d2.py', 'p3_depth_comparison.py']:
    exists = os.path.exists(f'/content/{fname}')
    print(f"  {fname}: {'✓' if exists else '✗ NOT FOUND'}")
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — Run P1 (CNN D2 vs test accuracy)
# ═══════════════════════════════════════════════════════════════════════════
CELL_3 = """
# Estimated time: 4-6 hours on T4
# Saves checkpoint after each LR point — safe to interrupt and resume

import subprocess, sys
result = subprocess.run(
    [sys.executable, '-u', '/content/p1_cnn_generalization_vs_d2.py'],
    capture_output=False,   # stream output to notebook
)
print(f"Exit code: {result.returncode}")
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 4 — Run P3 (depth comparison)
# ═══════════════════════════════════════════════════════════════════════════
CELL_4 = """
# Estimated time: 2-3 hours on T4
# Loads CNN EoS from P1 cache if available

import subprocess, sys
result = subprocess.run(
    [sys.executable, '-u', '/content/p3_depth_comparison.py'],
    capture_output=False,
)
print(f"Exit code: {result.returncode}")
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 5 — Quick diagnostic: plot results from saved .npz (if run was cut short)
# ═══════════════════════════════════════════════════════════════════════════
CELL_5 = """
import numpy as np
import matplotlib.pyplot as plt

DRIVE_DIR = '/content/drive/MyDrive/chaos_generalization'

# Load P1 results (whatever has been saved so far)
try:
    data = np.load(f'{DRIVE_DIR}/p1_cnn_results.npz')
    lr_fracs = data['lr_fracs']
    D2_mean  = data['D2_mean']
    acc_mean = data['acc_mean']
    D2_std   = data['D2_std']
    acc_std  = data['acc_std']

    fig, (ax1_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4))

    ax2_left = ax1_left.twinx()
    ax1_left.errorbar(lr_fracs * 100, D2_mean, yerr=D2_std,
                      color='steelblue', marker='o', label='D₂')
    ax2_left.errorbar(lr_fracs * 100, acc_mean, yerr=acc_std,
                      color='tomato', marker='s', linestyle='--', label='Test acc')
    ax1_left.set_xlabel('LR (% EoS)')
    ax1_left.set_ylabel('D₂', color='steelblue')
    ax2_left.set_ylabel('Test accuracy', color='tomato')
    ax1_left.set_title('D₂ and Accuracy vs. LR')
    ax1_left.axvspan(20, 40, alpha=0.1, color='green')

    ax_right.scatter(D2_mean, acc_mean, c=lr_fracs * 100,
                     cmap='viridis', s=80)
    ax_right.set_xlabel('D₂')
    ax_right.set_ylabel('Test accuracy')
    ax_right.set_title('D₂ vs. Accuracy')

    plt.tight_layout()
    plt.savefig(f'{DRIVE_DIR}/p1_interim_plot.pdf', bbox_inches='tight')
    plt.show()

    print(f"\\nLR fracs covered so far: {lr_fracs * 100}")
    print(f"D2 values: {D2_mean.round(3)}")
    print(f"Acc values: {acc_mean.round(4)}")

    if len(D2_mean) >= 3:
        best_acc = lr_fracs[np.argmax(acc_mean)] * 100
        best_d2  = lr_fracs[np.nanargmax(D2_mean)] * 100
        print(f"\\nPeak accuracy at {best_acc:.0f}% EoS")
        print(f"Peak D₂ at {best_d2:.0f}% EoS")

except FileNotFoundError:
    print("P1 results not yet saved. Run Cell 3 first.")
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — Resume P1 from checkpoint (if session dropped)
# ═══════════════════════════════════════════════════════════════════════════
CELL_6 = """
# This cell modifies p1 to skip already-computed LR fractions
# Only needed if the session was interrupted

import numpy as np, os

DRIVE_DIR = '/content/drive/MyDrive/chaos_generalization'
checkpoint = f'{DRIVE_DIR}/p1_cnn_results.npz'

if os.path.exists(checkpoint):
    data = np.load(checkpoint)
    done_fracs = set(data['lr_fracs'].tolist())
    print(f"Already completed: {sorted([f*100 for f in done_fracs])} % EoS")
    print("To resume, filter out done_fracs from lr_fractions in p1 script")
    print("Or just re-run — checkpoint skips are not auto-handled; manual edit needed")
else:
    print("No checkpoint found. Run from scratch.")
"""

# Print all cells for easy copy-paste
if __name__ == '__main__':
    cells = [
        ("CELL 1 — Mount Drive + verify GPU", CELL_1),
        ("CELL 2 — Upload scripts", CELL_2),
        ("CELL 3 — Run P1 (CNN D2 vs accuracy)", CELL_3),
        ("CELL 4 — Run P3 (depth comparison)", CELL_4),
        ("CELL 5 — Quick diagnostic plot", CELL_5),
        ("CELL 6 — Resume from checkpoint", CELL_6),
    ]

    for title, code in cells:
        print(f"\n{'='*60}")
        print(f"# {title}")
        print(f"{'='*60}")
        print(code)
