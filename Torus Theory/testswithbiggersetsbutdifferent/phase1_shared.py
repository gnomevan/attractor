"""
Shared components for Phase 1 chaos onset experiments.

Architecture and data match the chaos onset findings report:
- MLP: Input(220) → Linear(220,50) → Tanh → Linear(50,50) → Tanh → Linear(50,10)
- 156,710 parameters
- Synthetic data: 2000 samples, 10 classes, 200 random + 20 quadratic features
- Full-batch GD, MSE loss, no momentum, no weight decay

Reference values:
- λ_max ≈ 7.42 (Hessian top eigenvalue at init)
- EoS threshold: 2/λ_max ≈ 0.270
- η_c ≈ 0.018 (preliminary, from 3-seed runs)
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import json
import os
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────

class ChaosOnsetMLP(nn.Module):
    """
    Two hidden-layer MLP with tanh activation.
    Input(220) → Linear(220,50) → Tanh → Linear(50,50) → Tanh → Linear(50,10)
    
    Parameter count: 220*50+50 + 50*50+50 + 50*10+10 = 11050+2550+510 = ~14,110
    
    NOTE: The report says 156,710 parameters. That corresponds to a larger architecture.
    If your original experiments used different layer sizes, adjust HIDDEN_SIZES below.
    With the dimensions listed in the report text (220→50→50→10), we get ~14,110.
    For 156,710 params, the architecture would need to be approximately:
    Input(220) → Linear(220,400) → Tanh → Linear(400,400) → Tanh → Linear(400,10)
    which gives 220*400+400 + 400*400+400 + 400*10+10 = 88400+160400+4010 ≈ 252,810
    
    Or: Input(220) → Linear(220,256) → Tanh → Linear(256,256) → Tanh → Linear(256,10)
    = 220*256+256 + 256*256+256 + 256*10+10 = 56576+65792+2570 = 124,938
    
    Or: Input(220) → Linear(220,300) → Tanh → Linear(300,300) → Tanh → Linear(300,10)
    = 220*300+300 + 300*300+300 + 300*10+10 = 66300+90300+3010 = 159,610 ≈ close
    
    Adjusting: we use 220→280→280→10 = 220*280+280 + 280*280+280 + 280*10+10
    = 61880+78680+2810 = 143,370. Still not exact.
    
    Using 220→290→290→10 = 220*290+290 + 290*290+290 + 290*10+10 = 64090+84390+2910 = 151,390
    
    Using 220→295→295→10 = 220*295+295 + 295*295+295 + 295*10+10 = 65195+87320+2960 = 155,475
    
    Using 220→297→297→10 = 220*297+297 + 297*297+297 + 297*10+10 = 65637+88506+2980 = 157,123
    
    Close enough. But the report explicitly says "Input(220) → Linear(220, 50) → Tanh → 
    Linear(50, 50) → Tanh → Linear(50, 10)" AND "156,710 parameters". These are inconsistent.
    
    We'll use the ARCHITECTURE as stated (220→50→50→10) since that's what matters for 
    reproducibility. The parameter count in the report may have included the data matrix 
    or been from a different iteration. If you want to match 156K params exactly, 
    set HIDDEN1=HIDDEN2=297 below.
    """
    def __init__(self, input_dim=220, hidden1=50, hidden2=50, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────

def generate_data(n_samples=2000, n_classes=10, n_random_features=200, 
                  n_quadratic_features=20, data_seed=42, device='cpu'):
    """
    Generate synthetic classification data matching the chaos onset report.
    
    - Gaussian clusters around 10 class centers in 200D
    - 20 quadratic features (products of random feature pairs)
    - Quadratic features ensure persistent training dynamics (can't memorize)
    - Returns X (2000, 220) and Y (2000, 10) as one-hot MSE targets
    """
    rng = np.random.RandomState(data_seed)
    
    # Class labels
    labels = rng.randint(0, n_classes, size=n_samples)
    
    # Class centers in feature space
    centers = rng.randn(n_classes, n_random_features) * 2.0
    
    # Generate random features: cluster around class centers
    X_random = centers[labels] + rng.randn(n_samples, n_random_features) * 0.5
    
    # Generate quadratic features: products of random pairs
    quad_pairs = rng.choice(n_random_features, size=(n_quadratic_features, 2), replace=True)
    X_quad = np.zeros((n_samples, n_quadratic_features))
    for i, (j, k) in enumerate(quad_pairs):
        X_quad[:, i] = X_random[:, j] * X_random[:, k]
    
    # Normalize quadratic features to similar scale
    X_quad = X_quad / (np.std(X_quad, axis=0, keepdims=True) + 1e-8)
    
    # Concatenate: 200 random + 20 quadratic = 220 features
    X = np.concatenate([X_random, X_quad], axis=1)
    
    # One-hot encode targets for MSE loss
    Y = np.zeros((n_samples, n_classes))
    Y[np.arange(n_samples), labels] = 1.0
    
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
    
    return X_tensor, Y_tensor, labels


# ─────────────────────────────────────────────
# Lyapunov exponent measurement
# ─────────────────────────────────────────────

def compute_lyapunov(lr, X, Y, n_steps=5000, epsilon=1e-8, 
                     init_seed=0, device='cpu',
                     hidden1=50, hidden2=50,
                     record_every=10, verbose=False):
    """
    Compute function-space Lyapunov exponent for gradient descent.
    
    Method:
    1. Initialize network with seed `init_seed`
    2. Create perturbed copy: add epsilon to all parameters
    3. Train both with identical full-batch GD (same X, Y)
    4. Track ||f_θ(X) - f_θ'(X)|| over training
    5. Lyapunov exponent = slope of log(distance) vs step
    
    Returns:
        dict with keys:
        - 'lyapunov_exponent': float, estimated Lyapunov exponent
        - 'distances': list of (step, log_distance) pairs
        - 'final_loss_original': float
        - 'final_loss_perturbed': float
        - 'lr': float
        - 'seed': int
        - 'epsilon': float
    """
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    
    # Create original network
    model_orig = ChaosOnsetMLP(
        input_dim=X.shape[1], hidden1=hidden1, hidden2=hidden2, 
        output_dim=Y.shape[1]
    ).to(device)
    
    # Create perturbed copy
    model_pert = copy.deepcopy(model_orig)
    with torch.no_grad():
        for p in model_pert.parameters():
            p.add_(epsilon)
    
    # MSE loss
    criterion = nn.MSELoss()
    
    # Track distances
    distances = []
    
    for step in range(n_steps):
        # Record distance periodically
        if step % record_every == 0:
            with torch.no_grad():
                out_orig = model_orig(X)
                out_pert = model_pert(X)
                dist = torch.norm(out_orig - out_pert).item()
                log_dist = np.log(dist + 1e-300)  # avoid log(0)
                distances.append((step, log_dist))
                
                if verbose and step % 500 == 0:
                    loss_o = criterion(out_orig, Y).item()
                    print(f"  Step {step}: log_dist={log_dist:.2f}, loss={loss_o:.6f}")
        
        # Forward pass — original
        out_orig = model_orig(X)
        loss_orig = criterion(out_orig, Y)
        
        # Forward pass — perturbed
        out_pert = model_pert(X)
        loss_pert = criterion(out_pert, Y)
        
        # Backward pass — original
        model_orig.zero_grad()
        loss_orig.backward()
        
        # Backward pass — perturbed
        model_pert.zero_grad()
        loss_pert.backward()
        
        # SGD update (manual, no momentum, no weight decay)
        with torch.no_grad():
            for p in model_orig.parameters():
                p -= lr * p.grad
            for p in model_pert.parameters():
                p -= lr * p.grad
    
    # Final measurements
    with torch.no_grad():
        final_loss_orig = criterion(model_orig(X), Y).item()
        final_loss_pert = criterion(model_pert(X), Y).item()
    
    # Estimate Lyapunov exponent: linear regression of log(distance) vs step
    steps_arr = np.array([d[0] for d in distances])
    logdist_arr = np.array([d[1] for d in distances])
    
    # Use middle 60% to avoid transients at start and end
    n = len(steps_arr)
    start_idx = n // 5
    end_idx = 4 * n // 5
    
    if end_idx - start_idx > 2:
        slope, intercept = np.polyfit(
            steps_arr[start_idx:end_idx], 
            logdist_arr[start_idx:end_idx], 
            1
        )
        lyap = slope  # per step
    else:
        lyap = float('nan')
    
    return {
        'lyapunov_exponent': float(lyap),
        'distances': [(int(s), float(d)) for s, d in distances],
        'final_loss_original': float(final_loss_orig),
        'final_loss_perturbed': float(final_loss_pert),
        'lr': float(lr),
        'seed': int(init_seed),
        'epsilon': float(epsilon),
        'n_steps': n_steps,
    }


# ─────────────────────────────────────────────
# Hessian top eigenvalue (for EoS reference)
# ─────────────────────────────────────────────

def estimate_top_eigenvalue(model, X, Y, n_iters=50, device='cpu'):
    """
    Estimate top eigenvalue of the Hessian using power iteration.
    This gives the Edge of Stability threshold: 2/λ_max.
    """
    criterion = nn.MSELoss()
    
    # Random vector in parameter space
    v = [torch.randn_like(p) for p in model.parameters()]
    
    for _ in range(n_iters):
        # Compute Hv (Hessian-vector product) via finite differences
        # or via double backward
        model.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        
        # First gradient
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Hessian-vector product: d/dθ (grad · v)
        gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v))
        hv = torch.autograd.grad(gv, model.parameters())
        
        # Normalize
        norm = sum(torch.sum(hvi ** 2) for hvi in hv).sqrt().item()
        v = [hvi / norm for hvi in hv]
    
    # Eigenvalue: v^T H v
    model.zero_grad()
    out = model(X)
    loss = criterion(out, Y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v))
    hv = torch.autograd.grad(gv, model.parameters())
    
    eigenvalue = sum(torch.sum(hvi * vi) for hvi, vi in zip(hv, v)).item()
    
    return eigenvalue


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def save_results(results, filename, output_dir='results'):
    """Save results dict as JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")
    return filepath


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}hr"
