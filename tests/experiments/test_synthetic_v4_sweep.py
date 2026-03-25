"""Synthetic experiment sweep — match author Rebuttal conditions.

Author conditions (OpenReview Rebuttal):
  - 1000 samples, dim=1024 (hidden dimension), λ=0.01, MCC=0.85

Unknown parameters swept:
  - AE architecture (layers, hidden_dim, activation)
  - B_true sparsity (shared_fraction)
  - Learning rate
  - Mixing function depth

Run: python tests/experiments/test_synthetic_v4_sweep.py
GPU not required.
"""

import sys
import os
import json
import time
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np

from data.synthetic import generate_synthetic_data
from evaluation.synthetic_eval import compute_mcc_fast


# ============================================================
# AE with configurable activation
# ============================================================

class SweepAE(nn.Module):
    """AE with configurable activation for sweep experiments."""

    def __init__(self, n_h, n_z, hidden_dim, num_layers, activation='leakyrelu'):
        super().__init__()
        act_fn = {
            'leakyrelu': lambda: nn.LeakyReLU(0.2),
            'gelu': lambda: nn.GELU(),
            'relu': lambda: nn.ReLU(),
        }[activation]

        def build(in_d, out_d):
            layers = []
            dims = [in_d] + [hidden_dim] * (num_layers - 1) + [out_d]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act_fn())
            return nn.Sequential(*layers)

        self.encoder = build(n_h, n_z)
        self.decoder = build(n_z, n_h)

    def forward(self, H):
        Z = self.encoder(H)
        return self.decoder(Z)

    def encode(self, H):
        return self.encoder(H)


# ============================================================
# Training with stochastic Jacobian L1
# ============================================================

def train_ae_synthetic(ae, X, lam=0.01, lr=1e-3, epochs=200, batch_size=128,
                       jac_sample_rows=64, verbose=False):
    """Train AE with Jacobian L1 on synthetic data. Returns trained AE."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ae = ae.to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)

    # Normalize
    X_mean = X.mean(0).to(device)
    X_std = X.std(0).clamp(min=1e-8).to(device)
    X_norm = ((X.to(device) - X_mean) / X_std)

    n = X_norm.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_rec, total_jac = 0.0, 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch = X_norm[perm[i:i + batch_size]]
            if batch.shape[0] < 2:
                continue

            Z = ae.encoder(batch)
            X_hat = ae.decoder(Z)

            rec_loss = ((batch - X_hat) ** 2).mean()

            # Stochastic Jacobian L1
            if lam > 0:
                Z_jac = Z.detach().requires_grad_(True)
                X_dec = ae.decoder(Z_jac)
                n_h = X_dec.shape[1]
                k = min(jac_sample_rows, n_h)
                rows = torch.randperm(n_h)[:k]

                jac_l1 = 0.0
                for r in rows:
                    grad = torch.autograd.grad(
                        X_dec[:, r].sum(), Z_jac,
                        create_graph=True, retain_graph=True
                    )[0]
                    jac_l1 = jac_l1 + grad.abs().mean()
                jac_l1 = jac_l1 / k
            else:
                jac_l1 = torch.tensor(0.0)

            loss = rec_loss + lam * jac_l1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()

            total_rec += rec_loss.item()
            total_jac += jac_l1.item() if isinstance(jac_l1, torch.Tensor) else jac_l1
            n_batches += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: rec={total_rec/n_batches:.4f}, jac={total_jac/n_batches:.4f}")
        elif not verbose and (epoch + 1) % 100 == 0:
            print(f"    ep{epoch+1}/{epochs}", end="", flush=True)

    if not verbose:
        print()  # newline after progress dots
    return ae, X_mean.cpu(), X_std.cpu()


def evaluate_mcc(ae, X, Z_true, X_mean, X_std):
    """Compute MCC between estimated and true latents."""
    device = next(ae.parameters()).device
    ae.eval()
    with torch.no_grad():
        X_d = X.to(device)
        X_norm = (X_d - X_mean.to(device)) / X_std.to(device)
        Z_hat = ae.encode(X_norm)

    Z_hat_np = Z_hat.cpu().numpy()
    Z_true_np = Z_true.numpy()

    mcc, perm = compute_mcc_fast(Z_true_np, Z_hat_np)
    return mcc


# ============================================================
# Sweep runner
# ============================================================

def run_single_experiment(config):
    """Run one synthetic experiment and return MCC."""
    t0 = time.time()

    # Generate data
    X, Z, B_true, groups, mixing = generate_synthetic_data(
        dim=config['dim'],
        num_samples=config['num_samples'],
        seed=config.get('seed', 42),
        num_layers=config['mixing_layers'],
        shared_fraction=config['shared_fraction'],
    )

    b_nonzero_frac = B_true.float().mean().item()

    # Build AE
    ae = SweepAE(
        n_h=config['dim'],
        n_z=config['dim'],
        hidden_dim=config['ae_hidden'],
        num_layers=config['ae_layers'],
        activation=config['ae_activation'],
    )

    # Train
    ae, X_mean, X_std = train_ae_synthetic(
        ae, X,
        lam=config['lam'],
        lr=config['lr'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=config.get('verbose', False),
    )

    # Evaluate
    mcc = evaluate_mcc(ae, X, Z, X_mean, X_std)
    elapsed = time.time() - t0

    return {
        **config,
        'mcc': mcc,
        'b_nonzero_frac': b_nonzero_frac,
        'time_sec': elapsed,
    }


# ============================================================
# Main sweep
# ============================================================

if __name__ == '__main__':
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Fixed author conditions
    BASE = dict(
        dim=1024,
        num_samples=1000,
        lam=0.01,
        epochs=200,
        batch_size=128,
        seed=42,
    )

    all_results = []

    # =========================================
    # Phase 1: Baseline (our current defaults)
    # =========================================
    print("=" * 60)
    print("Phase 1: Baseline (dim=1024, n=1000, λ=0.01)")
    print("=" * 60)

    config = {**BASE, 'ae_hidden': 2048, 'ae_layers': 3, 'ae_activation': 'leakyrelu',
              'shared_fraction': 1/3, 'mixing_layers': 1, 'lr': 1e-3, 'verbose': True}
    result = run_single_experiment(config)
    result['phase'] = 'baseline'
    all_results.append(result)
    print(f"  MCC = {result['mcc']:.4f} (B nonzero = {result['b_nonzero_frac']:.2%}, {result['time_sec']:.0f}s)")
    print()

    # =========================================
    # Phase 2: AE architecture sweep
    # =========================================
    print("=" * 60)
    print("Phase 2: AE architecture sweep")
    print("=" * 60)

    ae_configs = list(itertools.product(
        ['leakyrelu', 'gelu', 'relu'],  # activation
        [2, 3],                          # layers
        [1024, 2048],                    # hidden_dim
    ))

    for act, layers, hidden in ae_configs:
        config = {**BASE, 'ae_hidden': hidden, 'ae_layers': layers, 'ae_activation': act,
                  'shared_fraction': 1/3, 'mixing_layers': 1, 'lr': 1e-3}
        result = run_single_experiment(config)
        result['phase'] = 'ae_sweep'
        all_results.append(result)
        print(f"  {act:12s} L={layers} H={hidden:4d}  MCC={result['mcc']:.4f}  ({result['time_sec']:.0f}s)")

    best_ae = max([r for r in all_results if r['phase'] == 'ae_sweep'], key=lambda r: r['mcc'])
    print(f"\n  Best AE: {best_ae['ae_activation']} L={best_ae['ae_layers']} H={best_ae['ae_hidden']} MCC={best_ae['mcc']:.4f}")
    print()

    # =========================================
    # Phase 3: B_true sparsity sweep
    # =========================================
    print("=" * 60)
    print("Phase 3: B_true sparsity sweep (using best AE)")
    print("=" * 60)

    for sf in [0.1, 0.2, 1/3, 0.5]:
        config = {**BASE, 'ae_hidden': best_ae['ae_hidden'], 'ae_layers': best_ae['ae_layers'],
                  'ae_activation': best_ae['ae_activation'],
                  'shared_fraction': sf, 'mixing_layers': 1, 'lr': 1e-3}
        result = run_single_experiment(config)
        result['phase'] = 'sparsity_sweep'
        all_results.append(result)
        print(f"  shared={sf:.2f}  B_nonzero={result['b_nonzero_frac']:.2%}  MCC={result['mcc']:.4f}  ({result['time_sec']:.0f}s)")
    print()

    # =========================================
    # Phase 4: Learning rate sweep
    # =========================================
    print("=" * 60)
    print("Phase 4: LR sweep (using best AE + best sparsity)")
    print("=" * 60)

    best_sp = max([r for r in all_results if r['phase'] == 'sparsity_sweep'], key=lambda r: r['mcc'])

    for lr in [1e-3, 5e-4, 1e-4, 1e-5]:
        config = {**BASE, 'ae_hidden': best_ae['ae_hidden'], 'ae_layers': best_ae['ae_layers'],
                  'ae_activation': best_ae['ae_activation'],
                  'shared_fraction': best_sp['shared_fraction'], 'mixing_layers': 1, 'lr': lr}
        result = run_single_experiment(config)
        result['phase'] = 'lr_sweep'
        all_results.append(result)
        print(f"  lr={lr:.0e}  MCC={result['mcc']:.4f}  ({result['time_sec']:.0f}s)")
    print()

    # =========================================
    # Phase 5: Nonlinear mixing
    # =========================================
    print("=" * 60)
    print("Phase 5: Nonlinear mixing (best overall config)")
    print("=" * 60)

    best_lr = max([r for r in all_results if r['phase'] == 'lr_sweep'], key=lambda r: r['mcc'])

    for ml in [1, 3]:
        config = {**BASE, 'ae_hidden': best_ae['ae_hidden'], 'ae_layers': best_ae['ae_layers'],
                  'ae_activation': best_ae['ae_activation'],
                  'shared_fraction': best_sp['shared_fraction'],
                  'mixing_layers': ml, 'lr': best_lr['lr']}
        result = run_single_experiment(config)
        result['phase'] = 'mixing_sweep'
        all_results.append(result)
        print(f"  mixing_layers={ml}  MCC={result['mcc']:.4f}  ({result['time_sec']:.0f}s)")
    print()

    # =========================================
    # Summary
    # =========================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_overall = max(all_results, key=lambda r: r['mcc'])
    print(f"Best MCC: {best_overall['mcc']:.4f}")
    print(f"  AE: {best_overall['ae_activation']} L={best_overall['ae_layers']} H={best_overall['ae_hidden']}")
    print(f"  shared_fraction: {best_overall['shared_fraction']}")
    print(f"  lr: {best_overall['lr']}")
    print(f"  mixing_layers: {best_overall['mixing_layers']}")
    print(f"  B_nonzero: {best_overall['b_nonzero_frac']:.2%}")
    print()

    if best_overall['mcc'] > 0.75:
        print("*** SUCCESS: MCC > 0.75 achieved! ***")
    else:
        print(f"*** FAILURE: Best MCC = {best_overall['mcc']:.4f} < 0.75 ***")
        print("AE + Jacobian L1 does not reproduce author results under any tested condition.")

    # Save all results
    # Convert non-serializable values
    for r in all_results:
        for k, v in r.items():
            if isinstance(v, float) and (v != v):  # NaN
                r[k] = None

    output_path = os.path.join(RESULTS_DIR, 'synthetic_v4_sweep.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")
