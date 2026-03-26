"""Training loop for GIN-based nonlinear ICA.

Two-stage approach:
  Stage 1: FastICA (linear ICA) for pre-processing
  Stage 2: GIN (nonlinear ICA via NLL) on ICA output

The GIN learns the residual nonlinear transformation that ICA cannot capture.
"""

import copy
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from sklearn.decomposition import FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from models.gin_model import GINWrapper


def stage1_ica(H_np, n_components=None, random_state=42):
    """Stage 1: Linear ICA pre-processing.

    Args:
        H_np: (n_samples, n_dim) numpy array of observations.
        n_components: Number of ICA components. Default: same as input dim.
        random_state: Random seed.

    Returns:
        Z_ica: (n_samples, n_dim) numpy array of ICA-transformed data.
        ica_model: Fitted FastICA model (for later use).
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    n_dim = H_np.shape[1]
    if n_components is None:
        n_components = n_dim

    ica = FastICA(
        n_components=n_components,
        max_iter=2000,
        tol=1e-4,
        random_state=random_state,
    )
    Z_ica = ica.fit_transform(H_np)
    return Z_ica, ica


def stage2_gin(Z_ica_np, n_coupling_layers=8, subnet_hidden=None,
               n_epochs=500, lr=5e-4, weight_decay=0.01,
               batch_size=256, device='cpu', verbose=True):
    """Stage 2: GIN refinement on ICA output.

    Trains GIN with Laplace NLL to learn residual nonlinear transformation.

    Args:
        Z_ica_np: (n_samples, n_dim) numpy array from Stage 1.
        n_coupling_layers: Number of GIN coupling blocks.
        subnet_hidden: Hidden dim for GIN subnets. Default: max(dim, 10).
        n_epochs: Training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization (critical for preventing MCC collapse).
        batch_size: Batch size.
        device: 'cpu' or 'cuda'.
        verbose: Print progress.

    Returns:
        gin: Trained GINWrapper model.
        loss_history: List of per-epoch NLL values.
    """
    n_dim = Z_ica_np.shape[1]
    n_samples = Z_ica_np.shape[0]

    if subnet_hidden is None:
        subnet_hidden = max(n_dim, 10)

    # Build GIN
    gin = GINWrapper(
        n_dim=n_dim,
        n_coupling_layers=n_coupling_layers,
        subnet_hidden=subnet_hidden,
    ).to(device)

    # Normalize input
    Z_ica = torch.tensor(Z_ica_np, dtype=torch.float32)
    z_mean = Z_ica.mean(0)
    z_std = Z_ica.std(0).clamp(min=1e-8)
    Z_norm = ((Z_ica - z_mean) / z_std).to(device)

    # Optimizer with weight decay (critical finding)
    optimizer = optim.Adam(gin.inn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_history = []
    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        gin.train()
        perm = torch.randperm(n_samples)
        epoch_nll = 0.0
        n_batch = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size].to(device)
            z_batch = Z_norm[idx]

            # Forward: Z_ica -> Z_hat
            z_hat, _ = gin.inn(z_batch)

            # NLL: Laplace(0, 1) prior
            nll = (z_hat.abs() + np.log(2)).sum(dim=1).mean()

            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(gin.inn.parameters(), 1.0)
            optimizer.step()

            epoch_nll += nll.item()
            n_batch += 1

        scheduler.step()
        avg_nll = epoch_nll / n_batch
        loss_history.append(avg_nll)

        # Track best model
        if avg_nll < best_loss:
            best_loss = avg_nll
            best_state = copy.deepcopy(gin.state_dict())

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  GIN Epoch {epoch+1}/{n_epochs}: NLL={avg_nll:.4f}")

    # Restore best model and set norm stats
    if best_state is not None:
        gin.load_state_dict(best_state)
    gin.set_norm_stats(z_mean, z_std)

    return gin, loss_history


def two_stage_train(H_np, n_coupling_layers=8, subnet_hidden=None,
                    n_epochs=500, lr=5e-4, weight_decay=0.01,
                    batch_size=256, device='cpu', verbose=True):
    """Full two-stage pipeline: ICA -> GIN.

    Args:
        H_np: (n_samples, n_dim) numpy array of observations.
        **kwargs: Passed to stage2_gin.

    Returns:
        gin: Trained GINWrapper model.
        ica_model: Fitted FastICA model.
        Z_ica: ICA output (numpy).
        loss_history: GIN training history.
    """
    n_dim = H_np.shape[1]

    if verbose:
        print(f"Stage 1: FastICA (dim={n_dim})...")
    Z_ica, ica_model = stage1_ica(H_np)
    if verbose:
        print(f"  ICA complete.")

    if verbose:
        print(f"Stage 2: GIN refinement (coupling={n_coupling_layers}, "
              f"subnet_h={subnet_hidden or max(n_dim,10)})...")
    gin, loss_history = stage2_gin(
        Z_ica,
        n_coupling_layers=n_coupling_layers,
        subnet_hidden=subnet_hidden,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    return gin, ica_model, Z_ica, loss_history
