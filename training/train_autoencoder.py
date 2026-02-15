"""Training loop for the sparsity-regularized autoencoder."""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models.autoencoder import SparsityRegularizedAE
from .jacobian_utils import stochastic_jacobian_l1


def train_autoencoder(H_train, config, verbose=True):
    """Train the sparsity-regularized autoencoder on concatenated hidden states.

    Args:
        H_train: (num_samples, n_h) tensor of concatenated hidden states
        config: ThoughtCommConfig with AE hyperparameters
        verbose: whether to print progress

    Returns:
        model: trained SparsityRegularizedAE
        loss_history: dict with keys 'rec', 'jac', 'total' (lists of per-epoch losses)
    """
    model = SparsityRegularizedAE(
        n_h=config.n_h,
        n_z=config.n_z,
        hidden_dim=config.ae_hidden,
        num_layers=config.ae_num_layers,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.ae_lr)

    dataset = TensorDataset(H_train.float().to(config.device))
    loader = DataLoader(dataset, batch_size=config.ae_batch_size, shuffle=True)

    loss_history = {"rec": [], "jac": [], "total": []}

    iterator = tqdm(range(config.ae_epochs), desc="AE Training") if verbose else range(config.ae_epochs)
    for epoch in iterator:
        epoch_rec = 0.0
        epoch_jac = 0.0
        num_batches = 0

        for (H_batch,) in loader:
            # Forward pass
            H_rec, Z_hat = model(H_batch)

            # Reconstruction loss (Eq. 7, first term)
            L_rec = F.mse_loss(H_batch, H_rec)

            # Stochastic Jacobian L1 (Eq. 7, second term)
            Z_for_jac = model.encode(H_batch)
            L_jac = stochastic_jacobian_l1(
                model.decoder, Z_for_jac,
                num_sample_rows=config.jacobian_sample_rows,
            )

            loss = L_rec + config.jacobian_l1_weight * L_jac

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_rec += L_rec.item()
            epoch_jac += L_jac.item()
            num_batches += 1

        avg_rec = epoch_rec / num_batches
        avg_jac = epoch_jac / num_batches
        loss_history["rec"].append(avg_rec)
        loss_history["jac"].append(avg_jac)
        loss_history["total"].append(avg_rec + config.jacobian_l1_weight * avg_jac)

        if verbose and (epoch + 1) % 20 == 0:
            tqdm.write(
                f"Epoch {epoch + 1}/{config.ae_epochs}: "
                f"rec={avg_rec:.6f}, jac={avg_jac:.4f}, "
                f"total={loss_history['total'][-1]:.6f}"
            )

    return model, loss_history


def train_autoencoder_baseline(H_train, config, verbose=True):
    """Train autoencoder WITHOUT Jacobian sparsity (baseline for comparison).

    Args:
        H_train: (num_samples, n_h) tensor
        config: ThoughtCommConfig

    Returns:
        model: trained SparsityRegularizedAE (same architecture, no sparsity)
        loss_history: dict with 'rec' key
    """
    model = SparsityRegularizedAE(
        n_h=config.n_h,
        n_z=config.n_z,
        hidden_dim=config.ae_hidden,
        num_layers=config.ae_num_layers,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.ae_lr)
    dataset = TensorDataset(H_train.float().to(config.device))
    loader = DataLoader(dataset, batch_size=config.ae_batch_size, shuffle=True)

    loss_history = {"rec": []}

    iterator = tqdm(range(config.ae_epochs), desc="Baseline AE") if verbose else range(config.ae_epochs)
    for epoch in iterator:
        epoch_rec = 0.0
        num_batches = 0

        for (H_batch,) in loader:
            H_rec, _ = model(H_batch)
            loss = F.mse_loss(H_batch, H_rec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_rec += loss.item()
            num_batches += 1

        loss_history["rec"].append(epoch_rec / num_batches)

    return model, loss_history
