"""Training loop for the sparsity-regularized autoencoder."""

import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models.autoencoder import SparsityRegularizedAE
from .jacobian_utils import stochastic_jacobian_l1, stochastic_jacobian_group_l1


def train_autoencoder(H_train, config, verbose=True):
    """Train the sparsity-regularized autoencoder on concatenated hidden states.

    Includes gradient clipping, cosine LR schedule, and early stopping
    with best model restoration.

    Args:
        H_train: (num_samples, n_h) tensor of concatenated hidden states
        config: ThoughtCommConfig with AE hyperparameters
        verbose: whether to print progress

    Returns:
        model: trained SparsityRegularizedAE (best model by total loss)
        loss_history: dict with keys 'rec', 'jac', 'total' (lists of per-epoch losses)
        norm_stats: dict with 'mean' and 'std' tensors for denormalization
    """
    model = SparsityRegularizedAE(
        n_h=config.n_h,
        n_z=config.n_z,
        hidden_dim=config.ae_hidden,
        num_layers=config.ae_num_layers,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.ae_lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.ae_epochs, eta_min=config.ae_lr_min
    )

    H = H_train.float()
    H_mean = H.mean(dim=0)
    H_std = H.std(dim=0).clamp(min=1e-8)
    H_normalized = (H - H_mean) / H_std
    norm_stats = {"mean": H_mean, "std": H_std}

    dataset = TensorDataset(H_normalized.to(config.device))
    loader = DataLoader(dataset, batch_size=config.ae_batch_size, shuffle=True)

    loss_history = {"rec": [], "jac": [], "total": []}

    # Early stopping state
    best_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    # Regularization type (fixed for entire training)
    reg_type = getattr(config, "jacobian_reg_type", "l1")
    if reg_type == "group_l1":
        jac_weight = getattr(config, "jacobian_group_weight", 1.0)
    else:
        jac_weight = config.jacobian_l1_weight

    if verbose:
        tqdm.write(f"Jacobian reg: {reg_type}, weight={jac_weight}")

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

            # Jacobian regularization (Eq. 7, second term)
            Z_for_jac = model.encode(H_batch)

            if reg_type == "group_l1":
                L_jac = stochastic_jacobian_group_l1(
                    model.decoder, Z_for_jac,
                    num_agents=config.num_agents,
                    hidden_size=config.hidden_size,
                )
            else:
                L_jac = stochastic_jacobian_l1(
                    model.decoder, Z_for_jac,
                    num_sample_rows=config.jacobian_sample_rows,
                )

            loss = L_rec + jac_weight * L_jac

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.ae_grad_clip
            )
            optimizer.step()

            epoch_rec += L_rec.item()
            epoch_jac += L_jac.item()
            num_batches += 1

        scheduler.step()

        avg_rec = epoch_rec / num_batches
        avg_jac = epoch_jac / num_batches
        avg_total = avg_rec + jac_weight * avg_jac
        loss_history["rec"].append(avg_rec)
        loss_history["jac"].append(avg_jac)
        loss_history["total"].append(avg_total)

        # Early stopping: track best model
        if avg_total < best_loss:
            best_loss = avg_total
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"Epoch {epoch + 1}/{config.ae_epochs}: "
                f"rec={avg_rec:.6f}, jac={avg_jac:.4f}, "
                f"total={avg_total:.6f}, lr={current_lr:.2e}, "
                f"best={best_loss:.6f}, patience={patience_counter}/{config.ae_patience}"
            )

        if patience_counter >= config.ae_patience:
            if verbose:
                tqdm.write(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {config.ae_patience} epochs). "
                    f"Best total loss: {best_loss:.6f}"
                )
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if verbose:
            tqdm.write(f"Restored best model (total loss: {best_loss:.6f})")

    # Embed norm_stats in model so encode() auto-normalizes raw inputs
    model.set_norm_stats(H_mean, H_std)

    return model, loss_history, norm_stats


def train_autoencoder_baseline(H_train, config, verbose=True, norm_stats=None):
    """Train autoencoder WITHOUT Jacobian sparsity (baseline for comparison).

    Args:
        H_train: (num_samples, n_h) tensor
        config: ThoughtCommConfig
        norm_stats: if provided, use these stats for normalization (from train_autoencoder)

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

    H = H_train.float()
    if norm_stats is not None:
        H_mean = norm_stats["mean"]
        H_std = norm_stats["std"]
    else:
        H_mean = H.mean(dim=0)
        H_std = H.std(dim=0).clamp(min=1e-8)
        norm_stats = {"mean": H_mean, "std": H_std}
    H = (H - H_mean) / H_std

    dataset = TensorDataset(H.to(config.device))
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

    # Embed norm_stats in model so encode() auto-normalizes raw inputs
    model.set_norm_stats(H_mean, H_std)

    return model, loss_history, norm_stats
