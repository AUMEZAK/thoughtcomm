"""Jacobian computation utilities for sparsity-regularized autoencoder.

Provides:
1. Stochastic Jacobian L1 estimation for training (fast, unbiased)
2. Full Jacobian computation for B-matrix extraction (post-training)
3. Binary pattern B(J_f) computation and visualization
"""

import torch
import torch.nn as nn


def stochastic_jacobian_l1(decoder: nn.Module, Z_hat: torch.Tensor,
                           num_sample_rows: int = 64) -> torch.Tensor:
    """Estimate ||J_{f_hat}||_1 using stochastic row sampling.

    Instead of computing the full (n_h, n_z) Jacobian, we sample random
    output dimensions and compute their gradients w.r.t. Z_hat.

    Args:
        decoder: the decoder network f_hat
        Z_hat: (batch, n_z) encoded latents (will be detached and re-attached)
        num_sample_rows: number of output dimensions to sample per step

    Returns:
        jacobian_l1: scalar estimate of ||J||_1, differentiable
    """
    Z = Z_hat.detach().requires_grad_(True)
    H_rec = decoder(Z)  # (batch, n_h)
    n_h = H_rec.shape[1]

    # Sample random output dimensions
    row_indices = torch.randint(0, n_h, (num_sample_rows,), device=Z.device)

    jacobian_rows = []
    for idx in row_indices:
        grad_output = torch.zeros_like(H_rec)
        grad_output[:, idx] = 1.0

        grads = torch.autograd.grad(
            outputs=H_rec,
            inputs=Z,
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
        )[0]  # (batch, n_z)
        jacobian_rows.append(grads)

    # (num_sample_rows, batch, n_z)
    J_sampled = torch.stack(jacobian_rows, dim=0)

    # L1 norm, averaged over batch, scaled to full Jacobian
    l1_norm = J_sampled.abs().mean() * (n_h / num_sample_rows)
    return l1_norm


def compute_full_jacobian(decoder: nn.Module, Z_hat: torch.Tensor) -> torch.Tensor:
    """Compute full Jacobian of decoder w.r.t. input.

    Uses torch.func for efficient vectorized computation.

    Args:
        Z_hat: (batch, n_z)

    Returns:
        J: (batch, n_h, n_z) full Jacobian matrices
    """
    from torch.func import jacrev, vmap

    def single_decode(z):
        return decoder(z.unsqueeze(0)).squeeze(0)

    J = vmap(jacrev(single_decode))(Z_hat)
    return J


def compute_full_jacobian_batched(decoder: nn.Module, Z_all: torch.Tensor,
                                  sub_batch: int = 8,
                                  device: str = "cuda") -> torch.Tensor:
    """Compute full Jacobian in sub-batches for memory efficiency.

    Args:
        decoder: the decoder network
        Z_all: (N, n_z) all latent codes
        sub_batch: process this many at a time
        device: computation device

    Returns:
        J: (N, n_h, n_z) full Jacobian on CPU
    """
    J_list = []
    decoder_device = next(decoder.parameters()).device

    for i in range(0, len(Z_all), sub_batch):
        Z_sub = Z_all[i:i + sub_batch].to(device)
        J_sub = compute_full_jacobian(decoder, Z_sub)
        J_list.append(J_sub.cpu())
        if device == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(J_list, dim=0)


def compute_binary_pattern(decoder: nn.Module, Z_batch: torch.Tensor,
                           threshold: float = 0.01,
                           sub_batch: int = 8,
                           device: str = "cuda") -> torch.Tensor:
    """Compute binary Jacobian pattern B(J_{f_hat}).

    Evaluates the Jacobian on a batch of samples and checks which entries
    are ever non-zero (above threshold).

    Args:
        decoder: trained decoder network
        Z_batch: (N, n_z) representative latent codes
        threshold: absolute value threshold for "non-zero"
        sub_batch: sub-batch size for memory efficiency
        device: computation device

    Returns:
        B: (n_h, n_z) binary matrix (int tensor)
    """
    J = compute_full_jacobian_batched(decoder, Z_batch, sub_batch, device)
    # Take max absolute value across all samples
    J_max = J.abs().max(dim=0).values  # (n_h, n_z)
    B = (J_max > threshold).int()
    return B
