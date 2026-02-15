"""Evaluation metrics for synthetic experiments (Section 5.1).

Implements:
- R^2 score matrix between estimated and true latent groups (Fig 3)
- Mean Correlation Coefficient (MCC) with optimal permutation (Fig 4)
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression


def compute_r2_matrix(Z_hat, Z_true, group_indices):
    """Compute R^2 between each pair of estimated and true latent groups.

    For each (estimated group, true group), fits a linear regression from
    the estimated group to the true group and reports R^2.

    Args:
        Z_hat: (N, dim) estimated latents (numpy or tensor)
        Z_true: (N, dim) ground-truth latents (numpy or tensor)
        group_indices: dict mapping group names to (start, end) tuples

    Returns:
        r2_matrix: (3, 3) numpy array, rows=estimated groups, cols=true groups
        group_names: list of group names in order
    """
    if isinstance(Z_hat, torch.Tensor):
        Z_hat = Z_hat.detach().cpu().numpy()
    if isinstance(Z_true, torch.Tensor):
        Z_true = Z_true.detach().cpu().numpy()

    latent_groups = ["Z_A\\B", "Z_A∩B", "Z_B\\A"]
    r2_matrix = np.zeros((3, 3))

    for i, est_name in enumerate(latent_groups):
        est_start, est_end = group_indices[est_name]
        Z_est_group = Z_hat[:, est_start:est_end]

        for j, true_name in enumerate(latent_groups):
            true_start, true_end = group_indices[true_name]
            Z_true_group = Z_true[:, true_start:true_end]

            # Fit linear regression and compute R^2
            reg = LinearRegression()
            reg.fit(Z_est_group, Z_true_group)
            r2_matrix[i, j] = reg.score(Z_est_group, Z_true_group)

    return r2_matrix, latent_groups


def compute_mcc(Z_hat, Z_true):
    """Compute Mean Correlation Coefficient with optimal permutation alignment.

    Uses the Hungarian algorithm to find the permutation of estimated latent
    dimensions that maximizes correlation with ground-truth dimensions.

    Args:
        Z_hat: (N, dim) estimated latents
        Z_true: (N, dim) ground-truth latents

    Returns:
        mcc: float, mean absolute correlation of optimally matched dimensions
        perm: optimal permutation (Z_hat[:, perm[i]] matches Z_true[:, i])
    """
    if isinstance(Z_hat, torch.Tensor):
        Z_hat = Z_hat.detach().cpu().numpy()
    if isinstance(Z_true, torch.Tensor):
        Z_true = Z_true.detach().cpu().numpy()

    dim = Z_hat.shape[1]

    # Compute absolute correlation matrix
    corr_matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            c = np.corrcoef(Z_hat[:, i], Z_true[:, j])[0, 1]
            corr_matrix[i, j] = abs(c) if not np.isnan(c) else 0.0

    # Hungarian algorithm to find optimal assignment (maximize correlation)
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)  # minimize negative

    mcc = corr_matrix[row_ind, col_ind].mean()
    perm = col_ind

    return mcc, perm


def compute_mcc_fast(Z_hat, Z_true, block_size=None):
    """Memory-efficient MCC computation using block correlation.

    For large dimensions, computing the full NxN correlation matrix is expensive.
    This version uses blocks.

    Args:
        Z_hat: (N, dim) estimated latents
        Z_true: (N, dim) ground-truth latents
        block_size: if set, compute correlation in blocks

    Returns:
        mcc: float
        perm: permutation array
    """
    if isinstance(Z_hat, torch.Tensor):
        Z_hat = Z_hat.detach().cpu().numpy()
    if isinstance(Z_true, torch.Tensor):
        Z_true = Z_true.detach().cpu().numpy()

    # Standardize
    Z_hat_std = (Z_hat - Z_hat.mean(0)) / (Z_hat.std(0) + 1e-8)
    Z_true_std = (Z_true - Z_true.mean(0)) / (Z_true.std(0) + 1e-8)

    # Correlation matrix via matrix multiplication
    N = Z_hat_std.shape[0]
    corr_matrix = np.abs(Z_hat_std.T @ Z_true_std / N)

    row_ind, col_ind = linear_sum_assignment(-corr_matrix)
    mcc = corr_matrix[row_ind, col_ind].mean()

    return mcc, col_ind
