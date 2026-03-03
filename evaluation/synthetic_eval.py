"""Evaluation metrics for synthetic experiments (Section 5.1).

Implements:
- R^2 score matrix between estimated and true latent groups (Fig 3)
- Mean Correlation Coefficient (MCC) with optimal permutation (Fig 4)
- B matrix recovery from ICA mixing matrix or correlation analysis
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

    # col_ind[i] = Z_true dim matched by Z_hat dim i (Z_hat→Z_true mapping)
    # We need perm such that Z_hat[:, perm[j]] ~ Z_true[:, j] (Z_true→Z_hat mapping)
    # This is the inverse permutation
    perm = np.argsort(col_ind)

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

    # col_ind[i] = Z_true dim matched by Z_hat dim i (Z_hat→Z_true mapping)
    # Invert to get perm[j] = Z_hat dim for Z_true dim j
    perm = np.argsort(col_ind)

    return mcc, perm


def recover_b_matrix(X, Z_hat, n_obs_a, perm=None, method='agent_aware'):
    """Recover binary dependency matrix B from estimated latents.

    B[i,j] = 1 means observed variable X_i depends on latent Z_j.
    The ground truth has block structure:
      - X_A (rows 0:n_obs_a) depends on Z_private_a and Z_shared
      - X_B (rows n_obs_a:) depends on Z_shared and Z_private_b

    Args:
        X: (N, n_h) observed data (numpy)
        Z_hat: (N, n_z) estimated latents (numpy), already permuted if perm given
        n_obs_a: number of observed variables for agent A
        perm: permutation from MCC computation (Z_hat[:,perm[j]] ~ Z_true[:,j])
        method: 'agent_aware' (recommended) or 'correlation'

    Returns:
        B_est: (n_h, n_z) binary matrix
        scores: (n_h, n_z) continuous influence scores
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Z_hat, torch.Tensor):
        Z_hat = Z_hat.detach().cpu().numpy()

    # Apply permutation to align Z_hat with ground truth ordering
    if perm is not None:
        Z_hat = Z_hat[:, perm]

    n_h = X.shape[1]
    n_z = Z_hat.shape[1]
    n_obs_b = n_h - n_obs_a

    if method == 'agent_aware':
        return _recover_b_agent_aware(X, Z_hat, n_obs_a)
    elif method == 'correlation':
        return _recover_b_correlation(X, Z_hat, n_obs_a)
    else:
        raise ValueError(f"Unknown method: {method}")


def _recover_b_agent_aware(X, Z_hat, n_obs_a):
    """Agent-aware B matrix recovery.

    For each latent j, compute its influence on X_A vs X_B:
    - influence_A[j] = mean |corr(X_A_i, Z_j)| over i in agent A
    - influence_B[j] = mean |corr(X_B_i, Z_j)| over i in agent B

    Then classify each latent as:
    - A-only: influence_A >> influence_B
    - B-only: influence_B >> influence_A
    - Shared: both significant
    """
    n_h, n_z = X.shape[1], Z_hat.shape[1]
    n_obs_b = n_h - n_obs_a

    # Compute correlation matrix |corr(X_i, Z_j)|
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Z_std = (Z_hat - Z_hat.mean(0)) / (Z_hat.std(0) + 1e-8)
    N = X.shape[0]
    corr = np.abs(X_std.T @ Z_std / N)  # (n_h, n_z)

    # Per-latent influence on each agent
    inf_a = corr[:n_obs_a, :].mean(axis=0)   # (n_z,) mean influence on X_A
    inf_b = corr[n_obs_a:, :].mean(axis=0)   # (n_z,) mean influence on X_B

    # Adaptive thresholds based on the distribution of influences
    # For each latent, determine if it affects A, B, or both
    B_est = np.zeros((n_h, n_z), dtype=int)

    for j in range(n_z):
        ratio_ab = inf_a[j] / (inf_b[j] + 1e-10)
        ratio_ba = inf_b[j] / (inf_a[j] + 1e-10)

        # Per-row thresholding within the relevant agent block
        if ratio_ab > 1.5:
            # Primarily agent A's latent
            B_est[:n_obs_a, j] = 1
        elif ratio_ba > 1.5:
            # Primarily agent B's latent
            B_est[n_obs_a:, j] = 1
        else:
            # Shared latent — affects both agents
            B_est[:n_obs_a, j] = 1
            B_est[n_obs_a:, j] = 1

    return B_est, corr


def _recover_b_correlation(X, Z_hat, n_obs_a):
    """Correlation-based B matrix recovery with row-wise thresholding."""
    n_h, n_z = X.shape[1], Z_hat.shape[1]

    X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Z_std = (Z_hat - Z_hat.mean(0)) / (Z_hat.std(0) + 1e-8)
    N = X.shape[0]
    corr = np.abs(X_std.T @ Z_std / N)

    # Row-wise thresholding: keep top correlations per observed variable
    B_est = np.zeros((n_h, n_z), dtype=int)
    for i in range(n_h):
        thresh = np.percentile(corr[i, :], 50)  # median threshold
        B_est[i, :] = (corr[i, :] > thresh).astype(int)

    return B_est, corr


def evaluate_b_matrix(B_est, B_true):
    """Evaluate B matrix recovery quality.

    Args:
        B_est: (n_h, n_z) estimated binary matrix
        B_true: (n_h, n_z) ground truth binary matrix

    Returns:
        metrics: dict with accuracy, precision, recall, F1
    """
    if isinstance(B_est, torch.Tensor):
        B_est = B_est.numpy()
    if isinstance(B_true, torch.Tensor):
        B_true = B_true.numpy()

    accuracy = (B_est == B_true).mean()
    tp = (B_est * B_true).sum()
    fp = (B_est * (1 - B_true)).sum()
    fn = ((1 - B_est) * B_true).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
    }
