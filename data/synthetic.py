"""Synthetic data generation for identifiability experiments (Section 5.1).

Generates data from a structured invertible transformation of multivariate
Laplacian variables, with known shared/private latent structure.
"""

import torch
import torch.nn as nn
import numpy as np


class StructuredInvertibleMLP(nn.Module):
    """Invertible MLP with block-sparse Jacobian structure.

    The Jacobian respects the dependency: X_A depends on Z_{A\\B} and Z_{A∩B},
    X_B depends on Z_{A∩B} and Z_{B\\A}.
    """

    def __init__(self, dim, n_private_a, n_shared, n_private_b,
                 n_obs_a, n_obs_b, num_layers=3, seed=42):
        super().__init__()
        self.dim = dim
        self.n_private_a = n_private_a
        self.n_shared = n_shared
        self.n_private_b = n_private_b
        self.n_obs_a = n_obs_a
        self.n_obs_b = n_obs_b

        torch.manual_seed(seed)

        # X_A network: maps Z_{A\\B} ∪ Z_{A∩B} -> X_A
        input_a = n_private_a + n_shared
        self.net_a = self._build_invertible_net(input_a, n_obs_a, num_layers)

        # X_B network: maps Z_{A∩B} ∪ Z_{B\\A} -> X_B
        input_b = n_shared + n_private_b
        self.net_b = self._build_invertible_net(input_b, n_obs_b, num_layers)

    def _build_invertible_net(self, in_dim, out_dim, num_layers):
        """Build a network with invertible-friendly architecture."""
        layers = []
        dims = [in_dim] + [max(in_dim, out_dim)] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1], bias=True)
            # Initialize with well-conditioned random weights
            nn.init.orthogonal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, Z):
        """Map latent Z to observed X.

        Args:
            Z: (batch, dim) where columns are ordered as
               [Z_{A\\B} | Z_{A∩B} | Z_{B\\A}]

        Returns:
            X: (batch, n_obs_a + n_obs_b)
        """
        # Split latent into groups
        z_private_a = Z[:, :self.n_private_a]
        z_shared = Z[:, self.n_private_a:self.n_private_a + self.n_shared]
        z_private_b = Z[:, self.n_private_a + self.n_shared:]

        # X_A depends on Z_{A\\B} and Z_{A∩B}
        z_input_a = torch.cat([z_private_a, z_shared], dim=1)
        x_a = self.net_a(z_input_a)

        # X_B depends on Z_{A∩B} and Z_{B\\A}
        z_input_b = torch.cat([z_shared, z_private_b], dim=1)
        x_b = self.net_b(z_input_b)

        return torch.cat([x_a, x_b], dim=1)


def generate_synthetic_data(dim=128, num_samples=10000, seed=42):
    """Generate synthetic data with known latent structure.

    Args:
        dim: total dimensionality (latent and observed use same dim)
        num_samples: number of data points
        seed: random seed

    Returns:
        X: (num_samples, dim) observed variables
        Z: (num_samples, dim) ground-truth latent variables
        B_true: (dim, dim) ground-truth binary Jacobian pattern
        group_indices: dict mapping group names to (start, end) index tuples
        mixing_fn: the StructuredInvertibleMLP used for generation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split dim into 3 roughly equal groups
    n_private_a = dim // 3
    n_shared = dim // 3
    n_private_b = dim - n_private_a - n_shared

    # Observed dimensions split in half
    n_obs_a = dim // 2
    n_obs_b = dim - n_obs_a

    group_indices = {
        "Z_A\\B": (0, n_private_a),
        "Z_A∩B": (n_private_a, n_private_a + n_shared),
        "Z_B\\A": (n_private_a + n_shared, dim),
        "X_A": (0, n_obs_a),
        "X_B": (n_obs_a, dim),
    }

    # Sample latent variables from Laplace distribution
    Z = torch.distributions.Laplace(0, 1).sample((num_samples, dim))

    # Build structured mixing function
    mixing_fn = StructuredInvertibleMLP(
        dim, n_private_a, n_shared, n_private_b, n_obs_a, n_obs_b, seed=seed
    )
    mixing_fn.eval()

    # Generate observed data
    with torch.no_grad():
        X = mixing_fn(Z)

    # Build ground-truth binary Jacobian pattern
    B_true = torch.zeros(dim, dim, dtype=torch.int)
    # X_A rows [0:n_obs_a] depend on Z_{A\\B} [0:n_private_a] and Z_{A∩B} [n_private_a:n_private_a+n_shared]
    B_true[:n_obs_a, :n_private_a + n_shared] = 1
    # X_B rows [n_obs_a:dim] depend on Z_{A∩B} and Z_{B\\A}
    B_true[n_obs_a:, n_private_a:] = 1

    return X, Z, B_true, group_indices, mixing_fn


def generate_multi_setup_data(dimensions=None, num_samples=10000, seed=42):
    """Generate data for multiple dimensionality settings (for MCC sweep).

    Args:
        dimensions: list of dimensions to test. Default: [128,256,384,512,640,768,896,1024]
        num_samples: samples per setting
        seed: base seed

    Returns:
        datasets: dict mapping dim -> (X, Z, B_true, group_indices, mixing_fn)
    """
    if dimensions is None:
        dimensions = [128, 256, 384, 512, 640, 768, 896, 1024]

    datasets = {}
    for i, dim in enumerate(dimensions):
        datasets[dim] = generate_synthetic_data(
            dim=dim, num_samples=num_samples, seed=seed + i
        )

    return datasets
