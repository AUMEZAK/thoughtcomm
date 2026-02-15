"""Sparsity-regularized autoencoder with Jacobian L1 penalty (Section 4.1).

Loss: L_rec = ||H - f_hat(f_hat^{-1}(H))||_2^2 + lambda * ||J_{f_hat}||_1

The L1 penalty on the decoder Jacobian encourages sparse dependencies between
latent thoughts Z and observed hidden states H, enabling identification of
shared vs. private thoughts.
"""

import torch
import torch.nn as nn


class SparsityRegularizedAE(nn.Module):
    """Autoencoder with L1 regularization on the decoder's Jacobian.

    Encoder f_hat^{-1}: R^{n_h} -> R^{n_z}
    Decoder f_hat:       R^{n_z} -> R^{n_h}
    """

    def __init__(self, n_h: int, n_z: int, hidden_dim: int = 2048, num_layers: int = 3):
        """
        Args:
            n_h: input dimension (concatenated hidden states of all agents)
            n_z: latent dimension (number of latent thoughts)
            hidden_dim: width of hidden layers
            num_layers: depth of encoder and decoder
        """
        super().__init__()
        self.n_h = n_h
        self.n_z = n_z

        self.encoder = self._build_network(n_h, n_z, hidden_dim, num_layers)
        self.decoder = self._build_network(n_z, n_h, hidden_dim, num_layers)

    def _build_network(self, in_dim, out_dim, hidden_dim, num_layers):
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def encode(self, H: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to latent thoughts.

        Args:
            H: (batch, n_h) concatenated hidden states

        Returns:
            Z_hat: (batch, n_z) latent thoughts
        """
        return self.encoder(H)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        """Decode latent thoughts back to hidden states.

        Args:
            Z: (batch, n_z) latent thoughts

        Returns:
            H_rec: (batch, n_h) reconstructed hidden states
        """
        return self.decoder(Z)

    def forward(self, H: torch.Tensor):
        """Full forward pass: encode then decode.

        Args:
            H: (batch, n_h)

        Returns:
            H_rec: (batch, n_h) reconstructed hidden states
            Z_hat: (batch, n_z) latent thoughts
        """
        Z_hat = self.encode(H)
        H_rec = self.decode(Z_hat)
        return H_rec, Z_hat
