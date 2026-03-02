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

        # Normalization buffers (set after training via set_norm_stats).
        # encode() auto-normalizes raw inputs when _has_norm is True.
        # During training _has_norm is False so pre-normalized data passes through.
        self.register_buffer('_norm_mean', torch.zeros(n_h))
        self.register_buffer('_norm_std', torch.ones(n_h))
        self.register_buffer('_has_norm', torch.tensor(False))

    def _build_network(self, in_dim, out_dim, hidden_dim, num_layers):
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set normalization statistics after training.

        Once set, encode() will auto-normalize raw hidden state inputs.
        Stats are persisted in state_dict for checkpoint loading.

        Args:
            mean: (n_h,) per-dimension mean of training hidden states
            std: (n_h,) per-dimension std of training hidden states
        """
        self._norm_mean.copy_(mean.detach())
        self._norm_std.copy_(std.detach().clamp(min=1e-8))
        self._has_norm.fill_(True)

    def _normalize(self, H: torch.Tensor) -> torch.Tensor:
        """Normalize input if norm_stats have been set."""
        if self._has_norm:
            return (H - self._norm_mean.to(H.device)) / self._norm_std.to(H.device)
        return H

    def encode(self, H: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to latent thoughts.

        Auto-normalizes if norm_stats have been set via set_norm_stats().

        Args:
            H: (batch, n_h) concatenated hidden states (raw or pre-normalized)

        Returns:
            Z_hat: (batch, n_z) latent thoughts
        """
        return self.encoder(self._normalize(H))

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
