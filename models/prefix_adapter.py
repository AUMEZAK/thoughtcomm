"""Prefix adapter for latent thought injection (Section 4.3).

Maps personalized latent representation Z_tilde to prefix embeddings P
that are prepended to the LLM's token embeddings.

    P_t^(i) = g(Z_tilde^(i))  in R^{m x d}

where m = prefix_length, d = hidden_size of the LLM.
"""

import torch
import torch.nn as nn


class PrefixAdapter(nn.Module):
    """Adapter g: R^{n_z} -> R^{m x d} that converts latent thoughts to prefix embeddings."""

    def __init__(self, n_z, hidden_size, prefix_length=1, adapter_hidden=512):
        """
        Args:
            n_z: latent dimension (input)
            hidden_size: LLM hidden dimension (output per prefix token)
            prefix_length: m, number of prefix tokens (default 1)
            adapter_hidden: width of adapter hidden layers
        """
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size

        output_dim = prefix_length * hidden_size

        self.net = nn.Sequential(
            nn.Linear(n_z, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, output_dim),
        )

    def forward(self, Z_tilde):
        """Generate prefix embeddings from personalized latent.

        Args:
            Z_tilde: (batch, n_z) personalized latent representation

        Returns:
            prefix: (batch, prefix_length, hidden_size) prefix embeddings
        """
        out = self.net(Z_tilde)  # (batch, prefix_length * hidden_size)
        prefix = out.view(-1, self.prefix_length, self.hidden_size)
        return prefix
