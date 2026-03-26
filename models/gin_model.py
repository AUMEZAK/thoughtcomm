"""GIN (General Incompressible-flow Network) wrapper for nonlinear ICA.

Replaces the AE in ThoughtComm's pipeline with a volume-preserving
normalizing flow. GIN provides structural invertibility and efficient
Jacobian computation for latent thought extraction.

Reference:
- Sorrenson et al. 2020 "Disentanglement by Nonlinear ICA with General
  Incompressible-flow Networks (GIN)"
- Zheng et al. 2022 "On the Identifiability of Nonlinear ICA"
"""

import torch
import torch.nn as nn

try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
    FREIA_AVAILABLE = True
except ImportError:
    FREIA_AVAILABLE = False


def build_gin(n_dim, n_coupling_layers=8, subnet_hidden=None, clamp=2.0):
    """Build a GIN (volume-preserving normalizing flow).

    Args:
        n_dim: Input/output dimension (must be same for GIN).
        n_coupling_layers: Number of GINCouplingBlock layers.
        subnet_hidden: Hidden dim for subnet MLPs. Default: max(n_dim, 10).
        clamp: Soft clamping value for affine coefficients.

    Returns:
        inn: FrEIA SequenceINN model.
    """
    if not FREIA_AVAILABLE:
        raise ImportError("FrEIA is required. Install with: pip install FrEIA")

    if subnet_hidden is None:
        subnet_hidden = max(n_dim, 10)

    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, subnet_hidden),
            nn.ReLU(),
            nn.Linear(subnet_hidden, subnet_hidden),
            nn.ReLU(),
            nn.Linear(subnet_hidden, dims_out),
        )

    inn = Ff.SequenceINN(n_dim)
    for _ in range(n_coupling_layers):
        inn.append(Fm.GINCouplingBlock,
                   subnet_constructor=subnet_fc,
                   clamp=clamp)
        inn.append(Fm.PermuteRandom)

    return inn


class GINWrapper(nn.Module):
    """Wrapper around FrEIA GIN for ThoughtComm integration.

    Provides encode/decode interface compatible with the existing pipeline.
    Includes normalization buffers like SparsityRegularizedAE.
    """

    def __init__(self, n_dim, n_coupling_layers=8, subnet_hidden=None, clamp=2.0):
        super().__init__()
        self.n_dim = n_dim
        self.inn = build_gin(n_dim, n_coupling_layers, subnet_hidden, clamp)

        # Normalization buffers (same interface as SparsityRegularizedAE)
        self.register_buffer('_norm_mean', torch.zeros(n_dim))
        self.register_buffer('_norm_std', torch.ones(n_dim))
        self.register_buffer('_has_norm', torch.tensor(False))

    def set_norm_stats(self, mean, std):
        """Set normalization statistics after training."""
        self._norm_mean.copy_(mean.detach())
        self._norm_std.copy_(std.detach().clamp(min=1e-8))
        self._has_norm.fill_(True)

    def _normalize(self, H):
        if self._has_norm:
            return (H - self._norm_mean.to(H.device)) / self._norm_std.to(H.device)
        return H

    def encode(self, H):
        """Forward pass: observations H -> latent Z."""
        H_norm = self._normalize(H)
        Z, _ = self.inn(H_norm)
        return Z

    def decode(self, Z):
        """Inverse pass: latent Z -> observations H."""
        H, _ = self.inn(Z, rev=True)
        return H

    def forward(self, H):
        """Full forward pass (encode then decode for compatibility)."""
        Z = self.encode(H)
        H_rec = self.decode(Z)
        return H_rec, Z
