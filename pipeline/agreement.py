"""Agreement-based reweighting of latent thoughts (Section 4.2).

Given the binary Jacobian pattern B(J_f), determines which latent dimensions
are relevant to each agent, computes agreement levels, and constructs
personalized latent representations Z_tilde^(i).
"""

import torch
import torch.nn as nn


class AgreementReweighter(nn.Module):
    """Routes latent thoughts to agents based on structural agreement.

    For each latent dimension j:
    - alpha_j = number of agents for which dimension j is relevant
    - Dimensions with high alpha are "shared" thoughts
    - Dimensions with low alpha are "private" thoughts
    - Learnable weights w_{alpha} scale each agreement level

    Z_tilde^(i) = Z_hat * mask^(i) * w[alpha]
    """

    def __init__(self, B, config):
        """
        Args:
            B: (n_h, n_z) binary Jacobian pattern from trained autoencoder
            config: ThoughtCommConfig with num_agents, hidden_size
        """
        super().__init__()
        self.config = config
        self.register_buffer("B", B.int())

        # Determine which latent dims are relevant to each agent (Eq. 4)
        # Agent k's rows in B are [k*hidden_size : (k+1)*hidden_size]
        relevance_masks = []
        for k in range(config.num_agents):
            row_start = k * config.hidden_size
            row_end = (k + 1) * config.hidden_size
            agent_B = B[row_start:row_end, :]  # (hidden_size, n_z)
            relevant = (agent_B.sum(dim=0) > 0).float()  # (n_z,)
            relevance_masks.append(relevant)

        # (num_agents, n_z) - each row is a mask for one agent
        self.register_buffer("relevance_masks", torch.stack(relevance_masks, dim=0))

        # Compute agreement level alpha_j for each latent dim j (Eq. 8)
        alpha = self.relevance_masks.sum(dim=0).long()  # (n_z,)
        self.register_buffer("alpha", alpha)

        # Learnable weights per agreement level (Eq. 9)
        # With num_agents=3, alpha in {0, 1, 2, 3}
        self.w = nn.Parameter(torch.ones(config.num_agents + 1))

    def get_personalized_latent(self, Z_hat, agent_idx):
        """Construct personalized latent representation Z_tilde^(i) (Eq. 9-10).

        Args:
            Z_hat: (batch, n_z) latent thoughts
            agent_idx: which agent (0-indexed)

        Returns:
            Z_tilde: (batch, n_z) reweighted, agent-specific latent
        """
        mask = self.relevance_masks[agent_idx]  # (n_z,)
        weights = self.w[self.alpha]  # (n_z,) weight per dim based on its agreement level

        Z_tilde = Z_hat * mask.unsqueeze(0) * weights.unsqueeze(0)
        return Z_tilde

    def get_agreement_stats(self):
        """Return statistics about the agreement structure.

        Returns:
            dict with counts per agreement level and per-agent relevant dim counts
        """
        stats = {
            "total_dims": self.alpha.shape[0],
            "agreement_distribution": {},
            "per_agent_relevant": [],
        }

        for level in range(self.config.num_agents + 1):
            count = (self.alpha == level).sum().item()
            stats["agreement_distribution"][f"alpha={level}"] = count

        for k in range(self.config.num_agents):
            count = self.relevance_masks[k].sum().item()
            stats["per_agent_relevant"].append(int(count))

        return stats
