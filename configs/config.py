"""ThoughtComm configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThoughtCommConfig:
    """All hyperparameters for ThoughtComm reproduction."""

    # --- Model ---
    model_name: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 1024  # per-agent hidden dim (Qwen=1024, Phi-4=3072)
    num_agents: int = 3
    num_rounds: int = 2

    # --- Autoencoder ---
    n_z: int = 1024  # latent thought dimension
    ae_hidden: int = 2048  # AE hidden layer width
    ae_num_layers: int = 3  # encoder/decoder depth
    ae_lr: float = 1e-3
    ae_epochs: int = 200
    ae_batch_size: int = 64
    jacobian_l1_weight: float = 0.01  # lambda for ||J||_1
    jacobian_sample_rows: int = 64  # stochastic rows per training step
    jacobian_threshold: float = 0.01  # for binarizing B(J_f)

    # --- Prefix Adapter ---
    adapter_hidden: int = 512
    adapter_lr: float = 1e-4
    adapter_epochs: int = 50
    prefix_length: int = 1  # m = 1 (paper default)
    cosine_loss_weight: float = 1.0
    lm_loss_weight: float = 1.0
    max_gen_tokens_adapter: int = 50  # short continuation for adapter training

    # --- Data ---
    num_train: int = 500
    num_eval: int = 500
    math_level: int = 3  # MATH difficulty level

    # --- Generation ---
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # --- Hardware ---
    dtype: str = "bfloat16"
    device: str = "cuda"
    gradient_checkpointing: bool = True

    # --- Paths ---
    save_dir: str = "/content/drive/MyDrive/thoughtcomm_checkpoints/"
    local_save_dir: str = "./checkpoints/"

    @property
    def n_h(self) -> int:
        """Concatenated hidden state dimension for all agents."""
        return self.num_agents * self.hidden_size

    @property
    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)

    @classmethod
    def for_qwen_0_6b(cls, **kwargs) -> "ThoughtCommConfig":
        """Preset for Qwen-3-0.6B."""
        defaults = dict(
            model_name="Qwen/Qwen3-0.6B",
            hidden_size=1024,
            ae_hidden=2048,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_phi4_mini(cls, **kwargs) -> "ThoughtCommConfig":
        """Preset for Phi-4-mini-instruct (3.8B)."""
        defaults = dict(
            model_name="microsoft/Phi-4-mini-instruct",
            hidden_size=3072,
            ae_hidden=4096,
        )
        defaults.update(kwargs)
        return cls(**defaults)
