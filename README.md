# ThoughtComm: Thought Communication in Multiagent Collaboration

[![Paper](https://img.shields.io/badge/arXiv-2510.20733-b31b1b.svg)](https://arxiv.org/abs/2510.20733)
[![Conference](https://img.shields.io/badge/NeurIPS%202025-Spotlight-blue.svg)](https://neurips.cc/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

Unofficial reproduction of the NeurIPS 2025 Spotlight paper:

> **Thought Communication in Multiagent Collaboration**
> Yujia Zheng, Zhuokai Zhao, Zijian Li, Yaqi Xie, Mingze Gao, Lizhu Zhang, Kun Zhang
> *CMU, Meta AI, MBZUAI*
> [arXiv:2510.20733](https://arxiv.org/abs/2510.20733)

---

## Overview

Natural language is lossy, ambiguous, and indirect. ThoughtComm proposes a fundamentally different communication paradigm for LLM multi-agent systems: instead of exchanging text, agents share **latent thoughts** directly --- akin to telepathy.

The key insight is that agent hidden states are generated from underlying latent thoughts via an unknown function. By recovering these thoughts with theoretical identifiability guarantees, agents can communicate mind-to-mind, distinguishing shared beliefs from private reasoning.

### How It Works

```
                    Communication Round t

Agent 1 ──> H_t^(1) ─┐
Agent 2 ──> H_t^(2) ─┼──> H_t (concat) ──> Encoder ──> Z_hat_t (latent thoughts)
Agent 3 ──> H_t^(3) ─┘                                       │
                                                              v
                                                 Jacobian B(J_f) structure
                                                 ┌─────────────────────┐
                                                 │ Shared thoughts     │ (alpha=3: all agents)
                                                 │ Pairwise thoughts   │ (alpha=2: two agents)
                                                 │ Private thoughts    │ (alpha=1: one agent)
                                                 └─────────────────────┘
                                                              │
                                                   Agreement Reweighting
                                                              │
                                              Z_tilde^(i) per agent (personalized)
                                                              │
                                                   Prefix Adapter g()
                                                              │
                                              P_t^(i) prefix embedding
                                                              │
                                              Inject into next round generation
```

### Three-Stage Pipeline

1. **Latent Thought Extraction** --- A sparsity-regularized autoencoder maps concatenated agent hidden states `H_t` into latent thoughts `Z_hat_t`. The L1 penalty on the decoder Jacobian enforces identifiability (Eq. 7):

   ```
   L_rec = ||H_t - f_hat(Z_hat_t)||^2 + lambda * ||J_f_hat||_1
   ```

2. **Structural Recovery & Agreement Reweighting** --- The binary Jacobian pattern `B(J_f)` reveals which latent dimensions influence which agents. Each dimension is classified by its *agreement level* `alpha_j` (how many agents share it), and personalized representations `Z_tilde^(i)` are constructed with learnable weights per agreement level (Eq. 8-10).

3. **Prefix Injection** --- A lightweight adapter `g` maps each agent's personalized latent representation into a prefix embedding `P_t^(i)`, which is prepended to the token embeddings for the next generation round (Eq. 11-12). Only ~1.3M parameters are trained.

---

## Reproduction Targets

### Table 1: Main Results (MATH & GSM8K)

| Base Model | Method | MATH Acc (%) | MATH Cons (%) | GSM8K Acc (%) | GSM8K Cons (%) |
|---|---|---|---|---|---|
| Qwen 3-0.6B | Single Answer | 45.80 +/- 2.23 | N/A | 58.20 +/- 2.21 | N/A |
| | Multiagent Finetuning | 71.20 +/- 2.03 | 90.07 | 70.80 +/- 2.03 | 86.40 |
| | **ThoughtComm** | **85.00 +/- 1.60** | **91.20** | **75.80 +/- 1.92** | **89.27** |
| Phi-4-mini (3.84B) | Single Answer | 63.80 +/- 2.15 | N/A | 81.60 +/- 1.73 | N/A |
| | Multiagent Finetuning | 60.20 +/- 2.19 | 78.89 | 82.16 +/- 1.71 | 91.24 |
| | **ThoughtComm** | **74.60 +/- 1.95** | **84.73** | **84.20 +/- 1.63** | **94.73** |

### Figures to Reproduce

- **Fig 3**: R^2 matrix showing disentanglement (ours vs. baseline) on synthetic data
- **Fig 4**: MCC across dimensions (128 to 1024), all above 0.75 identifiability threshold
- **Fig 5**: Prefix length ablation (m = 1, 4, 8, 16) --- stable performance across lengths
- **Fig 6**: Debate round scaling (2 to 6 rounds) --- ThoughtComm gains in both accuracy and consensus

---

## Quick Start (Google Colab)

Run the 5 notebooks **sequentially** on Google Colab. Each notebook saves checkpoints to Google Drive, enabling session recovery.

| # | Notebook | Description | Time | GPU |
|---|----------|-------------|------|-----|
| 1 | `01_synthetic_experiment.ipynb` | Validate identifiability theory (Fig 3, 4) | ~15 min | No |
| 2 | `02_collect_hidden_states.ipynb` | Run 3-agent debate on 500 problems, collect hidden states | 2-8 hr | Yes |
| 3 | `03_train_autoencoder.ipynb` | Train sparsity-regularized AE, compute B matrix | ~30 min | Optional |
| 4 | `04_train_adapter.ipynb` | Train prefix adapter with L_comm loss | 1-2 hr | Yes |
| 5 | `05_full_evaluation.ipynb` | Full evaluation: baselines + ThoughtComm (Table 1) | 2-4 hr | Yes |

Each notebook starts with:
```python
!git clone https://github.com/AUMEZAK/thoughtcomm.git
%cd thoughtcomm
!pip install -e . -q
```

Checkpoints are saved to Google Drive every 50 examples to handle Colab disconnections.

---

## Supported Models

| Model | Hidden Size | `n_h` (3 agents) | Parameters |
|-------|-----------|-----------------|------------|
| **Qwen-3-0.6B** (`Qwen/Qwen3-0.6B`) | 1024 | 3072 | 0.6B |
| **Phi-4-mini-instruct** (`microsoft/Phi-4-mini-instruct`) | 3072 | 9216 | 3.8B |

---

## Local Installation

```bash
git clone https://github.com/AUMEZAK/thoughtcomm.git
cd thoughtcomm
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.40
- CUDA GPU recommended (T4 minimum, A100 recommended for Phi-4)

---

## Project Structure

```
thoughtcomm/
├── configs/
│   └── config.py                  # ThoughtCommConfig dataclass (all hyperparameters)
├── data/
│   ├── synthetic.py               # Synthetic data with structured invertible MLP
│   ├── math_data.py               # MATH dataset loader (Level-3 filtering)
│   └── gsm8k_data.py              # GSM8K dataset loader
├── models/
│   ├── autoencoder.py             # SparsityRegularizedAE (encoder + decoder with LeakyReLU)
│   ├── prefix_adapter.py          # PrefixAdapter (MLP with GELU, Z_tilde -> prefix P)
│   └── model_utils.py             # HuggingFace model loading, hidden state extraction
├── pipeline/
│   ├── debate.py                  # MultiAgentDebate (3 agents, multi-round orchestration)
│   ├── hidden_state_collector.py  # Collect H_t with Drive checkpointing
│   ├── agreement.py               # AgreementReweighter (B matrix -> personalized Z_tilde)
│   └── thought_comm.py            # ThoughtCommPipeline (full inference)
├── training/
│   ├── train_autoencoder.py       # AE training (reconstruction + stochastic Jacobian L1)
│   ├── train_adapter.py           # Adapter training (cosine similarity + LM loss)
│   └── jacobian_utils.py          # Stochastic Jacobian estimation, B matrix extraction
├── evaluation/
│   ├── math_eval.py               # \boxed{} answer parsing, sympy normalization
│   ├── gsm8k_eval.py              # #### answer parsing, numeric extraction
│   ├── synthetic_eval.py          # R^2 matrix, MCC computation
│   └── metrics.py                 # Accuracy, consensus, bootstrap std
├── utils/
│   ├── prompts.py                 # Debate prompt templates
│   └── memory.py                  # GPU memory management
├── notebooks/                     # 5 Google Colab notebooks (sequential pipeline)
├── requirements.txt
└── setup.py
```

---

## Key Implementation Details

### Stochastic Jacobian Estimation
Computing the full Jacobian `J_f ∈ R^{n_h x n_z}` is expensive. During training, we sample 64 random output rows per step and compute their gradients, scaling the L1 penalty accordingly. The full Jacobian is computed only once post-training via `torch.func.jacrev` + `vmap` for B matrix extraction.

### Hidden State Extraction
We extract the **last layer, last token** hidden state from each agent's generated response via a separate forward pass with `output_hidden_states=True`. This avoids storing all intermediate states during generation.

### Memory-Efficient Design
- Single model instance shared across 3 agents (sequential execution)
- bfloat16 precision (float16 on T4)
- `torch.cuda.empty_cache()` after each agent generation
- Gradient checkpointing during adapter training

### Agreement-Based Reweighting
The binary Jacobian pattern B reveals thought-sharing structure:
- **alpha=1** (private): thought used by only 1 agent
- **alpha=2** (pairwise shared): thought shared between 2 agents
- **alpha=3** (globally shared): thought used by all 3 agents

Learnable weights `w_alpha` per agreement level enable the model to emphasize shared consensus or private specialization.

---

## Datasets

- **MATH** ([hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math)): Level-3 problems, 500 train + 500 eval. Answers parsed from `\boxed{}` format.
- **GSM8K** ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)): 500 train + 500 eval. Answers parsed from `####` delimiter.

---

## Differences from Original Paper

This is an **unofficial reproduction** based solely on the paper description. Key differences:

- **Models**: We use Qwen-3-0.6B and Phi-4-mini (the paper also evaluates Qwen-3-1.7B, LLaMA 3-8B, DeepSeek-R1-Distill-Llama-8B)
- **No Multiagent Finetuning baseline**: We compare against Single Answer and Debate-only (Multiagent Finetuning requires separate full model finetuning)
- **No official code**: All implementation is from scratch based on the paper's methodology sections and appendix

---

## Citation

```bibtex
@inproceedings{zheng2025thought,
  title={Thought Communication in Multiagent Collaboration},
  author={Zheng, Yujia and Zhao, Zhuokai and Li, Zijian and Xie, Yaqi and Gao, Mingze and Zhang, Lizhu and Zhang, Kun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

This reproduction is for research and educational purposes only.
