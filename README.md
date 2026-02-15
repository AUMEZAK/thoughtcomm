# ThoughtComm: Thought Communication in Multiagent Collaboration

Reproduction of the NeurIPS 2025 Spotlight paper [arXiv:2510.20733](https://arxiv.org/abs/2510.20733).

## Overview

ThoughtComm enables LLM agents to communicate via **latent thoughts** instead of natural language. The framework:

1. **Extracts latent thoughts** from agent hidden states using a sparsity-regularized autoencoder
2. **Identifies shared vs. private thoughts** via Jacobian structure analysis
3. **Injects personalized thought representations** back into agents via prefix adaptation

## Quick Start (Google Colab)

Run the notebooks in order:

| Notebook | Description | Time | GPU Required |
|----------|-------------|------|--------------|
| `01_synthetic_experiment.ipynb` | Validate identifiability theory (Fig 3, 4) | ~15 min | No |
| `02_collect_hidden_states.ipynb` | Run debate, collect hidden states | 2-8 hr | Yes |
| `03_train_autoencoder.ipynb` | Train sparsity-regularized AE | ~30 min | Optional |
| `04_train_adapter.ipynb` | Train prefix adapter | 1-2 hr | Yes |
| `05_full_evaluation.ipynb` | Full evaluation (Table 1) | 2-4 hr | Yes |

## Supported Models

- **Qwen-3-0.6B** (`Qwen/Qwen3-0.6B`)
- **Phi-4-mini-instruct** (`microsoft/Phi-4-mini-instruct`)

## Installation

```bash
git clone https://github.com/AUMEZAK/thoughtcomm.git
cd thoughtcomm
pip install -e .
```

## Project Structure

```
thoughtcomm/
├── configs/         # Hyperparameter configuration
├── data/            # Dataset loaders (MATH, GSM8K, synthetic)
├── models/          # Autoencoder, prefix adapter, model utilities
├── pipeline/        # Debate, hidden state collection, ThoughtComm inference
├── training/        # AE training, adapter training, Jacobian utilities
├── evaluation/      # Answer parsing, grading, metrics
├── utils/           # Prompts, memory management
└── notebooks/       # Google Colab notebooks (5 sequential steps)
```

## Citation

```bibtex
@inproceedings{zheng2025thought,
  title={Thought Communication in Multiagent Collaboration},
  author={Zheng, Yujia and Zhao, Zhuokai and Li, Zijian and Xie, Yaqi and Gao, Mingze and Zhang, Lizhu and Zhang, Kun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
