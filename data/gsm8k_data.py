"""GSM8K dataset loader (Section 5.2).

Loads from HuggingFace openai/gsm8k and splits into train/eval sets.
"""

import random
import re


def load_gsm8k_dataset(num_train: int = 500, num_eval: int = 500,
                       seed: int = 42):
    """Load the GSM8K dataset.

    Args:
        num_train: number of training examples
        num_eval: number of evaluation examples
        seed: random seed for shuffling

    Returns:
        train_data: list of dicts with 'question', 'answer'
        eval_data: list of dicts with same keys
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")

    examples = []
    for ex in ds:
        answer = _extract_gsm8k_answer(ex["answer"])
        examples.append({
            "question": ex["question"],
            "answer": answer,
            "full_solution": ex["answer"],
        })

    random.seed(seed)
    random.shuffle(examples)

    if len(examples) < num_train + num_eval:
        print(f"Warning: only {len(examples)} GSM8K examples available. "
              f"Using all available.")
        mid = len(examples) // 2
        return examples[:mid], examples[mid:]

    train_data = examples[:num_train]
    eval_data = examples[num_train:num_train + num_eval]

    return train_data, eval_data


def _extract_gsm8k_answer(solution: str) -> str:
    """Extract the numeric answer after #### delimiter."""
    match = re.search(r"####\s*(.+)", solution)
    if match:
        return match.group(1).strip().replace(",", "")
    return solution.strip().split("\n")[-1]
