"""MATH dataset loader with level filtering (Section 5.2).

Loads from HuggingFace hendrycks/competition_math, filters for specified
difficulty level, and splits into train/eval sets.
"""

import random
from typing import Optional


def load_math_dataset(num_train: int = 500, num_eval: int = 500,
                      level: int = 3, seed: int = 42,
                      split: str = "train"):
    """Load and filter the MATH dataset.

    Args:
        num_train: number of training examples
        num_eval: number of evaluation examples
        level: MATH difficulty level to filter (1-5)
        seed: random seed for shuffling
        split: HF dataset split to use

    Returns:
        train_data: list of dicts with 'question', 'answer', 'level', 'type'
        eval_data: list of dicts with same keys
    """
    from datasets import load_dataset

    ds = load_dataset("hendrycks/competition_math", split=split)

    # Filter for specified level
    level_str = f"Level {level}"
    filtered = []
    for ex in ds:
        if ex.get("level") == level_str:
            filtered.append({
                "question": ex["problem"],
                "answer": _extract_boxed(ex["solution"]),
                "full_solution": ex["solution"],
                "level": ex["level"],
                "type": ex.get("type", ""),
            })

    random.seed(seed)
    random.shuffle(filtered)

    if len(filtered) < num_train + num_eval:
        print(f"Warning: only {len(filtered)} Level-{level} problems available, "
              f"requested {num_train + num_eval}. Using all available.")
        mid = len(filtered) // 2
        return filtered[:mid], filtered[mid:]

    train_data = filtered[:num_train]
    eval_data = filtered[num_train:num_train + num_eval]

    return train_data, eval_data


def _extract_boxed(solution: str) -> str:
    """Extract the final \\boxed{...} answer from a MATH solution string."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        idx = solution.rfind("\\boxed ")
        if idx == -1:
            return solution.strip().split("\n")[-1]

    start = solution.index("{", idx)
    depth = 1
    i = start + 1
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    return solution[start + 1:i - 1].strip()
