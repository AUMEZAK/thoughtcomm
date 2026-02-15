"""GSM8K answer parsing and grading (Section 5.2).

Extracts numeric answers from #### delimiter or \\boxed{} format.
"""

import re
from .math_eval import extract_boxed_answer


def extract_gsm8k_answer(text):
    """Extract numeric answer from GSM8K response.

    Tries \\boxed{} first (our agents use this format),
    then #### delimiter (ground truth format).

    Args:
        text: generated or ground truth response

    Returns:
        answer string (numeric), or None
    """
    # Try boxed format first
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return _clean_number(boxed)

    # Try #### format
    match = re.search(r"####\s*(.+)", text)
    if match:
        return _clean_number(match.group(1).strip())

    # Try to find the last number in the text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return _clean_number(numbers[-1])

    return None


def _clean_number(s):
    """Clean a numeric string: remove commas, whitespace."""
    return s.strip().replace(",", "").replace("$", "").replace("%", "")


def grade_gsm8k_answer(predicted, ground_truth):
    """Compare predicted GSM8K answer to ground truth.

    Both answers should be numeric strings.

    Args:
        predicted: predicted answer string
        ground_truth: ground truth numeric answer

    Returns:
        True if answers match numerically
    """
    if predicted is None:
        return False

    pred = _clean_number(str(predicted))
    truth = _clean_number(str(ground_truth))

    # String comparison
    if pred == truth:
        return True

    # Numeric comparison
    try:
        pred_val = float(pred)
        truth_val = float(truth)
        return abs(pred_val - truth_val) < 1e-6
    except (ValueError, TypeError):
        return False
