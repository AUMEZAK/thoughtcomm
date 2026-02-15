"""MATH answer parsing and grading (Section 5.2).

Extracts answers from \\boxed{} format and compares using string normalization
and symbolic computation (sympy).
"""

import re


def extract_boxed_answer(text):
    """Extract the last \\boxed{...} answer from generated text.

    Handles nested braces correctly.

    Args:
        text: generated response string

    Returns:
        answer string, or None if no \\boxed{} found
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed ")
        if idx == -1:
            return None

    try:
        start = text.index("{", idx)
    except ValueError:
        return None

    depth = 1
    i = start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[start + 1:i - 1].strip()


def normalize_answer(answer):
    """Normalize a mathematical answer string for comparison.

    Args:
        answer: raw answer string

    Returns:
        normalized string
    """
    if answer is None:
        return ""

    answer = answer.strip()
    # Remove enclosing $ signs
    answer = re.sub(r"^\$(.+)\$$", r"\1", answer)
    # Remove \text{}, \mathrm{}, \textbf{} wrappers
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\textbf\{([^}]*)\}", r"\1", answer)
    # Remove \left and \right
    answer = answer.replace("\\left", "").replace("\\right", "")
    # Normalize whitespace
    answer = " ".join(answer.split())

    return answer


def grade_answer(predicted, ground_truth):
    """Compare predicted answer to ground truth.

    Tries string comparison, then numeric, then symbolic (sympy).

    Args:
        predicted: predicted answer string (from \\boxed{})
        ground_truth: ground truth answer string

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False

    pred = normalize_answer(predicted)
    truth = normalize_answer(ground_truth)

    if pred == truth:
        return True

    # Try numeric comparison
    try:
        pred_val = float(pred.replace(",", ""))
        truth_val = float(truth.replace(",", ""))
        if abs(pred_val - truth_val) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # Try fraction comparison
    try:
        frac_pattern = r"\\frac\{([^}]*)\}\{([^}]*)\}"
        pred_match = re.search(frac_pattern, predicted)
        truth_match = re.search(frac_pattern, ground_truth)
        if pred_match and truth_match:
            pred_val = float(pred_match.group(1)) / float(pred_match.group(2))
            truth_val = float(truth_match.group(1)) / float(truth_match.group(2))
            if abs(pred_val - truth_val) < 1e-6:
                return True
    except (ValueError, ZeroDivisionError):
        pass

    # Try sympy symbolic comparison
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify
        pred_expr = parse_latex(predicted)
        truth_expr = parse_latex(ground_truth)
        if simplify(pred_expr - truth_expr) == 0:
            return True
    except Exception:
        pass

    return False
