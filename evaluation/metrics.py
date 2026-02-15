"""Evaluation metrics: accuracy, consensus, standard deviation (Section 5.2)."""

import numpy as np
from .math_eval import extract_boxed_answer, grade_answer
from .gsm8k_eval import extract_gsm8k_answer, grade_gsm8k_answer


def compute_accuracy(responses, ground_truths, dataset_type="math"):
    """Compute accuracy from agent responses and ground truths.

    Args:
        responses: list of str (one response per example, from final round)
        ground_truths: list of str (ground truth answers)
        dataset_type: 'math' or 'gsm8k'

    Returns:
        accuracy: float (0-100)
        correct_mask: list of bool
    """
    correct = []
    for resp, gt in zip(responses, ground_truths):
        if dataset_type == "math":
            pred = extract_boxed_answer(resp)
            is_correct = grade_answer(pred, gt)
        else:
            pred = extract_gsm8k_answer(resp)
            is_correct = grade_gsm8k_answer(pred, gt)
        correct.append(is_correct)

    accuracy = 100.0 * sum(correct) / len(correct) if correct else 0.0
    return accuracy, correct


def compute_consensus(all_agent_responses, dataset_type="math"):
    """Compute consensus: fraction of examples where all agents agree.

    Args:
        all_agent_responses: list of list of str — [num_examples][num_agents]
        dataset_type: 'math' or 'gsm8k'

    Returns:
        consensus: float (0-100)
    """
    unanimous = 0
    total = len(all_agent_responses)

    for example_responses in all_agent_responses:
        # Extract answers from all agents
        if dataset_type == "math":
            answers = [extract_boxed_answer(r) for r in example_responses]
        else:
            answers = [extract_gsm8k_answer(r) for r in example_responses]

        if None in answers:
            continue

        # Check if all agents agree
        all_same = True
        for i in range(1, len(answers)):
            if dataset_type == "math":
                if not grade_answer(answers[0], answers[i]):
                    all_same = False
                    break
            else:
                if not grade_gsm8k_answer(answers[0], answers[i]):
                    all_same = False
                    break

        if all_same:
            unanimous += 1

    return 100.0 * unanimous / total if total > 0 else 0.0


def compute_accuracy_with_std(responses_per_run, ground_truths, dataset_type="math"):
    """Compute accuracy with standard deviation across multiple runs.

    Args:
        responses_per_run: list of list of str — [num_runs][num_examples]
        ground_truths: list of str
        dataset_type: 'math' or 'gsm8k'

    Returns:
        mean_accuracy: float
        std_accuracy: float
    """
    accuracies = []
    for responses in responses_per_run:
        acc, _ = compute_accuracy(responses, ground_truths, dataset_type)
        accuracies.append(acc)

    return float(np.mean(accuracies)), float(np.std(accuracies))
