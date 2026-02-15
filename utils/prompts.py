"""Debate prompt templates for multi-agent communication.

Based on the prompt format from Subramaniam et al. (2025) Multiagent Finetuning.
"""

INITIAL_PROMPT = (
    "Can you solve the following math problem? {question} "
    "Provide a bullet point summary of your reasoning. "
    "Your final answer should be a single answer, in the form "
    "\\boxed{{answer}}, at the end of your response."
)

DEBATE_PROMPT = (
    "These are solutions to the problem from other agents:\n\n"
    "{other_responses}\n\n"
    "Using each response as additional advice, can you give an updated "
    "bullet by bullet answer to the following question: {question}\n"
    "Your final answer should be a single answer, in the form "
    "\\boxed{{answer}}, at the end of your response."
)

AGENT_RESPONSE_HEADER = "Agent {agent_id} solution:\n{response}\n"


def format_other_responses(responses, exclude_agent_idx):
    """Format other agents' responses for the debate prompt.

    Args:
        responses: list of str, one per agent
        exclude_agent_idx: index of the current agent (to exclude)

    Returns:
        Formatted string of other agents' responses
    """
    parts = []
    for i, resp in enumerate(responses):
        if i != exclude_agent_idx:
            parts.append(AGENT_RESPONSE_HEADER.format(agent_id=i + 1, response=resp))
    return "\n".join(parts)
