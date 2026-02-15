"""Full ThoughtComm inference pipeline (Section 4).

Integrates all components:
1. Multi-agent debate with hidden state extraction
2. Autoencoder encoding to latent thoughts
3. Agreement-based reweighting for personalized latents
4. Prefix adapter for thought injection
"""

import torch
from tqdm import tqdm

from .debate import MultiAgentDebate
from .agreement import AgreementReweighter
from ..models.autoencoder import SparsityRegularizedAE
from ..models.prefix_adapter import PrefixAdapter
from ..utils.memory import clear_gpu_memory


class ThoughtCommPipeline:
    """Full ThoughtComm inference: debate + thought communication."""

    def __init__(self, model, tokenizer, autoencoder, reweighter, adapter, config):
        """
        Args:
            model: loaded HF causal LM (frozen)
            tokenizer: HF tokenizer
            autoencoder: trained SparsityRegularizedAE (frozen)
            reweighter: AgreementReweighter (with trained weights)
            adapter: trained PrefixAdapter (frozen)
            config: ThoughtCommConfig
        """
        self.model = model
        self.tokenizer = tokenizer
        self.ae = autoencoder
        self.reweighter = reweighter
        self.adapter = adapter
        self.config = config
        self.device = config.device

        # Set all to eval mode
        self.ae.eval()
        self.reweighter.eval()
        self.adapter.eval()

        # Create debate pipeline
        self.debate = MultiAgentDebate(model, tokenizer, config)

    @torch.no_grad()
    def run(self, question):
        """Run ThoughtComm on a single question.

        Round 0: Standard debate (no prefix)
        Round 1+: Debate with prefix injection from latent thoughts

        Args:
            question: math problem text

        Returns:
            final_responses: list of str (one per agent, from last round)
            all_responses: list[list[str]] (rounds x agents)
        """
        def prefix_fn(round_idx, agent_idx, all_hidden_states):
            """Callback to compute prefix for a given agent at a given round."""
            prev_round = round_idx - 1
            if prev_round < 0 or prev_round >= len(all_hidden_states):
                return None

            # Concatenate previous round's hidden states
            H_t = torch.cat(all_hidden_states[prev_round], dim=0)  # (n_h,)
            H_t = H_t.unsqueeze(0).float().to(self.device)  # (1, n_h)

            # Encode to latent
            Z_hat = self.ae.encode(H_t)  # (1, n_z)

            # Personalized latent for this agent
            Z_tilde = self.reweighter.get_personalized_latent(Z_hat, agent_idx)

            # Generate prefix
            prefix = self.adapter(Z_tilde)  # (1, prefix_length, hidden_size)
            return prefix

        all_responses, all_hidden = self.debate.run_debate(
            question, extract_hidden=True, prefix_fn=prefix_fn
        )

        return all_responses[-1], all_responses

    def evaluate(self, eval_data, dataset_type="math"):
        """Run ThoughtComm on evaluation dataset.

        Args:
            eval_data: list of dicts with 'question', 'answer'
            dataset_type: 'math' or 'gsm8k'

        Returns:
            results: dict with:
                - 'final_responses': list[list[str]] — [num_examples][num_agents]
                - 'all_responses': list[list[list[str]]] — [num_examples][rounds][agents]
                - 'ground_truths': list[str]
        """
        results = {
            "final_responses": [],
            "all_responses": [],
            "ground_truths": [],
        }

        for example in tqdm(eval_data, desc="ThoughtComm Evaluation"):
            final_resp, all_resp = self.run(example["question"])
            results["final_responses"].append(final_resp)
            results["all_responses"].append(all_resp)
            results["ground_truths"].append(example["answer"])
            clear_gpu_memory()

        return results


def run_single_answer_baseline(model, tokenizer, eval_data, config,
                               dataset_type="math"):
    """Run single-answer baseline (no debate).

    Args:
        model: HF causal LM
        tokenizer: HF tokenizer
        eval_data: list of dicts
        config: ThoughtCommConfig
        dataset_type: 'math' or 'gsm8k'

    Returns:
        responses: list of str (one per example)
    """
    from ..utils.prompts import INITIAL_PROMPT

    responses = []
    device = next(model.parameters()).device

    for example in tqdm(eval_data, desc="Single Answer Baseline"):
        prompt = INITIAL_PROMPT.format(question=example["question"])
        messages = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = outputs[0, input_ids.shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(response)
        clear_gpu_memory()

    return responses


def run_debate_baseline(model, tokenizer, eval_data, config):
    """Run debate-only baseline (no ThoughtComm, no finetuning).

    Args:
        model: HF causal LM
        tokenizer: HF tokenizer
        eval_data: list of dicts
        config: ThoughtCommConfig

    Returns:
        results: dict with 'final_responses' and 'all_responses'
    """
    debate = MultiAgentDebate(model, tokenizer, config)

    results = {"final_responses": [], "all_responses": []}

    for example in tqdm(eval_data, desc="Debate Baseline"):
        all_resp, _ = debate.run_debate(example["question"], extract_hidden=False)
        results["final_responses"].append(all_resp[-1])
        results["all_responses"].append(all_resp)
        clear_gpu_memory()

    return results
