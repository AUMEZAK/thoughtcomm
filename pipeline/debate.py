"""Multi-agent debate pipeline (Section 5.2).

Orchestrates multiple LLM agents engaging in multi-round debate.
Uses a single model instance, running agents sequentially to save GPU memory.
"""

import torch
from utils.prompts import INITIAL_PROMPT, DEBATE_PROMPT, format_other_responses
from utils.memory import clear_gpu_memory
from models.model_utils import extract_last_hidden_state


class MultiAgentDebate:
    """Orchestrates multi-round debate between LLM agents."""

    def __init__(self, model, tokenizer, config):
        """
        Args:
            model: loaded HF causal LM (shared by all agents)
            tokenizer: HF tokenizer
            config: ThoughtCommConfig
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def run_debate(self, question, extract_hidden=False, prefix_fn=None):
        """Run full multi-agent debate for one question.

        Args:
            question: the math problem text
            extract_hidden: whether to extract hidden states
            prefix_fn: optional callable(round_idx, agent_idx, hidden_states)
                       -> prefix_embedding (1, m, hidden_size) or None.
                       Used by ThoughtComm to inject latent thoughts.

        Returns:
            all_responses: list[list[str]] — [num_rounds][num_agents]
            all_hidden_states: list[list[Tensor]] — [num_rounds][num_agents]
                               each tensor is (hidden_size,). Empty if extract_hidden=False.
        """
        conversations = [[] for _ in range(self.config.num_agents)]
        all_responses = []
        all_hidden_states = []

        for round_idx in range(self.config.num_rounds):
            round_responses = []
            round_hidden = []

            for agent_idx in range(self.config.num_agents):
                # Build prompt
                if round_idx == 0:
                    prompt = INITIAL_PROMPT.format(question=question)
                else:
                    other_resps = format_other_responses(
                        all_responses[round_idx - 1], agent_idx
                    )
                    prompt = DEBATE_PROMPT.format(
                        other_responses=other_resps, question=question
                    )

                conversations[agent_idx].append({"role": "user", "content": prompt})

                # Get prefix if ThoughtComm is active
                prefix = None
                if prefix_fn is not None and round_idx > 0:
                    prefix = prefix_fn(round_idx, agent_idx, all_hidden_states)

                # Generate response
                response, h_i = self._generate(
                    conversations[agent_idx],
                    extract_hidden=extract_hidden,
                    prefix_embedding=prefix,
                )

                conversations[agent_idx].append({"role": "assistant", "content": response})
                round_responses.append(response)
                if extract_hidden:
                    round_hidden.append(h_i)

                clear_gpu_memory()

            all_responses.append(round_responses)
            if extract_hidden:
                all_hidden_states.append(round_hidden)

        return all_responses, all_hidden_states

    def _generate(self, conversation, extract_hidden=False, prefix_embedding=None):
        """Generate a response for one agent, optionally with prefix injection.

        Args:
            conversation: list of message dicts for this agent
            extract_hidden: whether to extract hidden state
            prefix_embedding: optional (1, m, hidden_size) prefix to prepend

        Returns:
            response: generated text (str)
            h_i: (hidden_size,) hidden state or None
        """
        # Tokenize conversation
        input_ids = self.tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        if prefix_embedding is not None:
            response, h_i = self._generate_with_prefix(
                input_ids, prefix_embedding, extract_hidden
            )
        else:
            response, h_i = self._generate_standard(input_ids, extract_hidden)

        return response, h_i

    def _generate_standard(self, input_ids, extract_hidden):
        """Standard generation without prefix."""
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_ids = outputs[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        h_i = None
        if extract_hidden:
            h_i = extract_last_hidden_state(self.model, outputs)

        return response, h_i

    def _generate_with_prefix(self, input_ids, prefix_embedding, extract_hidden):
        """Generation with prefix embedding prepended to input embeddings.

        This implements the prefix injection from Section 4.3:
        P_t^(i) is prepended to the token embeddings.
        """
        embed_layer = self.model.get_input_embeddings()
        token_embeds = embed_layer(input_ids)  # (1, seq_len, hidden_size)

        prefix = prefix_embedding.to(token_embeds.dtype).to(token_embeds.device)
        combined_embeds = torch.cat([prefix, token_embeds], dim=1)

        attention_mask = torch.ones(
            1, combined_embeds.shape[1], dtype=torch.long, device=self.device
        )

        outputs = self.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode: skip prefix length in output tokens
        # Note: generate() with inputs_embeds may not include the prefix in output IDs
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        h_i = None
        if extract_hidden:
            # Run forward pass on full generated sequence for hidden state
            full_ids = outputs  # (1, gen_len) — these are just the generated token IDs
            # We need to reconstruct the full sequence with prefix for hidden state
            # Use the combined embeddings approach
            gen_embeds = embed_layer(outputs)
            full_embeds = torch.cat([prefix, gen_embeds], dim=1)
            full_mask = torch.ones(
                1, full_embeds.shape[1], dtype=torch.long, device=self.device
            )
            fwd_out = self.model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                output_hidden_states=True,
            )
            last_layer = fwd_out.hidden_states[-1]
            h_i = last_layer[0, -1, :].float().cpu()

        return response, h_i
