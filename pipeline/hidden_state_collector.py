"""Hidden state collection from multi-agent debate (Section 4.1).

Runs debate on training data, collects concatenated hidden states H_t
from all agents at each round, and saves checkpoints.
"""

import os
import torch
from tqdm import tqdm
from utils.memory import clear_gpu_memory, print_memory_stats


class HiddenStateCollector:
    """Collects concatenated hidden states H_t from multi-agent debate."""

    def __init__(self, debate_pipeline, config):
        """
        Args:
            debate_pipeline: MultiAgentDebate instance
            config: ThoughtCommConfig
        """
        self.debate = debate_pipeline
        self.config = config

    def collect(self, dataset, save_dir=None, checkpoint_every=50):
        """Run debate on all examples, collect hidden states.

        Args:
            dataset: list of dicts with 'question' and 'answer'
            save_dir: directory for checkpoint saving (None to skip)
            checkpoint_every: save every N examples

        Returns:
            H_all: (num_samples * num_rounds, n_h) concatenated hidden states
            metadata: list of dicts with question, round, responses, answer
        """
        H_list = []
        metadata = []
        start_idx = 0

        # Resume from checkpoint if available
        if save_dir and os.path.exists(os.path.join(save_dir, "checkpoint.pt")):
            ckpt = torch.load(os.path.join(save_dir, "checkpoint.pt"))
            H_list = ckpt["H_list"]
            metadata = ckpt["metadata"]
            start_idx = ckpt["next_idx"]
            print(f"Resuming from example {start_idx}/{len(dataset)}")

        for idx in tqdm(range(start_idx, len(dataset)), desc="Collecting hidden states"):
            example = dataset[idx]

            try:
                responses, hidden_states = self.debate.run_debate(
                    example["question"], extract_hidden=True
                )
            except Exception as e:
                print(f"Error on example {idx}: {e}. Skipping.")
                continue

            for round_idx in range(self.config.num_rounds):
                # Concatenate hidden states from all agents: (n_h,)
                H_t = torch.cat(hidden_states[round_idx], dim=0)
                H_list.append(H_t)
                metadata.append({
                    "example_idx": idx,
                    "round": round_idx,
                    "question": example["question"],
                    "answer": example["answer"],
                    "responses": responses[round_idx],
                })

            clear_gpu_memory()

            # Checkpoint
            if save_dir and (idx + 1) % checkpoint_every == 0:
                self._save_checkpoint(save_dir, H_list, metadata, idx + 1)
                print(f"  Checkpoint saved at example {idx + 1}")
                print_memory_stats("  ")

        H_all = torch.stack(H_list, dim=0)  # (total_samples, n_h)

        # Final save
        if save_dir:
            self._save_final(save_dir, H_all, metadata)

        return H_all, metadata

    def _save_checkpoint(self, save_dir, H_list, metadata, next_idx):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "H_list": H_list,
            "metadata": metadata,
            "next_idx": next_idx,
        }, os.path.join(save_dir, "checkpoint.pt"))

    def _save_final(self, save_dir, H_all, metadata):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "H": H_all,
            "metadata": metadata,
        }, os.path.join(save_dir, "hidden_states.pt"))
        print(f"Final hidden states saved: {H_all.shape}")

        # Remove checkpoint file
        ckpt_path = os.path.join(save_dir, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
