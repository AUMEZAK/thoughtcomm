"""Training loop for the prefix adapter (Section 4.3).

Loss function L_comm (Eq. 12):
    L_comm = sum_i sum_t [ (1 - cos(phi_bar(y_gen), phi_bar(y_ref)))
                           - log p(y_gen | context, P_t^(i)) ]

Trains the prefix adapter g and agreement weights w jointly.
The LLM and autoencoder remain frozen.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.memory import clear_gpu_memory


def precompute_reference_embeddings(metadata, tokenizer, embed_layer, config):
    """Pre-compute mean token embeddings phi_bar(y_ref) for all samples.

    Args:
        metadata: list of dicts with 'responses' key
        tokenizer: HF tokenizer
        embed_layer: model.get_input_embeddings()
        config: ThoughtCommConfig

    Returns:
        ref_embeddings: dict mapping sample_idx -> list of (hidden_size,) tensors
    """
    ref_embeddings = {}
    device = next(embed_layer.parameters()).device

    for idx, meta in enumerate(metadata):
        agent_embeds = []
        for agent_idx in range(config.num_agents):
            ref_text = meta["responses"][agent_idx]
            ref_ids = tokenizer.encode(
                ref_text, return_tensors="pt",
                max_length=config.max_gen_tokens_adapter,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                embeds = embed_layer(ref_ids)  # (1, seq_len, d)
                mean_embed = embeds.mean(dim=1).squeeze(0).float().cpu()  # (d,)

            agent_embeds.append(mean_embed)
        ref_embeddings[idx] = agent_embeds

    return ref_embeddings


def train_adapter(model, tokenizer, autoencoder, reweighter, adapter,
                  H_train, metadata, config, verbose=True):
    """Train prefix adapter and agreement weights.

    Args:
        model: frozen LLM
        tokenizer: HF tokenizer
        autoencoder: frozen, trained SparsityRegularizedAE
        reweighter: AgreementReweighter (w params trainable)
        adapter: PrefixAdapter (trainable)
        H_train: (num_samples, n_h) hidden states
        metadata: list of dicts with responses, questions
        config: ThoughtCommConfig

    Returns:
        adapter: trained PrefixAdapter
        reweighter: AgreementReweighter with trained weights
        loss_history: list of per-epoch average losses
    """
    device = config.device

    # Freeze LLM and autoencoder
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # Trainable parameters: adapter + agreement weights
    adapter = adapter.to(device)
    reweighter = reweighter.to(device)
    params = list(adapter.parameters()) + [reweighter.w]
    optimizer = torch.optim.Adam(params, lr=config.adapter_lr)

    embed_layer = model.get_input_embeddings()

    # Pre-compute reference embeddings
    if verbose:
        print("Pre-computing reference embeddings...")
    ref_embeddings = precompute_reference_embeddings(
        metadata, tokenizer, embed_layer, config
    )

    loss_history = []

    for epoch in range(config.adapter_epochs):
        total_loss = 0.0
        num_samples = 0

        iterator = range(len(H_train))
        if verbose:
            iterator = tqdm(iterator, desc=f"Adapter Epoch {epoch + 1}")

        for sample_idx in iterator:
            H_t = H_train[sample_idx].unsqueeze(0).float().to(device)  # (1, n_h)
            meta = metadata[sample_idx]

            # Encode to latent
            with torch.no_grad():
                Z_hat = autoencoder.encode(H_t)  # (1, n_z)

            loss_sample = torch.tensor(0.0, device=device, requires_grad=True)

            for agent_idx in range(config.num_agents):
                # Personalized latent
                Z_tilde = reweighter.get_personalized_latent(Z_hat, agent_idx)

                # Generate prefix
                prefix = adapter(Z_tilde)  # (1, prefix_length, hidden_size)

                # Build input
                conversation = [{"role": "user", "content": meta["question"]}]
                input_ids = tokenizer.apply_chat_template(
                    conversation, return_tensors="pt", add_generation_prompt=True
                ).to(device)

                token_embeds = embed_layer(input_ids)
                prefix_cast = prefix.to(token_embeds.dtype)
                combined = torch.cat([prefix_cast, token_embeds], dim=1)

                # Get reference response tokens
                ref_text = meta["responses"][agent_idx]
                ref_ids = tokenizer.encode(
                    ref_text, return_tensors="pt",
                    max_length=config.max_gen_tokens_adapter,
                    truncation=True,
                ).to(device)

                ref_embeds = embed_layer(ref_ids)
                full_embeds = torch.cat([prefix_cast, token_embeds, ref_embeds], dim=1)

                # Forward pass for LM loss
                outputs = model(inputs_embeds=full_embeds)
                logits = outputs.logits

                # LM loss: predict ref tokens given prefix + input
                prefix_and_input_len = config.prefix_length + input_ids.shape[1]
                shift_logits = logits[:, prefix_and_input_len - 1:-1, :]
                shift_labels = ref_ids
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                lm_loss = F.cross_entropy(
                    shift_logits[:, :min_len].reshape(-1, shift_logits.size(-1)),
                    shift_labels[:, :min_len].reshape(-1),
                )

                # Cosine similarity loss
                # Approximate generated token embeddings from argmax of logits
                gen_ids = logits[:, prefix_and_input_len:prefix_and_input_len + min_len, :].argmax(dim=-1)
                gen_embeds_mean = embed_layer(gen_ids).mean(dim=1).float()  # (1, d)

                ref_mean = ref_embeddings[sample_idx][agent_idx].unsqueeze(0).to(device)
                cosine_loss = 1.0 - F.cosine_similarity(gen_embeds_mean, ref_mean, dim=-1).mean()

                agent_loss = (config.cosine_loss_weight * cosine_loss +
                              config.lm_loss_weight * lm_loss)
                loss_sample = loss_sample + agent_loss

            loss_sample = loss_sample / config.num_agents

            optimizer.zero_grad()
            loss_sample.backward()
            optimizer.step()

            total_loss += loss_sample.item()
            num_samples += 1

            clear_gpu_memory()

        avg_loss = total_loss / max(num_samples, 1)
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{config.adapter_epochs}: avg loss = {avg_loss:.4f}")

    return adapter, reweighter, loss_history
