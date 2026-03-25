"""LLM loading and hidden state extraction utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name, dtype="bfloat16", device_map="auto"):
    """Load HuggingFace model and tokenizer.

    Args:
        model_name: HF model identifier
        dtype: precision ('bfloat16', 'float16', 'float32')
        device_map: device placement strategy

    Returns:
        model: AutoModelForCausalLM
        tokenizer: AutoTokenizer
    """
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


@torch.no_grad()
def extract_hidden_state(model, input_ids, attention_mask=None, layer_fraction=0.667):
    """Extract hidden state at specified depth and last token position.

    The paper authors extract from 2/3 model depth (OpenReview Rebuttal),
    not the final layer. Middle layers contain richer semantic representations.

    Args:
        model: HF causal LM
        input_ids: (1, seq_len) token IDs (full sequence including generated tokens)
        attention_mask: (1, seq_len) or None
        layer_fraction: fraction of model depth (0.667 = 2/3 depth, 1.0 = last layer)

    Returns:
        h: (hidden_size,) hidden state (float32, CPU)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # outputs.hidden_states: tuple of (num_layers+1,) tensors
    # Index 0 = embedding layer output, 1..N = transformer layer outputs
    num_layers = len(outputs.hidden_states) - 1  # exclude embedding layer
    target_layer = min(int(num_layers * layer_fraction), num_layers)

    h = outputs.hidden_states[target_layer][0, -1, :].float().cpu()
    return h


# Backward compatibility alias
def extract_last_hidden_state(model, input_ids, attention_mask=None):
    """Legacy wrapper — extracts from last layer."""
    return extract_hidden_state(model, input_ids, attention_mask, layer_fraction=1.0)
