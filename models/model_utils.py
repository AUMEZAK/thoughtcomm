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
def extract_last_hidden_state(model, input_ids, attention_mask=None):
    """Extract the last-layer hidden state at the last token position.

    This corresponds to H_t^(i) in the paper: the model state of agent i,
    which is the representation of the last generated token at the final layer.

    Args:
        model: HF causal LM
        input_ids: (1, seq_len) token IDs (full sequence including generated tokens)
        attention_mask: (1, seq_len) or None

    Returns:
        h: (hidden_size,) last-layer, last-token hidden state (float32, CPU)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # outputs.hidden_states: tuple of (num_layers+1,) tensors, each (batch, seq_len, hidden)
    last_layer = outputs.hidden_states[-1]  # (1, seq_len, hidden_size)
    h = last_layer[0, -1, :].float().cpu()   # (hidden_size,)
    return h
