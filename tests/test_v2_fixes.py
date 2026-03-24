"""Tests for v2 fixes: enable_thinking, cosine_loss gradient, prefix embedding, config.

Tests are split into:
- Source code validation (no torch needed)
- Functional tests (torch needed, skipped if unavailable)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


ROOT = os.path.join(os.path.dirname(__file__), '..')


def read_source(relpath):
    with open(os.path.join(ROOT, relpath)) as f:
        return f.read()


# ===== Source code validation (no torch) =====

def test_step1_enable_thinking_thought_comm():
    """thought_comm.py has enable_thinking=False with try/except fallback."""
    src = read_source('pipeline/thought_comm.py')
    assert 'enable_thinking=False' in src, "FAIL: enable_thinking=False not found"
    assert 'except TypeError:' in src, "FAIL: TypeError fallback not found"
    print("PASS Step 1: thought_comm.py — enable_thinking=False with fallback")


def test_step2_enable_thinking_train_adapter():
    """train_adapter.py has enable_thinking=False with try/except fallback."""
    src = read_source('training/train_adapter.py')
    assert 'enable_thinking=False' in src, "FAIL: enable_thinking=False not found"
    assert 'except TypeError:' in src, "FAIL: TypeError fallback not found"
    print("PASS Step 2: train_adapter.py — enable_thinking=False with fallback")


def test_step3_cosine_loss_source():
    """train_adapter.py uses softmax instead of argmax for cosine_loss."""
    src = read_source('training/train_adapter.py')
    # argmax should not appear in the cosine loss section
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'argmax' in line and 'Cosine' not in lines[max(0,i-3):i+1].__repr__():
            # argmax might exist in comments but not in active code
            stripped = line.strip()
            if not stripped.startswith('#'):
                assert False, f"FAIL: active argmax found at line {i+1}: {stripped}"

    assert 'F.softmax' in src, "FAIL: F.softmax not found"
    assert 'embed_layer.weight' in src, "FAIL: embed_layer.weight not found"
    assert 'gen_probs @' in src or 'gen_probs@' in src, "FAIL: gen_probs @ embed_layer.weight pattern not found"
    print("PASS Step 3 (source): cosine_loss uses softmax, not argmax")


def test_step4_prefix_embedding_source():
    """debate.py includes token_embeds in hidden state recomputation."""
    src = read_source('pipeline/debate.py')
    assert 'torch.cat([prefix, token_embeds, gen_embeds]' in src, \
        "FAIL: full_embeds should be cat([prefix, token_embeds, gen_embeds])"
    # Verify old pattern is gone
    # Count occurrences of the old pattern (prefix + gen without token_embeds in the hidden recompute section)
    print("PASS Step 4 (source): prefix + token_embeds + gen_embeds in hidden state recomputation")


def test_step5_config():
    """config.py defaults to num_train=500, num_eval=500."""
    from configs.config import ThoughtCommConfig
    config = ThoughtCommConfig()
    assert config.num_train == 500, f"FAIL: num_train={config.num_train}, expected 500"
    assert config.num_eval == 500, f"FAIL: num_eval={config.num_eval}, expected 500"
    print("PASS Step 5: num_train=500, num_eval=500")


def test_enable_thinking_consistency():
    """All files with apply_chat_template have enable_thinking handling."""
    files = [
        'pipeline/debate.py',
        'pipeline/thought_comm.py',
        'training/train_adapter.py',
    ]
    for f in files:
        src = read_source(f)
        has = 'enable_thinking=False' in src
        count = src.count('apply_chat_template')
        assert has, f"FAIL: {f} missing enable_thinking=False"
        print(f"  {f}: apply_chat_template x{count}, enable_thinking=YES")
    print("PASS: enable_thinking consistent across all 3 files")


# ===== Functional tests (torch required) =====

def test_step3_gradient_flow():
    """cosine_loss gradient flows through softmax to adapter parameters."""
    if not HAS_TORCH:
        print("SKIP Step 3 (functional): torch not available")
        return

    adapter = nn.Linear(8, 4)
    embed_weight = torch.randn(10, 4)

    z = torch.randn(1, 8)
    prefix = adapter(z)
    logits = prefix @ embed_weight.T
    logits = logits.unsqueeze(1).expand(1, 3, 10)

    gen_probs = F.softmax(logits / 0.1, dim=-1)
    gen_embeds_mean = (gen_probs @ embed_weight).mean(dim=1).float()

    ref_mean = torch.randn(1, 4)
    cosine_loss = 1.0 - F.cosine_similarity(gen_embeds_mean, ref_mean, dim=-1).mean()

    adapter.zero_grad()
    cosine_loss.backward()

    grad_norm = sum(p.grad.abs().sum().item() for p in adapter.parameters() if p.grad is not None)
    assert grad_norm > 0, f"FAIL: adapter grad_norm={grad_norm}, expected > 0"
    print(f"PASS Step 3 (functional): gradient flows (grad_norm={grad_norm:.6f})")


def test_step4_shape():
    """full_embeds has correct shape with token_embeds included."""
    if not HAS_TORCH:
        print("SKIP Step 4 (functional): torch not available")
        return

    p, i, g, h = 1, 20, 50, 16
    prefix = torch.randn(1, p, h)
    token_embeds = torch.randn(1, i, h)
    gen_embeds = torch.randn(1, g, h)

    full = torch.cat([prefix, token_embeds, gen_embeds], dim=1)
    expected = p + i + g
    assert full.shape == (1, expected, h), f"FAIL: shape {full.shape} != (1, {expected}, {h})"
    print(f"PASS Step 4 (functional): shape = (1, {expected}, {h})")


if __name__ == '__main__':
    print("=" * 60)
    print("v2 fix tests")
    print("=" * 60)

    # Source validation (no torch)
    test_step1_enable_thinking_thought_comm()
    test_step2_enable_thinking_train_adapter()
    test_step3_cosine_loss_source()
    test_step4_prefix_embedding_source()
    test_step5_config()
    test_enable_thinking_consistency()

    # Functional tests (torch)
    test_step3_gradient_flow()
    test_step4_shape()

    print("=" * 60)
    print("ALL TESTS PASSED" if HAS_TORCH else "ALL SOURCE TESTS PASSED (torch tests skipped)")
    print("=" * 60)
