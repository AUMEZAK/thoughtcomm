"""Tests for v3 fixes: 2/3 depth hidden state extraction + OpenReview parameters.

Also verifies v2 fixes are preserved.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ROOT = os.path.join(os.path.dirname(__file__), '..')


def read_source(relpath):
    with open(os.path.join(ROOT, relpath)) as f:
        return f.read()


# ===== v3 config =====

def test_config_v3_params():
    """Config has 2/3 depth, lr=1e-5, batch=128."""
    from configs.config import ThoughtCommConfig
    c = ThoughtCommConfig()
    assert c.hidden_state_layer_fraction == 0.667, f"Got {c.hidden_state_layer_fraction}"
    assert c.ae_lr == 1e-5, f"Got {c.ae_lr}"
    assert c.ae_batch_size == 128, f"Got {c.ae_batch_size}"
    print("PASS: config v3 params (fraction=0.667, lr=1e-5, batch=128)")


def test_config_target_layer():
    """target_layer_index computes correctly for various models."""
    from configs.config import ThoughtCommConfig
    c = ThoughtCommConfig()

    # Qwen3-0.6B: 28 layers → target = int(28 * 0.667) = 18
    assert c.target_layer_index(28) == 18, f"Got {c.target_layer_index(28)}"

    # LLaMA-3-8B: 32 layers → target = int(32 * 0.667) = 21
    assert c.target_layer_index(32) == 21, f"Got {c.target_layer_index(32)}"

    # Edge: fraction=1.0 → last layer
    c2 = ThoughtCommConfig(hidden_state_layer_fraction=1.0)
    assert c2.target_layer_index(28) == 28

    # Edge: 1 layer model
    assert c.target_layer_index(1) == 0

    print("PASS: target_layer_index (28→18, 32→21, edge cases OK)")


# ===== v3 model_utils =====

def test_model_utils_extract_hidden_state():
    """model_utils has extract_hidden_state with layer_fraction parameter."""
    src = read_source('models/model_utils.py')
    assert 'def extract_hidden_state(' in src, "extract_hidden_state function not found"
    assert 'layer_fraction' in src, "layer_fraction parameter not found"
    assert 'num_layers * layer_fraction' in src, "layer calculation not found"
    # Backward compat
    assert 'def extract_last_hidden_state(' in src, "backward compat alias missing"
    print("PASS: model_utils has extract_hidden_state with layer_fraction")


# ===== v3 debate.py =====

def test_debate_uses_target_layer():
    """debate.py uses config.target_layer_index for hidden state extraction."""
    src = read_source('pipeline/debate.py')

    # Both extraction points should use target_layer_index
    assert 'target_layer_index' in src, "target_layer_index not used in debate.py"
    assert src.count('target_layer_index') >= 2, \
        f"target_layer_index used {src.count('target_layer_index')} times, expected >=2"

    # No remaining hidden_states[-1] in extraction context
    # (hidden_states[-1] is OK for getting the last *step* in generate output,
    #  but not for selecting the *layer*)
    lines = src.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'hidden_states[-1]' in stripped and not stripped.startswith('#'):
            # This is OK if it's selecting the last generation step (tuple of layers)
            # Not OK if it's selecting the last layer directly for h_i
            if 'h_i' in stripped or 'last_layer' in stripped:
                assert False, f"Line {i+1}: hidden_states[-1] used for layer selection: {stripped}"

    print("PASS: debate.py uses target_layer_index (2 locations)")


# ===== v2 fixes preserved =====

def test_v2_enable_thinking_preserved():
    """enable_thinking=False still present in all 3 files."""
    for f in ['pipeline/debate.py', 'pipeline/thought_comm.py', 'training/train_adapter.py']:
        src = read_source(f)
        assert 'enable_thinking=False' in src, f"FAIL: {f} missing enable_thinking=False"
    print("PASS: v2 enable_thinking fixes preserved")


def test_v2_cosine_loss_preserved():
    """cosine_loss still uses softmax (not argmax)."""
    src = read_source('training/train_adapter.py')
    assert 'F.softmax' in src, "softmax not found"
    assert 'embed_layer.weight' in src, "embed_layer.weight not found"
    # No active argmax in cosine loss
    lines = src.split('\n')
    for line in lines:
        if 'argmax' in line and not line.strip().startswith('#'):
            assert False, f"Active argmax found: {line.strip()}"
    print("PASS: v2 cosine_loss softmax fix preserved")


def test_v2_prefix_embedding_preserved():
    """prefix + token_embeds + gen_embeds in hidden state recomputation."""
    src = read_source('pipeline/debate.py')
    assert 'torch.cat([prefix, token_embeds, gen_embeds]' in src
    print("PASS: v2 prefix embedding fix preserved")


def test_v2_data_scale_preserved():
    """num_train=500, num_eval=500."""
    from configs.config import ThoughtCommConfig
    c = ThoughtCommConfig()
    assert c.num_train == 500
    assert c.num_eval == 500
    print("PASS: v2 data scale preserved (500/500)")


if __name__ == '__main__':
    print("=" * 60)
    print("v3 fix tests")
    print("=" * 60)

    # v3 new
    test_config_v3_params()
    test_config_target_layer()
    test_model_utils_extract_hidden_state()
    test_debate_uses_target_layer()

    # v2 preserved
    test_v2_enable_thinking_preserved()
    test_v2_cosine_loss_preserved()
    test_v2_prefix_embedding_preserved()
    test_v2_data_scale_preserved()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
