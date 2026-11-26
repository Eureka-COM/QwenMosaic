import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REAL_MODEL = os.environ.get("MMLLM_MODEL", "")  # local path or HF id of Qwen/Qwen3.5-0.8B


def tiny_qwen35(seed: int = 0):
    """A randomly initialised Qwen3.5 text model with the real layer pattern (3 GDN + 1 attention) x 2."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        eos_token_id=7, pad_token_id=0, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    )
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    return model


def tiny_attention_only(kind: str = "qwen3", seed: int = 0):
    """A randomly initialised decoder with softmax attention in every layer: Qwen3 or Llama type
    (no layer_types in the config; RMSNorm multiplies by weight, not by 1 + weight)."""
    import transformers
    torch.manual_seed(seed)
    common = dict(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, eos_token_id=7, pad_token_id=0, tie_word_embeddings=True, max_position_embeddings=512)
    if kind == "qwen3":
        cfg = transformers.Qwen3Config(head_dim=16, **common)
        model = transformers.Qwen3ForCausalLM(cfg)
    else:
        cfg = transformers.LlamaConfig(**common)
        model = transformers.LlamaForCausalLM(cfg)
    with torch.no_grad():                      # norm gains away from 1, so that a wrong convention shows
        for layer in model.model.layers:
            for n in (layer.input_layernorm, layer.post_attention_layernorm):
                n.weight.copy_(0.5 + torch.rand_like(n.weight))
    return model.float().eval()


@pytest.fixture
def tiny_model():
    return tiny_qwen35()


@pytest.fixture(scope="session")
def real_model_path():
    if not REAL_MODEL:
        pytest.skip("set MMLLM_MODEL to a Qwen3.5 checkpoint to run real-model tests")
    return REAL_MODEL
