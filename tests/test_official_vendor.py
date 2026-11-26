"""The vendored upstream code is unmodified, and the two CPU-side primitives of
mmllm.official compute what the upstream ones compute."""

import pytest
import torch

from mmllm import official
from mmllm.official import DocumentLayout, MosaicBlock, MosaicsConfig, memory_mask


def test_vendored_files_match_upstream_manifest():
    info = official.verify()
    assert info["commit"] == "c2ca0e88c5c89fc18a0a9465d6f7a1c3c8ea8faf"
    assert info["files"] >= 40


def test_verify_detects_a_modified_file(tmp_path, monkeypatch):
    import shutil
    dst = tmp_path / "MemoryMosaics"
    shutil.copytree(official.VENDOR, dst)
    f = dst / "model" / "mosaics" / "memory.py"
    f.write_text(f.read_text() + "\n# edited\n")
    monkeypatch.setattr(official, "VENDOR", dst)
    try:
        official.verify()
    except RuntimeError as e:
        assert "model/mosaics/memory.py" in str(e)
    else:
        raise AssertionError("a modified file was not detected")


def test_classes_come_from_the_vendored_package():
    for obj in (MosaicBlock, MosaicsConfig, official.MosaicsMemory, memory_mask, DocumentLayout):
        assert obj.__module__.startswith("model."), obj.__module__
    import model
    assert str(model.__file__).startswith(str(official.VENDOR))


def _eager_flex(q, k, v, **kw):
    """PyTorch's FlexAttention itself, uncompiled (upstream wraps this same function in torch.compile)."""
    from torch.nn.attention.flex_attention import flex_attention
    return flex_attention(q, k, v, **kw)


def test_dense_attention_equals_upstream_flex_attention():
    """Without gradients PyTorch's FlexAttention runs on CPU: compare it to the dense path,
    for the short (sliding window, own position excluded) and long (skip) masks, which both
    leave some rows with nothing to read."""
    torch.manual_seed(0)
    B, H, T, d = 2, 3, 37, 8
    q, k, v = (torch.randn(B, H, T, d) for _ in range(3))
    tokens, q_lens = torch.zeros(1, T, dtype=torch.long), torch.tensor([T])
    for scope, kw in (("short", {"window": 6}), ("long", {"skip": 5}), ("short", {"window": 0})):
        mask = memory_mask(scope, False, tokens, q_lens, "causal", **kw)
        with torch.no_grad():
            ref = _eager_flex(q, k, v, block_mask=mask.block_mask, scale=1)
            got = official.dense_attention(q, k, v, block_mask=mask.block_mask, scale=1)
        assert torch.isfinite(got).all()
        assert (ref - got).abs().max().item() < 1e-5, scope
    # rows that may read nothing give exactly zero
    mask = memory_mask("long", False, tokens, q_lens, "causal", skip=5)
    got = official.dense_attention(q, k, v, block_mask=mask.block_mask, scale=1)
    assert got[:, :, :5].abs().max().item() == 0.0


def test_leaky_scan_equals_upstream_reference_loop():
    torch.manual_seed(0)
    B, T, c_wk, G = 2, 77, 3, 4
    omega = torch.rand(B, T, c_wk, dtype=torch.float64)
    omega[:, 20] = 0.0                                   # a reset, as upstream's chunk warm-up does
    u = torch.randn(B, T, c_wk * G, dtype=torch.float64, requires_grad=True)
    y_prev = torch.randn(B, c_wk * G, dtype=torch.float64)
    for prev in (None, y_prev):
        ref = official._upstream_reference(omega, u, c_wk, prev)
        got = official.leaky_avg_scan(omega, u, c_wk, prev)
        assert (ref - got).abs().max().item() < 1e-10
    g_ref, = torch.autograd.grad(official._upstream_reference(omega, u, c_wk).pow(2).sum(), u)
    g_got, = torch.autograd.grad(official.leaky_avg_scan(omega, u, c_wk).pow(2).sum(), u)
    assert (g_ref - g_got).abs().max().item() < 1e-9


def _block(window=8, skip=4):
    cfg = MosaicsConfig(dim=48, n_layers=1, head_dim=12, max_seq_len=256, vocab_size=50, ffn_dim_multiplier=1.3,
                        multiple_of=1, attn_bias_type="causal", memory_gate=True, layout="flat", norm_eps=1e-6,
                        short_memory_type="mosaics", short_mem_window_size=window, n_short_heads=4,
                        long_memory_type="mosaics", long_key_feature="leaky_average", long_mem_skip_size=skip,
                        n_long_heads=4, n_cluster_heads=0)
    torch.manual_seed(0)
    blk = MosaicBlock(cfg, 0)
    blk.init_weights()
    return blk


def _masks(T, window=8, skip=4):
    tokens, q_lens = torch.zeros(1, T, dtype=torch.long), torch.tensor([T])
    return {"short": memory_mask("short", False, tokens, q_lens, "causal", window=window),
            "long": memory_mask("long", False, tokens, q_lens, "causal", skip=skip)}


def _pos(B, T, start=None):
    lay = DocumentLayout()
    if start is None:
        lay.full_sequence("causal", torch.tensor([T] * B), T)
        return lay.positions("cpu", B, T)
    lay.decoding("causal", start, B)
    return lay.positions("cpu", B, 1)


def test_official_block_is_causal_and_trains_on_cpu():
    blk = _block()
    B, T = 2, 40
    x = torch.randn(B, T, 48, requires_grad=True)
    y, _ = blk(x, _masks(T), _pos(B, T))
    y.pow(2).mean().backward()                            # CPU backward through the upstream modules
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in blk.short_memory.parameters())
    x2 = x.detach().clone(); x2[:, -1] += 1.0
    with torch.no_grad():
        y2, _ = blk(x2, _masks(T), _pos(B, T))
    assert (y[:, :-1] - y2[:, :-1]).abs().max().item() == 0.0
    assert (y[:, -1] - y2[:, -1]).abs().max().item() > 0


def test_official_block_generation_cache_equals_full_forward():
    blk = _block()
    B, T, P = 2, 40, 25
    x = torch.randn(B, T, 48)
    with torch.no_grad():
        full, _ = blk(x, _masks(T), _pos(B, T))
        cache = blk.init_cache(B, 64, x.dtype, x.device)
        out, cache = blk(x[:, :P], _masks(P), _pos(B, P), cache)
        outs = [out]
        for t in range(P, T):
            o, cache = blk(x[:, t:t + 1], None, _pos(B, 1, start=t), cache)
            outs.append(o)
    assert (torch.cat(outs, 1) - full).abs().max().item() < 1e-5


def _dense_bf16_autocast(dev):
    torch.manual_seed(0)
    B, H, T, d = 1, 2, 24, 8
    q, k, v = (torch.randn(B, H, T, d, device=dev, requires_grad=True) for _ in range(3))
    mask = memory_mask("short", False, torch.zeros(1, T, dtype=torch.long, device=dev), torch.tensor([T], device=dev),
                       "causal", window=6)
    ref = official.dense_attention(q, k, v, block_mask=mask.block_mask, scale=1)
    with torch.autocast(dev, dtype=torch.bfloat16):
        out = official.dense_attention(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                                       block_mask=mask.block_mask, scale=1)
        assert out.dtype == torch.bfloat16
        out.float().pow(2).sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    assert (out.float() - ref).abs().max().item() < 0.05          # bf16 inputs, fp32 arithmetic


def test_dense_attention_under_bf16_autocast_cpu():
    """bf16 inputs inside a bf16 autocast context, with gradients: result comes back in the
    caller's dtype, computed in fp32. (This does NOT reproduce the first GPU run's failure:
    CPU autocast leaves softmax in bf16. The CUDA test below does.)"""
    _dense_bf16_autocast("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dense_attention_under_bf16_autocast_cuda():
    """First GPU run (results/2026-11-11_2230_mq-dev-0p8b): torch._safe_softmax records the
    dtype of its input (bf16) for its backward, while CUDA autocast runs the softmax inside it
    in fp32 -> "expected input and grad types to match ...". dense_attention now leaves
    autocast for its own arithmetic."""
    _dense_bf16_autocast("cuda")
