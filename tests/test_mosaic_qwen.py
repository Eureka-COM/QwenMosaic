"""MosaicQwen: every attention layer of a Qwen3.5-type model becomes an official MosaicBlock."""

import copy

import pytest
import torch
from conftest import tiny_attention_only, tiny_qwen35

from mmllm import official
from mmllm.mosaic_qwen import (LoRALinear, MosaicQwenLayer, MQConfig, _text_model, convert_to_mosaic_qwen,
                               official_config)

SMALL = dict(n_short_heads=2, n_long_heads=2, head_dim=16, window=8, skip=4, random_skip=4)


def _ids(B=2, T=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 300, (B, T), generator=g)


def _logits(model, ids):
    with torch.no_grad():
        return model(ids, use_cache=False).logits


def test_defaults_are_read_from_the_upstream_mmv2_config():
    c = MQConfig.from_upstream_mmv2()
    assert (c.n_short_heads, c.n_long_heads, c.head_dim) == (16, 16, 128)
    assert (c.window, c.skip, c.random_skip, c.layout, c.memory_gate) == (256, 64, 192, "flat", True)


def test_conversion_replaces_every_attention_layer_with_an_official_block():
    model = tiny_qwen35()
    ids = _ids()
    ref = _logits(model, ids)
    mq = convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    text = _text_model(model)
    assert mq.cfg.layers == [3, 7]
    for i in (3, 7):
        layer = text.layers[i]
        assert isinstance(layer, MosaicQwenLayer)
        assert type(layer.block) is official.MosaicBlock
        assert type(layer.block.short_memory) is official.MosaicsMemory
        assert type(layer.block.long_memory) is official.MosaicsMemory
        assert layer.block.layout == "flat"
    # the student holds no attention weights and no attention module
    assert mq.attention_parameters() == []
    assert not any(type(m).__name__ == "Qwen3_5Attention" for m in model.modules())
    # the teacher is the original model, exactly
    with mq.teacher():
        assert (_logits(model, ids) - ref).abs().max().item() == 0.0
    assert (_logits(model, ids) - ref).abs().max().item() > 1e-3        # the student is a different model
    assert mq.cfg.upstream_commit == official.upstream_info()["commit"]


def test_persistent_memory_and_norms_start_as_the_layers_own():
    """With the new memories silenced, a converted block is the original layer minus its attention."""
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    layer = mq.layers[3]
    orig = mq.state.orig[3]
    assert layer.block.feed_forward.w1.weight is orig.mlp.gate_proj.weight      # shared, not copied
    assert layer.block.feed_forward.w3.weight is orig.mlp.up_proj.weight
    assert layer.block.feed_forward.w2.weight is orig.mlp.down_proj.weight
    with torch.no_grad():
        for m in layer.memories():
            m.proj_out.weight.zero_()
        x = torch.randn(2, 20, 64)
        got = layer(x)
        want = x + orig.mlp(orig.post_attention_layernorm(x))
        h_block = layer.block.attention_norm(x)
        h_orig = orig.input_layernorm(x)
    assert (got - want).abs().max().item() < 1e-5
    assert (h_block - h_orig).abs().max().item() < 1e-5


def test_memory_residual_is_the_memory_part_of_the_official_block():
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    layer = mq.layers[7]
    x = torch.randn(2, 30, 64)
    with torch.no_grad():
        mix = layer.memory_residual(x)
        blk = layer.block
        want = (x + mix) + blk.feed_forward(blk.ffn_norm(x + mix))
        got = layer(x)
    assert mix.abs().max().item() > 0
    assert (got - want).abs().max().item() < 1e-5


def test_student_is_causal():
    model = tiny_qwen35()
    convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    ids = _ids()
    a = _logits(model, ids)
    ids2 = ids.clone(); ids2[:, -1] = 5
    b = _logits(model, ids2)
    assert (a[:, :-1] - b[:, :-1]).abs().max().item() == 0.0


@pytest.mark.parametrize("replace", ["attention", "all"])
def test_generation_cache_equals_full_forward(replace):
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(replace=replace, **SMALL))
    if replace == "all":
        assert mq.cfg.layers == list(range(8))
        assert not any(type(m).__name__ == "Qwen3_5GatedDeltaNet" for m in model.modules())
    ids = _ids(T=44)
    full = _logits(model, ids)
    with torch.no_grad():
        out = model(ids[:, :20], use_cache=True)
        pkv, parts = out.past_key_values, [out.logits]
        for t in range(20, 30):                                    # one token at a time
            out = model(ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values; parts.append(out.logits)
        out = model(ids[:, 30:44], past_key_values=pkv, use_cache=True)   # several tokens at once
        parts.append(out.logits)
    assert (torch.cat(parts, 1) - full).abs().max().item() < 2e-4
    assert pkv.get_seq_length() == 44
    with torch.no_grad():
        gen = model.generate(ids[:1, :12], max_new_tokens=6, do_sample=False)
    assert gen.shape == (1, 18)
    # greedy generation must agree with the full forward pass on its own output
    nxt = _logits(model, gen[:, :-1])[0, 11:].argmax(-1)
    assert torch.equal(nxt, gen[0, 12:])


def test_two_generations_do_not_share_state():
    model = tiny_qwen35()
    convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    a, b = _ids(B=1, T=30, seed=1), _ids(B=1, T=30, seed=2)
    fa, fb = _logits(model, a), _logits(model, b)
    with torch.no_grad():
        oa = model(a[:, :10], use_cache=True); ob = model(b[:, :10], use_cache=True)
        pa, pb = oa.past_key_values, ob.past_key_values
        la, lb = [oa.logits], [ob.logits]
        for t in range(10, 30):                                    # interleaved
            oa = model(a[:, t:t + 1], past_key_values=pa, use_cache=True); pa = oa.past_key_values; la.append(oa.logits)
            ob = model(b[:, t:t + 1], past_key_values=pb, use_cache=True); pb = ob.past_key_values; lb.append(ob.logits)
    assert (torch.cat(la, 1) - fa).abs().max().item() < 2e-4
    assert (torch.cat(lb, 1) - fb).abs().max().item() < 2e-4


def test_local_step_trains_only_that_layers_memories_and_reduces_the_loss():
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    params = mq.set_trainable(memories=True, norms=False, lora=False)
    assert {id(p) for p in params} == {id(p) for p in mq.memory_parameters()}
    opt = torch.optim.AdamW(params, lr=3e-3)
    ids = _ids(B=4, T=40)
    hist = []

    def local(i, x, mixed, out):
        with torch.enable_grad():
            pred = mq.layers[i].memory_residual(x.detach())
            loss = (pred - mixed).pow(2).mean() / mixed.pow(2).mean()
            loss.backward()
        hist[-1][i] = loss.item()

    base_before = copy.deepcopy(mq.state.orig[3].self_attn.state_dict())
    for _ in range(30):
        hist.append({})
        with torch.no_grad(), mq.teacher(capture=local):
            model(ids, use_cache=False)
        opt.step(); opt.zero_grad()
    assert hist[-1][3] < 0.8 * hist[0][3] and hist[-1][7] < 0.8 * hist[0][7], (hist[0], hist[-1])
    for k, v in mq.state.orig[3].self_attn.state_dict().items():
        assert torch.equal(v, base_before[k])


def test_training_with_cache_is_refused():
    model = tiny_qwen35()
    convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    model.train()
    with pytest.raises(RuntimeError, match="use_cache=False"):
        model(_ids(), use_cache=True)


def test_delta_round_trip_with_shards_and_split(tmp_path):
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    mq.add_lora(r=2)
    mq.set_trainable()
    with torch.no_grad():
        for p in mq.memory_parameters() + mq.lora_parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ids = _ids()
    want = _logits(model, ids)
    meta = mq.save_delta(tmp_path / "d", shard_mb=1)
    assert meta["attention_parameters_left"] == 0 and len(meta["shards"]) >= 1
    # force the row-splitting path on a small tensor
    import mmllm.mosaic_qwen as M
    model2 = tiny_qwen35()
    mq2 = convert_to_mosaic_qwen(model2, MQConfig(**SMALL))
    mq2.load_delta(tmp_path / "d")
    assert len(mq2.lora) == len(mq.lora) > 0
    assert (_logits(model2, ids) - want).abs().max().item() < 1e-5
    assert M.FORMAT_VERSION == 1


def test_lora_does_not_touch_the_teacher_and_merges_exactly():
    model = tiny_qwen35()
    ids = _ids()
    ref = _logits(model, ids)
    mq = convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    n = mq.add_lora(r=2)
    assert n > 0 and all(isinstance(m, LoRALinear) for m in mq.lora.values())
    before = _logits(model, ids)                       # B = 0: LoRA starts as a no-op
    with torch.no_grad():
        for m in mq.lora.values():
            m.lora_B.normal_(std=0.05)
    after = _logits(model, ids)
    assert (after - before).abs().max().item() > 1e-4
    with mq.teacher():
        assert (_logits(model, ids) - ref).abs().max().item() == 0.0
    with pytest.raises(RuntimeError):
        mq.merge_lora()                                # the teacher still shares the weights
    mq.drop_teacher()
    mq.merge_lora()
    assert not mq.lora and not any(isinstance(m, LoRALinear) for m in model.modules())
    assert (_logits(model, ids) - after).abs().max().item() < 1e-4
    with pytest.raises(RuntimeError):
        with mq.teacher():
            _logits(model, ids)


def test_official_config_reproduces_the_ffn_width():
    model = tiny_qwen35()
    tcfg = _text_model(model).config
    ocfg = official_config(tcfg, MQConfig(**SMALL))
    blk = official.MosaicBlock(ocfg, 0)
    assert blk.feed_forward.hidden_dim == tcfg.intermediate_size
    for dim, inter in ((1024, 3584), (5120, 17408), (2048, 6144)):
        tcfg.hidden_size, tcfg.intermediate_size = dim, inter
        o = official_config(tcfg, MQConfig())
        assert int(o.ffn_dim_multiplier * int(2 * 4 * dim / 3)) == inter


# ── rows of several documents: "doc_causal" and the cluster memory (plan section 10, step 2) ──

EOS = 5                                     # not produced by _ids (10..299)
DOCS = dict(attn_bias_type="doc_causal", eos_token_id=EOS, replace="all")
PLAIN = dict(short_key_feature="plain", long_key_feature="plain", cluster_key_feature="plain")


def _doc_ids(B=2, T=48, ends=((9, 19, 24, 35), (5, 26, 33)), seed=0):
    ids = _ids(B=B, T=T, seed=seed)
    for row, cols in enumerate(ends):
        for c in cols:
            ids[row, c] = EOS
    return ids


def test_doc_causal_on_single_document_rows_is_the_causal_model():
    ids = _ids()
    a = tiny_qwen35(); convert_to_mosaic_qwen(a, MQConfig(replace="all", **SMALL))
    b = tiny_qwen35(); mq = convert_to_mosaic_qwen(b, MQConfig(**DOCS, **SMALL))
    assert mq.state.doc_lens(2, 48, ids.device).tolist() == [48, 48]             # no ids seen yet: one document per row
    assert (_logits(a, ids) - _logits(b, ids)).abs().max().item() < 2e-5
    assert mq.state.doc_lens(2, 48, ids.device).tolist() == [48, 48]


def test_document_lengths_are_read_from_the_ids_of_the_call():
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(**DOCS, **SMALL))
    ids = _doc_ids()
    _logits(model, ids)
    # a document ends at its end-of-document token (which belongs to it) or at the end of the row
    assert mq.state.doc_lens(2, 48, ids.device).tolist() == [10, 10, 5, 11, 12, 6, 21, 7, 14]
    pos = mq.state.positions(2, 48, ids.device)
    assert pos.tokenid.view(2, 48)[0, :12].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2]
    assert pos.cluster_tokenid.view(2, 48)[0, [0, 10, 20, 25, 36]].tolist() == [0, 10, 20, 25, 36]
    assert pos.cluster_tokenid.view(2, 48)[1, [0, 6, 27, 34]].tolist() == [0, 6, 27, 34]


def test_without_a_cluster_memory_a_document_does_not_read_the_documents_before_it():
    """Plain keys: the leaky-average key of upstream runs over the row and is not reset at a
    document end, so with it the first tokens of a document still carry a trace of the one before."""
    model = tiny_qwen35()
    convert_to_mosaic_qwen(model, MQConfig(**DOCS, **PLAIN, **SMALL))
    ids = _doc_ids()
    a = _logits(model, ids)
    ids2 = ids.clone(); ids2[0, :9] = torch.randint(10, 300, (9,), generator=torch.Generator().manual_seed(3))
    b = _logits(model, ids2)
    assert (a[0, :10] - b[0, :10]).abs().max().item() > 1e-3          # the changed document itself
    assert (a[0, 10:] - b[0, 10:]).abs().max().item() == 0.0          # the documents after it
    assert (a[1] - b[1]).abs().max().item() == 0.0                    # the other row


def test_cluster_memory_reads_the_earlier_documents_of_the_row_and_only_those():
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(n_cluster_heads=2, **DOCS, **PLAIN, **SMALL))
    assert all(type(l.block.cluster_memory) is official.MosaicsMemory for l in mq.layers.values())
    assert sum(p.numel() for p in mq.memory_parameters()) > 0
    ids = _doc_ids()
    a = _logits(model, ids)
    ids2 = ids.clone(); ids2[0, :9] = torch.randint(10, 300, (9,), generator=torch.Generator().manual_seed(3))
    b = _logits(model, ids2)
    assert (a[0, 10:] - b[0, 10:]).abs().max().item() > 1e-4          # later documents read the changed one
    assert (a[1] - b[1]).abs().max().item() == 0.0                    # never another row
    ids3 = ids.clone(); ids3[0, 40] = 11 if ids[0, 40] != 11 else 12   # the future stays unread
    c = _logits(model, ids3)
    assert (a[0, :40] - c[0, :40]).abs().max().item() == 0.0
    with mq.without_cluster():                                        # switched off: documents are separate again
        a0, b0 = _logits(model, ids), _logits(model, ids2)
    assert (a0[0, 10:] - b0[0, 10:]).abs().max().item() == 0.0
    assert (a0[0, :10] - a[0, :10]).abs().max().item() == 0.0         # the first document has nothing to read
    assert (a0[0, 10:] - a[0, 10:]).abs().max().item() > 1e-4
    assert (_logits(model, ids) - a).abs().max().item() == 0.0        # and back on


@pytest.mark.parametrize("keys", ["leaky_average", "plain"])
def test_generation_cache_equals_full_forward_across_document_ends(keys):
    model = tiny_qwen35()
    extra = PLAIN if keys == "plain" else {}
    mq = convert_to_mosaic_qwen(model, MQConfig(n_cluster_heads=2, **DOCS, **extra, **SMALL))
    ids = _doc_ids(T=44)                     # row 0 ends documents at 9, 19 (last prompt token), 24, 35; row 1 at 5, 26, 33
    full = _logits(model, ids)

    def stepwise():
        with torch.no_grad():
            out = model(ids[:, :20], use_cache=True)
            pkv, parts = out.past_key_values, [out.logits]
            for t in range(20, 30):
                out = model(ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values; parts.append(out.logits)
            out = model(ids[:, 30:44], past_key_values=pkv, use_cache=True)
            parts.append(out.logits)
        return torch.cat(parts, 1)

    assert (stepwise() - full).abs().max().item() < 2e-4
    with mq.without_cluster():
        assert (stepwise() - _logits(model, ids)).abs().max().item() < 2e-4
    with torch.no_grad():
        gen = model.generate(ids[:1, :12], max_new_tokens=6, do_sample=False, eos_token_id=None, pad_token_id=0)
    nxt = _logits(model, gen[:, :-1])[0, 11:].argmax(-1)
    assert torch.equal(nxt, gen[0, 12:])


def test_memory_residual_matches_the_block_with_a_cluster_memory():
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(n_cluster_heads=2, **DOCS, **SMALL))
    ids = _doc_ids()
    _logits(model, ids)                                               # the layout of this batch
    layer = mq.layers[3]
    with torch.no_grad():
        x = torch.randn(2, 48, 64)
        res = layer.memory_residual(x)
        y = x + res
        want = y + layer.block.feed_forward(layer.block.ffn_norm(y))
        assert (layer(x) - want).abs().max().item() < 1e-5
        with mq.without_cluster():
            res0 = layer.memory_residual(x)
        assert (res0[0, :10] - res[0, :10]).abs().max().item() == 0.0
        assert (res0[0, 10:] - res[0, 10:]).abs().max().item() > 1e-5


def test_delta_round_trip_keeps_the_document_settings(tmp_path):
    model = tiny_qwen35()
    mq = convert_to_mosaic_qwen(model, MQConfig(n_cluster_heads=2, **DOCS, **SMALL))
    with torch.no_grad():
        for p in mq.memory_parameters():
            p.add_(torch.randn_like(p) * 0.01)
    ids = _doc_ids()
    ref = _logits(model, ids)
    mq.save_delta(tmp_path / "d")
    other = tiny_qwen35()
    meta_cfg = __import__("json").loads((tmp_path / "d" / "mq_config.json").read_text())["mq_config"]
    assert (meta_cfg["attn_bias_type"], meta_cfg["n_cluster_heads"], meta_cfg["eos_token_id"]) == ("doc_causal", 2, EOS)
    mq2 = convert_to_mosaic_qwen(other, MQConfig(**{k: v for k, v in meta_cfg.items() if k in MQConfig.__dataclass_fields__}))
    mq2.load_delta(tmp_path / "d")
    assert (_logits(other, ids) - ref).abs().max().item() < 1e-5


# ── other bases: attention in every layer (Qwen3, Llama type) ──

@pytest.mark.parametrize("kind", ["qwen3", "llama"])
def test_attention_only_models_convert_like_qwen35(kind):
    model = tiny_attention_only(kind)
    ids = _ids()
    ref = _logits(model, ids)
    mq = convert_to_mosaic_qwen(model, MQConfig(**SMALL))
    assert mq.cfg.layers == [0, 1, 2, 3]                             # every layer holds attention: every layer is replaced
    assert mq.attention_parameters() == []
    with mq.teacher():
        assert (_logits(model, ids) - ref).abs().max().item() == 0.0   # the teacher is the original model
    assert (_logits(model, ids) - ref).abs().max().item() > 1e-3
    layer, orig = mq.layers[1], mq.state.orig[1]
    assert layer.block.feed_forward.w1.weight is orig.mlp.gate_proj.weight
    with torch.no_grad():                                            # memories silenced: the layer minus its attention
        for m in layer.memories():
            m.proj_out.weight.zero_()
        x = torch.randn(2, 20, 64)
        want = x + orig.mlp(orig.post_attention_layernorm(x))
        assert (layer(x) - want).abs().max().item() < 1e-5
        assert (layer.block.attention_norm(x) - orig.input_layernorm(x)).abs().max().item() < 1e-5
    # a stage-1 step: the original mixer's output is captured and the memories get a gradient
    seen = {}

    def local(i, xin, mixed, _out):
        with torch.enable_grad():
            loss = (mq.layers[i].memory_residual(xin.detach()) - mixed).pow(2).mean()
            loss.backward()
        seen[i] = (loss.item(), mq.state.capture_ctx is not None)

    mq.set_trainable(memories=True, norms=True, lora=False)
    with torch.no_grad(), mq.teacher(capture=local):
        model(input_ids=ids, use_cache=False)
    assert sorted(seen) == [0, 1, 2, 3] and all(ok for _, ok in seen.values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mq.memory_parameters())
    # cached generation equals the full forward on these models too
    full = _logits(model, ids)
    with torch.no_grad():
        out = model(ids[:, :20], use_cache=True)
        pkv, parts = out.past_key_values, [out.logits]
        for t in range(20, 48):
            out = model(ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values; parts.append(out.logits)
    assert (torch.cat(parts, 1) - full).abs().max().item() < 2e-4


def test_a_delta_with_base_weights_says_so_and_carries_an_untied_head(tmp_path):
    """Stage 2 with every weight trained: the delta holds the base weights too, the output projection
    included when it is a weight of its own; loading it marks the handle (a teacher then has to be a
    separate copy). A delta of memories alone does not."""
    import transformers
    from mmllm.mosaic_qwen import HEAD_KEY

    def build():
        torch.manual_seed(0)
        cfg = transformers.Qwen3Config(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, eos_token_id=7,
                                       pad_token_id=0, tie_word_embeddings=False, max_position_embeddings=512)
        m = transformers.Qwen3ForCausalLM(cfg).float().eval()
        return m, convert_to_mosaic_qwen(m, MQConfig(replace="all", **SMALL))

    m, mq = build()
    assert m.lm_head.weight is not m.model.embed_tokens.weight
    mq.set_trainable(memories=True, norms=True, lora=False)
    meta = mq.save_delta(tmp_path / "mem")
    assert meta["base_tensors"] == 0
    mq.set_trainable(base=True)
    with torch.no_grad():
        m.lm_head.weight.add_(0.5)
        m.model.embed_tokens.weight.mul_(1.5)
    meta = mq.save_delta(tmp_path / "all")
    assert meta["base_tensors"] > 0
    ids = torch.randint(8, 320, (1, 24))
    with torch.no_grad():
        want = m(input_ids=ids, use_cache=False).logits

    m2, mq2 = build()
    mq2.load_delta(tmp_path / "mem")
    assert not mq2.base_changed
    with torch.no_grad():
        assert not torch.allclose(m2(input_ids=ids, use_cache=False).logits, want, atol=1e-3)
    mq2.load_delta(tmp_path / "all")
    assert mq2.base_changed
    assert torch.equal(m2.lm_head.weight, m.lm_head.weight)
    assert HEAD_KEY in mq.delta_state()
    with torch.no_grad():
        assert torch.allclose(m2(input_ids=ids, use_cache=False).logits, want, atol=1e-5)
