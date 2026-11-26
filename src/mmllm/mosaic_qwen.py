"""MosaicQwen: a Qwen3.5 / Qwen3.8 language model whose softmax-attention layers are ALL
replaced by official Memory Mosaics blocks. No attention is left in the converted model.

    original layer (Qwen3_5DecoderLayer, block_type "full_attention")
        x += attention(norm(x))                 softmax attention, RoPE, KV cache
        x += mlp(norm(x))

    converted layer = one official `MosaicBlock` (facebookresearch/MemoryMosaics,
    model/mosaics/block.py, unmodified), in the Memory Mosaics v2 configuration
    (configs/mosaics/memory_mosaics_v2.yaml: layout "flat")
        h  = attention_norm(x)
        x += short(h) + long(h)                 two official `MosaicsMemory` modules
        x += feed_forward(ffn_norm(x))          the persistent memory (SwiGLU)

    short  Mosaics memory over the last `window` tokens
    long   Mosaics memory over everything at least `skip` tokens back
    Both: key = unit-norm leaky average of the past, query = key * bandwidth(n),
    value looks one token ahead, no position encoding (see upstream memory.py).

What is new and what is kept
    new      the short and long memories of every converted layer (trained here)
    kept     the block's persistent memory and norms START as the layer's own Qwen MLP and
             RMSNorm weights (same function: SwiGLU, RMSNorm), so Qwen's knowledge stays;
             the other layers of the model (Gated DeltaNet + MLP) are untouched
    removed  q_proj / k_proj / v_proj / o_proj / q_norm / k_norm of every attention layer

Which layers: `replace="attention"` converts every "full_attention" layer (Qwen3.8-27B: 16
of 64; Qwen3.5-0.8B: 6 of 24). `replace="all"` converts every layer, Gated DeltaNet ones
included (a model made of Mosaics blocks only; far more to train).

Teacher. While training, the original layers are kept beside the converted ones (outside
the module tree: not in parameters(), not saved) and `with mq.teacher():` runs the
original model on the same weights. `mq.drop_teacher()` deletes them; export does it.

Generation uses the upstream caches (keys/values per memory + the leaky-average state),
stored on the Hugging Face cache object of the call, so `model.generate` works (greedy or
sampling; no beam search). Sequences are whole rows: padding inside a batch is not masked
out by the memories, so batch only equal-length prompts or generate one row at a time.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from . import official
from .official import DocumentLayout, LayerCache, MosaicBlock, MosaicsConfig, memory_mask

FORMAT_VERSION = 1
HEAD_KEY = "::lm_head.weight"     # in a delta: the output projection, when it is not tied to the embedding


# ── configuration ──

@dataclass
class MQConfig:
    """How the layers are converted. Defaults = upstream memory_mosaics_v2.yaml."""
    n_short_heads: int = 16
    n_long_heads: int = 16
    head_dim: int = 128
    window: int = 256                 # short memory: the last `window` tokens
    skip: int = 64                    # long memory: at least `skip` tokens back
    random_skip: int = 192            # training: add U{0..random_skip} to the skip each step
    layout: str = "flat"
    memory_gate: bool = True
    short_key_feature: str = "leaky_average"
    long_key_feature: str = "leaky_average"
    recurrence_chunk_size: int = 1024
    recurrence_padding: int = 256
    replace: str = "attention"        # "attention" | "all"
    # What the memories may read in a row of several documents (upstream attn_bias_type):
    #   "causal"      short and long read the whole row
    #   "doc_causal"  short and long read the current document only; documents end at
    #                 `eos_token_id` (or at the end of the row). The cluster memory, if it has
    #                 heads, reads the earlier documents of the row (upstream "two-stage").
    attn_bias_type: str = "causal"
    n_cluster_heads: int = 0
    cluster_key_feature: str = "leaky_average"
    eos_token_id: int | None = None
    init_seed: int = 42
    base_model: str = ""
    upstream_commit: str = ""
    layers: list[int] = field(default_factory=list)
    format_version: int = FORMAT_VERSION

    @classmethod
    def from_upstream_mmv2(cls, **overrides) -> "MQConfig":
        """Read the memory settings from the vendored upstream MMv2 config file."""
        m = official.mmv2_reference_config()
        assert m["short_memory_type"] == "mosaics" and m["long_memory_type"] == "mosaics"
        base = dict(n_short_heads=m["n_short_heads"], n_long_heads=m["n_long_heads"], head_dim=m["head_dim"],
                    window=m["short_mem_window_size"], skip=m["long_mem_skip_size"],
                    random_skip=m["random_long_mem_skip_size"], layout=m["layout"], memory_gate=m["memory_gate"],
                    long_key_feature=m["long_key_feature"])
        base.update(overrides)
        return cls(**base)


def official_config(text_cfg, mq: MQConfig, max_seq_len: int = 4096) -> MosaicsConfig:
    """The upstream MosaicsConfig of one converted layer of a Qwen text model."""
    dim, inter = text_cfg.hidden_size, text_cfg.intermediate_size
    base = int(2 * 4 * dim / 3)                       # upstream FeedForward: int(2 * ffn_exp * dim / 3)
    mult = (inter + 0.5) / base                       # so that int(mult * base) == inter (multiple_of = 1)
    cfg = MosaicsConfig(
        dim=dim, n_layers=text_cfg.num_hidden_layers, head_dim=mq.head_dim, max_seq_len=max_seq_len,
        vocab_size=text_cfg.vocab_size, ffn_dim_multiplier=mult, ffn_exp=4, multiple_of=1,
        norm_eps=text_cfg.rms_norm_eps, attn_bias_type=mq.attn_bias_type, memory_gate=mq.memory_gate, layout=mq.layout,
        short_memory_type="mosaics", long_memory_type="mosaics", cluster_memory_type="mosaics",
        n_short_heads=mq.n_short_heads, n_long_heads=mq.n_long_heads, n_cluster_heads=mq.n_cluster_heads,
        short_key_feature=mq.short_key_feature if mq.n_short_heads else None,
        long_key_feature=mq.long_key_feature if mq.n_long_heads else None,
        cluster_key_feature=mq.cluster_key_feature if mq.n_cluster_heads else None,
        recurrence_chunk_size=mq.recurrence_chunk_size, recurrence_padding=mq.recurrence_padding,
        short_mem_window_size=mq.window, long_mem_skip_size=mq.skip, random_long_mem_skip_size=mq.random_skip,
    )
    assert int(cfg.ffn_dim_multiplier * base) == inter
    return cfg


# ── per-call state shared by the converted layers ──

class MQState:
    def __init__(self, mq: MQConfig) -> None:
        self.mq = mq
        self.teacher = False
        self.capture: Callable | None = None       # teacher mode: f(layer_idx, x_in, mixer_out, x_out)
        self.collect: dict | None = None           # student mode: {layer_idx: block output}
        self.orig: dict[int, nn.Module] = {}       # the original layers (teacher only)
        self.skip_extra = 0                        # random long-memory skip of this training step
        self.capture_ctx: dict | None = None       # the arguments of the original mixer, during a capture call
        self.keep_orig: set[int] = set()           # layers the STUDENT still runs as the original (training schedule only)
        self.tokens: torch.Tensor | None = None    # input ids of the current forward (where documents end)
        self.cluster_off = False                   # skip the cluster memory (what the model does without it)
        self._masks: dict = {}
        self._pos: dict = {}
        self._doc: tuple | None = None             # doc_causal: (key, masks, positions, layout) of the current row layout

    def skip(self) -> int:
        return self.mq.skip + self.skip_extra

    @property
    def documentwise(self) -> bool:
        return self.mq.attn_bias_type == "doc_causal"

    def doc_lens(self, B: int, T: int, device) -> torch.Tensor:
        """Document lengths of the current (B, T) step, row-major, as upstream counts them: a
        document ends at the end-of-document token (which belongs to it) or at the end of its
        row. Without ids of this shape (a call with embeddings): one document per row."""
        t = self.tokens
        if t is None or tuple(t.shape) != (B, T) or self.mq.eos_token_id is None:
            return torch.full((B,), T, dtype=torch.long, device=device)
        return torch.tensor(official.tokens_to_seq_lens(t, self.mq.eos_token_id), dtype=torch.long, device=device)

    def _documents(self, B: int, T: int, device) -> tuple:
        lens = self.doc_lens(B, T, device)
        key = (B, T, str(device), self.mq.window, self.skip(), self.cluster_off, tuple(lens.tolist()))
        if self._doc is None or self._doc[0] != key:
            tokens = torch.empty(B, T, dtype=torch.long, device=device)
            m = {}
            if self.mq.n_short_heads:
                m["short"] = memory_mask("short", False, tokens, lens, "doc_causal", window=self.mq.window)
            if self.mq.n_long_heads:
                m["long"] = memory_mask("long", False, tokens, lens, "doc_causal", skip=self.skip())
            if self.mq.n_cluster_heads and not self.cluster_off:
                m["cluster"] = memory_mask("cluster", False, tokens, lens, "doc_causal", cluster_mem_skip=True)
            lay = DocumentLayout()
            lay.full_sequence("doc_causal", lens, T)
            self._doc = (key, m, lay.positions(device, B, T), lay)
        return self._doc

    def masks(self, B: int, T: int, device) -> dict:
        if self.documentwise:
            return self._documents(B, T, device)[1]
        key = (T, str(device), self.mq.window, self.skip())
        if key not in self._masks:
            if len(self._masks) > 8:
                self._masks.clear()
            tokens = torch.empty(1, T, dtype=torch.long, device=device)
            q_lens = torch.tensor([T], device=device)
            m = {}
            if self.mq.n_short_heads:
                m["short"] = memory_mask("short", False, tokens, q_lens, "causal", window=self.mq.window)
            if self.mq.n_long_heads:
                m["long"] = memory_mask("long", False, tokens, q_lens, "causal", skip=self.skip())
            self._masks[key] = m
        return self._masks[key]

    def positions(self, B: int, T: int, device):
        """Upstream Positions of a full-sequence step."""
        if self.documentwise:
            return self._documents(B, T, device)[2]
        key = (B, T, str(device))
        if key not in self._pos:
            if len(self._pos) > 8:
                self._pos.clear()
            lay = DocumentLayout()
            lay.full_sequence("causal", torch.tensor([T] * B, device=device), T)
            self._pos[key] = lay.positions(device, B, T)
        return self._pos[key]

    # ── generation: the document layout is carried from the prompt through the decoding steps ──

    def _end_documents(self, lay, B: int, col: int) -> None:
        """Rows whose token in column `col` of the current call is the end-of-document token start
        a new document with their next token (upstream LanguageModel.end_documents; the Mosaics
        caches themselves keep everything, only the layout moves)."""
        t = self.tokens
        if (self.documentwise and self.mq.eos_token_id is not None and t is not None and t.dim() == 2
                and t.shape[0] == B and -t.shape[1] <= col < t.shape[1]):
            lay.end_documents(t[:, col] == self.mq.eos_token_id)

    def prefill_positions(self, store: dict, B: int, T: int, device):
        """Positions of the prompt (a full-sequence step) and the layout the decoding steps continue.
        Computed by the first layer of the call and shared by the others."""
        pre = store.get("prefill")
        if pre is None:
            lay = DocumentLayout()
            lens = self.doc_lens(B, T, device) if self.documentwise else torch.tensor([T] * B, device=device)
            lay.full_sequence(self.mq.attn_bias_type, lens, T)
            pre = store["prefill"] = lay.positions(device, B, T)
            store["layout"] = lay
            self._end_documents(lay, B, -1)
        return pre

    def decoding_positions(self, store: dict, B: int, device, call_start: int, start: int):
        """Positions of the one token at `start`, in a call whose first token sits at `call_start`.
        The first layer of the call computes them in order and moves the layout past document
        ends; the other layers read the same objects."""
        cached = store.get("pos")
        if cached is None or cached[0] != call_start:
            cached = store["pos"] = (call_start, {})
        if start not in cached[1]:
            lay = store.get("layout")
            if lay is None:
                lay = store["layout"] = DocumentLayout()
            lay.decoding(self.mq.attn_bias_type, start, B)
            cached[1][start] = lay.positions(device, B, 1)
            self._end_documents(lay, B, start - call_start)
        return cached[1][start]


def _autocast(x: torch.Tensor):
    if x.is_cuda and x.dtype == torch.bfloat16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _grow(cache: LayerCache, need: int) -> LayerCache:
    """Make room for `need` positions in the upstream (keys, values, n) caches."""
    out = {}
    for scope in ("short", "long", "cluster"):
        c = getattr(cache, scope)
        if c is None:
            continue
        k, v, n = c
        if need > k.shape[1]:
            cap = max(need, 2 * k.shape[1])
            k2 = torch.empty((k.shape[0], cap, *k.shape[2:]), dtype=k.dtype, device=k.device)
            v2 = torch.empty_like(k2)
            k2[:, :n], v2[:, :n] = k[:, :n], v[:, :n]
            c = (k2, v2, n)
        out[scope] = c
    return LayerCache(**out)


_DECODE_ATTRS = ("_recurrence_state", "_prev_v")


class MosaicQwenLayer(GradientCheckpointingLayer):
    """One converted layer: an official MosaicBlock behind the Qwen decoder-layer interface."""

    def __init__(self, block: MosaicBlock, state: MQState, layer_idx: int, block_type: str) -> None:
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.block_type = block_type                 # what the layer was: "full_attention" | "linear_attention"
        object.__setattr__(self, "_st", state)       # shared, not a submodule

    # ── the memories alone (stage 1 trains them against the original mixer) ──

    def memories(self):
        return [m for m in (self.block.short_memory, self.block.long_memory, self.block.cluster_memory) if m is not None]

    def memory_residual(self, x: torch.Tensor) -> torch.Tensor:
        """What the block's memories add to the residual stream x (before its FFN), computed
        with the upstream modules exactly as MosaicBlock.forward does."""
        st, blk = self._st, self.block
        B, T, _ = x.shape
        masks, pos = st.masks(B, T, x.device), st.positions(B, T, x.device)
        cluster = None if (blk.disable_cluster or st.cluster_off) else blk.cluster_memory
        with _autocast(x):
            if blk.layout == "flat":
                h = blk.attention_norm(x)
                short, _ = blk._read(blk.short_memory, h, masks.get("short"), pos, None)
                long, _ = blk._read(blk.long_memory, h, masks.get("long"), pos, None)
                clus, _ = blk._read(cluster, h, masks.get("cluster"), pos, None)
                return long + short + clus
            y = x
            if blk.short_memory is not None:
                y = y + blk.short_memory(blk.short_norm(y), masks.get("short"), pos, None)[0]
            if blk.layout == "short_flat":
                h = blk.attention_norm(y)
                long, _ = blk._read(blk.long_memory, h, masks.get("long"), pos, None)
                clus, _ = blk._read(cluster, h, masks.get("cluster"), pos, None)
                return y + long + clus - x
            if blk.long_memory is not None:
                y = y + blk.long_memory(blk.attention_norm(y), masks.get("long"), pos, None)[0]
            if cluster is not None:
                y = y + cluster(blk.cluster_norm(y), masks.get("cluster"), pos, None)[0]
            return y - x

    # ── teacher: the original layer ──

    def _teacher(self, hidden_states, position_embeddings, attention_mask, position_ids, past_key_values, kwargs):
        st = self._st
        orig = st.orig.get(self.layer_idx)
        if orig is None:
            raise RuntimeError("the original layers were dropped: this model has no teacher")
        if st.capture is None:
            return orig(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        h = orig.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            mixed = orig.linear_attn(hidden_states=h, cache_params=past_key_values, attention_mask=attention_mask, **kwargs)
        else:
            mixed, _ = orig.self_attn(hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
                                      past_key_values=past_key_values, position_embeddings=position_embeddings, **kwargs)
        x = hidden_states + mixed
        out = x + orig.mlp(orig.post_attention_layernorm(x))
        # what the original mixer was called with (scripts/mq_stage1_control.py calls a second mixer the same way)
        st.capture_ctx = {"attention_mask": attention_mask, "position_ids": position_ids,
                          "position_embeddings": position_embeddings, "kwargs": kwargs}
        st.capture(self.layer_idx, hidden_states, mixed, out)
        st.capture_ctx = None
        return out

    # ── student: the Mosaics block ──

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None,
                past_key_values=None, **kwargs):
        st = self._st
        if st.teacher:
            return self._teacher(hidden_states, position_embeddings, attention_mask, position_ids, past_key_values, kwargs)
        if self.layer_idx in st.keep_orig:        # training only: this layer is not replaced yet (progressive replacement)
            return st.orig[self.layer_idx](hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                                           position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        x = hidden_states
        B, T, _ = x.shape
        blk = self.block
        with _autocast(x):
            if past_key_values is None:                       # training / evaluation: one full-sequence step
                out, _ = blk(x, st.masks(B, T, x.device), st.positions(B, T, x.device), None)
            else:
                if self.training and torch.is_grad_enabled():
                    raise RuntimeError("training goes through the full-sequence path: call the model with use_cache=False")
                out = self._forward_cached(x, past_key_values)
        out = out.to(x.dtype)
        if st.collect is not None:
            st.collect[self.layer_idx] = out
        return out

    def _forward_cached(self, x, pkv):
        st, blk, i = self._st, self.block, self.layer_idx
        B, T, _ = x.shape
        store = pkv.__dict__.setdefault("_mq", {"layers": {}})
        entry = store["layers"].get(i)
        if entry is None:                                     # prefill: the whole prompt in one step
            cache = blk.init_cache(B, max(2 * T, T + 256), x.dtype, x.device)
            if T == 1:
                out, cache = blk(x, None, st.decoding_positions(store, B, x.device, 0, 0), cache)
            else:
                out, cache = blk(x, st.masks(B, T, x.device), st.prefill_positions(store, B, T, x.device), cache)
            seen = T
        else:                                                 # decoding: one token at a time
            cache, seen = _grow(entry["cache"], entry["seen"] + T), entry["seen"]
            for m, saved in zip(self.memories(), entry["decode"]):
                for a, val in saved.items():
                    setattr(m, a, val)
            outs = []
            for t in range(T):
                o, cache = blk(x[:, t:t + 1], None, st.decoding_positions(store, B, x.device, seen, seen + t), cache)
                outs.append(o)
            out = outs[0] if T == 1 else torch.cat(outs, dim=1)
            seen += T
        store["layers"][i] = {"cache": cache, "seen": seen,
                              "decode": [{a: getattr(m, a) for a in _DECODE_ATTRS if hasattr(m, a)} for m in self.memories()]}
        if self.block_type == "full_attention":
            # Hugging Face reads the number of cached tokens from the attention layers' cache entries.
            dummy = x.new_zeros(B, 1, T, 1)
            pkv.update(dummy, dummy, i)
        return out


# ── LoRA on the layers that are kept ──

class LoRALinear(nn.Module):
    """base(x) + B A x * (alpha / r); the low-rank part is skipped in teacher mode."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, state: MQState) -> None:
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        object.__setattr__(self, "_st", state)

    @property
    def weight(self):
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        if self._st.teacher:
            return y
        with _autocast(x):
            xin = x if (x.is_cuda and x.dtype == torch.bfloat16) else x.to(self.lora_A.dtype)
            d = F.linear(F.linear(xin, self.lora_A), self.lora_B)
        return y + (d * self.scale).to(y.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        w = self.base.weight
        return (w.float() + (self.lora_B.float() @ self.lora_A.float()) * self.scale).to(w.dtype)


def _is_decoder_stack(cand) -> bool:
    """A stack of pre-norm decoder layers the conversion understands: every layer has an input
    norm, a mixer (self_attn or linear_attn), a second norm and a gated MLP (gate / up / down)."""
    layers = getattr(cand, "layers", None)
    if cand is None or layers is None or len(layers) == 0 or not hasattr(getattr(cand, "config", None), "num_hidden_layers"):
        return False
    first = layers[0]
    mlp = getattr(first, "mlp", None)
    return (all(hasattr(first, n) for n in ("input_layernorm", "post_attention_layernorm"))
            and all(hasattr(mlp, n) for n in ("gate_proj", "up_proj", "down_proj"))
            and (hasattr(first, "self_attn") or hasattr(first, "linear_attn") or isinstance(first, MosaicQwenLayer)))


def _text_model(model: nn.Module) -> nn.Module:
    for cand in (getattr(model, "model", None), getattr(getattr(model, "model", None), "language_model", None), model):
        if cand is not None and hasattr(cand, "layers") and hasattr(getattr(cand, "config", None), "layer_types"):
            return cand
    # Qwen3 / Llama-type models: no layer_types in the config, softmax attention in every layer
    for cand in (getattr(model, "model", None), model):
        if _is_decoder_stack(cand) or (cand is not None and hasattr(cand, "layers")
                                       and any(isinstance(l, MosaicQwenLayer) for l in cand.layers)):
            return cand
    raise ValueError("could not find a decoder stack the conversion understands")


def layer_types(text_cfg) -> list[str]:
    """What every layer's mixer is: the config's layer_types (Qwen3.5 / Qwen3.8), else attention everywhere."""
    lt = getattr(text_cfg, "layer_types", None)
    return list(lt) if lt else ["full_attention"] * text_cfg.num_hidden_layers


_ATTENTION_PARAM_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")


class MosaicQwen:
    """Handle of a converted model."""

    def __init__(self, model: nn.Module, layers: dict[int, MosaicQwenLayer], state: MQState, cfg: MQConfig) -> None:
        self.model, self.layers, self.state, self.cfg = model, layers, state, cfg
        self.base_changed = False                    # a loaded delta carried base weights (see load_delta)
        self.lora: dict[str, LoRALinear] = {}

    # ── parameters ──

    def memory_parameters(self) -> list[nn.Parameter]:
        return [p for l in self.layers.values() for m in l.memories() for p in m.parameters()]

    def norm_parameters(self) -> list[nn.Parameter]:
        out = []
        for l in self.layers.values():
            for name in ("attention_norm", "short_norm", "cluster_norm", "ffn_norm"):
                n = getattr(l.block, name, None)
                if n is not None:
                    out += list(n.parameters())
        return out

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.lora.values() for p in (m.lora_A, m.lora_B)]

    def num_memory_parameters(self) -> int:
        return sum(p.numel() for p in self.memory_parameters())

    def attention_parameters(self) -> list[str]:
        """Names of softmax-attention parameters still in the (student) model. Empty when
        every attention layer is converted: the proof that no attention is left."""
        text = _text_model(self.model)
        names = []
        for n, _ in text.named_parameters():
            parts = n.split(".")
            if "self_attn" in parts or any(p in _ATTENTION_PARAM_NAMES for p in parts):
                names.append(n)
        return names

    def freeze_base(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

    def set_trainable(self, memories: bool = True, norms: bool = True, lora: bool = True, base: bool = False) -> list[nn.Parameter]:
        """Freeze everything, then enable the chosen groups. Returns the trainable parameters.
        base=True trains every weight of the student (needs a separate teacher: see has_shared_teacher)."""
        for p in self.model.parameters():
            p.requires_grad_(base)
        for group, flag in ((self.memory_parameters(), memories), (self.norm_parameters(), norms),
                            (self.lora_parameters(), lora)):
            for p in group:
                p.requires_grad_(flag or base)
        seen, out = set(), []
        for p in self.model.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); out.append(p)
        return out

    # ── teacher ──

    @contextmanager
    def teacher(self, capture: Callable | None = None):
        """Run the original model (original attention, no LoRA)."""
        st = self.state
        prev = (st.teacher, st.capture)
        st.teacher, st.capture = True, capture
        try:
            yield
        finally:
            st.teacher, st.capture = prev

    @contextmanager
    def collect(self):
        """Record the output of every converted block of the student forward passes inside."""
        st = self.state
        prev, st.collect = st.collect, {}
        try:
            yield st.collect
        finally:
            st.collect = prev

    def keep_original(self, layers=()) -> None:
        """Progressive replacement (training only): the student runs these layers as the original ones,
        the others as Mosaics blocks. Empty = the converted model. Never part of a saved model."""
        layers = set(int(i) for i in layers)
        if layers - set(self.state.orig):
            raise RuntimeError("the original layers were dropped: nothing to keep")
        self.state.keep_orig = layers

    def has_teacher(self) -> bool:
        return len(self.state.orig) == len(self.layers)

    def detach_teacher_weights(self) -> None:
        """Give the teacher its own copy of the weights the converted blocks share with it
        (their FFN), so that training those weights does not move the teacher."""
        for i, l in self.layers.items():
            orig = self.state.orig[i]
            if orig.mlp.gate_proj.weight is l.block.feed_forward.w1.weight:
                orig.mlp = copy.deepcopy(orig.mlp)
                for p in orig.mlp.parameters():
                    p.requires_grad_(False)

    def drop_teacher(self) -> None:
        """Delete the original layers. After this the model holds no attention weights."""
        self.state.orig.clear()

    def sample_skip(self, step: int, seed: int = 0, rank: int = 0, quantum: int = 64) -> int:
        """Training augmentation of upstream MemoryMosaics.build_masks: a random extra
        long-memory skip, drawn per (seed, step, rank) from U{0..random_skip}.

        quantum > 1 rounds the draw down to a multiple of it (0, 64, 128, 192 by default).
        The compiled FlexAttention kernel is specialised on the mask rule, so every distinct
        skip may cost a compilation: a handful of values keeps short runs from spending
        their time compiling. quantum=1 is exactly upstream's draw."""
        if self.cfg.random_skip > 0:
            rng = random.Random(seed * 1_000_000_000 + step * 1000 + rank)
            extra = rng.randint(0, self.cfg.random_skip)
            self.state.skip_extra = extra - extra % max(1, quantum)
        return self.state.skip_extra

    def fixed_skip(self) -> None:
        self.state.skip_extra = 0

    @contextmanager
    def without_cluster(self):
        """Run the student without its cluster memory (the memory of earlier documents)."""
        st = self.state
        prev, st.cluster_off = st.cluster_off, True
        flags = [(layer.block, layer.block.disable_cluster) for layer in self.layers.values()]
        for blk, _ in flags:
            blk.disable_cluster = True                 # the upstream switch (covers generation too)
        try:
            yield
        finally:
            st.cluster_off = prev
            for blk, was in flags:
                blk.disable_cluster = was

    # ── LoRA ──

    def add_lora(self, r: int = 16, alpha: float | None = None, min_out: int = 64) -> int:
        """Wrap every Linear of the decoder layers (the memories excepted) in a LoRALinear.
        Returns the number of LoRA parameters."""
        alpha = float(alpha if alpha is not None else r)
        text = _text_model(self.model)
        skip = {id(m) for l in self.layers.values() for mem in l.memories() for m in mem.modules()}
        for li, layer in enumerate(text.layers):
            for name, mod in list(layer.named_modules()):
                if not isinstance(mod, nn.Linear) or id(mod) in skip or mod.out_features < min_out:
                    continue
                parent = layer
                *path, leaf = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                wrapped = LoRALinear(mod, r, alpha, self.state)
                setattr(parent, leaf, wrapped)
                self.lora[f"layers.{li}.{name}"] = wrapped
        return sum(p.numel() for p in self.lora_parameters())

    @torch.no_grad()
    def merge_lora(self) -> None:
        """Fold the low-rank parts into the base weights (drop the teacher first: they share weights)."""
        if self.state.orig:
            raise RuntimeError("drop_teacher() before merge_lora(): the teacher shares the base weights")
        text = _text_model(self.model)
        for key, wrapped in self.lora.items():
            _, li, name = key.split(".", 2)
            wrapped.base.weight.copy_(wrapped.merged_weight())
            parent = text.layers[int(li)]
            *path, leaf = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, leaf, wrapped.base)
        self.lora.clear()

    # ── saving: the delta (what conversion and training changed) ──

    def delta_state(self, include_trained_base: bool = True) -> dict[str, torch.Tensor]:
        """Memories and norms of the converted blocks, LoRA factors, and (optionally) every
        other parameter that is currently trainable. Keys are relative to the text model."""
        text = _text_model(self.model)
        keep = {id(p) for p in self.memory_parameters() + self.norm_parameters() + self.lora_parameters()}
        out = {}
        for n, p in text.named_parameters():
            if id(p) in keep or (include_trained_base and p.requires_grad):
                out[n] = p.detach()
        head = self._untied_head()
        if include_trained_base and head is not None and head.requires_grad:
            out[HEAD_KEY] = head.detach()
        return out

    def _untied_head(self) -> nn.Parameter | None:
        """The output projection, when it is a weight of its own (not the input embedding)."""
        head = getattr(self.model, "lm_head", None)
        if head is None:
            return None
        return None if head.weight is _text_model(self.model).get_input_embeddings().weight else head.weight

    def save_delta(self, path, dtype: torch.dtype | None = None, shard_mb: int = 90, extra: dict | None = None) -> dict:
        """Write the delta as safetensors shards of at most `shard_mb` MB (GitHub refuses
        files over 100 MB) plus mq_config.json. A tensor larger than a shard is split by rows."""
        from safetensors.torch import save_file
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        for old in path.glob("delta-*.safetensors"):
            old.unlink()
        limit = shard_mb * 1024 * 1024
        shards, cur, size, split = [], {}, 0, {}

        def flush():
            nonlocal cur, size
            if cur:
                name = f"delta-{len(shards):05d}.safetensors"
                save_file(cur, str(path / name)); shards.append(name)
                cur, size = {}, 0

        for name, t in self.delta_state().items():
            t = t.to("cpu", dtype if (dtype is not None and t.is_floating_point()) else t.dtype).contiguous()
            nbytes = t.numel() * t.element_size()
            pieces = [(name, t)]
            if nbytes > limit and t.dim() >= 1 and t.shape[0] > 1:
                rows = max(1, int(t.shape[0] * limit // nbytes))
                pieces = [(f"{name}::part{j:04d}", t[a:a + rows].contiguous())
                          for j, a in enumerate(range(0, t.shape[0], rows))]
                split[name] = len(pieces)
            for pname, pt in pieces:
                pb = pt.numel() * pt.element_size()
                if cur and size + pb > limit:
                    flush()
                cur[pname] = pt; size += pb
        flush()
        meta = {"mq_config": asdict(self.cfg), "shards": shards, "split": split,
                "base_tensors": len(self._base_keys(self.delta_state())),
                "lora": {k: {"r": m.r, "scale": m.scale} for k, m in self.lora.items()},
                "num_memory_parameters": self.num_memory_parameters(),
                "attention_parameters_left": len(self.attention_parameters())}
        if extra:
            meta["extra"] = extra
        (path / "mq_config.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        return meta

    def _base_keys(self, names) -> list[str]:
        """Of the names of a delta, those of base weights (neither memories, converted-block norms nor LoRA):
        present when the base was trained (stage 2 with --train all)."""
        own_ids = {id(q) for q in self.memory_parameters() + self.norm_parameters() + self.lora_parameters()}
        own = {n for n, q in _text_model(self.model).named_parameters() if id(q) in own_ids}
        return [k for k in names if k not in own]

    def load_delta(self, path, strict: bool = True) -> dict:
        """Load a delta. Sets self.base_changed when it carried base weights: the original layers kept
        by the conversion then no longer run the original model (they share those weights), and a
        teacher has to be a separate copy of the base model."""
        from safetensors.torch import load_file
        path = Path(path)
        meta = json.loads((path / "mq_config.json").read_text())
        if meta.get("lora") and not self.lora:
            any_ = next(iter(meta["lora"].values()))
            self.add_lora(r=any_["r"], alpha=any_["scale"] * any_["r"])
        sd = {}
        for s in meta["shards"]:
            sd.update(load_file(str(path / s)))
        for name, n in meta.get("split", {}).items():
            sd[name] = torch.cat([sd.pop(f"{name}::part{j:04d}") for j in range(n)], dim=0)
        text = _text_model(self.model)
        params = dict(text.named_parameters())
        head = self._untied_head()
        if head is not None:
            params[HEAD_KEY] = head
        missing = [k for k in sd if k not in params]
        if missing and strict:
            raise KeyError(f"delta has parameters the model lacks: {missing[:5]}")
        expected = {n for n, p in params.items() if id(p) in {id(q) for q in self.memory_parameters()}}
        absent = sorted(expected - set(sd))
        if absent and strict:
            raise KeyError(f"delta lacks memory parameters: {absent[:5]}")
        with torch.no_grad():
            for k, v in sd.items():
                if k in params:
                    params[k].copy_(v.to(params[k].dtype))
        self.base_changed = self.base_changed or bool(self._base_keys(sd))
        return meta


    # ── saving: the whole converted model ──

    def save_standalone(self, path, dtype: torch.dtype | None = None, shard_mb: int = 4000) -> dict:
        """Write the converted model as a self-contained checkpoint: every weight of the
        student (LoRA merged), no original attention. Load with load_standalone()."""
        from safetensors.torch import save_file
        if self.state.orig:
            raise RuntimeError("drop_teacher() first: a standalone checkpoint has no teacher")
        if self.lora:
            self.merge_lora()
        left = self.attention_parameters()
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        sd = self.model.state_dict()
        tied = []
        emb = sd.get("model.embed_tokens.weight")
        if emb is not None and "lm_head.weight" in sd and sd["lm_head.weight"].data_ptr() == emb.data_ptr():
            del sd["lm_head.weight"]; tied.append("lm_head.weight")
        limit = shard_mb * 1024 * 1024
        shards, cur, size = [], {}, 0
        for name, t in sd.items():
            t = t.to("cpu", dtype if (dtype is not None and t.is_floating_point()) else t.dtype).contiguous()
            nb = t.numel() * t.element_size()
            if cur and size + nb > limit:
                fn = f"model-{len(shards):05d}.safetensors"; save_file(cur, str(path / fn)); shards.append(fn); cur, size = {}, 0
            cur[name] = t; size += nb
        if cur:
            fn = f"model-{len(shards):05d}.safetensors"; save_file(cur, str(path / fn)); shards.append(fn)
        self.model.config.save_pretrained(str(path))
        meta = {"mq_config": asdict(self.cfg), "shards": shards, "tied": tied, "standalone": True,
                "attention_parameters_left": len(left), "parameters": sum(p.numel() for p in self.model.parameters()),
                "num_memory_parameters": self.num_memory_parameters()}
        (path / "mq_config.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        return meta


def load_standalone(path, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> MosaicQwen:
    """Build the converted architecture from the saved config and load a save_standalone
    checkpoint. The original Qwen checkpoint is not needed (and no attention is created
    for the converted layers beyond the moment of construction)."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    path = Path(path)
    meta = json.loads((path / "mq_config.json").read_text())
    assert meta.get("standalone"), "not a standalone checkpoint (use load_mosaic_qwen for a delta)"
    config = AutoConfig.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_config(config).to(dtype)
    known = set(MQConfig.__dataclass_fields__)
    mq = convert_to_mosaic_qwen(model, MQConfig(**{k: v for k, v in meta["mq_config"].items() if k in known}),
                                keep_teacher=False, memory_dtype=dtype)
    # one shard at a time: holding every shard next to the model doubles the memory
    # (measured: the 0.8B in fp32 did not fit in 6 GB; a 27B would need over 100 GB of RAM)
    expected, seen, unexpected = set(model.state_dict()), set(), []
    for s in meta["shards"]:
        part = load_file(str(path / s))
        unexpected += [k for k in part if k not in expected]
        model.load_state_dict({k: v for k, v in part.items() if k in expected}, strict=False)
        seen |= set(part)
        del part
    missing = sorted(expected - seen - set(meta.get("tied", [])))
    if unexpected or missing:
        raise KeyError(f"checkpoint does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.to(device).eval()
    return mq


def _qwen_norm_weight(norm: nn.Module) -> torch.Tensor:
    """The gain g of an RMSNorm y = g * x / rms(x). Qwen3.5's module multiplies by (1 + weight),
    Qwen3's and Llama's by weight (upstream's RMSNorm: by weight). Which of the two a module
    does is read from what it computes on a probe input; the digits come from its weight."""
    w = norm.weight.detach()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, w.numel(), generator=gen)
    x = (torch.sign(x) * (0.5 + x.abs())).to(device=w.device, dtype=w.dtype)     # no entry near zero
    with torch.no_grad():
        y = norm(x).float()
    xf = x.float()
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
    gain = (y / (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)))[0]
    for cand in (1.0 + w.float(), w.float()):
        if bool(((gain - cand).abs() <= 0.05 + 0.02 * cand.abs()).all()):
            return cand
    raise ValueError(f"{type(norm).__name__} is not an RMSNorm of the form gain * x / rms(x)")


def convert_to_mosaic_qwen(model: nn.Module, mq: MQConfig | None = None, keep_teacher: bool = True,
                           memory_dtype: torch.dtype = torch.float32, max_seq_len: int = 4096) -> MosaicQwen:
    """Replace the attention layers of a Qwen3.5 / Qwen3.8 model by official MosaicBlocks.

    The new memories are initialised as upstream does (LanguageModel.init_weights: truncated
    normal, std = dim^-0.5 / sqrt(2 * (depth + 1))), from `mq.init_seed`. The block's FFN
    uses the layer's own MLP weights (the very same Parameters: no copy) and its norms start
    as the layer's RMSNorms.
    """
    official.verify()
    mq = copy.deepcopy(mq) if mq is not None else MQConfig.from_upstream_mmv2()
    text = _text_model(model)
    tcfg = text.config
    kinds = layer_types(tcfg)
    assert mq.replace in ("attention", "all"), mq.replace
    ocfg = official_config(tcfg, mq, max_seq_len=max_seq_len)
    state = MQState(mq)
    device = next(model.parameters()).device
    official.prepare_device(device)
    layers: dict[int, MosaicQwenLayer] = {}
    gen_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=gen_devices):
        torch.manual_seed(mq.init_seed)
        for i, layer in enumerate(text.layers):
            kind = kinds[i]
            if not (kind == "full_attention" or mq.replace == "all"):
                continue
            block = MosaicBlock(ocfg, layer_id=i)
            block.init_weights(None, factor=official.layer_init_factor(ocfg.init_method, i, ocfg.n_layers, ocfg.dim))
            # persistent memory = the layer's own MLP (the same Parameters, no copy)
            ff, mlp = block.feed_forward, layer.mlp
            # upstream FeedForward is w2(silu(w1 x) * w3 x) without biases: the layer's MLP must be that
            act = getattr(mlp, "act_fn", None)
            probe = torch.linspace(-4.0, 4.0, 33)
            if act is None or not torch.allclose(act(probe), F.silu(probe), atol=1e-6):
                raise ValueError(f"layer {i}: the MLP's activation is {act!r}, not SiLU; the Mosaics block's FFN cannot hold it")
            if any(getattr(mlp, n).bias is not None for n in ("gate_proj", "up_proj", "down_proj")):
                raise ValueError(f"layer {i}: the MLP has biases; the Mosaics block's FFN has none")
            assert ff.w1.weight.shape == mlp.gate_proj.weight.shape, (ff.w1.weight.shape, mlp.gate_proj.weight.shape)
            ff.w1.weight, ff.w3.weight, ff.w2.weight = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
            # the new parts: memories and norms (never .to() the whole block: its FFN is shared)
            for name in ("short_memory", "long_memory", "cluster_memory", "attention_norm", "short_norm", "cluster_norm",
                         "ffn_norm"):
                part = getattr(block, name, None)
                if part is not None:
                    part.to(device=device, dtype=memory_dtype)
            with torch.no_grad():
                for name in ("attention_norm", "short_norm", "cluster_norm"):
                    n = getattr(block, name, None)
                    if n is not None:
                        n.weight.copy_(_qwen_norm_weight(layer.input_layernorm))
                block.ffn_norm.weight.copy_(_qwen_norm_weight(layer.post_attention_layernorm))
            wrapper = MosaicQwenLayer(block, state, i, kind)
            if keep_teacher:
                state.orig[i] = layer
            text.layers[i] = wrapper
            layers[i] = wrapper
    if not layers:
        raise ValueError("no layer to convert")
    mq.layers = sorted(layers)
    mq.upstream_commit = official.upstream_info()["commit"]
    if not mq.base_model:
        mq.base_model = str(getattr(getattr(model, "config", None), "_name_or_path", "") or "")
    handle = MosaicQwen(model, layers, state, mq)
    object.__setattr__(model, "_mq_handle", handle)     # reachable from the model, outside the module tree

    # The ids of every forward: under "doc_causal" the memories need to know where documents end.
    def _ids(args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        return ids if (torch.is_tensor(ids) and ids.dtype == torch.long and ids.dim() == 2) else None

    def _outer(_m, args, kwargs):                       # a call of the whole model: ids or none
        state.tokens = _ids(args, kwargs)

    def _inner(_m, args, kwargs):                       # the decoder stack called directly with ids
        ids = _ids(args, kwargs)
        if ids is not None:
            state.tokens = ids

    model.register_forward_pre_hook(_outer, with_kwargs=True)
    if text is not model:
        text.register_forward_pre_hook(_inner, with_kwargs=True)
    return handle


def load_mosaic_qwen(base_model: str, delta: str | None = None, dtype: torch.dtype = torch.float32,
                     device: str | torch.device = "cpu", keep_teacher: bool = False,
                     memory_dtype: torch.dtype | None = None, **from_pretrained_kwargs) -> tuple[MosaicQwen, object]:
    """Load a base Qwen checkpoint, convert it, and load a trained delta (save_delta).

    The conversion settings come from the delta's mq_config.json. With keep_teacher=False
    the original attention weights are deleted right after conversion.
    Returns (handle, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, **from_pretrained_kwargs)
    model.to(device).eval()
    mq = None
    if delta is not None:
        meta = json.loads((Path(delta) / "mq_config.json").read_text())
        known = set(MQConfig.__dataclass_fields__)
        mq = MQConfig(**{k: v for k, v in meta["mq_config"].items() if k in known})
    handle = convert_to_mosaic_qwen(model, mq, keep_teacher=keep_teacher, memory_dtype=memory_dtype or dtype)
    if delta is not None:
        handle.load_delta(delta)
    tok = AutoTokenizer.from_pretrained(base_model)
    return handle, tok
