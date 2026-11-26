"""The official Memory Mosaics code (facebookresearch/MemoryMosaics), loaded as it is.

third_party/MemoryMosaics/ holds an UNMODIFIED copy of the upstream `model/` package
(commit and per-file SHA-256 in third_party/MemoryMosaics/UPSTREAM.json; `verify()` checks
them). Everything in this repository that is called a Memory Mosaics memory, block or mask
is the upstream class, imported from there:

    from mmllm.official import MosaicBlock, MosaicsConfig, MosaicsMemory, memory_mask, ...

Two device-specific primitives of the upstream code cannot run everywhere, so this module
installs a dispatcher in front of each. The upstream files stay untouched; on a CUDA
machine with the CUDA toolkit both dispatchers call the upstream primitive.

1. FlexAttention (model.mosaics.memory.flex_attention, a torch.compile'd
   torch.nn.attention.flex_attention). PyTorch has no CPU backward for it.
     backend "official"  the upstream call (CUDA)
     backend "dense"     the same mathematics on a dense score matrix: softmax over the
                         positions the block mask's mask_mod allows, rows that may read
                         nothing give 0 (what FlexAttention does).
     backend "auto"      (default) CUDA with gradients, i.e. training: the upstream call.
                         CPU: dense. CUDA without gradients (evaluation, generation), while
                         the score matrix stays under MQ_DENSE_NOGRAD_GB (1.5): dense, because
                         prompts of many lengths would make the compiled kernel rebuild.
                         If the upstream call fails: dense from then on (logged once).
2. The leaky-average scan y_t = omega_t * y_{t-1} + u_t (model.mosaics.leaky_average).
   Upstream runs a JIT-built CUDA kernel (needs nvcc + ninja) and otherwise a Python loop
   over time, `leaky_avg_reference`. Here the non-kernel path is `leaky_avg_scan`, a
   parallel prefix scan that computes the same recurrence in log2(T) tensor steps; if the
   kernel cannot be built on CUDA, upstream's FORCE_REFERENCE switch is turned on so that
   this path is used there too (logged once).

Environment:
    MQ_ATTN_BACKEND   auto (default) | official | dense
    MQ_LEAKY_BACKEND  auto (default) | kernel | scan | loop
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "MemoryMosaics"


def upstream_info() -> dict:
    return json.loads((VENDOR / "UPSTREAM.json").read_text())


def verify() -> dict:
    """Check every vendored file against UPSTREAM.json. Returns {"commit", "files"}; raises
    RuntimeError naming the files that differ (a modified copy is not the official code)."""
    info = upstream_info()
    bad = []
    for rel, sha in info["files"].items():
        p = VENDOR / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"third_party/MemoryMosaics differs from upstream {info['commit'][:7]}: {bad}")
    return {"commit": info["commit"], "files": len(info["files"])}


def _import_upstream():
    existing = sys.modules.get("model")
    if existing is not None and not str(getattr(existing, "__file__", "")).startswith(str(VENDOR)):
        raise ImportError("a different top-level package named 'model' is already imported; "
                          "the official Memory Mosaics code needs that name")
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import model.common.mask as mask
    import model.common.positions as positions
    import model.mosaics.block as block
    import model.mosaics.config as config
    import model.mosaics.leaky_average as leaky
    import model.mosaics.memory as memory
    from model.common.init import layer_init_factor
    return mask, positions, block, config, leaky, memory, layer_init_factor


_mask, _positions, _block, _config, _leaky, _memory, layer_init_factor = _import_upstream()

MosaicBlock = _block.MosaicBlock
LayerCache = _block.LayerCache
MosaicsConfig = _config.MosaicsConfig
MosaicsMemory = _memory.MosaicsMemory
LeakyAverage = _leaky.LeakyAverage
memory_mask = _mask.memory_mask
tokens_to_seq_lens = _mask.tokens_to_seq_lens
set_block_mask_compile = _mask.set_block_mask_compile
FlexAttentionBias = _mask.FlexAttentionBias
DocumentLayout = _positions.DocumentLayout
Positions = _positions.Positions


# ── 1. attention read ──

_upstream_flex = _memory.flex_attention
_state = {"attn": os.environ.get("MQ_ATTN_BACKEND", "auto"),
          "leaky": os.environ.get("MQ_LEAKY_BACKEND", "auto"),
          "attn_fallback": None, "leaky_fallback": None, "kernel_checked": False}


def dense_attention(query, key, value, block_mask=None, scale=None, kernel_options=None):
    """What flex_attention computes, on a dense (L, S) score matrix.

    query (B, H, L, E), key/value (B or 1, H, S, E). The mask is block_mask.mask_mod
    evaluated on every (q, kv) pair; upstream mask rules do not depend on batch or head.
    """
    L, S = query.shape[-2], key.shape[-2]
    if scale is None:
        scale = query.shape[-1] ** -0.5
    dtype = torch.float32 if query.dtype in (torch.bfloat16, torch.float16) else query.dtype
    # Always in fp32, outside whatever autocast context the caller is in. Under CUDA bf16
    # autocast the matmul makes bf16 scores; torch._safe_softmax records that dtype for its
    # backward while autocast runs the softmax inside it in fp32, and the backward then fails
    # ("expected input and grad types to match ..."). CPU autocast does not do this.
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.matmul(query.to(dtype), key.to(dtype).transpose(-2, -1)) * scale
        if block_mask is not None:
            q_idx = torch.arange(L, device=query.device).view(L, 1)
            kv_idx = torch.arange(S, device=query.device).view(1, S)
            allowed = block_mask.mask_mod(0, 0, q_idx, kv_idx)
            if scores.requires_grad:
                scores = scores.masked_fill(~allowed, float("-inf"))
            else:
                scores.masked_fill_(~allowed, float("-inf"))
        probs = torch._safe_softmax(scores, dim=-1)
        out = torch.matmul(probs, value.to(dtype))
    return out.to(query.dtype)


_DENSE_NOGRAD_BYTES = int(float(os.environ.get("MQ_DENSE_NOGRAD_GB", "1.5")) * 2**30)


def _flex_dispatch(query, key, value, block_mask=None, scale=None, kernel_options=None):
    backend = _state["attn"]
    if backend == "dense" or (backend == "auto" and (not query.is_cuda or _state["attn_fallback"])):
        return dense_attention(query, key, value, block_mask=block_mask, scale=scale)
    if backend == "auto" and not (torch.is_grad_enabled() and (query.requires_grad or key.requires_grad
                                                               or value.requires_grad)):
        # Evaluation and generation: prompts of many lengths, and decoding steps whose memory
        # grows by one entry per token. The compiled kernel would be rebuilt for new shapes;
        # the dense path computes the same thing and is cheap without a graph to keep.
        if query.shape[0] * query.shape[1] * query.shape[-2] * key.shape[-2] * 4 <= _DENSE_NOGRAD_BYTES:
            return dense_attention(query, key, value, block_mask=block_mask, scale=scale)
    try:
        return _upstream_flex(query, key, value, block_mask=block_mask, scale=scale,
                              kernel_options=kernel_options)
    except Exception as e:                                   # noqa: BLE001
        if backend == "official":
            raise
        _state["attn_fallback"] = f"{type(e).__name__}: {str(e)[:300]}"
        warnings.warn("official FlexAttention call failed; using the dense attention path from now on: "
                      + _state["attn_fallback"])
        return dense_attention(query, key, value, block_mask=block_mask, scale=scale)


_memory.flex_attention = _flex_dispatch


# ── the whole language models (for training from scratch) ──
# The upstream families as they are: MemoryMosaics (model.mosaics.model) and Transformer
# (model.transformer.model), on the shared LanguageModel. The upstream attention layer of the
# Transformer calls the same FlexAttention; it goes through the same dispatcher, so it runs
# on CPU (tests) and falls back the same way. Nothing else is changed.

def _import_families():
    import model.common.attention as attention
    import model.mosaics.model as mosaics_model
    import model.transformer.config as transformer_config
    import model.transformer.model as transformer_model
    return attention, mosaics_model, transformer_model, transformer_config


_attention, _mm_model, _tf_model, _tf_config = _import_families()
_attention.flex_attention = _flex_dispatch
MemoryMosaics = _mm_model.MemoryMosaics
Transformer = _tf_model.Transformer
TransformerConfig = _tf_config.TransformerConfig

FAMILY_CONFIGS = {"mosaics_v2": ("mosaics", "memory_mosaics_v2.yaml"),
                  "transformer": ("transformer", "transformer.yaml")}


def official_model_section(name: str) -> dict:
    """The `model:` section of an upstream config, by name (FAMILY_CONFIGS)."""
    import yaml
    folder, file = FAMILY_CONFIGS[name]
    return dict(yaml.safe_load((VENDOR / "configs" / folder / file).read_text())["model"])


def build_official_lm(name: str, vocab_size: int, max_seq_len: int, **overrides):
    """An upstream language model from its upstream config, with `overrides` (e.g. a smaller
    dim and head count) applied on top. Returns (model, config, the settings used)."""
    section = official_model_section(name)
    section.update(overrides)
    section.update(vocab_size=vocab_size, max_seq_len=max_seq_len)
    if name == "transformer":
        cfg = TransformerConfig(**section)
        model = Transformer(cfg)
    else:
        cfg = MosaicsConfig(**section)
        model = MemoryMosaics(cfg)
    return model, cfg, section


# ── 2. leaky-average scan ──

_upstream_reference = _leaky.leaky_avg_reference


def leaky_avg_scan(omega: torch.Tensor, u: torch.Tensor, c_wk: int,
                   y_prev: torch.Tensor | None = None) -> torch.Tensor:
    """y_t = omega_t * y_{t-1} + u_t (y_{-1} = y_prev or 0): same contract as upstream's
    leaky_avg_reference, computed as a parallel prefix scan over the affine maps
    y -> a*y + b, whose composition is (a2, b2) o (a1, b1) = (a1*a2, a2*b1 + b2).

    omega (B, T, c_wk): one decay per group of C // c_wk consecutive channels. u (B, T, C).
    """
    B, T, C = u.shape
    G = C // c_wk
    a = omega.to(u.dtype)
    b = u.reshape(B, T, c_wk, G)
    if y_prev is not None:
        first = b[:, :1] + a[:, :1, :, None] * y_prev.to(u.dtype).reshape(B, 1, c_wk, G)
        b = torch.cat([first, b[:, 1:]], dim=1)
    shift = 1
    while shift < T:
        a_prev = F.pad(a[:, :-shift], (0, 0, shift, 0), value=1.0)
        b_prev = F.pad(b[:, :-shift], (0, 0, 0, 0, shift, 0), value=0.0)
        b = a[..., None] * b_prev + b
        a = a * a_prev
        shift *= 2
    return b.reshape(B, T, C)


def _reference_dispatch(omega, u, c_wk, y_prev=None):
    if _state["leaky"] == "loop" or u.shape[1] < 4:
        return _upstream_reference(omega, u, c_wk, y_prev)
    return leaky_avg_scan(omega, u, c_wk, y_prev)


_leaky.leaky_avg_reference = _reference_dispatch
if _state["leaky"] in ("scan", "loop"):
    _leaky.FORCE_REFERENCE = True


def prepare_device(device) -> dict:
    """Decide the backends for `device` once, before the first real forward: on CUDA, try
    to build and run the upstream leaky-average kernel; if that fails (no nvcc, no ninja,
    unsupported GPU) switch to the scan. Returns what will be used, for the run's log."""
    device = torch.device(device)
    set_block_mask_compile(False)       # sequence lengths vary; upstream does the same for generation
    if device.type == "cuda" and not _state["kernel_checked"] and _state["leaky"] in ("auto", "kernel"):
        _state["kernel_checked"] = True
        try:
            la = LeakyAverage(n_embd=64, n_head=2)
            x = torch.randn(1, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
            dl = torch.zeros(1, 8, 2, device=device, dtype=torch.bfloat16)
            y = la(x, dl, gate_logits=dl)
            y.float().sum().backward()
            ref = la._post_normalize(_upstream_reference(*la._prepare_inputs(x.detach().float(), dl.float(), dl.float()), 2))
            err = (y.detach().float() - ref).abs().max().item()
            if not err < 0.05:
                raise RuntimeError(f"kernel output differs from the reference (max abs diff {err})")
        except Exception as e:                               # noqa: BLE001
            if _state["leaky"] == "kernel":
                raise
            _state["leaky_fallback"] = f"{type(e).__name__}: {str(e)[:300]}"
            _leaky.FORCE_REFERENCE = True
            warnings.warn("official leaky-average CUDA kernel unavailable; using the scan: " + _state["leaky_fallback"])
    return backends(device)


def backends(device=None) -> dict:
    cuda = device is not None and torch.device(device).type == "cuda"
    attn = _state["attn"]
    if attn == "auto":
        attn = "official (training) / dense (no-grad)" if (cuda and not _state["attn_fallback"]) else "dense"
    leaky = _state["leaky"]
    if leaky == "auto":
        leaky = "kernel" if (cuda and not _leaky.FORCE_REFERENCE) else "scan"
    return {"attention": attn, "leaky_average": leaky, "attention_fallback": _state["attn_fallback"],
            "leaky_fallback": _state["leaky_fallback"], "upstream_commit": upstream_info()["commit"]}


def set_backends(attention: str | None = None, leaky: str | None = None) -> None:
    if attention is not None:
        assert attention in ("auto", "official", "dense"), attention
        _state["attn"] = attention
    if leaky is not None:
        assert leaky in ("auto", "kernel", "scan", "loop"), leaky
        _state["leaky"] = leaky
        _leaky.FORCE_REFERENCE = leaky in ("scan", "loop")


def mmv2_reference_config() -> dict:
    """The `model:` section of upstream configs/mosaics/memory_mosaics_v2.yaml."""
    import yaml
    return yaml.safe_load((VENDOR / "configs" / "mosaics" / "memory_mosaics_v2.yaml").read_text())["model"]
