"""Start the Mosaics memories from the original mixer's weights instead of random values.

The official blocks and their code are untouched: this only chooses the values the memories'
own parameters start from (they are trained afterwards as before). Work on converting pretrained
transformers to other sequence mixers reports that the starting point decides how far a short
distillation gets (weight transfer in RADLADS; Taylor-Calibrate; the kernel-trick stage of
"Attention to Mamba"). What can be carried over to a Mosaics memory:

  values / output  ("ov")   A head of the original mixer writes  O_h V_h x  into the stream for the
        positions it reads. A memory head writes  proj_out_h * normalize(proj_v_h x) * exp(value_scale_h).
        proj_v and proj_out are set to the best low-rank factors of O_h V_h (SVD), value_interp to 0
        (the value is the token itself, as in attention), and value_scale to the log of the typical
        norm of proj_v_h x on real inputs, shared between the memories of the layer.
  keys  ("ovk": the above, plus)   A memory compares keys made by ONE projection (the query is the
        key times a bandwidth); attention compares q = Q x with k = K x. The symmetric part of
        Q_h^T K_h is the closest such kernel: its leading non-negative eigen-directions become the
        key projection. Position information (RoPE) has no counterpart and is dropped.

Decay and gate rows, bandwidths and the memory gate keep upstream's initial values.
"""
from __future__ import annotations

import math

import torch

from .mosaic_qwen import MosaicQwen, _text_model


def _teacher_heads(orig, kind: str, cfg):
    """Per head of the original mixer: (O_h (dim, r), V_h (r, dim), Q_h or None, K_h or None)."""
    out = []
    if kind == "linear_attention":
        la = orig.linear_attn
        vdim = la.out_proj.in_features
        w = la.in_proj_qkv.weight.detach().float()
        kdim = (w.shape[0] - vdim) // 2
        nv = int(getattr(cfg, "linear_num_value_heads", 1)); nk = int(getattr(cfg, "linear_num_key_heads", nv))
        q, k, v = w[:kdim], w[kdim:2 * kdim], w[2 * kdim:]
        o = la.out_proj.weight.detach().float()
        vh, kh = vdim // nv, kdim // nk
        for h in range(nv):
            g = h * nk // nv
            out.append((o[:, h * vh:(h + 1) * vh], v[h * vh:(h + 1) * vh], q[g * kh:(g + 1) * kh], k[g * kh:(g + 1) * kh]))
        return out
    at = orig.self_attn
    nh, nkv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    o, v = at.o_proj.weight.detach().float(), at.v_proj.weight.detach().float()
    q, k = at.q_proj.weight.detach().float(), at.k_proj.weight.detach().float()
    hd = o.shape[1] // nh
    for h in range(nh):
        g = h * nkv // nh
        out.append((o[:, h * hd:(h + 1) * hd], v[g * hd:(g + 1) * hd], q[h * hd:(h + 1) * hd], k[g * hd:(g + 1) * hd]))
    return out


def _low_rank(o: torch.Tensor, v: torch.Tensor, r: int):
    """Best rank-r factors (A (dim, r), B (r, dim)) of o @ v, the singular values split evenly."""
    qo, ro = torch.linalg.qr(o)                 # o = qo ro
    qv, rv = torch.linalg.qr(v.T)               # v = rv^T qv^T
    u, s, wh = torch.linalg.svd(ro @ rv.T)
    r = min(r, s.numel())
    root = s[:r].sqrt()
    return (qo @ u[:, :r]) * root, (root[:, None] * wh[:r]) @ qv.T


def _key_dirs(q: torch.Tensor, k: torch.Tensor, r: int) -> torch.Tensor:
    """(r, dim): leading non-negative eigen-directions of the symmetric part of q^T k, scaled by sqrt(eigenvalue)."""
    basis, _ = torch.linalg.qr(torch.cat([q, k]).T)            # (dim, rank): everything lives in this span
    a, b = q @ basis, k @ basis
    s = (a.T @ b + b.T @ a) / 2
    lam, vec = torch.linalg.eigh(s)
    order = torch.argsort(lam, descending=True)[:r]
    lam, vec = lam[order].clamp_min(0), vec[:, order]
    return (vec * lam.sqrt()).T @ basis.T


@torch.no_grad()
def teacher_init(mq: MosaicQwen, mode: str = "ov") -> dict:
    """Set the memories' projections from the original mixers (see the module docstring).
    Needs the original layers (keep_teacher=True). Returns what was set, per layer."""
    if mode not in ("ov", "ovk"):
        raise ValueError(mode)
    cfg = _text_model(mq.model).config
    report = {}
    for i, layer in mq.layers.items():
        orig = mq.state.orig[i]
        heads = _teacher_heads(orig, layer.block_type, cfg)
        info = {"teacher_heads": len(heads), "memories": {}}
        for mem in layer.memories():
            H, d = mem.n_heads, mem.head_dim
            slots = H * d
            r = slots // len(heads)
            if r == 0:
                info["memories"][mem.scope] = "skipped: more teacher heads than value dimensions"
                continue
            pv, po = mem.proj_v.weight, mem.proj_out.weight
            at = 0
            for o, v, q, k in heads:
                a, b = _low_rank(o.to(pv.device), v.to(pv.device), r)
                n = a.shape[1]
                po[:, at:at + n] = a.to(po.dtype)
                pv[at:at + n] = b.to(pv.dtype)
                if n < r:                       # the head has fewer directions than its share: the rest stays silent
                    po[:, at + n:at + r] = 0
                    pv[at + n:at + r] = 0
                if mode == "ovk":
                    kd = _key_dirs(q.to(pv.device), k.to(pv.device), r)
                    mem.proj_kf.weight[at:at + kd.shape[0]] = kd.to(mem.proj_kf.weight.dtype)
                at += r
            mem.value_interp.zero_()
            info["memories"][mem.scope] = {"heads": H, "head_dim": d, "rank_per_teacher_head": r, "filled": at, "of": slots}
        report[str(i)] = info
    return {"mode": mode, "layers": report}


@torch.no_grad()
def calibrate_value_scale(mq: MosaicQwen, model, rows: torch.Tensor, device) -> dict:
    """value_scale <- log(mean |proj_v_h x| / number of memories in the layer), on the original model's
    hidden states for `rows`: a memory normalises its values, so the size the original values had is
    put back here; the layer's memories share it."""
    sums = {i: {m.scope: torch.zeros(m.n_heads, device=device) for m in l.memories()} for i, l in mq.layers.items()}
    count = {i: 0 for i in mq.layers}

    def cap(i, x, _mixed, _out):
        layer = mq.layers[i]
        if layer.block.layout != "flat":
            raise NotImplementedError("value-scale calibration: only the flat layout (all memories read attention_norm(x))")
        h = layer.block.attention_norm(x.float())
        for m in layer.memories():
            xv = m.proj_v(h.to(m.proj_v.weight.dtype))[..., : m.n_heads * m.head_dim]
            sums[i][m.scope] += xv.view(-1, m.n_heads, m.head_dim).float().norm(dim=-1).mean(0)
        count[i] += 1

    for r in rows:
        with mq.teacher(capture=cap):
            model(input_ids=r[None].to(device), use_cache=False)
    out = {}
    for i, layer in mq.layers.items():
        mems = layer.memories()
        for m in mems:
            mean = (sums[i][m.scope] / max(count[i], 1)).clamp_min(1e-6)
            m.value_scale.copy_(torch.log(mean / len(mems)).to(m.value_scale.dtype))
        out[str(i)] = {m.scope: round(float(m.value_scale.float().mean()), 3) for m in mems}
    return out


@torch.no_grad()
def fit_output_gain(mq: MosaicQwen, model, rows: torch.Tensor, device) -> dict:
    """After teacher_init + calibrate_value_scale: one gain per layer, the least-squares factor a that makes
    a * (what the layer's memories add) closest to the original mixer's output on `rows`; log(a) is added
    to the value_scale of the layer's memories. A value's typical length overstates what a weighted average
    of many values has (diagnostic 6 started 6-7 times too large); this puts the size right, so the error
    before training is at most 1. A layer whose best factor is not positive is left silent (factor 1e-3)."""
    num = {i: 0.0 for i in mq.layers}
    den = {i: 0.0 for i in mq.layers}
    mq.fixed_skip()

    def cap(i, x, mixed, _out):
        y = mq.layers[i].memory_residual(x).float()
        num[i] += (y * mixed.float()).sum().item()
        den[i] += y.pow(2).sum().item()

    for r in rows:
        with mq.teacher(capture=cap):
            model(input_ids=r[None].to(device), use_cache=False)
    out = {}
    for i, layer in mq.layers.items():
        a = max(num[i] / max(den[i], 1e-12), 1e-3)
        for m in layer.memories():
            m.value_scale.add_(math.log(a))
        out[str(i)] = round(a, 4)
    return out
