"""Training the upstream language models from scratch, side by side.

Memory Mosaics v2 (upstream `memory_mosaics_v2.yaml`) and the Transformer baseline (upstream
`transformer.yaml`) are built from their upstream configs with the same overrides (width, head
count) and trained on the same token stream, in the same order, with the same optimizer and
schedule: what differs is only what the upstream configs make differ (attention with RoPE versus
the short and long Mosaics memories). Both read within the current document (doc_causal), as
the v2 paper does. The model code is the upstream code (mmllm.official); this module only feeds
it and measures it.

    data      TokenFile: a uint16 stream of documents (BOS ... EOS) cut into rows
    training  chunked_ce: next-token loss without materializing every logit
    measures  heldout_loss, position_loss (how the loss falls along a document: the
              in-context property the papers are about), recall (copying a fact seen
              `distance` tokens earlier)
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import official

BOS, EOS = 1, 2          # the upstream tokenizer (Mistral-7B-v0.1): <s> = 1, </s> = 2


# ── data ──

class TokenFile:
    """A uint16 stream of documents in a file, cut into rows of seq_len + 1 tokens (inputs and
    the targets one ahead). Rows are visited in a permutation fixed by `seed`, the same for
    every model trained on the file; a new permutation begins after the last row."""

    def __init__(self, path, seq_len: int) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_rows = (len(self.data) - 1) // seq_len
        if self.n_rows < 1:
            raise ValueError(f"{path}: {len(self.data)} tokens, fewer than one row of {seq_len + 1}")

    def row(self, i: int) -> torch.Tensor:
        a = i * self.seq_len
        return torch.from_numpy(self.data[a:a + self.seq_len + 1].astype(np.int64))

    def batch(self, step: int, bsz: int, seed: int) -> torch.Tensor:
        """Rows step*bsz .. step*bsz + bsz - 1 of the visiting order: (bsz, seq_len + 1)."""
        out = []
        for k in range(step * bsz, step * bsz + bsz):
            epoch, j = divmod(k, self.n_rows)
            perm = _perm(self.n_rows, seed * 1000 + epoch)
            out.append(self.row(int(perm[j])))
        return torch.stack(out)


_PERMS: dict = {}


def _perm(n: int, seed: int) -> np.ndarray:
    key = (n, seed)
    if key not in _PERMS:
        _PERMS.clear()
        _PERMS[key] = np.random.default_rng(seed).permutation(n)
    return _PERMS[key]


def doc_lens(inputs: torch.Tensor) -> list[int]:
    """Document lengths of a batch of rows, row-major (upstream rule: a document ends at EOS)."""
    return official.tokens_to_seq_lens(inputs, EOS)


# ── model ──

class Checkpointed(torch.nn.Module):
    """A block that recomputes its activations in the backward pass instead of keeping them.
    The block is kept as `_orig_mod`, the name upstream looks for (LanguageModel.blocks)."""

    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = block

    def forward(self, h, mask, pos, cache):
        if cache is None and self.training and torch.is_grad_enabled():
            return checkpoint(self._orig_mod, h, mask, pos, None, use_reentrant=False)
        return self._orig_mod(h, mask, pos, cache)


def build(family: str, vocab_size: int, seq_len: int, dim: int, heads: int, n_layers: int | None = None,
          grad_ckpt: bool = True, **extra):
    """The upstream model `family` ("mosaics_v2" | "transformer") from its upstream config, with
    the width and the head count changed (heads x head_dim = dim is kept by choosing heads) and
    documents kept apart. Returns (model, the settings used)."""
    over = {"dim": dim, "attn_bias_type": "doc_causal", "fused_cross_entropy": False, **extra}
    if n_layers is not None:
        over["n_layers"] = n_layers
    if family == "mosaics_v2":
        over.update(n_short_heads=heads, n_long_heads=heads)
    elif family == "transformer":
        over.update(n_heads=heads)
    else:
        raise ValueError(family)
    model, _cfg, section = official.build_official_lm(family, vocab_size=vocab_size, max_seq_len=seq_len, **over)
    model.init_weights()
    if grad_ckpt:
        for i, layer in enumerate(model.layers):
            model.layers[i] = Checkpointed(layer)
    return model, section


def mask_for(model, inputs: torch.Tensor, step: int = 0, seed: int = 0):
    return model.create_attention_mask(inputs, doc_lens(inputs), step=step, seed=seed, compile_mask=False)


def chunked_ce(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 4096,
               reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy of the output projection of h (bsz, T, dim) against targets (bsz, T), a
    chunk of positions at a time; in training each chunk's logits are recomputed in the backward
    pass, so at most one chunk of (chunk x vocab) logits exists at once."""
    hf, tf = h.reshape(-1, h.shape[-1]), targets.reshape(-1)

    def piece(hc, tc):
        logits = (hc @ weight.t().to(hc.dtype)).float()
        return F.cross_entropy(logits, tc, reduction="none")

    parts = []
    for a in range(0, hf.shape[0], chunk):
        hc, tc = hf[a:a + chunk], tf[a:a + chunk]
        if torch.is_grad_enabled() and hc.requires_grad:
            parts.append(checkpoint(piece, hc, tc, use_reentrant=False))
        else:
            parts.append(piece(hc, tc))
    loss = torch.cat(parts)
    if reduction == "none":
        return loss.view_as(targets)
    return loss.mean()


def token_losses(model, rows: torch.Tensor, amp) -> torch.Tensor:
    """Per-position next-token loss of rows (n, T + 1): (n, T). Each row is evaluated on its own
    (one row per call), documents kept apart within it."""
    out = []
    with torch.no_grad():
        for r in rows:
            x, y = r[None, :-1], r[None, 1:]
            with amp():
                h = model(x, mask_for(model, x), return_hidden=True)
                out.append(chunked_ce(h, model.output.weight, y, reduction="none")[0].float().cpu())
    return torch.stack(out)


# ── measures ──

POSITION_BUCKETS = [(1, 16), (17, 64), (65, 256), (257, 1024), (1025, 2048), (2049, 4096)]
EARLY, LATE = (50, 100), (1000, 2000)      # positions in a document (1-based) for the in-context score


def heldout_loss(model, rows: torch.Tensor, amp) -> float:
    return float(token_losses(model, rows, amp).mean())


def position_loss(model, docs: torch.Tensor, amp, early=EARLY, late=LATE) -> dict:
    """docs: (n, L + 1), one long document per row starting at its BOS. Mean loss per position
    bucket, and per document the in-context score loss(EARLY) - loss(LATE): how much the loss
    falls between position 50-100 and 1000-2000 of the same document."""
    losses = token_losses(model, docs, amp)            # position t predicts token t + 1 (1-based t)
    n, T = losses.shape
    buckets = {f"{a}-{b}": round(float(losses[:, a - 1:min(b, T)].mean()), 4) for a, b in POSITION_BUCKETS if a <= T}
    if late[0] > T:
        raise ValueError(f"documents of {T} positions are shorter than the late window {late}")
    early = losses[:, early[0] - 1:early[1]].mean(1)
    late = losses[:, late[0] - 1:min(late[1], T)].mean(1)
    score = (early - late).tolist()
    return {"buckets": buckets, "icl_score": round(sum(score) / n, 4), "icl_per_doc": [round(v, 4) for v in score],
            "early": round(float(early.mean()), 4), "late": round(float(late.mean()), 4),
            "early_per_doc": [round(v, 4) for v in early.tolist()], "late_per_doc": [round(v, 4) for v in late.tolist()],
            "docs": n, "length": T}


NAMES = ["Arlen", "Bexley", "Corwin", "Dalia", "Elric", "Fenna", "Garrick", "Hollis", "Imara", "Joss",
         "Kestrel", "Lorne", "Maren", "Nolan", "Oriel", "Perrin", "Quill", "Rowan", "Sable", "Tamsin"]


def recall_cases(encode, filler: list[int], distances, n: int, seed: int = 7) -> list[dict]:
    """n cases per distance: a fact "The access code for <name> is <5 digits>." placed `distance`
    tokens before the same words again; the model has to continue with the digits. One document
    per case (BOS + filler + fact + filler + cue), the same for every model."""
    rng = random.Random(seed)
    cases = []
    for d in distances:
        for _ in range(n):
            name = rng.choice(NAMES)
            code = "".join(rng.choice("0123456789") for _ in range(5))
            fact = encode(f"\nThe access code for {name} is {code}.")
            cue_alone = encode(f"\nThe access code for {name} is")
            i = 0                                      # the cue is the common prefix, token for token
            while i < min(len(fact), len(cue_alone)) and fact[i] == cue_alone[i]:
                i += 1
            cue, answer = fact[:i], fact[i:]
            lead = 64
            start = rng.randrange(0, max(1, len(filler) - lead - d - 1))
            ids = [BOS] + filler[start:start + lead] + fact + filler[start + lead:start + lead + d] + cue + answer
            cases.append({"distance": d, "code": code, "ids": ids, "answer_len": len(answer)})
    return cases


def recall(model, cases: list[dict], amp, device) -> dict:
    """Teacher-forced: a case is right when every answer token is the model's first choice."""
    res = {}
    with torch.no_grad():
        for c in cases:
            ids = torch.tensor(c["ids"], device=device)
            x, y = ids[None, :-1], ids[None, 1:]
            with amp():
                h = model(x, mask_for(model, x), return_hidden=True)
            k = c["answer_len"]
            logits = (h[0, -k:] @ model.output.weight.t().to(h.dtype)).float()
            ok = bool((logits.argmax(-1) == y[0, -k:]).all())
            lp = float(F.log_softmax(logits, -1).gather(1, y[0, -k:, None]).mean())
            r = res.setdefault(str(c["distance"]), {"n": 0, "right": 0, "logprob": 0.0})
            r["n"] += 1; r["right"] += ok; r["logprob"] += lp
    return {d: {"n": r["n"], "acc": round(r["right"] / r["n"], 3), "mean_logprob": round(r["logprob"] / r["n"], 3)}
            for d, r in res.items()}


def bootstrap_mean(values, n: int = 10_000, seed: int = 0) -> dict:
    t = torch.tensor(values, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    m = t[torch.randint(0, len(t), (n, len(t)), generator=g)].mean(1)
    lo, hi = torch.quantile(m, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"mean": round(float(t.mean()), 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(values)}


def lr_at(step: int, total: int, lr: float, warmup: int, min_ratio: float) -> float:
    """Linear warm-up, then cosine decay to min_ratio * lr (the schedule of the upstream configs)."""
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
