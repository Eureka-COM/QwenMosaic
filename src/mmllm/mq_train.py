"""Shared pieces of the MosaicQwen training stages (scripts/mq_stage1.py, mq_stage2.py).

    Progress      one JSON line per event in <out>/progress.jsonl (the runner tails it)
    Budget        --max-minutes: stop cleanly, with a checkpoint, before the time limit
    lr_at         linear warm-up, cosine decay to lr * min_ratio
    load_base     the base Qwen model + tokenizer
    Teacher       teacher forward passes: the original layers kept beside the converted ones
                  (one copy of the weights), or a separate copy when the base is trained
    evaluate      the numbers every stage reports, on fixed held-out rows:
                    kl           KL(teacher || student) per token, nats
                    nll_teacher  next-token loss of the original model
                    nll_student  next-token loss of the converted model
                    ppl_ratio    exp(nll_student - nll_teacher)   (1.0 = no loss)
                    top1         share of positions where both pick the same next token
                    rel_mse      per converted layer, teacher-forced: |memories(x) - mixer(x)|^2 / |mixer(x)|^2
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .mosaic_qwen import MosaicQwen, _text_model


class Progress:
    def __init__(self, out: str | Path, resume: bool = False) -> None:
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "progress.jsonl"
        if not resume and self.path.exists():
            self.path.unlink()
        self.t0 = time.time()

    def log(self, event: str, step: int, echo: bool = True, **fields) -> dict:
        rec = {"event": event, "step": step, "elapsed_min": round((time.time() - self.t0) / 60, 2), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if echo:
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        return rec


class Budget:
    """The time a run may take. Also the polite stop: the runner stops a run (/content/STOP) with SIGTERM;
    the first SIGTERM only marks the budget as spent, so the training loop ends at the next step and saves
    (to its store too) instead of losing everything since the last periodic save. A second SIGTERM ends
    the process at once."""

    def __init__(self, max_minutes: float | None, reserve_min: float = 3.0, catch_term: bool = True) -> None:
        self.t0, self.max, self.reserve = time.time(), max_minutes, reserve_min
        self.terminated = False
        if catch_term:
            import signal
            import threading
            if threading.current_thread() is threading.main_thread():
                def _on_term(signum, frame):
                    if self.terminated:                   # asked twice: stop now
                        raise SystemExit(143)
                    self.terminated = True
                    print("SIGTERM: stopping after this step (saving first)", flush=True)
                signal.signal(signal.SIGTERM, _on_term)

    def minutes(self) -> float:
        return (time.time() - self.t0) / 60

    def exhausted(self) -> bool:
        if self.terminated:
            return True
        return self.max is not None and self.max > 0 and self.minutes() > max(self.max - self.reserve, self.max * 0.9)


def lr_at(step: int, total: int, lr: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_base(model_path: str, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = None
    if device.type == "cuda":
        try:        # weights go from disk to the GPU without a full copy in RAM (a 27B model is 50 GB)
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, low_cpu_mem_usage=True,
                                                         device_map={"": device.index or 0})
        except Exception as e:  # noqa: BLE001
            print(f"[mq] direct-to-GPU loading failed ({type(e).__name__}: {str(e)[:120]}); loading through RAM", flush=True)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def hidden_states(model, ids: torch.Tensor) -> torch.Tensor:
    return _text_model(model)(input_ids=ids, use_cache=False).last_hidden_state


class Teacher:
    """Runs the original model. `separate` is an independent copy of the base model (needed
    when the student's base weights are trained); otherwise the originals kept by the
    conversion are used, on the same weights as the student."""

    def __init__(self, mq: MosaicQwen, separate=None) -> None:
        self.mq, self.separate = mq, separate
        self._hooks_out: dict | None = None
        if separate is not None:
            text = _text_model(separate)
            for i in mq.layers:
                text.layers[i].register_forward_hook(self._hook(i))

    def _hook(self, i):
        def f(_m, _a, out):
            if self._hooks_out is not None:
                self._hooks_out[i] = out[0] if isinstance(out, tuple) else out
        return f

    @torch.no_grad()
    def hidden(self, ids: torch.Tensor, layer_outputs: bool = False):
        """(final hidden states, {layer: output of that layer} or None)."""
        outs = {} if layer_outputs else None
        if self.separate is not None:
            self._hooks_out = outs
            try:
                h = hidden_states(self.separate, ids)
            finally:
                self._hooks_out = None
            return h, outs
        cap = (lambda i, x, mixed, out: outs.__setitem__(i, out)) if layer_outputs else None
        with self.mq.teacher(capture=cap):
            h = hidden_states(self.mq.model, ids)
        return h, outs

    def lm_head(self):
        return (self.separate or self.mq.model).lm_head


def kl_nll(student_logits, teacher_logits, target):
    t_logp = F.log_softmax(teacher_logits.float(), -1)
    s_logp = F.log_softmax(student_logits.float(), -1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    nll_s = -s_logp.gather(-1, target[:, None]).squeeze(-1)
    nll_t = -t_logp.gather(-1, target[:, None]).squeeze(-1)
    agree = (t_logp.argmax(-1) == s_logp.argmax(-1)).float()
    return kl, nll_s, nll_t, agree


HARD_NLL = 8.0        # positions where the teacher gave the actual next token less than e^-8 (0.03%)


@torch.no_grad()
def evaluate(mq: MosaicQwen, teacher: Teacher, rows: torch.Tensor, device, chunk: int | None = None,
             layer_mse: bool = True, row_bsz: int = 1) -> dict:
    """See the module docstring. rows (N, T) of token ids; every position but the last is scored."""
    if chunk is None:        # positions per vocabulary projection (each is vocab x 4 bytes, several copies)
        chunk = 256 if torch.device(device).type == "cuda" else 48
    was_training = mq.model.training
    mq.model.eval()
    mq.fixed_skip()
    tot = {"kl": 0.0, "nll_student": 0.0, "nll_teacher": 0.0, "top1": 0.0}
    n = 0
    dg = collections.defaultdict(float)
    mse_num = {i: 0.0 for i in mq.layers}
    mse_den = {i: 0.0 for i in mq.layers}

    def cap(i, x, mixed, out):
        pred = mq.layers[i].memory_residual(x)
        mse_num[i] += (pred.float() - mixed.float()).pow(2).sum().item()
        mse_den[i] += mixed.float().pow(2).sum().item()

    for a in range(0, rows.shape[0], row_bsz):
        ids = rows[a:a + row_bsz].to(device)
        if layer_mse and teacher.separate is None:
            with mq.teacher(capture=cap):
                h_t = hidden_states(mq.model, ids)
        else:
            h_t, _ = teacher.hidden(ids)
        h_s = hidden_states(mq.model, ids)
        target = ids[:, 1:].reshape(-1)
        h_t, h_s = h_t[:, :-1].reshape(-1, h_t.shape[-1]), h_s[:, :-1].reshape(-1, h_s.shape[-1])
        for c in range(0, target.shape[0], chunk):
            s_logits, t_logits = mq.model.lm_head(h_s[c:c + chunk]), teacher.lm_head()(h_t[c:c + chunk])
            tgt = target[c:c + chunk]
            kl, nll_s, nll_t, agree = kl_nll(s_logits, t_logits, tgt)
            tot["kl"] += kl.sum().item(); tot["nll_student"] += nll_s.sum().item()
            tot["nll_teacher"] += nll_t.sum().item(); tot["top1"] += agree.sum().item()
            n += kl.numel()
            # Is the teacher too sure of itself on this text? (27B stage 2: the student's loss on the
            # text fell below the teacher's while it only ever imitates the teacher.)
            t_logp, s_logp = F.log_softmax(t_logits.float(), -1), F.log_softmax(s_logits.float(), -1)
            del s_logits, t_logits
            dg["entropy_teacher"] += -(t_logp.exp() * t_logp).sum().item()
            dg["entropy_student"] += -(s_logp.exp() * s_logp).sum().item()
            dg["correct_teacher"] += (t_logp.argmax(-1) == tgt).sum().item()
            dg["correct_student"] += (s_logp.argmax(-1) == tgt).sum().item()
            for temp in (1.25, 1.5):
                dg[f"nll_teacher_temperature_{temp}"] += F.nll_loss(F.log_softmax(t_logp / temp, -1), tgt, reduction="sum").item()
            hard = nll_t > HARD_NLL
            dg["hard_positions"] += int(hard.sum())
            dg["nll_teacher_hard"] += nll_t[hard].sum().item(); dg["nll_student_hard"] += nll_s[hard].sum().item()
            dg["nll_teacher_rest"] += nll_t[~hard].sum().item(); dg["nll_student_rest"] += nll_s[~hard].sum().item()
            del t_logp, s_logp
    out = {k: round(v / max(n, 1), 5) for k, v in tot.items()}
    out["ppl_ratio"] = round(math.exp(out["nll_student"] - out["nll_teacher"]), 4)
    out["positions"] = n
    nh = dg.pop("hard_positions")
    cal = {k: round(v / max(n, 1), 5) for k, v in dg.items() if not k.endswith(("_hard", "_rest"))}
    cal.update(hard_nll=HARD_NLL, hard_positions=nh,
               nll_teacher_hard=round(dg["nll_teacher_hard"] / max(nh, 1), 4), nll_student_hard=round(dg["nll_student_hard"] / max(nh, 1), 4),
               nll_teacher_rest=round(dg["nll_teacher_rest"] / max(n - nh, 1), 5), nll_student_rest=round(dg["nll_student_rest"] / max(n - nh, 1), 5))
    out["calibration"] = cal
    if layer_mse and teacher.separate is None:
        out["rel_mse"] = {str(i): round(mse_num[i] / max(mse_den[i], 1e-30), 5) for i in mq.layers}
        out["rel_mse_mean"] = round(sum(mse_num[i] / max(mse_den[i], 1e-30) for i in mq.layers) / len(mq.layers), 5)
    if was_training:
        mq.model.train()
    return out


def save_optimizer(opt, path: Path, step: int, tokens: int) -> None:
    torch.save({"opt": opt.state_dict(), "step": step, "tokens": tokens}, path)


def load_optimizer(opt, path: Path) -> tuple[int, int]:
    blob = torch.load(path, map_location="cpu")
    opt.load_state_dict(blob["opt"])
    return blob["step"], blob["tokens"]


def resume_point(resume_dir: Path, opt, stage: int) -> tuple[int, int]:
    """(step, tokens) to continue from: the optimizer checkpoint if there is one, else what
    the delta recorded (then the optimizer starts fresh). A delta of an earlier stage does
    not count: a new stage starts at step 0."""
    resume_dir = Path(resume_dir)
    if (resume_dir / "optim.pt").exists():
        return load_optimizer(opt, resume_dir / "optim.pt")
    extra = json.loads((resume_dir / "delta" / "mq_config.json").read_text()).get("extra", {})
    if extra.get("stage") != stage:
        return 0, 0
    return int(extra.get("step", 0)), int(extra.get("tokens", 0))


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def gpu_stats(detail: bool = False) -> dict:
    """Peak memory allocated by PyTorch; with detail also what PyTorch holds and what the rest of
    the device uses (other libraries' own allocations: these also count against the card)."""
    if not torch.cuda.is_available():
        return {}
    out = {"gpu_gb_peak": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    if detail:
        free, total = torch.cuda.mem_get_info()
        held = torch.cuda.memory_reserved()
        out.update(gpu_gb_held_peak=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                   gpu_gb_not_torch=round((total - free - held) / 2**30, 2), gpu_gb_free=round(free / 2**30, 2))
    return out


def vocab_loss_backward(student_head, teacher_head, hs_sel, ht_sel, target, *, ce_weight: float = 0.0,
                        kl_weight: float = 1.0, scale: float = 1.0, chunk: int = 128, amp=None, extra=None):
    """kl_weight * KL(teacher || student) + ce_weight * cross-entropy over the vocabulary at the selected
    positions, and its backward pass, without ever holding positions x vocabulary tensors for
    all positions at once.

    hs_sel (N, d): the student's hidden states at the selected positions, attached to the graph
    of the student's forward pass. ht_sel (N, d): the teacher's. The loss is computed a chunk of
    positions at a time on a detached copy of hs_sel; each chunk's backward stops there and adds
    to that copy's gradient (and to the head's own weights, if they are trained). Then one
    backward pass carries the gradient through the model. This is the chain rule, not an
    approximation: every row of hs_sel belongs to one chunk.

    kl_weight 0 with ce_weight > 0 trains on the text alone (no distillation); the KL is still
    measured and returned.

    scale multiplies the gradient (1 / accumulation steps). extra: another scalar loss on the
    same graph, already scaled, that goes back in the same pass (the hidden-state term).
    Returns (kl, ce) as floats (means over positions)."""
    if kl_weight <= 0 and ce_weight <= 0 and extra is None:
        raise ValueError("no loss: kl_weight and ce_weight are both 0")
    n = hs_sel.shape[0]
    leaf = hs_sel.detach().requires_grad_(True)
    kl_sum, ce_sum = 0.0, 0.0
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        with torch.no_grad():
            with (amp if amp is not None else contextlib.nullcontext()):
                t = teacher_head(ht_sel[sl])
            t_logp = F.log_softmax(t.float(), -1)
            del t
        with (amp if amp is not None else contextlib.nullcontext()):
            s_logits = student_head(leaf[sl])
        s_logp = F.log_softmax(s_logits.float(), -1)
        del s_logits
        kl = (t_logp.exp() * (t_logp - s_logp)).sum()
        ce = F.nll_loss(s_logp, target[sl], reduction="sum")
        loss = kl_weight * kl + ce_weight * ce
        (loss * (scale / n)).backward()
        kl_sum += kl.item(); ce_sum += ce.item()
        del t_logp, s_logp, kl, ce, loss
    if extra is not None:
        torch.autograd.backward([hs_sel, extra], [leaf.grad, None])
    else:
        hs_sel.backward(leaf.grad)
    return kl_sum / n, ce_sum / n
