"""What does writing the standalone model cost? (docs/plan-mosaic-qwen.md §4: "the exported
model, read back, gives the same output".)

The trained student is: base weights + LoRA (kept apart) + memories. The export folds the
LoRA into the base weights and saves everything in one dtype. Folding a small low-rank update
into bf16 weights can round part of it away. This script measures that, on the same held-out
positions, in fp32 arithmetic (so that only the saved weights differ):

    in memory      delta loaded, LoRA not folded
    saved fp32     folded, saved as fp32, read back
    saved bf16     folded, saved as bf16, read back

For each: KL from the teacher, perplexity ratio to the teacher, top-1 agreement with the
teacher; and against "in memory": KL, largest logit difference, top-1 agreement.

    python scripts/mq_roundtrip.py --model PATH --delta DIR --data SRC ... --out DIR
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm.mosaic_qwen import load_mosaic_qwen, load_standalone  # noqa: E402
from mmllm.mq_data import RowPacker, load_source  # noqa: E402
from mmllm.mq_train import pick_device  # noqa: E402


@torch.no_grad()
def row_logits(model, row, device):
    return model(row[None].to(device), use_cache=False).logits[0, :-1].float().cpu()      # (T-1, vocab)


class Sums:
    """Running means over positions (one row at a time: a row of logits is 0.25 GB at 256 tokens)."""

    def __init__(self):
        self.n, self.s = 0, {}

    def add(self, n, **kw):
        self.n += n
        for k, v in kw.items():
            self.s[k] = self.s.get(k, 0.0) + float(v)

    def mean(self, k):
        return self.s[k] / self.n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--delta", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--rows", type=int, default=6); ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out", required=True); ap.add_argument("--work", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--step", default="all")    # all | prepare | saved_fp32 | saved_bf16 (one process each on a small machine)
    ap.add_argument("--shard-mb", type=int, default=300)       # small files: little memory while writing
    ap.add_argument("--max-minutes", type=float, default=None)  # accepted for the runner; unused
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work or out / "work"); work.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    t0 = time.time()

    def note(msg, **kw):
        line = {"event": msg, "elapsed_min": round((time.time() - t0) / 60, 2), **kw}
        print(json.dumps(line, ensure_ascii=False), flush=True)
        with open(out / "progress.jsonl", "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = RowPacker(tok, [load_source(s, a.docs) for s in a.data], a.seq_len, seed=0).eval_rows(a.rows)
    prepare = a.step in ("all", "prepare")
    mq = None
    if prepare:
        mq, _ = load_mosaic_qwen(a.model, a.delta, dtype=torch.float32, device=device, keep_teacher=True)
        note("loaded", rows=list(rows.shape), lora_modules=len(mq.lora))
    ref = work / "ref"; ref.mkdir(exist_ok=True)           # teacher log-probs and in-memory logits, per row, on disk
    res = {"positions": int(rows.shape[0] * (rows.shape[1] - 1)), "variants": {}}
    if not prepare:
        res = json.loads((out / "roundtrip.json").read_text())

    def measure(name, model, first=False):
        acc = Sums()
        mx = 0.0
        for i, r in enumerate(rows):
            tgt = r[1:]
            if first:
                with mq.teacher():
                    t_logp = F.log_softmax(row_logits(model, r, device), -1)
                torch.save(t_logp, ref / f"t{i}.pt")
            else:
                t_logp = torch.load(ref / f"t{i}.pt")
            logits = row_logits(model, r, device)
            logp = F.log_softmax(logits, -1)
            n = tgt.numel()
            acc.add(n, kl_t=(t_logp.exp() * (t_logp - logp)).sum(-1).sum(), nll=F.nll_loss(logp, tgt, reduction="sum"),
                    nll_t=F.nll_loss(t_logp, tgt, reduction="sum"), top1_t=(logp.argmax(-1) == t_logp.argmax(-1)).sum())
            if first:
                torch.save(logits, ref / f"m{i}.pt")
            else:
                m = torch.load(ref / f"m{i}.pt")
                mx = max(mx, (logits - m).abs().max().item())
                m_logp = F.log_softmax(m, -1)
                acc.add(0, kl_m=(m_logp.exp() * (m_logp - logp)).sum(-1).sum(), top1_m=(logp.argmax(-1) == m.argmax(-1)).sum())
                del m, m_logp
            del t_logp, logits, logp
            gc.collect()
        v = {"kl_teacher": round(acc.mean("kl_t"), 5), "nll": round(acc.mean("nll"), 5),
             "ppl_ratio": round(float(torch.tensor(acc.mean("nll") - acc.mean("nll_t")).exp()), 4),
             "top1_teacher": round(acc.mean("top1_t"), 4)}
        if not first:
            v.update(kl_in_memory=round(acc.mean("kl_m"), 6), top1_in_memory=round(acc.mean("top1_m"), 4),
                     max_logit_diff=round(mx, 4))
        res["nll_teacher"] = round(acc.mean("nll_t"), 5)
        res["variants"][name] = v
        note(name, **v)
        (out / "roundtrip.json").write_text(json.dumps(res, indent=1))

    if prepare:
        measure("in_memory", mq.model, first=True)
        mq.drop_teacher()
        for name, dt in (("saved_fp32", None), ("saved_bf16", torch.bfloat16)):
            mq.save_standalone(work / name, dtype=dt, shard_mb=a.shard_mb)     # folds the LoRA (in fp32) the first time
            gc.collect()
        del mq
        gc.collect()
    for name in ("saved_fp32", "saved_bf16"):
        if a.step not in ("all", name):
            continue
        m2 = load_standalone(work / name, dtype=torch.float32, device=device)
        measure(name, m2.model)
        del m2
        gc.collect()
        shutil.rmtree(work / name, ignore_errors=True)
    if not any((work / n).exists() for n in ("saved_fp32", "saved_bf16")):
        shutil.rmtree(ref, ignore_errors=True)
    note("done")


if __name__ == "__main__":
    main()
