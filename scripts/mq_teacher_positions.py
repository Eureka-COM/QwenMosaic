#!/usr/bin/env python3
"""Where is the ORIGINAL model wrong? Its loss position by position, on the rows the stages use.

Why: on 8,192-token rows the original Qwen3.8-27B (the teacher) scored a loss of 3.27, against
2.04 on 2,048-token rows, with 12% of the positions above 8 nats (stage 3, first evaluation).
A student that only imitates the teacher then "beats" it. This script does not touch the Mosaics
at all: it loads the original model and reports

  - which positions are bad: which document, how far into the row, how far after a boundary;
  - what the model predicted there (its top token and how sure it was);
  - whether the same tokens stay bad when the documents before them are taken away;
  - whether autocast changes anything (training runs the teacher under autocast, evaluation not).

It is built to run NEXT TO a training job on the same GPU: the model stays in CPU memory and each
layer visits the GPU only for its own forward pass; --gpu-fraction caps what this process may take.

    python scripts/mq_teacher_positions.py --model /content/models/Qwen3.8-27B \
        --data ... --docs 60000 --seq-len 8192 --max-doc 8192 --eval-rows 2 --train-rows 8 \
        --out /content/out/s3/teacher_positions
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("MMLLM_SRC") or str(ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mmllm.mosaic_qwen import _text_model  # noqa: E402
from mmllm.mq_data import RowPacker, load_source  # noqa: E402


def load(path: str, dtype, dev: torch.device):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).eval()
    text = _text_model(model)
    if dev.type == "cuda":
        for name, mod in text.named_children():
            if name != "layers":
                mod.to(dev)
        model.lm_head.to(dev)

        def pre(m, _a):
            m.to(dev)

        def post(m, _a, _out):
            m.to("cpu")

        for layer in text.layers:
            layer.register_forward_pre_hook(pre)
            layer.register_forward_hook(post)
    return tok, model, text


@torch.no_grad()
def score(model, text, rows: list[list[int]], pad: int, dev, autocast: bool = False, chunk: int = 512) -> list[dict]:
    """For every row: loss of each target (index j = 1..len-1), the top token and its probability."""
    T = max(len(r) for r in rows)
    batch = torch.full((len(rows), T), pad, dtype=torch.long)
    for b, r in enumerate(rows):
        batch[b, :len(r)] = torch.tensor(r)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (autocast and dev.type == "cuda") else contextlib.nullcontext()
    with ctx:
        h = text(input_ids=batch.to(dev), use_cache=False).last_hidden_state
    out = []
    for b, r in enumerate(rows):
        n = len(r) - 1
        tgt = torch.tensor(r[1:], device=dev)
        nll, top, ptop, ent = torch.empty(n), torch.empty(n, dtype=torch.long), torch.empty(n), torch.empty(n)
        for c in range(0, n, chunk):
            e = min(c + chunk, n)
            logp = F.log_softmax(model.lm_head(h[b, c:e]).float(), -1)
            nll[c:e] = -logp.gather(1, tgt[c:e, None])[:, 0].cpu()
            p, t = logp.max(-1)
            top[c:e], ptop[c:e] = t.cpu(), p.exp().cpu()
            ent[c:e] = -(logp.exp() * logp).sum(-1).cpu()
            del logp
        # index 0 has no prediction: pad so that every array is indexed by the target's position
        z = torch.zeros(1)
        out.append({"nll": torch.cat([z, nll]), "top": torch.cat([torch.zeros(1, dtype=torch.long), top]),
                    "ptop": torch.cat([z, ptop]), "ent": torch.cat([z, ent])})
    del h
    return out


def segments(ids: list[int], sep: int) -> list[tuple[int, int]]:
    """(start, end) of every document in the row; ids[end] is its separator (or the row ends)."""
    segs, start = [], 0
    for i, t in enumerate(ids):
        if t == sep:
            if i > start:
                segs.append((start, i))
            start = i + 1
    if start < len(ids):
        segs.append((start, len(ids)))
    return segs


def _tok(tok, i: int) -> str:
    return repr(tok.decode([int(i)]))[1:-1][:24]


def describe(tok, name: str, ids: list[int], res: dict, sep: int, hard: float, say) -> dict:
    nll, top, ptop, ent = res["nll"], res["top"], res["ptop"], res["ent"]
    T = len(ids)
    scored = torch.ones(T, dtype=torch.bool)
    scored[0] = False
    is_hard = (nll > hard) & scored
    n, nh = int(scored.sum()), int(is_hard.sum())
    mean = float(nll[scored].mean())
    rest = float(nll[scored & ~is_hard].mean()) if n > nh else float("nan")
    acc = float((top[scored] == torch.tensor(ids)[scored]).float().mean())
    say(f"\n===== {name}: {T} tokens, loss {mean:.3f}, top-1 correct {acc:.3f}, entropy {float(ent[scored].mean()):.3f}")
    say(f"  bad positions (loss > {hard:g}): {nh} = {100 * nh / n:.1f}%   their loss {float(nll[is_hard].mean()) if nh else 0:.2f}   the rest {rest:.3f}")

    segs = segments(ids, sep)
    seg_rows = []
    say("  documents:  start  len | loss  bad | first 32 tokens: loss bad | the rest: loss | text")
    for s, e in segs:
        idx = torch.arange(max(s, 1), e)
        if len(idx) == 0:
            continue
        first, later = idx[idx < s + 32], idx[idx >= s + 32]
        row = {"start": s, "len": e - s, "loss": round(float(nll[idx].mean()), 3), "bad": int(is_hard[idx].sum()),
               "loss_first32": round(float(nll[first].mean()), 3) if len(first) else None,
               "bad_first32": int(is_hard[first].sum()),
               "loss_later": round(float(nll[later].mean()), 3) if len(later) else None,
               "loss_of_separator": round(float(nll[e]), 2) if e < T else None,
               "head": tok.decode(ids[s:s + 14]).replace("\n", "\\n")[:60]}
        seg_rows.append(row)
        say(f"    {s:5d} {e - s:5d} | {row['loss']:6.3f} {row['bad']:4d} | {row['loss_first32']} {row['bad_first32']} | "
            f"{row['loss_later']} | sep {row['loss_of_separator']} | {row['head']}")

    by_pos = []
    for a in range(0, T, 1024):
        idx = torch.arange(max(a, 1), min(a + 1024, T))
        by_pos.append({"from": a, "loss": round(float(nll[idx].mean()), 3), "bad": int(is_hard[idx].sum())})
    say("  by position in the row (each 1,024): " + "  ".join(f"{b['from']}:{b['loss']}/{b['bad']}" for b in by_pos))

    offset = torch.zeros(T, dtype=torch.long)
    for s, e in segs:
        offset[s:min(e + 1, T)] = torch.arange(min(e + 1, T) - s)
    by_off = []
    for lo, hi in ((0, 8), (8, 32), (32, 128), (128, 512), (512, 10 ** 9)):
        m = scored & (offset >= lo) & (offset < hi)
        if int(m.sum()):
            by_off.append({"from": lo, "n": int(m.sum()), "loss": round(float(nll[m].mean()), 3), "bad": int((is_hard & m).sum())})
    say("  by distance from the start of its document: " + "  ".join(f"{b['from']}+:{b['loss']}/{b['bad']}of{b['n']}" for b in by_off))

    runs, cur = [], 0
    for v in is_hard.tolist():
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    in_long = sum(r for r in runs if r >= 8)
    say(f"  runs of bad positions: {len(runs)} runs, longest {max(runs) if runs else 0}, {in_long} of {nh} bad positions sit in runs of 8 or more")

    hard_idx = is_hard.nonzero()[:, 0].tolist()
    tops = collections.Counter(_tok(tok, top[j]) for j in hard_idx).most_common(12)
    tgts = collections.Counter(_tok(tok, ids[j]) for j in hard_idx).most_common(12)
    sure = float((ptop[is_hard] > 0.9).float().mean()) if nh else 0.0
    say(f"  at the bad positions the model's top token had probability {float(ptop[is_hard].mean()) if nh else 0:.3f} on average ({100 * sure:.0f}% above 0.9)")
    say("  its top tokens there: " + ", ".join(f"{t}×{c}" for t, c in tops))
    say("  the real tokens there: " + ", ".join(f"{t}×{c}" for t, c in tgts))
    samples = []
    for j in hard_idx[:: max(1, len(hard_idx) // 14)][:14]:
        samples.append({"pos": j, "offset": int(offset[j]), "loss": round(float(nll[j]), 1), "real": _tok(tok, ids[j]),
                        "top": _tok(tok, top[j]), "p_top": round(float(ptop[j]), 3),
                        "before": tok.decode(ids[max(0, j - 10):j]).replace("\n", "\\n")[-48:]})
        x = samples[-1]
        say(f"    pos {x['pos']:5d} (+{x['offset']}) loss {x['loss']:5.1f} real [{x['real']}] top [{x['top']}] p {x['p_top']} | …{x['before']}")
    return {"tokens": T, "loss": round(mean, 4), "bad": nh, "bad_share": round(nh / n, 4), "loss_rest": round(rest, 4),
            "top1_correct": round(acc, 4), "documents": seg_rows, "by_position": by_pos, "by_offset": by_off,
            "runs": {"count": len(runs), "longest": max(runs) if runs else 0, "bad_in_runs_of_8_or_more": in_long},
            "top_tokens_at_bad": tops, "real_tokens_at_bad": tgts, "samples": samples,
            "nll": [round(float(v), 2) for v in nll.tolist()]}


def _as_result(row: dict) -> dict:
    """A dumped row (arrays indexed by target - 1) in the shape `describe` takes (indexed by target)."""
    z = torch.zeros(1)
    nll = torch.cat([z, torch.tensor(row["nll"])])
    return {"nll": nll, "top": torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(row["top"])]),
            "ptop": torch.cat([z, torch.tensor(row["ptop"])]), "ent": torch.zeros_like(nll)}


def from_dump(args, out: Path, say) -> None:
    dump = json.loads(Path(args.dump).read_text())
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model or dump["model"])
    sep, rows = dump["sep"], dump["rows"]
    say(f"dump of {dump['model']}: {list(rows)}  (separator token {sep} = {_tok(tok, sep)})")
    result = {"model": dump["model"], "hard": args.hard, "rows": {}, "alone": {}, "autocast": {}}
    for nm, row in rows.items():
        if nm.endswith(("_tail", "_autocast")):
            continue
        result["rows"][nm] = describe(tok, nm, row["ids"], _as_result(row), sep, args.hard, say)
        result["rows"][nm].pop("nll")
    say("\n===== the second half of each row, alone (nothing before it)")
    for nm, row in rows.items():
        if not nm.endswith("_tail"):
            continue
        base = rows[nm[:-5]]
        start = len(base["ids"]) - len(row["ids"])
        alone = torch.tensor(row["nll"])
        full = torch.tensor(base["nll"])[start:start + len(alone)]
        hf, ha = full > args.hard, alone > args.hard
        x = {"from": start, "tokens": len(row["ids"]), "loss_in_row": round(float(full.mean()), 3),
             "loss_alone": round(float(alone.mean()), 3), "bad_in_row": int(hf.sum()), "bad_alone": int(ha.sum()),
             "bad_in_both": int((hf & ha).sum())}
        result["alone"][nm[:-5]] = x
        say(f"  {nm[:-5]}: tokens {start}..{start + len(row['ids'])}: loss in the row {x['loss_in_row']} / alone {x['loss_alone']};"
            f" bad in the row {x['bad_in_row']} / alone {x['bad_alone']} / in both {x['bad_in_both']}")
        for a in range(0, len(alone), 512):       # where along the tail the two part
            f, al = full[a:a + 512], alone[a:a + 512]
            say(f"      tokens {start + a:5d}+: in the row {float(f.mean()):6.3f} ({int((f > args.hard).sum())} bad) / alone {float(al.mean()):6.3f} ({int((al > args.hard).sum())} bad)")
    say("\n===== the same row under autocast (as the teacher runs during training)")
    for nm, row in rows.items():
        if not nm.endswith("_autocast"):
            continue
        a, b = torch.tensor(rows[nm[:-9]]["nll"]), torch.tensor(row["nll"])
        x = {"loss_plain": round(float(a.mean()), 3), "loss_autocast": round(float(b.mean()), 3),
             "bad_plain": int((a > args.hard).sum()), "bad_autocast": int((b > args.hard).sum()),
             "largest_difference": round(float((a - b).abs().max()), 3)}
        result["autocast"][nm[:-9]] = x
        say(f"  {nm[:-9]}: {x}")
    (out / "teacher_positions_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--dump", default=None, help="analyse what mq_teacher_dump.py wrote instead of running the model")
    ap.add_argument("--tokenizer", default=None, help="with --dump: where to take the tokenizer from")
    ap.add_argument("--docs", type=int, default=60_000)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--max-doc", type=int, default=None)
    ap.add_argument("--eval-rows", type=int, default=2)
    ap.add_argument("--train-rows", type=int, default=0, help="also build the first N training rows (seed 0)")
    ap.add_argument("--train-pick", default="", help="which of them to score, e.g. 6,8 (default: all)")
    ap.add_argument("--hard", type=float, default=8.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--gpu-fraction", type=float, default=0.12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s: str) -> None:
        print(s, flush=True)
        lines.append(s)
        (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.dump:
        from_dump(args, out, say)
        return
    if not args.model or not args.data:
        ap.error("--model and --data are needed (or --dump)")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    t0 = time.time()
    tok, model, text = load(args.model, torch.bfloat16 if args.dtype == "bf16" else torch.float32, dev)
    say(f"model loaded in {time.time() - t0:.0f} s: {args.model} ({args.dtype}, {len(text.layers)} layers)")

    sources = [load_source(s, args.docs) for s in args.data]
    packer = RowPacker(tok, sources, args.seq_len, seed=0, max_doc=args.max_doc)
    sep = packer.sep_id
    rows = {f"eval{i}": r for i, r in enumerate(packer.eval_rows(args.eval_rows).tolist())}
    pick = {int(x) for x in args.train_pick.split(",") if x}
    for i in range(1, args.train_rows + 1):
        r = packer.batch(1)[0].tolist()
        if not pick or i in pick:
            rows[f"train{i}"] = r
    say(f"rows: {list(rows)}  (separator token {sep} = {_tok(tok, sep)})")

    result = {"model": args.model, "seq_len": args.seq_len, "hard": args.hard, "rows": {}, "alone": {}, "autocast": {}}
    names = list(rows)
    scored = {}
    for a in range(0, len(names), 2):
        part = names[a:a + 2]
        t1 = time.time()
        for nm, res in zip(part, score(model, text, [rows[n] for n in part], sep, dev)):
            scored[nm] = res
        say(f"scored {part} in {time.time() - t1:.0f} s" + (f" (GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB)" if dev.type == "cuda" else ""))
    for nm in names:
        result["rows"][nm] = describe(tok, nm, rows[nm], scored[nm], sep, args.hard, say)

    # the same tokens without what came before them: the documents of the second half, alone
    tails = {}
    for nm in names:
        ids = rows[nm]
        cut = next((i for i in range(len(ids) // 2, len(ids) - 64) if ids[i] == sep), None)
        if cut is not None:
            tails[nm] = (cut + 1, ids[cut + 1:])
    tn = list(tails)
    say("\n===== the second half of each row, alone (nothing before it)")
    for a in range(0, len(tn), 2):
        part = tn[a:a + 2]
        for nm, res in zip(part, score(model, text, [tails[n][1] for n in part], sep, dev)):
            start, tail = tails[nm]
            full = scored[nm]["nll"][start + 1:start + len(tail)]
            alone = res["nll"][1:len(tail)]
            hf, ha = full > args.hard, alone > args.hard
            result["alone"][nm] = {"from": start, "tokens": len(tail), "loss_in_row": round(float(full.mean()), 3),
                                   "loss_alone": round(float(alone.mean()), 3), "bad_in_row": int(hf.sum()),
                                   "bad_alone": int(ha.sum()), "bad_in_both": int((hf & ha).sum())}
            x = result["alone"][nm]
            say(f"  {nm}: tokens {start}..{start + len(tail)}: loss in the row {x['loss_in_row']} / alone {x['loss_alone']};"
                f" bad in the row {x['bad_in_row']} / alone {x['bad_alone']} / in both {x['bad_in_both']}")

    if dev.type == "cuda":
        say("\n===== the same rows under autocast (as the teacher runs during training)")
        part = names[:2]
        for nm, res in zip(part, score(model, text, [rows[n] for n in part], sep, dev, autocast=True)):
            a, b = scored[nm]["nll"][1:], res["nll"][1:]
            result["autocast"][nm] = {"loss_plain": round(float(a.mean()), 3), "loss_autocast": round(float(b.mean()), 3),
                                      "bad_plain": int((a > args.hard).sum()), "bad_autocast": int((b > args.hard).sum()),
                                      "largest_difference": round(float((a - b).abs().max()), 3)}
            say(f"  {nm}: {result['autocast'][nm]}")

    (out / "teacher_positions.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    say(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
