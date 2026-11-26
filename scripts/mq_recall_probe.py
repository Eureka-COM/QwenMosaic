#!/usr/bin/env python3
"""Why did the 27B miss the recall test? The same question asked in four ways, on the teacher and
on the weights of every stage, with the probabilities at the answer position instead of just hit/miss.

The final evaluation (scripts/mq_eval.py) scored 0/20 at 6,000 tokens. Rebuilding its prompts showed
that every miss has a conversation before the question and that no 6,000-token prompt was free of one
(docs/results/mq-27b-recall-review-2026-11-12.md). This script separates the candidates:

  mixed   the evaluation's own 80 prompts (filler with conversations), now with probabilities
  plain   the same construction with filler that has no conversation in it: does the long memory
          itself reach 1,500 / 3,000 / 6,000 tokens back?
  chat    the real use: the note and the question inside a user turn, the answer in the assistant
          turn (the template's own framing, thinking off, the answer begun for the model)
  path    the last-position logits of a long prompt three ways: full sequence with the official
          FlexAttention, full sequence with the dense read, and the generation (cache) path

Nothing is generated for the first three: the answer is appended and one forward pass gives, at the
answer position, the probability of the first token of the code, of <|im_end|>, the top token, and
whether every token of the code is the top token (that is exactly "greedy decoding writes the code").
The weights are swapped in one after the other (--delta name=dir, in training order), the base model
is loaded once.

    python scripts/mq_recall_probe.py --model /content/models/Qwen3.8-27B \
        --delta s1=/content/prev/s1/delta --delta s2=/content/prev/s2/delta --delta s3=/content/prev/s3/delta \
        --data ... --docs 8000 --out /content/out/probe
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mq_eval  # noqa: E402  (the evaluation's own prompt builder and names)
from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import _text_model, load_mosaic_qwen  # noqa: E402
from mmllm.mq_data import RowPacker, load_source  # noqa: E402
from mmllm.mq_train import Budget, Progress, pick_device, pick_dtype, write_json  # noqa: E402

PAD_TAIL = 8          # every sequence of a group is padded to prompt + PAD_TAIL tokens: one shape per group


def filler_of(packer: RowPacker, rows: int) -> list[int]:
    ids = packer.eval_rows(rows, seed=99).reshape(-1).tolist()
    return [t for t in ids if t != packer.sep_id]


def state_before(tok, ids: list[int], upto: int) -> str:
    """What the model has seen before position `upto`, as far as conversations go."""
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    state = "no conversation yet"
    for j in range(upto):
        if ids[j] == im_start:
            state = "inside a turn: " + tok.decode(ids[j + 1:j + 2]).strip()
        elif ids[j] == im_end:
            state = "after a turn ended"
    return state


def chat_cases(tok, filler: list[int], distances: list[int], n: int, seed: int) -> list[dict]:
    """The note and the question inside ONE user turn, the answer begun in the assistant turn.
    Token ids are spliced so that every prompt of a distance has the same length."""
    mark = "@@MQ_SPLIT@@"
    text = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    pre, post = text.split(mark)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids       # noqa: E731
    P = enc(pre)
    rng = random.Random(seed)
    cases = []
    for d in distances:
        total = d + 420
        for _ in range(n):
            name = rng.choice(mq_eval.NAMES)
            code = "".join(rng.choice("0123456789") for _ in range(5))
            fact = enc(f"\nNote: The access code for {name} is {code}.\n")
            q = enc(f"\n\nQuestion: What is the access code for {name}?")
            turn, begun = enc(post), enc(f"The access code for {name} is")
            lead = total - len(P) - len(fact) - d - len(q) - len(turn) - len(begun)
            start = rng.randrange(0, max(1, len(filler) - d - lead - 8))
            free = P + filler[start:start + lead] + fact + filler[start + lead:start + lead + d] + q + turn
            cases.append({"set": "chat", "distance": d, "name": name, "code": code, "ids": free + begun, "free_ids": free,
                          "before_question": "the question is in a user turn"})
    return cases


def build_cases(tok, sources, args) -> list[dict]:
    mixed_filler = filler_of(RowPacker(tok, sources, args.seq_len, seed=args.seed), 5)
    plain_filler = filler_of(RowPacker(tok, [s for s in sources if not s.is_chat], args.seq_len, seed=args.seed),
                             max(5, (max(args.plain_distances) + 600) // args.seq_len + 2))
    cases = []
    for c in mq_eval.recall_cases(tok, mixed_filler, args.mixed_distances, args.n_mixed, args.seed):
        q = tok(f"\nQuestion: What is the access code for {c['name']}?\nAnswer: The access code for {c['name']} is",
                add_special_tokens=False).input_ids
        cases.append({"set": "mixed", **c, "before_question": state_before(tok, c["ids"], len(c["ids"]) - len(q))})
    for c in mq_eval.recall_cases(tok, plain_filler, args.plain_distances, args.n_plain, args.seed + 1):
        cases.append({"set": "plain", **c, "before_question": "no conversation yet"})
    cases += chat_cases(tok, plain_filler, args.chat_distances, args.n_chat, args.seed + 2)
    for c in cases:
        c["answer"] = tok(f" {c['code']}", add_special_tokens=False).input_ids
    return cases


@torch.no_grad()
def score(model, text, tok, group: list[dict], device, bsz: int) -> list[dict]:
    """One forward pass per batch over prompt + answer. Returns one record per case."""
    im_end, eos = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
    P = len(group[0]["ids"])
    assert all(len(c["ids"]) == P for c in group)
    out = []
    for a in range(0, len(group), bsz):
        part = group[a:a + bsz]
        x = torch.full((len(part), P + PAD_TAIL), eos, dtype=torch.long)
        for b, c in enumerate(part):
            seq = c["ids"] + c["answer"][:PAD_TAIL]
            x[b, :len(seq)] = torch.tensor(seq)
        h = text(input_ids=x.to(device), use_cache=False).last_hidden_state[:, P - 1:]
        logp = F.log_softmax(model.lm_head(h).float(), -1).cpu()           # (b, PAD_TAIL + 1, V)
        for b, c in enumerate(part):
            ans = c["answer"][:PAD_TAIL]
            # the code is written as a space token and then one token per digit: the first position decides
            # "answer or end the turn", the first DIGIT (given the space) is the retrieval itself
            k0 = next(k for k, t in enumerate(ans) if any(ch.isdigit() for ch in tok.decode([t])))
            first, digit = logp[b, 0], logp[b, k0]
            top = int(first.argmax())
            hits = [int(logp[b, k].argmax()) == ans[k] for k in range(len(ans))]
            out.append({"p_first": round(float(first[ans[0]].exp()), 5), "p_end": round(float(first[im_end].exp()), 5),
                        "p_eos": round(float(first[eos].exp()), 5), "top": tok.decode([top]), "p_top": round(float(first[top].exp()), 4),
                        "p_digit": round(float(digit[ans[k0]].exp()), 5), "rank_digit": int((digit > digit[ans[k0]]).sum()),
                        "digits_right": bool(all(hits[k0:])), "writes_the_code": bool(all(hits)),
                        "logp_digits": round(float(sum(logp[b, k, ans[k]] for k in range(k0, len(ans)))), 3)})
        del h, logp
    return out


@torch.no_grad()
def last_logits(model, text, ids: torch.Tensor, cache: bool) -> torch.Tensor:
    h = text(input_ids=ids, use_cache=cache).last_hidden_state[:, -1]
    return model.lm_head(h).float()[0].cpu()


def path_check(mq, model, text, tok, cases: list[dict], device, last: str) -> list[dict]:
    """Full sequence with the default backend, full sequence with the dense read, generation path."""
    res = []
    for c in cases:
        ids = torch.tensor([c["ids"]], device=device)
        rec = {"set": c["set"], "distance": c["distance"], "tokens": len(c["ids"])}
        ref = last_logits(model, text, ids, cache=False)
        variants = {"generation path": lambda: last_logits(model, text, ids, cache=True)}
        if device.type == "cuda":
            def dense():
                official.set_backends(attention="dense")
                try:
                    return last_logits(model, text, ids, cache=False)
                finally:
                    official.set_backends(attention="auto")
            variants["dense read"] = dense
        for name, fn in variants.items():
            try:
                got = fn()
                rec[name] = {"max_abs_diff": round(float((got - ref).abs().max()), 4), "same_top": bool(got.argmax() == ref.argmax()),
                             "kl": round(float(F.kl_div(F.log_softmax(got, -1), F.log_softmax(ref, -1), log_target=True, reduction="sum")), 5)}
            except Exception as e:                           # noqa: BLE001
                rec[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        rec["logit_scale"] = round(float(ref.abs().mean()), 3)
        with torch.no_grad():
            gen = mq_eval._generate(model, tok, ids, 8)
        rec["generated"], rec["generated_ok"] = gen, c["code"] in gen
        rec["forward_pass_says"] = c["scores"][last]["writes_the_code"] if last in c.get("scores", {}) else None
        res.append(rec)
    return res


def summarize(cases: list[dict], models: list[str]) -> dict:
    out = {}
    keys = sorted({(c["set"], c["distance"]) for c in cases}, key=lambda k: (k[0], k[1]))
    for s, d in keys:
        cs = [c for c in cases if c["set"] == s and c["distance"] == d]
        row = {"n": len(cs)}
        for m in models:
            rs = [c["scores"][m] for c in cs if m in c.get("scores", {})]
            if rs:
                mean = lambda k: round(sum(r[k] for r in rs) / len(rs), 3)      # noqa: E731
                row[m] = {"writes_the_code": sum(r["writes_the_code"] for r in rs), "digits_right": sum(r["digits_right"] for r in rs),
                          "p_digit": mean("p_digit"), "p_first": mean("p_first"), "p_end": mean("p_end")}
        out[f"{s}/{d}"] = row
    return out


def by_context(cases: list[dict], models: list[str]) -> dict:
    out = {}
    for c in cases:
        if c["set"] != "mixed":
            continue
        key = f"{c['before_question']} / {c['distance']}"
        row = out.setdefault(key, {"n": 0, **{m: {"writes_the_code": 0, "digits_right": 0, "p_digit": 0.0, "p_end": 0.0} for m in models}})
        row["n"] += 1
        for m in models:
            r = c["scores"].get(m)
            if r:
                for k in ("writes_the_code", "digits_right", "p_digit", "p_end"):
                    row[m][k] += r[k]
    for row in out.values():
        for m in models:
            row[m]["p_digit"] = round(row[m]["p_digit"] / row["n"], 3); row[m]["p_end"] = round(row[m]["p_end"] / row["n"], 3)
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--delta", action="append", required=True, help="name=dir, in training order (the first one sets the configuration)")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--docs", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed-distances", default="32,512,1500,6000")
    ap.add_argument("--plain-distances", default="32,512,1500,3000,6000")
    ap.add_argument("--chat-distances", default="512,1500,6000")
    ap.add_argument("--n-mixed", type=int, default=20)
    ap.add_argument("--n-plain", type=int, default=10)
    ap.add_argument("--n-chat", type=int, default=8)
    ap.add_argument("--n-generate", type=int, default=3, help="chat prompts per distance that are also generated freely (last weights and teacher)")
    ap.add_argument("--n-path", type=int, default=2, help="prompts per set at the longest distance for the path check")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    for k in ("mixed_distances", "plain_distances", "chat_distances"):
        setattr(args, k, [int(x) for x in getattr(args, k).split(",") if x])
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    out = Path(args.out)
    prog = Progress(out)
    budget = Budget(args.max_minutes)
    deltas = [d.split("=", 1) for d in args.delta]

    mq, tok = load_mosaic_qwen(args.model, deltas[0][1], dtype=dtype, device=device, keep_teacher=True)
    model, text = mq.model, _text_model(mq.model)
    mq.fixed_skip()
    sources = [load_source(s, args.docs) for s in args.data]
    cases = build_cases(tok, sources, args)
    for c in cases:
        c["scores"] = {}
    groups: dict = {}
    for c in cases:
        groups.setdefault((c["set"], c["distance"]), []).append(c)
    prog.log("start", 0, layers=mq.cfg.layers, deltas=[n for n, _ in deltas], cases=len(cases),
             groups={f"{s}/{d}": [len(g), len(g[0]["ids"])] for (s, d), g in groups.items()}, backends=official.backends(device))

    def bsz_for(n_tokens: int) -> int:
        return 4 if n_tokens <= 2200 else (2 if n_tokens <= 7000 else 1)

    def run(name: str) -> None:
        for (s, d), g in groups.items():
            if budget.exhausted():
                prog.log("out of time", 0, model=name, group=f"{s}/{d}")
                return
            t = time.time()
            for c, r in zip(g, score(model, text, tok, g, device, bsz_for(len(g[0]["ids"])))):
                c["scores"][name] = r
            prog.log("scored", 0, model=name, group=f"{s}/{d}", seconds=round(time.time() - t, 1),
                     writes_the_code=sum(c["scores"][name]["writes_the_code"] for c in g),
                     digits_right=sum(c["scores"][name]["digits_right"] for c in g), n=len(g),
                     p_digit=round(sum(c["scores"][name]["p_digit"] for c in g) / len(g), 3),
                     p_end=round(sum(c["scores"][name]["p_end"] for c in g) / len(g), 3))

    def save(step: str) -> None:
        models = ["teacher"] + [n for n, _ in deltas]
        keep = [{k: v for k, v in c.items() if k not in ("ids", "free_ids", "answer")} | {"prompt_tokens": len(c["ids"])} for c in cases]
        write_json(out / "probe.json", {"step": step, "summary": summarize(cases, models), "mixed_by_context": by_context(cases, models),
                                        "path_check": path, "generated": generated, "cases": keep})

    path, generated = [], []
    with mq.teacher():
        run("teacher")
    save("teacher")
    for i, (name, d) in enumerate(deltas):
        if i:
            mq.load_delta(d)
        mq.fixed_skip()
        run(name)
        save(name)

    # the last weights: the generation path against the full-sequence pass, and a few free answers
    if not budget.exhausted():
        longest = {s: max(d for (s2, d) in groups if s2 == s) for s in ("mixed", "plain")}
        probe = [c for s, d in longest.items() for c in groups[(s, d)][:args.n_path]]
        probe += groups[("plain", sorted(args.plain_distances)[len(args.plain_distances) // 2])][:1]
        path = path_check(mq, model, text, tok, probe, device, deltas[-1][0])
        prog.log("path", 0, checks=path)
        save("path")
    for (s, d), g in groups.items():
        if s != "chat" or budget.exhausted():
            continue
        for c in g[:args.n_generate]:
            ids = torch.tensor([c["free_ids"]], device=device)
            with torch.no_grad():
                with mq.teacher():
                    t_out = mq_eval._generate(model, tok, ids, 24)
                s_out = mq_eval._generate(model, tok, ids, 24)
            generated.append({"distance": d, "code": c["code"], "teacher": t_out, "student": s_out,
                              "teacher_ok": c["code"] in t_out, "student_ok": c["code"] in s_out})
    save("done")
    models = ["teacher"] + [n for n, _ in deltas]
    summary = summarize(cases, models)
    write_json(out / "result.json", {"status": "done", "summary": summary, "mixed_by_context": by_context(cases, models),
                                     "path_check": path, "generated": generated, "backends": official.backends(device)})
    prog.log("done", 1, status="done", summary=summary, generated=[{k: g[k] for k in ("distance", "teacher_ok", "student_ok")} for g in generated])


if __name__ == "__main__":
    main()
