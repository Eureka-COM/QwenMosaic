#!/usr/bin/env python3
"""General benchmarks: the converted model against the original, on the same questions.

docs/plan-mosaic-qwen.md §4 ("27B の判定"): after the numbers of the judgement, compare with the
original 27B on general benchmarks. Two are run here, both in the chat format the model is used in
(the tokenizer's template, thinking off):

  MMLU    multiple choice, no examples. The assistant's turn is begun with "The answer is (" and the
          next-token scores of A, B, C, D are compared: one forward pass per question and model,
          nothing generated. A fixed random sample of the 14,042 test questions.
  GSM8K   arithmetic word problems, answered by greedy generation ("... #### 42"). The first
          questions of the test set, teacher and student one after the other on each question,
          until --n-gsm or --gsm-minutes is reached.

Teacher = the original layers kept by the conversion (the same weights everywhere else, LoRA off);
student = the converted model with --delta.

    python scripts/mq_bench.py --model /content/models/Qwen3.8-27B --delta /content/prev/s2/delta \
        --mmlu /content/data/mmlu/all/test-00000-of-00001.parquet \
        --gsm /content/data/gsm8k/main/test-00000-of-00001.parquet --out /content/out/bench
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mq_eval  # noqa: E402  (greedy generation as in the evaluation)
from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import _text_model, load_mosaic_qwen  # noqa: E402
from mmllm.mq_train import Budget, Progress, pick_device, pick_dtype, write_json  # noqa: E402

LETTERS = "ABCD"
GSM_ASK = "\n\nSolve the problem step by step. End with the final answer in the form '#### <number>'."


def chat(tok, user: str, begun: str = "") -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text + begun, add_special_tokens=False).input_ids


def mmlu_prompt(tok, row: dict) -> list[int]:
    subject = str(row["subject"]).replace("_", " ")
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    user = (f"The following is a multiple choice question about {subject}.\n\n{row['question']}\n\n{options}\n\n"
            "Answer with a single letter (A, B, C or D).")
    return chat(tok, user, "The answer is (")


def number_in(text: str) -> str | None:
    """The number after the last '####'; without one, the last number in the text."""
    part = text.rsplit("####", 1)[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", part)
    if not nums:
        return None
    n = (nums[0] if "####" in text else nums[-1]).replace(",", "")
    try:
        f = float(n)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def read_rows(path: str) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


@torch.no_grad()
def letter_scores(model, text, ids: list[int], letter_ids: list[int], device) -> list[float]:
    h = text(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state[:, -1]
    return model.lm_head(h).float()[0, letter_ids].cpu().tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--mmlu", default=None)
    ap.add_argument("--gsm", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mmlu", type=int, default=1000)
    ap.add_argument("--n-gsm", type=int, default=100)
    ap.add_argument("--gsm-max-new", type=int, default=256)
    ap.add_argument("--gsm-minutes", type=float, default=22.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    out = Path(args.out)
    prog = Progress(out)
    budget = Budget(args.max_minutes)
    mq, tok = load_mosaic_qwen(args.model, args.delta, dtype=dtype, device=device, keep_teacher=True)
    model, text = mq.model, _text_model(mq.model)
    mq.fixed_skip()
    prog.log("start", 0, layers=mq.cfg.layers, delta=args.delta, backends=official.backends(device))
    result: dict = {"delta": args.delta, "backends": official.backends(device)}

    # ── MMLU ──
    if args.mmlu:
        rows = read_rows(args.mmlu)
        pick = random.Random(args.seed).sample(range(len(rows)), min(args.n_mmlu, len(rows)))
        letter_ids = [tok(x, add_special_tokens=False).input_ids for x in LETTERS]
        assert all(len(x) == 1 for x in letter_ids), "A-D are not single tokens"
        letter_ids = [x[0] for x in letter_ids]
        items, t0 = [], time.time()
        for k, i in enumerate(pick):
            if budget.exhausted():
                break
            row = rows[i]
            ids = mmlu_prompt(tok, row)
            with mq.teacher():
                ts = letter_scores(model, text, ids, letter_ids, device)
            ss = letter_scores(model, text, ids, letter_ids, device)
            gold = int(row["answer"])
            items.append({"i": i, "subject": row["subject"], "gold": gold, "teacher": ts.index(max(ts)), "student": ss.index(max(ss)),
                          "tokens": len(ids)})
            if (k + 1) % 100 == 0 or k + 1 == len(pick):
                n = len(items)
                prog.log("mmlu", k + 1, n=n, teacher=round(sum(x["teacher"] == x["gold"] for x in items) / n, 4),
                         student=round(sum(x["student"] == x["gold"] for x in items) / n, 4),
                         same_answer=round(sum(x["teacher"] == x["student"] for x in items) / n, 4),
                         seconds_per_question=round((time.time() - t0) / n, 3))
        n = len(items)
        tr, sr = [x["teacher"] == x["gold"] for x in items], [x["student"] == x["gold"] for x in items]
        result["mmlu"] = {"n": n, "sampled_from": len(rows), "teacher": round(sum(tr) / max(n, 1), 4), "student": round(sum(sr) / max(n, 1), 4),
                          "same_answer": round(sum(x["teacher"] == x["student"] for x in items) / max(n, 1), 4),
                          "both_right": sum(a and b for a, b in zip(tr, sr)), "only_teacher_right": sum(a and not b for a, b in zip(tr, sr)),
                          "only_student_right": sum(b and not a for a, b in zip(tr, sr)), "both_wrong": sum(not a and not b for a, b in zip(tr, sr))}
        write_json(out / "mmlu_items.json", items)
        write_json(out / "bench.json", result)

    # ── GSM8K ──
    if args.gsm:
        rows = read_rows(args.gsm)[:args.n_gsm]
        items, t0, new_tokens = [], time.time(), {"teacher": 0, "student": 0}
        seconds = {"teacher": 0.0, "student": 0.0}
        for k, row in enumerate(rows):
            if budget.exhausted() or (time.time() - t0) / 60 > args.gsm_minutes:
                break
            ids = torch.tensor([chat(tok, row["question"] + GSM_ASK)], device=device)
            gold = number_in(row["answer"])
            rec = {"k": k, "gold": gold, "prompt_tokens": ids.shape[1]}
            for who in ("teacher", "student"):
                t1 = time.time()
                with torch.no_grad():
                    if who == "teacher":
                        with mq.teacher():
                            txt = mq_eval._generate(model, tok, ids, args.gsm_max_new)
                    else:
                        txt = mq_eval._generate(model, tok, ids, args.gsm_max_new)
                seconds[who] += time.time() - t1
                new_tokens[who] += len(tok(txt, add_special_tokens=False).input_ids)
                rec[who] = txt
                rec[who + "_answer"] = number_in(txt)
                rec[who + "_ok"] = rec[who + "_answer"] is not None and rec[who + "_answer"] == gold
                rec[who + "_finished"] = "####" in txt
            items.append(rec)
            if (k + 1) % 5 == 0 or k + 1 == len(rows):
                n = len(items)
                prog.log("gsm", k + 1, n=n, teacher=round(sum(x["teacher_ok"] for x in items) / n, 4),
                         student=round(sum(x["student_ok"] for x in items) / n, 4),
                         tokens_per_second={w: round(new_tokens[w] / max(seconds[w], 1e-6), 1) for w in seconds},
                         minutes=round((time.time() - t0) / 60, 2))
                write_json(out / "gsm_items.json", items)
        n = len(items)
        tr, sr = [x["teacher_ok"] for x in items], [x["student_ok"] for x in items]
        result["gsm8k"] = {"n": n, "asked_for": args.n_gsm, "max_new_tokens": args.gsm_max_new,
                           "teacher": round(sum(tr) / max(n, 1), 4), "student": round(sum(sr) / max(n, 1), 4),
                           "both_right": sum(a and b for a, b in zip(tr, sr)), "only_teacher_right": sum(a and not b for a, b in zip(tr, sr)),
                           "only_student_right": sum(b and not a for a, b in zip(tr, sr)), "both_wrong": sum(not a and not b for a, b in zip(tr, sr)),
                           "finished_with_answer": {w: sum(x[w + "_finished"] for x in items) for w in ("teacher", "student")},
                           "tokens_per_second": {w: round(new_tokens[w] / max(seconds[w], 1e-6), 1) for w in seconds}}
        write_json(out / "gsm_items.json", items)
        write_json(out / "bench.json", result)

    write_json(out / "result.json", {"status": "done", **result})
    prog.log("done", 1, status="done", **{k: v for k, v in result.items() if k in ("mmlu", "gsm8k")})


if __name__ == "__main__":
    main()
