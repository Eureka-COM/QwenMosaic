"""Evaluate a MosaicQwen against the original model it was converted from.

    python scripts/mq_eval.py --model Qwen/Qwen3.5-0.8B --delta runs/s2/delta \
        --data data/cosmopedia-100k --out runs/eval

Three parts, all teacher (original model) versus student (converted model):

1. metrics.json    held-out rows: KL(teacher || student), next-token loss of both, perplexity
                   ratio, top-1 agreement, per-layer teacher-forced error (mq_train.evaluate)
2. recall.json     a fact ("The access code for <name> is <5 digits>.") is placed `distance`
                   tokens before a question about it; the answer must be produced from the
                   context. Exact match of the code, per distance, for both models. Beyond
                   the short window only the long memory can carry the code.
3. samples.json    greedy continuations of a few prompts (chat template), side by side

Without --delta the freshly converted, untrained model is evaluated (the starting point).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import load_mosaic_qwen  # noqa: E402
from mmllm.mq_data import RowPacker, load_source  # noqa: E402
from mmllm.mq_train import Budget, Progress, Teacher, evaluate, load_base, pick_device, pick_dtype, write_json  # noqa: E402

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "日本で一番高い山は何ですか。標高も答えてください。",
    "List three differences between a list and a tuple in Python.",
    "次の文を英語に訳してください：「記憶は予測のためにある。」",
]
NAMES = ["Arlen", "Bexley", "Corwin", "Dalia", "Elric", "Fenna", "Garrick", "Hollis", "Imara", "Joss",
         "Kestrel", "Lorne", "Maren", "Nolan", "Oriel", "Perrin", "Quill", "Rowan", "Sable", "Tamsin"]


def _generate(model, tok, ids, max_new_tokens):
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def recall_cases(tok, filler_ids, distances, n, seed):
    """n cases per distance: (prompt ids, code). `distance` = tokens between the fact and the question."""
    rng = random.Random(seed)
    cases = []
    for d in distances:
        total = d + 320                                   # every prompt of one distance has the same length
        for j in range(n):
            name = rng.choice(NAMES)
            code = "".join(rng.choice("0123456789") for _ in range(5))
            fact = tok(f"\nNote: The access code for {name} is {code}.\n", add_special_tokens=False).input_ids
            q = tok(f"\nQuestion: What is the access code for {name}?\nAnswer: The access code for {name} is",
                    add_special_tokens=False).input_ids
            lead = total - d - len(fact) - len(q)
            start = rng.randrange(0, max(1, len(filler_ids) - d - lead - 8))
            before = filler_ids[start:start + lead]
            between = filler_ids[start + lead:start + lead + d]
            cases.append({"distance": d, "name": name, "code": code, "ids": before + fact + between + q})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--delta", default=None)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--eval-rows", type=int, default=16)
    ap.add_argument("--recall-distances", default="32,512,1500")
    ap.add_argument("--recall-n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--docs", type=int, default=20_000)
    ap.add_argument("--max-minutes", type=float, default=None)
    # conversion settings, used only without --delta
    ap.add_argument("--short-heads", type=int, default=None)
    ap.add_argument("--long-heads", type=int, default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    out = Path(args.out)
    prog = Progress(out)
    budget = Budget(args.max_minutes)

    if args.delta is None and (args.short_heads or args.long_heads):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from mmllm.mosaic_qwen import MQConfig, convert_to_mosaic_qwen
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
        kw = {k: v for k, v in (("n_short_heads", args.short_heads), ("n_long_heads", args.long_heads)) if v}
        mq = convert_to_mosaic_qwen(model, MQConfig.from_upstream_mmv2(base_model=args.model, **kw), keep_teacher=True)
    else:
        mq, tok = load_mosaic_qwen(args.model, args.delta, dtype=dtype, device=device, keep_teacher=True)
    model = mq.model
    # A delta with base weights (stage 2 with --train all): the originals kept by the conversion share those
    # weights and no longer run the original model. The teacher is then a separate copy of the base model.
    separate = None
    if mq.base_changed:
        mq.drop_teacher()
        separate, _ = load_base(args.model, dtype, device)
    teacher = Teacher(mq, separate=separate)

    def teacher_generate(ids, n):
        if separate is not None:
            return _generate(separate, tok, ids, n)
        with mq.teacher():
            return _generate(model, tok, ids, n)
    prog.log("start", 0, layers=mq.cfg.layers, delta=args.delta, backends=official.backends(device),
             attention_parameters_in_student=len(mq.attention_parameters()))

    sources = [load_source(s, args.docs) for s in args.data]
    packer = RowPacker(tok, sources, args.seq_len, seed=args.seed)
    rows = packer.eval_rows(args.eval_rows)
    metrics = evaluate(mq, teacher, rows, device, layer_mse=separate is None)
    write_json(out / "metrics.json", {"delta": args.delta, "seq_len": args.seq_len, "rows": args.eval_rows,
                                      "teacher": "separate copy" if separate is not None else "shared weights", **metrics})
    prog.log("metrics", 1, **{k: v for k, v in metrics.items() if k != "rel_mse"})

    # recall across distances
    distances = [int(x) for x in args.recall_distances.split(",") if x]
    filler = packer.eval_rows(max(4, (max(distances) + 600) // args.seq_len + 2), seed=99).reshape(-1).tolist()
    filler = [t for t in filler if t != packer.sep_id]
    recall = {"cases": [], "summary": {}}
    for case in recall_cases(tok, filler, distances, args.recall_n, args.seed):
        if budget.exhausted():
            break
        ids = torch.tensor([case["ids"]], device=device)
        with torch.no_grad():
            t_out = teacher_generate(ids, 8)
            s_out = _generate(model, tok, ids, 8)
        recall["cases"].append({"distance": case["distance"], "code": case["code"], "prompt_tokens": len(case["ids"]),
                                "teacher": t_out, "student": s_out,
                                "teacher_ok": case["code"] in t_out, "student_ok": case["code"] in s_out})
    for d in distances:
        cs = [c for c in recall["cases"] if c["distance"] == d]
        if cs:
            recall["summary"][str(d)] = {"n": len(cs), "teacher": round(sum(c["teacher_ok"] for c in cs) / len(cs), 3),
                                         "student": round(sum(c["student_ok"] for c in cs) / len(cs), 3)}
    write_json(out / "recall.json", recall)
    prog.log("recall", 2, **recall["summary"])

    samples = []
    if not args.no_samples:
        for p in PROMPTS:
            if budget.exhausted():
                break
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            with torch.no_grad():
                t_out = teacher_generate(ids, args.max_new_tokens)
                s_out = _generate(model, tok, ids, args.max_new_tokens)
            samples.append({"prompt": p, "teacher": t_out, "student": s_out})
        write_json(out / "samples.json", samples)
    write_json(out / "result.json", {"status": "done", "metrics": metrics, "recall": recall["summary"],
                                     "samples": len(samples), "backends": official.backends(device)})
    prog.log("done", 3, status="done")


if __name__ == "__main__":
    main()
