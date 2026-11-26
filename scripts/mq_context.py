"""Does a model learn from related documents placed before a target document?
(docs/plan-mosaic-qwen.md section 10, step 2: the context-growth evaluation)

    python scripts/mq_context.py --model Qwen/Qwen3.5-0.8B --data fresh/cosmopedia@4 ... \
        --student all=runs/s3-all/delta --student docs=runs/s2-docs/delta --out runs/context

The documents of --data are grouped by content (the same k-means as the training rows of
`--rows cluster`). They must be documents no training run has read: give files other than
the training files. A case is a target document and the related documents of its cluster.
For k = 0, 1, 2, 4, 8, max documents placed before the target, the next-token loss on the
target (nats per token, its first token left out) is measured for

    teacher            the original model
    NAME               each --student NAME=DELTA
    NAME/no-cluster    the same student with its cluster memory switched off (if it has one)

and once more with unrelated documents (other clusters of the same source) instead of the
related ones, as many as the largest k.

    G = loss(k = 0) - loss(k = max)      what the related documents were worth

Writes <out>/context.json: per model the mean loss at every k, G with a bootstrap interval,
G of the unrelated documents, G relative to the teacher, the paired differences of G between
models; and every case's numbers. Rows are padded on the right to a multiple of --pad-to
tokens (the models are causal: the padding changes nothing before it) so that the compiled
attention sees a handful of lengths instead of one per row.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import load_mosaic_qwen  # noqa: E402
from mmllm.mq_data import ClusterRowPacker, context_row, load_source  # noqa: E402
from mmllm.mq_train import Budget, Progress, hidden_states, pick_device, pick_dtype, write_json  # noqa: E402


@torch.no_grad()
def target_loss(model, ids: list[int], start: int, device, pad_id: int, pad_to: int, chunk: int = 256) -> float:
    """Mean next-token loss (nats) on ids[start + 1:], every token before it in the context."""
    n = len(ids)
    padded = ids + [pad_id] * (-n % pad_to)
    h = hidden_states(model, torch.tensor([padded], device=device))[0, start:n - 1]
    tgt = torch.tensor(ids[start + 1:], device=device)
    total = 0.0
    for c in range(0, tgt.shape[0], chunk):
        total += F.cross_entropy(model.lm_head(h[c:c + chunk]).float(), tgt[c:c + chunk], reduction="sum").item()
    return total / tgt.shape[0]


def bootstrap(values: list[float], seed: int = 0, n: int = 10_000) -> dict:
    """Mean and 95% percentile interval of the mean, resampling the cases."""
    t = torch.tensor(values, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(values), (n, len(values)), generator=g)
    means = t[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"mean": round(t.mean().item(), 5), "lo": round(lo, 5), "hi": round(hi, 5)}


def summarize(cases: list[dict], models: list[str], ks: list[str]) -> dict:
    out = {"models": {}, "pairs": {}}
    gain = {m: [c["loss"][m]["0"] - c["loss"][m]["max"] for c in cases] for m in models}
    for m in models:
        out["models"][m] = {
            "loss": {k: round(sum(c["loss"][m][k] for c in cases) / len(cases), 5) for k in ks},
            "G": bootstrap(gain[m]),
            "G_unrelated": bootstrap([c["loss"][m]["0"] - c["loss"][m]["unrelated"] for c in cases]),
            "G_by_source": {s: round(sum(g for g, c in zip(gain[m], cases) if c["source"] == s)
                                     / max(1, sum(c["source"] == s for c in cases)), 5)
                            for s in sorted({c["source"] for c in cases})},
        }
    t = out["models"].get("teacher", {}).get("G", {}).get("mean")
    for m in models:
        if t:
            out["models"][m]["G_over_teacher"] = round(out["models"][m]["G"]["mean"] / t, 4)
        for o in models:
            if o != m:
                out["pairs"][f"{m} - {o}"] = bootstrap([a - b for a, b in zip(gain[m], gain[o])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--student", action="append", required=True, help="NAME=DELTA_DIR, repeatable")
    ap.add_argument("--data", action="append", required=True, help="documents no training run has read")
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=6000, help="documents read per source")
    ap.add_argument("--seq-len", type=int, default=16384, help="longest row (context + target)")
    ap.add_argument("--n-per-source", type=int, default=20)
    ap.add_argument("--ks", default="0,1,2,4,8")
    ap.add_argument("--min-context-docs", type=int, default=8)
    ap.add_argument("--target-min", type=int, default=256)
    ap.add_argument("--target-max", type=int, default=1024)
    ap.add_argument("--context-doc-tokens", type=int, default=2048)
    ap.add_argument("--cluster-size", type=int, default=60)
    ap.add_argument("--cluster-cache", default=None)
    ap.add_argument("--pad-to", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    out = Path(args.out)
    prog = Progress(out)
    budget = Budget(args.max_minutes)
    students = [s.split("=", 1) for s in args.student]
    assert all(len(s) == 2 for s in students), "--student NAME=DELTA_DIR"
    ks = [int(k) for k in args.ks.split(",") if k]
    assert ks[0] == 0 and max(ks) <= args.min_context_docs, "--ks starts at 0 and stays within --min-context-docs"

    cases, packer, pad_id, models = None, None, None, []
    for si, (name, delta) in enumerate(students):
        mq, tok = load_mosaic_qwen(args.model, delta, dtype=dtype, device=device, keep_teacher=True)
        model = mq.model.eval()
        mq.fixed_skip()
        if cases is None:
            sources = [load_source(s, args.docs) for s in args.data]
            packer = ClusterRowPacker(tok, sources, args.seq_len, seed=args.seed, cluster_size=args.cluster_size,
                                      holdout_frac=1.0, cache_dir=args.cluster_cache)
            # padding that does not end a document: the target document simply goes on
            pad_id = next(t for t in tok("\n\n the of", add_special_tokens=False).input_ids + [0, 1, 2]
                          if t != packer.sep_id)
            cases = packer.context_cases(args.n_per_source, args.seq_len, (args.target_min, args.target_max),
                                         args.context_doc_tokens, args.min_context_docs, seed=args.seed)
            for c in cases:
                c["loss"] = {}
            prog.log("start", 0, cases=len(cases), data=packer.describe(), backends=official.backends(device),
                     students=[n for n, _ in students])
        variants = [(name, None)]
        if mq.cfg.n_cluster_heads:
            variants.append((f"{name}/no-cluster", mq.without_cluster))
        if si == 0:
            variants.insert(0, ("teacher", mq.teacher))
        for label, ctx in variants:
            models.append(label)
            for ci, c in enumerate(cases):
                rows = {str(k): context_row(c, k) for k in ks}
                rows["max"] = context_row(c, len(c["related"]))
                rows["unrelated"] = context_row(c, len(c["unrelated"]), unrelated=True)
                res = {}
                for k, (ids, start) in rows.items():
                    if ctx is None:
                        res[k] = target_loss(model, ids, start, device, pad_id, args.pad_to)
                    else:
                        with ctx():
                            res[k] = target_loss(model, ids, start, device, pad_id, args.pad_to)
                c["loss"][label] = {k: round(v, 5) for k, v in res.items()}
                if budget.exhausted():
                    raise SystemExit("out of time: no summary written")
            prog.log("model", len(models), model=label,
                     loss_k0=round(sum(c["loss"][label]["0"] for c in cases) / len(cases), 4),
                     loss_max=round(sum(c["loss"][label]["max"] for c in cases) / len(cases), 4))
        del mq, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    keys = [str(k) for k in ks] + ["max", "unrelated"]
    summary = summarize(cases, models, keys)
    rng = random.Random(0)
    write_json(out / "context.json", {
        "args": vars(args), "students": dict(students), "n_cases": len(cases), "ks": keys, **summary,
        "context_docs_max": {"mean": round(sum(len(c["related"]) for c in cases) / len(cases), 2),
                             "min": min(len(c["related"]) for c in cases), "max": max(len(c["related"]) for c in cases)},
        "cases": [{"source": c["source"], "cluster": c["cluster"], "doc": c["doc"], "target_tokens": len(c["target"]),
                   "related_docs": len(c["related"]), "related_tokens": sum(len(d) for d in c["related"]),
                   "unrelated_docs": len(c["unrelated"]), "loss": c["loss"]} for c in cases],
        "example_target": packer.tok.decode(rng.choice(cases)["target"][:60]) if cases else None,
    })
    prog.log("done", len(models), **{m: summary["models"][m]["G"] for m in models})


if __name__ == "__main__":
    main()
