"""Before training on rows of related documents: build the clusters on the real data and look.

    python scripts/mq_cluster_check.py --model /content/models/Qwen3.5-0.8B --data ... --docs 40000 \
        --seq-len 16384 --max-doc 8192 --out runs/clusters [--fresh-data ... --fresh-docs 6000]

Writes <out>/clusters.json: how long the clustering took, the clusters of every source (how
many, their sizes, how many are held out), a few rows as the training will see them (the
first characters of each document of the row, and how many documents a row holds), and, with
--fresh-data, how many cases the context-growth evaluation finds there (scripts/mq_context.py).
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm.mq_data import ClusterRowPacker, load_source  # noqa: E402
from mmllm.mq_train import Progress, write_json  # noqa: E402


def row_documents(packer, rng, source, held_out=False, chars=90):
    """One training row of `source` as (number of documents, tokens, the head of each document)."""
    heads, n_tok = [], 0
    for i in packer._cluster_docs(rng, source, held_out):
        doc = source.docs[i]
        ids = packer._encode(source, doc)
        text = doc if isinstance(doc, str) else " ".join(str(m.get("content", m.get("value", ""))) for m in doc)
        heads.append(text[:chars].replace("\n", " "))
        n_tok += len(ids) + 1
        if n_tok >= packer.seq_len:
            break
    return {"documents": len(heads), "tokens": min(n_tok, packer.seq_len), "heads": heads}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=40_000)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--max-doc", type=int, default=8192)
    ap.add_argument("--cluster-size", type=int, default=60)
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--cluster-cache", default=None)
    ap.add_argument("--rows", type=int, default=3, help="example rows per source")
    ap.add_argument("--fresh-data", action="append", default=[])
    ap.add_argument("--fresh-docs", type=int, default=6000)
    ap.add_argument("--n-per-source", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    prog = Progress(out)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    t0 = time.time()
    sources = [load_source(s, args.docs) for s in args.data]
    t1 = time.time()
    packer = ClusterRowPacker(tok, sources, args.seq_len, max_doc=args.max_doc, cluster_size=args.cluster_size,
                              holdout_frac=args.holdout_frac, cache_dir=args.cluster_cache)
    t2 = time.time()
    res = {"args": vars(args), "load_s": round(t1 - t0, 1), "cluster_s": round(t2 - t1, 1), "data": packer.describe(),
           "rows": {}}
    prog.log("clusters", 1, cluster_s=res["cluster_s"], sources=[(e["name"], e["clusters"]) for e in res["data"]["sources"]])
    rng = random.Random(0)
    for s in sources:
        rows = [row_documents(packer, rng, s) for _ in range(max(args.rows, 20))]
        res["rows"][s.name] = {"documents_per_row_mean": round(sum(r["documents"] for r in rows) / len(rows), 1),
                               "documents_per_row_min": min(r["documents"] for r in rows),
                               "examples": rows[: args.rows]}
    t3 = time.time()
    packer.batch(4)
    res["batch_of_4_rows_s"] = round(time.time() - t3, 2)

    if args.fresh_data:
        t4 = time.time()
        fresh = [load_source(s, args.fresh_docs) for s in args.fresh_data]
        fp = ClusterRowPacker(tok, fresh, args.seq_len, cluster_size=args.cluster_size, holdout_frac=1.0,
                              cache_dir=args.cluster_cache)
        cases = fp.context_cases(args.n_per_source, args.seq_len)
        by = {}
        for c in cases:
            by.setdefault(c["source"], []).append(c)
        res["context_cases"] = {
            "seconds": round(time.time() - t4, 1), "data": fp.describe(),
            "by_source": {k: {"cases": len(v),
                              "related_docs_mean": round(sum(len(c["related"]) for c in v) / len(v), 1),
                              "related_tokens_mean": round(sum(sum(map(len, c["related"])) for c in v) / len(v)),
                              "target_tokens_mean": round(sum(len(c["target"]) for c in v) / len(v)),
                              "example": {"target": tok.decode(v[0]["target"][:40]),
                                          "related": [tok.decode(d[:24]) for d in v[0]["related"][:4]],
                                          "unrelated": [tok.decode(d[:24]) for d in v[0]["unrelated"][:2]]}}
                          for k, v in by.items()}}
    write_json(out / "clusters.json", res)
    prog.log("done", 2, cluster_s=res["cluster_s"])


if __name__ == "__main__":
    main()
