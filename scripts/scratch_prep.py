"""Tokenize the training text once, for every model trained from scratch on it.

    python scripts/scratch_prep.py --tokenizer TOKENIZER.json --train A.parquet [--train B.parquet ...] \
        --eval C.parquet --train-tokens 1.5e9 --out /content/scratch-data

Writes into --out:
    train.bin       uint16 token stream: every document as BOS ... EOS, in file order, until
                    --train-tokens (the rows are shuffled at training time, the same for all models)
    heldout.bin     the same from the --eval file, --heldout-tokens long (never trained on)
    long_docs.npy   (--long-docs, --long-len + 1) uint16: documents of the --eval file at least
                    --long-len + 1 tokens long, one per row, from their BOS (for position_loss)
    filler.npy      uint16 tokens of --eval documents with no BOS/EOS (for the recall probe)
    meta.json       counts, files, tokenizer hash
    progress.jsonl  how far the tokenizing has got

--fetch REPO:FILE downloads a file of a Hugging Face dataset first (retried in a fresh process
when a try stalls for --fetch-timeout seconds), into --fetch-dir.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm.mq_train import Budget, Progress, write_json  # noqa: E402

BOS, EOS = 1, 2


def fetch(spec: str, dest: Path, timeout: int, tries: int, prog: Progress) -> Path:
    repo, file = spec.split(":", 1)
    target = dest / file
    if target.exists():
        return target
    code = ("from huggingface_hub import hf_hub_download; "
            f"print(hf_hub_download({repo!r}, {file!r}, repo_type='dataset', local_dir={str(dest)!r}))")
    for i in range(1, tries + 1):
        env = None
        if i % 2 == 0:                                   # every other try without the Xet transfer
            import os
            env = {**os.environ, "HF_HUB_DISABLE_XET": "1"}
        prog.log("fetch_try", i, file=spec)
        try:
            subprocess.run([sys.executable, "-c", code], check=True, timeout=timeout, env=env)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            prog.log("fetch_retry", i, file=spec, reason=type(e).__name__)
            time.sleep(20)
            continue
        if target.exists():
            prog.log("fetched", i, file=spec, bytes=target.stat().st_size)
            return target
    raise SystemExit(f"could not fetch {spec}")


def documents(path: Path, column: str, batch: int = 2048):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    for rb in f.iter_batches(batch_size=batch, columns=[column]):
        yield [t for t in rb.column(0).to_pylist() if t]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="a tokenizer.json, or REPO for REPO/tokenizer.json on the Hugging Face "
                    "hub (the upstream one: mistralai/Mistral-7B-v0.1)")
    ap.add_argument("--train", action="append", default=[])
    ap.add_argument("--eval", required=False)
    ap.add_argument("--fetch", action="append", default=[], help="REPO:FILE of a dataset to download first")
    ap.add_argument("--fetch-dir", default="/content/data/fineweb-edu")
    ap.add_argument("--fetch-timeout", type=int, default=900)
    ap.add_argument("--fetch-tries", type=int, default=8)
    ap.add_argument("--column", default="text")
    ap.add_argument("--train-tokens", type=float, default=1.5e9)
    ap.add_argument("--heldout-tokens", type=float, default=2e6)
    ap.add_argument("--long-docs", type=int, default=256)
    ap.add_argument("--long-len", type=int, default=4096)
    ap.add_argument("--filler-tokens", type=int, default=200_000)
    ap.add_argument("--out", required=True, help="the run's folder (progress); the data too unless --data-out")
    ap.add_argument("--data-out", default=None, help="where the data goes (the runner sets --out to the run's folder)")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    prog = Progress(Path(args.out))
    out = Path(args.data_out or args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_minutes)
    for spec in args.fetch:
        fetch(spec, Path(args.fetch_dir), args.fetch_timeout, args.fetch_tries, prog)
    if not Path(args.tokenizer).exists():
        from huggingface_hub import hf_hub_download
        args.tokenizer = hf_hub_download(args.tokenizer, "tokenizer.json", local_dir=str(out / "tokenizer-src"))
    if not args.train or not args.eval:
        raise SystemExit("--train and --eval are needed")

    import shutil
    from tokenizers import Tokenizer
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.tokenizer, out / "tokenizer.json")          # the data carries its tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    tok_sha = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()

    def encode_all(texts):
        return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]

    # training stream
    limit = int(args.train_tokens)
    n, docs = 0, 0
    t0 = time.time()
    with open(out / "train.bin", "wb") as fh:
        for path in args.train:
            for texts in documents(Path(path), args.column):
                for ids in encode_all(texts):
                    a = np.asarray([BOS, *ids, EOS], dtype=np.uint16)
                    fh.write(a.tobytes()); n += len(a); docs += 1
                    if n >= limit:
                        break
                if n >= limit or budget.exhausted():
                    break
                if docs % 50_000 < len(texts):
                    prog.log("train", docs, tokens=n, tok_per_s=round(n / max(time.time() - t0, 1e-6)))
            if n >= limit or budget.exhausted():
                break
    if budget.exhausted():
        raise SystemExit("out of time while tokenizing")
    prog.log("train_done", docs, tokens=n, short_of_request=n < limit)

    # held-out material, from a file no training reads
    held, long_rows, filler = [], [], []
    hn = 0
    for texts in documents(Path(args.eval), args.column):
        for ids in encode_all(texts):
            if len(long_rows) < args.long_docs and len(ids) + 1 >= args.long_len + 1:
                long_rows.append([BOS, *ids[:args.long_len]])
            elif hn < args.heldout_tokens:
                a = [BOS, *ids, EOS]; held.extend(a); hn += len(a)
            elif len(filler) < args.filler_tokens:
                filler.extend(ids)
        if len(long_rows) >= args.long_docs and hn >= args.heldout_tokens and len(filler) >= args.filler_tokens:
            break
    np.asarray(held, dtype=np.uint16).tofile(out / "heldout.bin")
    np.save(out / "long_docs.npy", np.asarray(long_rows, dtype=np.uint16))
    np.save(out / "filler.npy", np.asarray(filler[:args.filler_tokens], dtype=np.uint16))
    meta = {"train_tokens": n, "train_docs": docs, "heldout_tokens": hn, "long_docs": len(long_rows),
            "long_len": args.long_len, "filler_tokens": min(len(filler), args.filler_tokens),
            "train_files": args.train, "eval_file": args.eval, "tokenizer": str(out / "tokenizer.json"), "tokenizer_sha256": tok_sha,
            "vocab_size": tok.get_vocab_size(), "bos": BOS, "eos": EOS, "minutes": round(budget.minutes(), 1)}
    write_json(out / "meta.json", meta)
    if Path(args.out) != out:
        write_json(Path(args.out) / "meta.json", meta)
    prog.log("done", docs, **{k: v for k, v in meta.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
