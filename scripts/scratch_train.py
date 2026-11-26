"""Train one upstream language model from scratch (Memory Mosaics v2 or the Transformer baseline).

Both families are trained by this one script with the same data, order, optimizer and schedule,
so that what differs between two runs is what the upstream configs make differ.

    python scripts/scratch_train.py --family mosaics_v2 --dim 512 --heads 4 --data /content/scratch-data \
        --tokens 1.2e9 --seq-len 4096 --bsz 4 --accum 4 --lr 6e-4 --duty 0.7 --out runs/mm

Checkpoints (--eval-at, shares of --tokens): held-out loss, the loss along long documents
(position buckets and the in-context score), recall of a fact 32-3,500 tokens back. Each is
written to progress.jsonl ("eval" events) and evals.jsonl, and the weights and the optimizer
state are saved (--store), so a stopped run resumes where it was. A failed upload to the store
is logged ("store_failed") and retried, and never ends the run: the checkpoint stays on the local disk.

--duty d (0 < d <= 1): after every optimizer step, wait (1/d - 1) times as long as the step took,
so the GPU computes about a share d of the time (lower average load and temperature).
--gpu-lock FILE: runs that name the same file take turns on the GPU (one optimizer step, or one
evaluation, at a time), and the wait after a step is taken while still holding the turn. With --duty d
the GPU as a whole then computes about a share d of the time, however many runs share it (without the
lock, two runs at d = 0.7 each keep it busy almost all the time).
--lr-from FILE: read the learning rate from a JSON file ({"lr": ...}) written by an earlier run.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm import official, scratch  # noqa: E402
from mmllm.mq_train import Budget, Progress, gpu_stats, pick_device, write_json  # noqa: E402
from mmllm.store import open_store  # noqa: E402

# Uploading a checkpoint to the store: tries, and the wait before the n-th retry (n x this).
STORE_TRIES = 3
STORE_RETRY_SEC = float(os.environ.get("MQ_STORE_RETRY_SEC", "30"))
# After giving up the GPU turn, pause this long so that a run waiting for it gets the next turn.
TURN_YIELD_SEC = 0.02


class GpuTurn:
    """Take turns on one GPU with the other runs that name the same lock file (--gpu-lock)."""

    def __init__(self, path) -> None:
        self.fd = None if path is None else os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        self.waited = 0.0

    @contextmanager
    def __call__(self):
        if self.fd is None:
            yield
            return
        t = time.time()
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.waited += time.time() - t
        try:
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            time.sleep(TURN_YIELD_SEC)


def gpu_temp() -> dict:
    try:
        return {"gpu_temp_c": torch.cuda.temperature(), "gpu_util": torch.cuda.utilization()}
    except Exception:                                    # noqa: BLE001 (no NVML: nothing to add)
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["mosaics_v2", "transformer"])
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4, help="per memory (mosaics_v2) or attention heads (transformer)")
    ap.add_argument("--head-dim", type=int, default=None, help="default: the upstream config's (128)")
    ap.add_argument("--layers", type=int, default=None, help="default: the upstream config's (24)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-from", default=None)
    ap.add_argument("--warmup-frac", type=float, default=0.05, help="upstream configs: 200 of 4000 steps")
    ap.add_argument("--min-ratio", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--data-seed", type=int, default=0, help="the order of the rows: keep it the same across models")
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--gpu-lock", default=None)
    ap.add_argument("--eval-at", default="0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--eval-rows", type=int, default=32, help="held-out rows of --seq-len")
    ap.add_argument("--pos-docs", type=int, default=128, help="long held-out documents for position_loss")
    ap.add_argument("--pos-len", type=int, default=2048)
    ap.add_argument("--icl-early", default="50,100", help="positions (1-based, inclusive) of the early window")
    ap.add_argument("--icl-late", default="1000,2000")
    ap.add_argument("--recall-distances", default="32,512,1500,3500")
    ap.add_argument("--recall-n", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--store", default=None)
    ap.add_argument("--save-every-min", type=float, default=30)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--stop-after", type=float, default=None, help="end (and save) after this many tokens, "
                    "keeping the schedule of --tokens (speed measurements)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    out = Path(args.out)
    budget = Budget(args.max_minutes)
    if args.lr_from:
        args.lr = float(json.loads(Path(args.lr_from).read_text())["lr"])
    data = Path(args.data)
    meta = json.loads((data / "meta.json").read_text())
    train = scratch.TokenFile(data / "train.bin", args.seq_len)
    held = scratch.TokenFile(data / "heldout.bin", args.seq_len)
    eval_rows = torch.stack([held.row(i) for i in range(min(args.eval_rows, held.n_rows))])
    long_docs = torch.from_numpy(np.load(data / "long_docs.npy")[:args.pos_docs, :args.pos_len + 1].astype(np.int64))
    filler = np.load(data / "filler.npy").astype(np.int64).tolist()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(meta["tokenizer"]) if Path(meta["tokenizer"]).exists() else Tokenizer.from_file(str(data / "tokenizer.json"))
    encode = lambda s: tk.encode(s, add_special_tokens=False).ids                            # noqa: E731
    distances = [int(x) for x in args.recall_distances.split(",") if x and int(x) + 160 < args.seq_len]
    icl_early = tuple(int(x) for x in args.icl_early.split(","))
    icl_late = tuple(int(x) for x in args.icl_late.split(","))
    cases = scratch.recall_cases(encode, filler, distances, args.recall_n)

    torch.manual_seed(args.model_seed)
    extra = {"head_dim": args.head_dim} if args.head_dim else {}
    if device.type == "cuda":
        official.prepare_device(device)
    model, section = scratch.build(args.family, meta["vocab_size"], args.seq_len, args.dim, args.heads, args.layers,
                                   grad_ckpt=not args.no_grad_ckpt, seed=args.model_seed, **extra)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.tok_embeddings.weight.numel() + (0 if section.get("weight_tying") else model.output.weight.numel())
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else nullcontext

    tokens_per_step = args.bsz * args.accum * args.seq_len
    total_steps = max(1, int(args.tokens // tokens_per_step))
    warmup = max(1, int(args.warmup_frac * total_steps))
    eval_steps = sorted({max(0, min(total_steps, round(float(f) * total_steps))) for f in args.eval_at.split(",") if f})
    stop_step = min(total_steps, int(args.stop_after // tokens_per_step)) if args.stop_after else total_steps

    # resume (weights, optimizer, step) from the store or the out dir
    step, done_evals = 0, []
    store = open_store(args.store)
    ck = out / "ckpt"
    if store is not None and not (ck / "state.json").exists():
        store.pull(ck)
    resumed = None
    if (ck / "state.json").exists():
        st = json.loads((ck / "state.json").read_text())
        model.load_state_dict(torch.load(ck / "model.pt", map_location=device))
        opt.load_state_dict(torch.load(ck / "optim.pt", map_location=device))
        step, done_evals = st["step"], st.get("evals_done", [])
        resumed = st
    prog = Progress(out, resume=resumed is not None)

    def save(tag: str) -> None:
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ck / "model.pt")
        torch.save(opt.state_dict(), ck / "optim.pt")
        write_json(ck / "state.json", {"step": step, "tokens": step * tokens_per_step, "tag": tag, "evals_done": done_evals,
                                       "family": args.family, "lr": args.lr})
        pushed = store is None
        for attempt in range(0 if store is None else STORE_TRIES):
            # A failed upload must not end a long run: the checkpoint is already on the local disk
            # (a restart resumes from it), and the next save tries the store again.
            try:
                store.push(ck)
                pushed = True
                break
            except Exception as e:                           # noqa: BLE001
                prog.log("store_failed", step, tag=tag, attempt=attempt + 1, error=f"{type(e).__name__}: {str(e)[:300]}")
                if attempt + 1 < STORE_TRIES:
                    time.sleep(STORE_RETRY_SEC * (attempt + 1))
        prog.log("saved", step, tokens=step * tokens_per_step, tag=tag, stored=pushed)

    def run_eval() -> dict:
        model.eval()
        t = time.time()
        res = {"tokens": step * tokens_per_step, "heldout_loss": round(scratch.heldout_loss(model, eval_rows.to(device), amp), 4)}
        pos = scratch.position_loss(model, long_docs.to(device), amp, early=icl_early, late=icl_late)
        res.update(position=pos["buckets"], icl_score=pos["icl_score"], early=pos["early"], late=pos["late"])
        res["recall"] = scratch.recall(model, cases, amp, device)
        res["eval_min"] = round((time.time() - t) / 60, 2)
        model.train()
        with (out / "evals.jsonl").open("a") as f:
            f.write(json.dumps({**res, "step": step, "icl_per_doc": pos["icl_per_doc"], "early_per_doc": pos["early_per_doc"],
                                "late_per_doc": pos["late_per_doc"]}) + "\n")
        prog.log("eval", step, **res)
        return res

    write_json(out / "run.json", {"args": vars(args), "device": str(device), "settings": section, "parameters": n_params,
                                  "embedding_parameters": n_embed, "non_embedding_parameters": n_params - n_embed,
                                  "total_steps": total_steps, "tokens_per_step": tokens_per_step, "warmup": warmup,
                                  "eval_steps": eval_steps, "data": meta, "backends": official.backends(device),
                                  "resumed": resumed})
    prog.log("start", step, family=args.family, parameters=n_params, non_embedding=n_params - n_embed,
             total_steps=total_steps, tokens_per_step=tokens_per_step, lr=args.lr, duty=args.duty, gpu_lock=args.gpu_lock,
             resumed=bool(resumed),
             backends=official.backends(device))

    model.train()
    turn = GpuTurn(args.gpu_lock)
    stopped, last_save, t_log, tok_log = "done", time.time(), time.time(), step * tokens_per_step
    wait_log = 0.0
    loss_acc, n_acc = 0.0, 0
    while True:
        if step in eval_steps and step not in done_evals:
            with turn():
                r = run_eval()
            done_evals.append(step)
            if not np.isfinite(r["heldout_loss"]):
                prog.log("nonfinite", step); stopped = "nonfinite"; break
            if step > 0:
                save("eval"); last_save = time.time()
        if step >= stop_step:
            if stop_step < total_steps:
                stopped = "stop_after"
            break
        if budget.exhausted():
            stopped = "budget"
            break
        with turn():
            t_step = time.time()
            lr = scratch.lr_at(step, total_steps, args.lr, warmup, args.min_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            for k in range(args.accum):
                rows = train.batch(step * args.accum + k, args.bsz, args.data_seed).to(device)
                x, y = rows[:, :-1], rows[:, 1:]
                mask = scratch.mask_for(model, x, step=step, seed=args.model_seed)
                with amp():
                    h = model(x, mask, return_hidden=True)
                    loss = scratch.chunked_ce(h, model.output.weight, y) / args.accum
                loss.backward()
                loss_acc += float(loss) * args.accum; n_acc += 1
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float("inf")))
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            finite = bool(np.isfinite(loss_acc))
            if args.duty < 1.0 and finite:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                time.sleep((time.time() - t_step) * (1.0 / args.duty - 1.0))
        if not finite:
            prog.log("nonfinite", step); stopped = "nonfinite"; break
        if step % args.log_every == 0:
            now = time.time()
            extra = {} if turn.fd is None else {"turn_wait_share": round((turn.waited - wait_log) / max(now - t_log, 1e-6), 3)}
            prog.log("step", step, tokens=step * tokens_per_step, loss=round(loss_acc / n_acc, 4), lr=round(lr, 8),
                     grad_norm=round(gnorm, 3), tok_per_s=round((step * tokens_per_step - tok_log) / max(now - t_log, 1e-6), 1),
                     **extra, **gpu_stats(), **gpu_temp())
            t_log, tok_log, loss_acc, n_acc, wait_log = now, step * tokens_per_step, 0.0, 0, turn.waited
        if time.time() - last_save > args.save_every_min * 60:
            save("periodic"); last_save = time.time()
    save(stopped)
    write_json(out / "result.json", {"status": stopped, "step": step, "total_steps": total_steps, "tokens": step * tokens_per_step,
                                     "family": args.family, "lr": args.lr, "parameters": n_params,
                                     "minutes": round(budget.minutes(), 1), **gpu_stats()})
    prog.log("done" if stopped == "done" else "stopped", step, status=stopped)


if __name__ == "__main__":
    main()
