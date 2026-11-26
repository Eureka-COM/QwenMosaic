"""MosaicQwen stage 1: train the new memories of every converted layer, layer by layer.

Convert the model (every attention layer -> one official Memory Mosaics block), then run
the ORIGINAL model once per batch (no gradient). At each converted layer, the layer's
input x and the output of its original mixer m(x) are handed to that layer's new memories,
which take one gradient step on

    | short(h) + long(h) - m(x) |^2 / | m(x) |^2          h = attention_norm(x)

Every layer has its own loss and its own parameters, so nothing is back-propagated through
the rest of the model: one forward pass of the base model per batch, and the memories of
all layers learn at once. (Same idea as step 1 of RADLADS, arXiv:2505.03005.)

    python scripts/mq_stage1.py --model Qwen/Qwen3.5-0.8B --data data/cosmopedia-100k \
        --seq-len 2048 --bsz 4 --tokens 20e6 --out runs/s1

Output in --out: progress.jsonl (one line per logged step / evaluation), delta/ (the
trained memories: safetensors shards + mq_config.json), optim.pt (to resume), result.json.
Resume with --resume <an earlier --out>.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import MQConfig, convert_to_mosaic_qwen  # noqa: E402
from mmllm.mq_data import load_source, make_packer, sep_token_id  # noqa: E402
from mmllm.mq_train import (Budget, Progress, Teacher, evaluate, gpu_stats, load_base, load_optimizer,  # noqa: E402
                            lr_at, pick_device, pick_dtype, resume_point, save_optimizer, write_json)
from mmllm.store import open_store  # noqa: E402


def add_conversion_args(ap: argparse.ArgumentParser) -> None:
    d = MQConfig.from_upstream_mmv2()
    ap.add_argument("--short-heads", type=int, default=d.n_short_heads)
    ap.add_argument("--long-heads", type=int, default=d.n_long_heads)
    ap.add_argument("--head-dim", type=int, default=d.head_dim)
    ap.add_argument("--window", type=int, default=d.window)
    ap.add_argument("--skip", type=int, default=d.skip)
    ap.add_argument("--random-skip", type=int, default=d.random_skip)
    ap.add_argument("--layout", default=d.layout, choices=["flat", "short_flat", "short_long_cluster"])
    ap.add_argument("--replace", default="attention", choices=["attention", "all"])
    ap.add_argument("--init-seed", type=int, default=42)
    # rows of several documents (plan section 10, step 2)
    ap.add_argument("--bias", default="causal", choices=["causal", "doc_causal"],
                    help="doc_causal: short and long memories read the current document only")
    ap.add_argument("--cluster-heads", type=int, default=0,
                    help="heads of the cluster memory, which reads the earlier documents of the row (needs --bias doc_causal)")
    ap.add_argument("--cluster-key-feature", default="leaky_average", choices=["leaky_average", "plain"])


def add_row_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--rows", default="random", choices=["random", "cluster"],
                    help="random: documents in random order; cluster: every row holds related documents (one k-means cluster)")
    ap.add_argument("--cluster-size", type=int, default=60, help="documents per cluster on average (--rows cluster)")
    ap.add_argument("--holdout-frac", type=float, default=0.05, help="share of the documents, as whole clusters, kept for evaluation")
    ap.add_argument("--cluster-cache", default=None, help="directory where the clusters are kept between runs")


def row_packer(args, tok, sources, max_doc=None):
    kw = ({"cluster_size": args.cluster_size, "holdout_frac": args.holdout_frac, "cache_dir": args.cluster_cache}
          if args.rows == "cluster" else {})
    return make_packer(args.rows, tok, sources, args.seq_len, seed=args.seed, max_doc=max_doc, **kw)


def conversion_config(args, tok=None) -> MQConfig:
    if args.cluster_heads and args.bias != "doc_causal":
        raise SystemExit("--cluster-heads needs --bias doc_causal (the cluster memory reads earlier documents)")
    docs = {}
    if args.bias == "doc_causal":
        if tok is None:
            raise SystemExit("--bias doc_causal needs the tokenizer (where documents end)")
        docs = dict(attn_bias_type="doc_causal", n_cluster_heads=args.cluster_heads,
                    cluster_key_feature=args.cluster_key_feature, eos_token_id=sep_token_id(tok))
    return MQConfig.from_upstream_mmv2(
        n_short_heads=args.short_heads, n_long_heads=args.long_heads, head_dim=args.head_dim, window=args.window,
        skip=args.skip, random_skip=args.random_skip, layout=args.layout, replace=args.replace,
        init_seed=args.init_seed, base_model=args.model, **docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True, help="PATH[:FIELD][@WEIGHT], repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--store", default=None, help="dir:PATH or github-release:TAG: every save is also pushed there, "
                    "and a run starts from what is there (large models; see mmllm/store.py)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-doc", type=int, default=None)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--tokens", type=float, default=None, help="total training tokens (overrides --steps)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min-ratio", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--no-train-norms", action="store_true")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-rows", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every-min", type=float, default=20)
    ap.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--shard-mb", type=int, default=90)
    ap.add_argument("--no-optim-save", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-quantum", type=int, default=64, help="random long-memory skip in multiples of this (1 = upstream)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--docs", type=int, default=200_000, help="documents read per source")
    ap.add_argument("--init-gain", action="store_true", help="with --init teacher-*: fit one output gain per layer by "
                    "least squares on real inputs (the error before training is then at most 1)")
    ap.add_argument("--init", default="upstream", choices=["upstream", "teacher-ov", "teacher-ovk"],
                    help="where the memories start: upstream's random values, or the original mixer's weights "
                         "(values and output; with -ovk the keys too). See mmllm/teacher_init.py")
    add_conversion_args(ap)
    add_row_args(ap)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    prog = Progress(out, resume=bool(args.resume) and Path(args.resume).resolve() == out.resolve())
    budget = Budget(args.max_minutes)

    model, tok = load_base(args.model, dtype, device)
    mq = convert_to_mosaic_qwen(model, conversion_config(args, tok), keep_teacher=True)
    params = mq.set_trainable(memories=True, norms=not args.no_train_norms, lora=False)
    by_layer = {i: [p for m in l.memories() for p in m.parameters()] +
                   ([] if args.no_train_norms else [p for n in ("attention_norm", "short_norm", "cluster_norm") if getattr(l.block, n, None) is not None
                                                    for p in getattr(l.block, n).parameters()])
                for i, l in mq.layers.items()}
    # stage 1 never changes ffn_norm (the FFN is not part of the local loss)
    for l in mq.layers.values():
        l.block.ffn_norm.weight.requires_grad_(False)
    params = [p for ps in by_layer.values() for p in ps]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    sources = [load_source(s, args.docs) for s in args.data]
    packer = row_packer(args, tok, sources, max_doc=args.max_doc)
    eval_rows = packer.eval_rows(args.eval_rows)
    tokens_per_step = args.bsz * args.seq_len
    total_steps = int(args.tokens // tokens_per_step) if args.tokens else args.steps
    teacher = Teacher(mq)
    init_report = None
    if args.init != "upstream" and not args.resume:
        from mmllm.teacher_init import calibrate_value_scale, teacher_init
        init_report = teacher_init(mq, args.init.split("-")[1])
        init_report["value_scale"] = calibrate_value_scale(mq, model, packer.eval_rows(4, seed=4242), device)
        if args.init_gain:
            from mmllm.teacher_init import fit_output_gain
            init_report["gain"] = fit_output_gain(mq, model, packer.eval_rows(4, seed=4242), device)

    store = open_store(args.store)
    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir is None and store is not None and store.pull(out / "delta"):
        resume_dir = out
    step, tokens, resumed_step = 0, 0, None
    if resume_dir is not None:
        mq.load_delta(resume_dir / "delta")
        step, tokens = resume_point(resume_dir, opt, stage=1)
        resumed_step = step if not (resume_dir / "optim.pt").exists() else None
        for _ in range(step):                        # same data order as an uninterrupted run
            packer.batch(args.bsz)

    info = {"stage": 1, "args": vars(args), "device": str(device), "dtype": str(dtype),
            "backends": official.backends(device), "mq_config": mq.cfg.__dict__,
            "memory_parameters": mq.num_memory_parameters(), "trainable": sum(p.numel() for p in params),
            "attention_parameters_in_student": len(mq.attention_parameters()),
            "data": packer.describe(), "total_steps": total_steps, "tokens_per_step": tokens_per_step,
            "init": args.init, "init_report": init_report}
    write_json(out / "run.json", info)
    prog.log("start", step, layers=mq.cfg.layers, memory_parameters=info["memory_parameters"],
             total_steps=total_steps, backends=info["backends"],
             resumed_from=str(resume_dir) if resume_dir is not None else None, store=store.describe() if store else None)

    def save(tag: str) -> None:
        mq.save_delta(out / "delta", dtype=torch.bfloat16 if args.save_dtype == "bf16" else None,
                      shard_mb=args.shard_mb, extra={"stage": 1, "step": step, "tokens": tokens, "tag": tag})
        if not args.no_optim_save:
            save_optimizer(opt, out / "optim.pt", step, tokens)
        if store is not None:
            store.push(out / "delta")

    def run_eval() -> dict:
        m = evaluate(mq, teacher, eval_rows, device)
        prog.log("eval", step, tokens=tokens, **m, **gpu_stats())
        return m

    losses: dict[int, float] = {}

    def local_step(i, x, mixed, _out):
        with torch.enable_grad():
            pred = mq.layers[i].memory_residual(x.detach())
            target = mixed.detach().float()
            loss = (pred.float() - target).pow(2).mean() / target.pow(2).mean().clamp_min(1e-12)
            loss.backward()
        losses[i] = loss.item()

    last_eval = run_eval() if step == 0 else None
    last_save = time.time()
    t_log, tok_log = time.time(), tokens
    stopped = "done"
    model.eval()
    while step < total_steps:
        if budget.exhausted():
            stopped = "budget"
            break
        lr = lr_at(step, total_steps, args.lr, args.warmup, args.lr_min_ratio)
        if resumed_step is not None:                 # no optimizer state: ease back in
            lr *= min(1.0, (step - resumed_step + 1) / 20)
        for g in opt.param_groups:
            g["lr"] = lr
        ids = packer.batch(args.bsz).to(device)
        mq.sample_skip(step, args.seed, quantum=args.skip_quantum)
        losses.clear()
        with torch.no_grad(), mq.teacher(capture=local_step):
            model(input_ids=ids, use_cache=False)
        if not all(torch.isfinite(torch.tensor(v)) for v in losses.values()):
            opt.zero_grad(set_to_none=True)
            prog.log("nonfinite", step, loss_layers={str(k): v for k, v in losses.items()})
            raise SystemExit("non-finite loss: stopping")
        if args.clip > 0:
            for ps in by_layer.values():
                torch.nn.utils.clip_grad_norm_(ps, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += tokens_per_step
        if step % args.log_every == 0 or step == total_steps:
            now = time.time()
            prog.log("step", step, tokens=tokens, loss=round(sum(losses.values()) / len(losses), 5),
                     loss_layers={str(k): round(v, 4) for k, v in sorted(losses.items())}, lr=round(lr, 7),
                     tok_per_s=round((tokens - tok_log) / max(now - t_log, 1e-6), 1), **gpu_stats())
            t_log, tok_log = now, tokens
        if step % args.eval_every == 0 and step < total_steps:
            last_eval = run_eval()
        if (time.time() - last_save) / 60 >= args.save_every_min:
            save("periodic"); last_save = time.time()
            prog.log("saved", step, tokens=tokens)

    mq.fixed_skip()
    last_eval = run_eval()
    save(stopped)
    write_json(out / "result.json", {"stage": 1, "status": stopped, "step": step, "total_steps": total_steps,
                                     "tokens": tokens, "eval": last_eval, "minutes": round(budget.minutes(), 1),
                                     "backends": official.backends(device)})
    prog.log("done" if stopped == "done" else "stopped", step, tokens=tokens, status=stopped)


if __name__ == "__main__":
    main()
