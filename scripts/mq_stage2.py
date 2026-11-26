"""MosaicQwen stage 2 (and stage 3): make the whole converted model follow the original.

Stage 1 fitted each layer's memories to the original mixer on the ORIGINAL model's hidden
states. Here the converted model runs on its own hidden states and is trained end to end:

    loss = KL( original(next token) || converted(next token) )          on --positions per row
         + --ce-weight * next-token cross-entropy                       (0 by default)
    (--kl-weight 0 --ce-weight 1: no distillation, the text alone; the KL is still measured)
         + --hidden-weight * mean over converted layers of
               | converted layer output - original layer output |^2 / | original |^2   (0 by default)

What is trained (--train):
    mem        the memories and norms of the converted blocks
    mem+lora   those, plus LoRA (rank --lora-r) on every Linear of every layer   [default]
    all        every weight (the teacher is then a separate copy of the base model)

The teacher is the original model, on the same weights as the student (the low-rank parts
are switched off for it). Stage 3 is this script with a longer --seq-len.

    python scripts/mq_stage2.py --model Qwen/Qwen3.5-0.8B --delta-in runs/s1/delta \
        --data data/cosmopedia-100k --seq-len 2048 --bsz 4 --tokens 20e6 --out runs/s2

Output in --out: progress.jsonl, delta/ (memories + LoRA; load with mmllm.load_mosaic_qwen),
optim.pt, result.json. Resume with --resume <an earlier --out>.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import MQConfig, convert_to_mosaic_qwen  # noqa: E402
from mmllm.mq_data import load_source  # noqa: E402
from mmllm.mq_train import (Budget, Progress, Teacher, evaluate, gpu_stats, hidden_states, load_base,  # noqa: E402
                            lr_at, pick_device, pick_dtype, resume_point, save_optimizer, vocab_loss_backward, write_json)
from mmllm.store import open_store  # noqa: E402
from mq_stage1 import add_conversion_args, add_row_args, conversion_config, row_packer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--delta-in", default=None, help="stage-1 (or earlier stage-2) delta directory")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--store", default=None, help="dir:PATH or github-release:TAG (see mq_stage1.py)")
    ap.add_argument("--train", default="mem+lora", choices=["mem", "mem+lora", "all"])
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-doc", type=int, default=None)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--positions", type=int, default=512, help="positions per row that get a KL term")
    ap.add_argument("--tokens", type=float, default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4, help="memories, norms and LoRA")
    ap.add_argument("--lr-base", type=float, default=1e-5, help="base weights (--train all)")
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.0)
    ap.add_argument("--kl-weight", type=float, default=1.0, help="0 with --ce-weight > 0: no distillation, the text alone "
                    "(the KL to the teacher is still measured)")
    ap.add_argument("--progressive", type=float, default=0.0,
                    help="progressive replacement: over this share of the steps the layers go from original to Mosaics. At step t "
                         "every layer is still the original with probability max(0, 1 - t / (share * steps)), drawn anew each "
                         "step; after that, and in every evaluation and in what is saved, all layers are Mosaics blocks")
    ap.add_argument("--hidden-weight", type=float, default=0.0)
    ap.add_argument("--loss-chunk", type=int, default=128, help="positions per vocabulary projection in the loss")
    ap.add_argument("--grad-checkpoint", action="store_true")
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
    ap.add_argument("--docs", type=int, default=200_000)
    add_conversion_args(ap)
    add_row_args(ap)
    args = ap.parse_args()
    if args.grad_checkpoint and args.hidden_weight > 0:
        ap.error("--hidden-weight needs the layer outputs, which gradient checkpointing does not keep")
    if args.progressive and (args.train == "all" or args.hidden_weight > 0):
        ap.error("--progressive works with --train mem / mem+lora and without --hidden-weight")

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    if args.train == "all" and args.dtype == "auto":
        dtype = torch.float32                       # trained weights are kept in fp32 (bf16 forward via autocast)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    prog = Progress(out, resume=bool(args.resume) and Path(args.resume).resolve() == out.resolve())
    budget = Budget(args.max_minutes)

    model, tok = load_base(args.model, dtype, device)
    store = open_store(args.store)
    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir is None and store is not None and store.pull(out / "delta"):
        resume_dir = out
    src = str(resume_dir / "delta") if resume_dir is not None else args.delta_in
    if src:
        meta = json.loads((Path(src) / "mq_config.json").read_text())
        known = set(MQConfig.__dataclass_fields__)
        mq_cfg = MQConfig(**{k: v for k, v in meta["mq_config"].items() if k in known})
    else:
        mq_cfg = conversion_config(args, tok)
    separate = None
    if args.train == "all":
        separate, _ = load_base(args.model, pick_dtype("auto", device), device)
    mq = convert_to_mosaic_qwen(model, mq_cfg, keep_teacher=separate is None)
    if args.train == "mem+lora":
        mq.add_lora(r=args.lora_r, alpha=args.lora_alpha)
    if src:
        mq.load_delta(src)
    mq.set_trainable(memories=True, norms=True, lora=args.train == "mem+lora", base=args.train == "all")
    fast_ids = {id(p) for p in mq.memory_parameters() + mq.norm_parameters() + mq.lora_parameters()}
    fast = [p for p in model.parameters() if p.requires_grad and id(p) in fast_ids]
    slow = [p for p in model.parameters() if p.requires_grad and id(p) not in fast_ids]
    groups = [{"params": fast, "lr": args.lr, "base_lr": args.lr}]
    if slow:
        groups.append({"params": slow, "lr": args.lr_base, "base_lr": args.lr_base})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    trainable = fast + slow
    teacher = Teacher(mq, separate=separate)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    sources = [load_source(s, args.docs) for s in args.data]
    packer = row_packer(args, tok, sources, max_doc=args.max_doc)
    eval_rows = packer.eval_rows(args.eval_rows)
    tokens_per_step = args.bsz * args.seq_len * args.accum
    total_steps = int(args.tokens // tokens_per_step) if args.tokens else args.steps

    step, tokens, resumed_step = 0, 0, None
    if resume_dir is not None:
        step, tokens = resume_point(resume_dir, opt, stage=2)
        resumed_step = step if not (resume_dir / "optim.pt").exists() else None
        for _ in range(step * args.accum):
            packer.batch(args.bsz)

    info = {"stage": 2, "args": vars(args), "device": str(device), "dtype": str(dtype),
            "backends": official.backends(device), "mq_config": mq.cfg.__dict__,
            "memory_parameters": mq.num_memory_parameters(), "lora_parameters": sum(p.numel() for p in mq.lora_parameters()),
            "trainable": sum(p.numel() for p in trainable), "teacher": "separate copy" if separate is not None else "shared weights",
            "attention_parameters_in_student": len(mq.attention_parameters()),
            "data": packer.describe(), "total_steps": total_steps, "tokens_per_step": tokens_per_step}
    write_json(out / "run.json", info)
    prog.log("start", step, layers=mq.cfg.layers, trainable=info["trainable"], total_steps=total_steps,
             backends=info["backends"], delta_in=src, train=args.train,
             resumed_from=str(resume_dir) if resume_dir is not None else None, store=store.describe() if store else None)

    def save(tag: str) -> None:
        mq.save_delta(out / "delta", dtype=torch.bfloat16 if args.save_dtype == "bf16" else None,
                      shard_mb=args.shard_mb, extra={"stage": 2, "step": step, "tokens": tokens, "tag": tag,
                                                     "train": args.train, "seq_len": args.seq_len})
        if not args.no_optim_save:
            save_optimizer(opt, out / "optim.pt", step, tokens)
        if store is not None:
            store.push(out / "delta")

    def run_eval() -> dict:
        m = evaluate(mq, teacher, eval_rows, device, layer_mse=separate is None)
        prog.log("eval", step, tokens=tokens, **m, **gpu_stats(detail=True))
        return m

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    last_eval = run_eval() if step == 0 else None
    if device.type == "cuda":
        torch.cuda.empty_cache()                 # hand back what the evaluation held; training packs tighter
    first_step = step
    last_save = time.time()
    t_log, tok_log = time.time(), tokens
    stopped = "done"
    agg = {"kl": 0.0, "ce": 0.0, "hid": 0.0, "n": 0}
    P = min(args.positions, args.seq_len - 1)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    gen_prog = torch.Generator(device="cpu").manual_seed(args.seed + 29)
    kept_sum = 0
    while step < total_steps:
        if budget.exhausted():
            stopped = "budget"
            break
        ease = min(1.0, (step - resumed_step + 1) / 20) if resumed_step is not None else 1.0
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, g["base_lr"], args.warmup, args.lr_min_ratio) * ease
        model.train()
        for _ in range(args.accum):
            ids = packer.batch(args.bsz).to(device)
            B, T = ids.shape
            sel = torch.stack([torch.randperm(T - 1, generator=gen)[:P] for _ in range(B)]).to(device)   # (B, P)
            rows = torch.arange(B, device=device)[:, None]
            target = ids[rows, sel + 1].reshape(-1)
            mq.fixed_skip()
            with amp:
                h_t, outs_t = teacher.hidden(ids, layer_outputs=args.hidden_weight > 0)
                ht_sel = h_t[rows, sel].reshape(B * P, -1)
            del h_t
            mq.sample_skip(step, args.seed, quantum=args.skip_quantum)
            if args.progressive > 0:
                p_orig = max(0.0, 1.0 - step / max(1.0, args.progressive * total_steps))
                draw = torch.rand(len(mq.layers), generator=gen_prog)
                mq.keep_original([i for i, u in zip(mq.layers, draw.tolist()) if u < p_orig])
                kept_sum += len(mq.state.keep_orig)
            with amp:
                if args.hidden_weight > 0:
                    with mq.collect() as outs_s:
                        h_s = hidden_states(model, ids)
                else:
                    h_s, outs_s = hidden_states(model, ids), None
                hs_sel = h_s[rows, sel].reshape(B * P, -1)
            hid = None
            if outs_s is not None:
                hid = torch.zeros((), device=device)
                for i in mq.layers:
                    a, b = outs_s[i].float(), outs_t[i].float()
                    hid = hid + (a - b).pow(2).mean() / b.pow(2).mean().clamp_min(1e-12)
                hid = hid / len(mq.layers)
            # The vocabulary projection of the sampled positions is positions x 248k x 4 bytes per
            # tensor (0.5 GB at 512 positions, and the loss needs half a dozen of them): computed a
            # chunk of positions at a time, with the same gradient (see vocab_loss_backward).
            kl, ce = vocab_loss_backward(model.lm_head, teacher.lm_head(), hs_sel, ht_sel, target,
                                         ce_weight=args.ce_weight, kl_weight=args.kl_weight, scale=1.0 / args.accum,
                                         chunk=args.loss_chunk, amp=amp,
                                         extra=None if hid is None else hid * (args.hidden_weight / args.accum))
            hid = float(hid) if hid is not None else 0.0
            if not math.isfinite(kl) or not math.isfinite(ce) or not math.isfinite(hid):
                prog.log("nonfinite", step, kl=kl, ce=ce)
                raise SystemExit("non-finite loss: stopping")
            agg["kl"] += kl; agg["ce"] += ce; agg["hid"] += hid; agg["n"] += 1
            # nothing of this micro-step may stay alive during the next one's forward pass (the
            # accumulated gradients already do)
            del h_s, hs_sel, ht_sel, outs_s, outs_t
            mq.keep_original(())
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += tokens_per_step
        early = step <= first_step + 12          # every step at first, with the memory in detail: a card that
        if step % args.log_every == 0 or step == total_steps or early:      # is too small shows here
            now = time.time()
            n = max(agg["n"], 1)
            prog.log("step", step, tokens=tokens, kl=round(agg["kl"] / n, 5), ce=round(agg["ce"] / n, 4),
                     hidden=round(agg["hid"] / n, 5), lr=round(opt.param_groups[0]["lr"], 7),
                     tok_per_s=round((tokens - tok_log) / max(now - t_log, 1e-6), 1),
                     **({"orig_layers": round(kept_sum / n, 1)} if args.progressive > 0 else {}), **gpu_stats(detail=early))
            agg = {"kl": 0.0, "ce": 0.0, "hid": 0.0, "n": 0}
            kept_sum = 0
            t_log, tok_log = now, tokens
        if step % args.eval_every == 0 and step < total_steps:
            last_eval = run_eval()
        if (time.time() - last_save) / 60 >= args.save_every_min:
            save("periodic"); last_save = time.time()
            prog.log("saved", step, tokens=tokens)

    mq.fixed_skip()
    last_eval = run_eval()
    save(stopped)
    write_json(out / "result.json", {"stage": 2, "status": stopped, "step": step, "total_steps": total_steps,
                                     "tokens": tokens, "eval": last_eval, "minutes": round(budget.minutes(), 1),
                                     "train": args.train, "backends": official.backends(device)})
    prog.log("done" if stopped == "done" else "stopped", step, tokens=tokens, status=stopped)


if __name__ == "__main__":
    main()
