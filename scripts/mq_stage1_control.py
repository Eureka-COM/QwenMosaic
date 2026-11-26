"""Stage-1 control: is the floor of the layer-wise matching the Mosaics memories, or the procedure?

Stage 1 trains, for every converted layer, the Mosaics memories to reproduce the output of the
layer's original mixer (softmax attention or Gated DeltaNet) from the same input. On Qwen3.5 and
Qwen3.8 the relative error stops near 0.25. This script trains, beside the memories and on the
very same batches, a CONTROL for every layer: a freshly initialised mixer of the same type as the
original (its weights drawn anew, its own copy of the input norm), with the same loss, optimizer,
schedule and data. The control is trained at every learning rate of --control-lr (the same new
weights at each); the one that ends lowest is "the control", so that a learning rate that suits
the memories but not an attention layer does not pass for a limit of the procedure.

    control reaches ~0, memories stay at ~0.25   the floor is what Mosaics memories can express
                                                 of this teacher's layers (a matter of fit)
    control also stalls                          the floor belongs to the procedure or the budget

    python scripts/mq_stage1_control.py --model Qwen/Qwen3.5-0.8B --data SRC ... --replace all \
        --short-heads 8 --long-heads 8 --seq-len 2048 --bsz 8 --tokens 10e6 --out runs/fit

Works on any model the conversion understands (Qwen3.5 / Qwen3.8, and attention-only models of
the Qwen3 / Llama type). Writes progress.jsonl (step, eval) and result.json: per layer and on
average, the relative error of the memories and of the control on held-out rows, split by the type
of the original layer. Nothing is saved as a model: this is a measurement.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import _autocast, _text_model, convert_to_mosaic_qwen  # noqa: E402
from mmllm.mq_data import load_source  # noqa: E402
from mmllm.mq_train import Budget, Progress, gpu_stats, load_base, lr_at, pick_device, pick_dtype, write_json  # noqa: E402
from mq_stage1 import add_conversion_args, add_row_args, conversion_config, row_packer  # noqa: E402


def fresh_mixers(model, layers, kinds, device, seed: int):
    """For every converted layer: a newly initialised mixer of the original's type and a copy of the
    original input norm. The new weights come from building the model class again from its config
    (the library's own initialisation), not from the trained model."""
    torch.manual_seed(seed)
    blank = type(model)(copy.deepcopy(model.config))
    blank_layers = _text_model(blank).layers
    out = {}
    for i, orig in layers.items():
        src = blank_layers[i]
        mixer = src.linear_attn if kinds[i] == "linear_attention" else src.self_attn
        norm = copy.deepcopy(orig.input_layernorm)
        out[i] = torch.nn.ModuleDict({"mixer": mixer, "norm": norm}).to(device=device, dtype=torch.float32)
        for p in out[i].parameters():
            p.requires_grad_(True)
    del blank
    return out


def control_output(ctrl, kind: str, x, ctx):
    """The control mixer called exactly as the original was (MosaicQwenLayer._teacher)."""
    with _autocast(x):
        h = ctrl["norm"](x)
        if kind == "linear_attention":
            return ctrl["mixer"](hidden_states=h, cache_params=None, attention_mask=ctx["attention_mask"], **ctx["kwargs"])
        y, _ = ctrl["mixer"](hidden_states=h, attention_mask=ctx["attention_mask"], position_ids=ctx["position_ids"],
                             past_key_values=None, position_embeddings=ctx["position_embeddings"], **ctx["kwargs"])
        return y


def rel(pred, target):
    target = target.detach().float()
    return (pred.float() - target).pow(2).mean() / target.pow(2).mean().clamp_min(1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_conversion_args(ap)
    add_row_args(ap)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-doc", type=int, default=None)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--tokens", type=float, default=10e6)
    ap.add_argument("--lr", type=float, default=1e-3, help="memories (as in stage 1)")
    ap.add_argument("--control-lr", default=None, help="control mixers: one learning rate or several, comma-separated "
                    "(a control per rate, all from the same new weights; default: --lr)")
    ap.add_argument("--lr-min-ratio", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--eval-rows", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-seed", type=int, default=1234)
    ap.add_argument("--skip-quantum", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--docs", type=int, default=200_000)
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    control_lrs = [float(x) for x in str(args.control_lr).split(",")] if args.control_lr else [args.lr]
    tags = [f"{x:g}" for x in control_lrs]
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    prog = Progress(out)
    budget = Budget(args.max_minutes)

    model, tok = load_base(args.model, dtype, device)
    mq = convert_to_mosaic_qwen(model, conversion_config(args, tok), keep_teacher=True)
    mq.set_trainable(memories=True, norms=True, lora=False)
    for l in mq.layers.values():
        l.block.ffn_norm.weight.requires_grad_(False)
    mm_params = {i: [p for m in l.memories() for p in m.parameters()] +
                    [p for n in ("attention_norm", "short_norm", "cluster_norm") if getattr(l.block, n, None) is not None
                     for p in getattr(l.block, n).parameters()]
                 for i, l in mq.layers.items()}
    kinds = {i: l.block_type for i, l in mq.layers.items()}
    ctrls = {t: fresh_mixers(model, mq.state.orig, kinds, device, args.control_seed) for t in tags}   # same seed: same start
    ctrl_params = {t: {i: list(c.parameters()) for i, c in ctrls[t].items()} for t in tags}
    adamw = lambda groups, lr: torch.optim.AdamW([p for ps in groups.values() for p in ps], lr=lr, betas=(0.9, 0.95),
                                                 weight_decay=0.0)
    opt_mm = adamw(mm_params, args.lr)
    opt_ct = {t: adamw(ctrl_params[t], lr) for t, lr in zip(tags, control_lrs)}

    sources = [load_source(s, args.docs) for s in args.data]
    packer = row_packer(args, tok, sources, max_doc=args.max_doc)
    eval_rows = packer.eval_rows(args.eval_rows)
    tokens_per_step = args.bsz * args.seq_len
    total_steps = int(args.tokens // tokens_per_step)

    n_mm = {i: sum(p.numel() for m in mq.layers[i].memories() for p in m.parameters()) for i in mq.layers}
    n_ct = {i: sum(p.numel() for p in c["mixer"].parameters()) for i, c in ctrls[tags[0]].items()}
    write_json(out / "run.json", {"args": vars(args), "device": str(device), "dtype": str(dtype),
                                  "backends": official.backends(device), "layers": mq.cfg.layers, "kinds": kinds,
                                  "memory_parameters_per_layer": n_mm, "control_parameters_per_layer": n_ct,
                                  "data": packer.describe(), "total_steps": total_steps})
    prog.log("start", 0, layers=len(mq.layers), total_steps=total_steps, backends=official.backends(device),
             memory_parameters=sum(n_mm.values()), control_parameters=sum(n_ct.values()))

    def summarize(per_layer: dict) -> dict:
        res = {"mean": round(sum(per_layer.values()) / len(per_layer), 5)}
        for kind in sorted(set(kinds.values())):
            vals = [v for i, v in per_layer.items() if kinds[i] == kind]
            res[kind] = round(sum(vals) / len(vals), 5)
        return res

    @torch.no_grad()
    def run_eval() -> dict:
        num = {k: {i: 0.0 for i in mq.layers} for k in ("mm", *tags)}
        den = {i: 0.0 for i in mq.layers}
        mq.fixed_skip()

        def cap(i, x, mixed, _o):
            t = mixed.float()
            den[i] += t.pow(2).sum().item()
            num["mm"][i] += (mq.layers[i].memory_residual(x).float() - t).pow(2).sum().item()
            for tag in tags:
                y = control_output(ctrls[tag][i], kinds[i], x, mq.state.capture_ctx)
                num[tag][i] += (y.float() - t).pow(2).sum().item()

        for r in eval_rows:
            with mq.teacher(capture=cap):
                model(input_ids=r[None].to(device), use_cache=False)

        def table(k):
            per = {i: num[k][i] / max(den[i], 1e-12) for i in mq.layers}
            return {"layers": {str(i): round(v, 5) for i, v in per.items()}, **summarize(per)}

        res = {"mm": table("mm"), "controls": {t: table(t) for t in tags}}
        best = min(tags, key=lambda t: res["controls"][t]["mean"])       # the control = the rate that is lowest now
        res["control"], res["control_lr"] = res["controls"][best], best
        return res

    losses = {k: {} for k in ("mm", *tags)}

    def local_step(i, x, mixed, _o):
        ctx = mq.state.capture_ctx
        with torch.enable_grad():
            a = rel(mq.layers[i].memory_residual(x.detach()), mixed)
            a.backward()
            losses["mm"][i] = a.item()
            for tag in tags:
                b = rel(control_output(ctrls[tag][i], kinds[i], x.detach(), ctx), mixed)
                b.backward()
                losses[tag][i] = b.item()

    step, tokens, stopped = 0, 0, "done"
    def log_eval() -> dict:
        res = run_eval()
        prog.log("eval", step, tokens=tokens, mm=res["mm"]["mean"], control=res["control"]["mean"],
                 control_lr=res["control_lr"], controls={t: res["controls"][t]["mean"] for t in tags},
                 **{f"{k}_{kind}": v for k in ("mm", "control") for kind, v in res[k].items() if kind not in ("layers", "mean")})
        return res

    last = log_eval()
    t_log, tok_log = time.time(), 0
    model.eval()
    while step < total_steps:
        if budget.exhausted():
            stopped = "budget"
            break
        for opt, base in ((opt_mm, args.lr), *((opt_ct[t], lr) for t, lr in zip(tags, control_lrs))):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, base, args.warmup, args.lr_min_ratio)
        ids = packer.batch(args.bsz).to(device)
        mq.sample_skip(step, args.seed, quantum=args.skip_quantum)
        with torch.no_grad(), mq.teacher(capture=local_step):
            model(input_ids=ids, use_cache=False)
        if not all(torch.isfinite(torch.tensor(v)) for v in losses["mm"].values()):
            prog.log("nonfinite", step, mm=losses["mm"])
            raise SystemExit("non-finite loss of the memories: stopping")
        if args.clip > 0:
            for group in (mm_params, *ctrl_params.values()):
                for ps in group.values():
                    torch.nn.utils.clip_grad_norm_(ps, args.clip)
        for opt in (opt_mm, *opt_ct.values()):
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        tokens += tokens_per_step
        if step % args.log_every == 0 or step == total_steps:
            now = time.time()
            prog.log("step", step, tokens=tokens, mm=round(sum(losses["mm"].values()) / len(losses["mm"]), 5),
                     controls={t: round(sum(losses[t].values()) / len(losses[t]), 5) for t in tags},
                     tok_per_s=round((tokens - tok_log) / max(now - t_log, 1e-6), 1), **gpu_stats())
            t_log, tok_log = now, tokens
        if step % args.eval_every == 0 and step < total_steps:
            last = log_eval()
    last = log_eval()
    write_json(out / "result.json", {"status": stopped, "step": step, "tokens": tokens, "kinds": kinds, **last,
                                     "memory_parameters_per_layer": n_mm, "control_parameters_per_layer": n_ct,
                                     "lr": args.lr, "control_lrs": control_lrs, "minutes": round(budget.minutes(), 1),
                                     "backends": official.backends(device)})
    prog.log("done" if stopped == "done" else "stopped", step, tokens=tokens, status=stopped)


if __name__ == "__main__":
    main()
