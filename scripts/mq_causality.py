"""Can the converted model see the future? (It must not: a memory whose value is "what came
next" answers the question if position t may read its own entry.)

Changes the tokens after a cut and compares the hidden states up to the cut, which must not
move. For the student without gradients (the path evaluation uses), the student with gradients
(the path training uses: on CUDA the compiled FlexAttention and the CUDA leaky-average
kernel), with the fixed and with a sampled long-memory skip, and for the teacher. The memories
are randomised first so that every path carries a signal.

    python scripts/mq_causality.py --model PATH --seq 2048 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import MQConfig, convert_to_mosaic_qwen  # noqa: E402
from mmllm.mq_train import hidden_states, load_base, pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--delta", default=None, help="trained memories (default: random ones)")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--cuts", default="40,300,1023,1024,1025,1500")
    ap.add_argument("--short-heads", type=int, default=None); ap.add_argument("--long-heads", type=int, default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="auto")
    ap.add_argument("--noise", type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    cuda = dev.type == "cuda"
    dtype = {"auto": torch.bfloat16 if cuda else torch.float32, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    official.prepare_device(dev)
    model, tok = load_base(a.model, dtype, dev)
    over = {k: v for k, v in (("n_short_heads", a.short_heads), ("n_long_heads", a.long_heads)) if v}
    mq = convert_to_mosaic_qwen(model, MQConfig.from_upstream_mmv2(base_model=a.model, **over))
    if a.delta:
        mq.load_delta(a.delta)
    else:
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in mq.memory_parameters():
                p.add_((torch.randn(p.shape, generator=g) * a.noise * (p.float().abs().mean().item() + 0.02)).to(p.device, p.dtype))
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1000, 50000, (1, a.seq), generator=g).to(dev)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else torch.autocast("cpu", enabled=False)
    res = {"model": a.model, "seq": a.seq, "device": str(dev), "dtype": str(dtype), "backends": official.backends(dev),
           "memories": a.delta or "random", "modes": {}}

    def run(mode, x):
        if mode == "teacher":
            with torch.no_grad(), amp, mq.teacher():
                return hidden_states(model, x).float()
        if mode.startswith("student, no gradient"):
            model.eval()
            with torch.no_grad(), amp:
                return hidden_states(model, x).float()
        model.train()
        for p in mq.memory_parameters():
            p.requires_grad_(True)
        with torch.enable_grad(), amp:
            return hidden_states(model, x).float().detach()

    for mode, skip in (("teacher", None), ("student, no gradient, fixed skip", None), ("student, no gradient, sampled skip", 3),
                       ("student, with gradient, fixed skip", None), ("student, with gradient, sampled skip", 3),
                       ("student, with gradient, sampled skip (another)", 8)):
        if skip is None:
            mq.fixed_skip()
        else:
            mq.sample_skip(skip, 0, quantum=64)
        base = run(mode, ids)
        again = run(mode, ids)
        m = {"skip_extra": getattr(mq.state, "skip_extra", None), "same_input_twice": float((base - again).abs().max()),
             "scale": float(base.abs().mean()), "cuts": {}}
        for cut in [int(c) for c in a.cuts.split(",") if int(c) < a.seq - 1]:
            other = ids.clone()
            other[0, cut + 1:] = torch.randint(1000, 50000, (a.seq - cut - 1,), generator=g).to(dev)
            h = run(mode, other)
            d = (h[0, :cut + 1] - base[0, :cut + 1]).abs()
            after = (h[0, cut + 1:] - base[0, cut + 1:]).abs().max()
            first = int((d.amax(-1) > 10 * max(m["same_input_twice"], 1e-6)).nonzero()[0]) if (d.amax(-1) > 10 * max(m["same_input_twice"], 1e-6)).any() else None
            m["cuts"][cut] = {"max_change_up_to_cut": float(d.max()), "first_changed_position": first,
                              "max_change_after_cut": float(after)}
        m["sees_the_future"] = any(c["first_changed_position"] is not None for c in m["cuts"].values())
        res["modes"][mode] = m
        print(json.dumps({mode: m}), flush=True)
        (out / "causality.json").write_text(json.dumps(res, indent=1))
    res["any_mode_sees_the_future"] = any(m["sees_the_future"] for m in res["modes"].values())
    (out / "causality.json").write_text(json.dumps(res, indent=1))
    (out / "progress.jsonl").write_text(json.dumps({"event": "done", "step": 1}) + "\n")
    print("ANY MODE SEES THE FUTURE:", res["any_mode_sees_the_future"], flush=True)


if __name__ == "__main__":
    main()
