"""Preflight on the machine that will train: can the official Memory Mosaics code run here
at full speed, and how fast is a training step?

    python scripts/mq_env.py --model /content/models/Qwen3.5-0.8B --out runs/env

Writes <out>/env.json:
    packages, GPU, nvcc / ninja / gcc
    leaky_kernel     the upstream CUDA kernel built and matches the reference   (else: scan fallback, with the reason)
    flex_attention   the upstream compiled FlexAttention ran forward + backward and matches
                     the dense path                                              (else: dense fallback, with the reason)
    block_step       seconds per forward+backward of one converted-layer memory pair at --seq
    model            (with --model) teacher == original, one stage-1 step, tokens per second,
                     peak GPU memory; and (with --stage2) one stage-2 step with LoRA
                     (+ gradient checkpointing with --grad-checkpoint)
The run never fails because a fast path is missing: it reports which path will be used.
"""

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _ver(cmd):
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        lines = [l for l in subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.splitlines()
                 if any(c.isdigit() for c in l)]
        return (lines[0] if cmd[0] == "gcc" else lines[-1]).strip()[:120] if lines else "?"
    except Exception as e:  # noqa: BLE001
        return f"error: {e!r}"[:120]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--short-heads", type=int, default=16)
    ap.add_argument("--long-heads", type=int, default=16)
    ap.add_argument("--replace", default="attention", choices=["attention", "all"])
    ap.add_argument("--bias", default="causal", choices=["causal", "doc_causal"])
    ap.add_argument("--cluster-heads", type=int, default=0)
    ap.add_argument("--doc-len", type=int, default=900, help="mean document length of the test rows (--bias doc_causal)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--positions", type=int, default=512)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.cuda.is_available(),
           "nvcc": _ver(["nvcc", "--version"]), "ninja": _ver(["ninja", "--version"]), "gcc": _ver(["gcc", "--version"])}
    dev = torch.device("cuda" if res["cuda"] else "cpu")
    if res["cuda"]:
        p = torch.cuda.get_device_properties(0)
        res.update(gpu=torch.cuda.get_device_name(0), gpu_mem_gb=round(p.total_memory / 2**30, 1),
                   torch_cuda=torch.version.cuda)
    for mod in ("transformers", "fla", "safetensors"):
        try:
            res[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            res[mod] = f"missing: {e!r}"[:100]

    def dump():
        (out / "env.json").write_text(json.dumps(res, indent=1))

    from mmllm import official
    from mmllm.official import DocumentLayout, MosaicBlock, MosaicsConfig, memory_mask
    res["upstream"] = official.verify()
    res["backends_before"] = official.prepare_device(dev)
    res["leaky_kernel"] = {"used": res["backends_before"]["leaky_average"] == "kernel",
                           "fallback_reason": res["backends_before"]["leaky_fallback"]}
    dump()

    # one converted-layer-sized pair of memories: official attention path vs the dense path
    dim = 1024
    cfg = MosaicsConfig(dim=dim, n_layers=1, head_dim=128, max_seq_len=a.seq, vocab_size=10, ffn_dim_multiplier=1.0,
                        multiple_of=1, attn_bias_type="causal", memory_gate=True, layout="flat", norm_eps=1e-6,
                        short_memory_type="mosaics", short_mem_window_size=256, n_short_heads=8,
                        long_memory_type="mosaics", long_key_feature="leaky_average", long_mem_skip_size=64,
                        n_long_heads=8, n_cluster_heads=0)
    torch.manual_seed(0)
    blk = MosaicBlock(cfg, 0); blk.init_weights(); blk.to(dev)
    T, B = a.seq, 2
    tokens, q_lens = torch.empty(1, T, dtype=torch.long, device=dev), torch.tensor([T], device=dev)
    masks = {"short": memory_mask("short", False, tokens, q_lens, "causal", window=256),
             "long": memory_mask("long", False, tokens, q_lens, "causal", skip=64)}
    lay = DocumentLayout(); lay.full_sequence("causal", torch.tensor([T] * B, device=dev), T)
    pos = lay.positions(dev, B, T)
    x = torch.randn(B, T, dim, device=dev)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if res["cuda"] else torch.autocast("cpu", enabled=False)

    def run(n=3):
        ts, y = [], None
        for _ in range(n):
            if res["cuda"]:
                torch.cuda.synchronize()
            t0 = time.time()
            with amp:
                y, _ = blk(x, masks, pos)
            y.float().pow(2).mean().backward()
            if res["cuda"]:
                torch.cuda.synchronize()
            ts.append(time.time() - t0)
        return y.detach().float(), ts

    flex = {}
    try:
        y_auto, ts = run()
        flex.update(first_call_s=round(ts[0], 2), step_s=round(min(ts[1:]), 4))
        if res["cuda"]:      # what costs a rebuild of the compiled kernel: another skip? another length?
            keep = (x, masks, pos)
            masks = dict(masks, long=memory_mask("long", False, tokens, q_lens, "causal", skip=64 + 128))
            _, ts2 = run(2)
            flex.update(other_skip_first_call_s=round(ts2[0], 2), other_skip_step_s=round(ts2[1], 4))
            T2 = T - 128
            tokens2, q2 = torch.empty(1, T2, dtype=torch.long, device=dev), torch.tensor([T2], device=dev)
            masks = {"short": memory_mask("short", False, tokens2, q2, "causal", window=256),
                     "long": memory_mask("long", False, tokens2, q2, "causal", skip=64)}
            lay2 = DocumentLayout(); lay2.full_sequence("causal", torch.tensor([T2] * B, device=dev), T2)
            pos, x = lay2.positions(dev, B, T2), keep[0][:, :T2].contiguous()
            _, ts3 = run(2)
            flex.update(other_length_first_call_s=round(ts3[0], 2), other_length_step_s=round(ts3[1], 4))
            x, masks, pos = keep
        used = official.backends(dev)
        flex.update(backend=used["attention"], fallback_reason=used["attention_fallback"])
        # The dense path holds rows x heads x T x T scores per memory (8.6 GB at 8,192 tokens, 34 GB at
        # 16,384): compare with it only where it fits. On 2026-11-13 the comparison at 16,384 ran a 32 GB
        # card out of memory and the stage measurements after it crawled in the spilled memory.
        dense_gb = B * 16 * T * T * 4 / 2**30
        if dense_gb <= 12:
            official.set_backends(attention="dense")
            try:
                y_dense, ts_d = run(2)
            finally:
                official.set_backends(attention="auto")      # the model test below must use the real path
            flex.update(dense_step_s=round(min(ts_d), 4), max_diff_vs_dense=round((y_auto - y_dense).abs().max().item(), 4),
                        rel_diff_vs_dense=round(((y_auto - y_dense).norm() / y_dense.norm()).item(), 5))
            del y_dense
        else:
            flex["dense_comparison"] = f"skipped: {dense_gb:.0f} GB of scores at {T} tokens"
    except Exception as e:  # noqa: BLE001
        flex["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    res["flex_attention"] = flex
    res["backends"] = official.backends(dev)
    del blk, x
    gc.collect()
    if res["cuda"]:
        torch.cuda.empty_cache()                 # nothing of the block test may stay on the card
    dump()

    if a.model:
        from mmllm.mosaic_qwen import MQConfig, convert_to_mosaic_qwen
        from mmllm.mq_train import hidden_states, load_base, vocab_loss_backward
        m = {}
        t0 = time.time()
        dtype = torch.bfloat16 if res["cuda"] else torch.float32
        model, tok = load_base(a.model, dtype, dev)
        m["load_s"] = round(time.time() - t0, 1)
        m["parameters"] = sum(p.numel() for p in model.parameters())
        ids = torch.tensor([tok("The capital of France is Paris. The capital of Japan is", add_special_tokens=False).input_ids], device=dev)
        with torch.no_grad():
            ref = model(input_ids=ids, use_cache=False).logits.float()
        from mmllm.mq_data import sep_token_id
        sep = sep_token_id(tok)
        docs = dict(attn_bias_type="doc_causal", n_cluster_heads=a.cluster_heads, eos_token_id=sep) if a.bias == "doc_causal" else {}
        mq = convert_to_mosaic_qwen(model, MQConfig.from_upstream_mmv2(n_short_heads=a.short_heads, n_long_heads=a.long_heads,
                                                                       replace=a.replace, base_model=a.model, **docs))

        def make_rows(bsz):
            """Random rows; under doc_causal cut into documents of about --doc-len tokens, differently
            at every call (as in training, where every step has its own document layout)."""
            hi = min(50000, model.get_input_embeddings().num_embeddings)
            r = torch.randint(min(1000, hi // 2), hi, (bsz, a.seq), device=dev)
            if a.bias == "doc_causal":
                r[torch.rand(bsz, a.seq, device=dev) < 1.0 / a.doc_len] = sep
            return r
        with torch.no_grad(), mq.teacher():
            t = model(input_ids=ids, use_cache=False).logits.float()
        m.update(teacher_max_diff=float((t - ref).abs().max()), next_token=tok.decode(int(ref[0, -1].argmax())),
                 replace=a.replace, bias=a.bias, cluster_heads=a.cluster_heads, layers=mq.cfg.layers,
                 memory_parameters=mq.num_memory_parameters(),
                 attention_parameters_in_student=len(mq.attention_parameters()))
        res["model"] = m; dump()
        params = mq.set_trainable(memories=True, norms=True, lora=False)
        opt = torch.optim.AdamW(params, lr=1e-4)
        losses = {}

        def local(i, xin, mixed, _o):
            with torch.enable_grad():
                pred = mq.layers[i].memory_residual(xin.detach())
                loss = (pred.float() - mixed.float()).pow(2).mean() / mixed.float().pow(2).mean().clamp_min(1e-12)
                loss.backward()
            losses[i] = loss.item()

        def gb(reserved=False):
            if not res["cuda"]:
                return None
            f = torch.cuda.max_memory_reserved if reserved else torch.cuda.max_memory_allocated
            return round(f() / 2**30, 2)

        # stage 1 at the asked batch size; out of memory is a result here: halve and measure again
        bsz, tried = a.bsz, []
        while True:
            ids = make_rows(bsz)
            if res["cuda"]:
                torch.cuda.reset_peak_memory_stats()
            ts = []
            try:
                for _ in range(a.steps):
                    t0 = time.time()
                    ids = make_rows(bsz)
                    with torch.no_grad(), mq.teacher(capture=local):
                        model(input_ids=ids, use_cache=False)
                    opt.step(); opt.zero_grad(set_to_none=True)
                    if res["cuda"]:
                        torch.cuda.synchronize()
                    ts.append(time.time() - t0)
                break
            except torch.OutOfMemoryError as e:
                oom = str(e)[:200]
            # (outside the except block: the traceback keeps the failed step's tensors alive)
            tried.append({"bsz": bsz, "error": oom, "gpu_gb_peak": gb()})
            m["stage1_out_of_memory"] = tried; dump()
            opt.zero_grad(set_to_none=True)
            del ids
            gc.collect(); torch.cuda.empty_cache()
            if bsz == 1:
                raise RuntimeError(f"stage 1 does not fit even at batch size 1: {oom}")
            bsz //= 2
        a.bsz = bsz
        m["stage1"] = {"seq": a.seq, "bsz": bsz, "first_step_s": round(ts[0], 2), "step_s": round(min(ts), 3),
                       "tokens_per_s": round(a.seq * bsz / min(ts)), "loss": round(sum(losses.values()) / len(losses), 4),
                       "gpu_gb_peak": gb(), "gpu_gb_reserved": gb(True)}
        dump()
        if a.stage2:
            del opt
            mq.add_lora(r=a.lora_r)
            params = mq.set_trainable(memories=True, norms=True, lora=True)
            opt = torch.optim.AdamW(params, lr=1e-5)
            if a.grad_checkpoint:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            P = min(a.positions, a.seq - 1)
            sel = torch.stack([torch.randperm(a.seq - 1)[:P] for _ in range(a.bsz)]).to(dev)
            rows = torch.arange(a.bsz, device=dev)[:, None]
            if res["cuda"]:
                torch.cuda.reset_peak_memory_stats()
            ts = []
            try:
                for _ in range(a.steps):
                    t0 = time.time()
                    ids = make_rows(a.bsz)
                    # the loss as scripts/mq_stage2.py computes it (the vocabulary projection a chunk of
                    # positions at a time): the memory measured here is the memory of the training
                    with amp:
                        with torch.no_grad(), mq.teacher():
                            ht_sel = hidden_states(model, ids)[rows, sel].reshape(a.bsz * P, -1)
                        hs_sel = hidden_states(model, ids)[rows, sel].reshape(a.bsz * P, -1)
                    kl, _ = vocab_loss_backward(model.lm_head, model.lm_head, hs_sel, ht_sel,
                                                ids[rows, sel + 1].reshape(-1), chunk=128, amp=amp)
                    del hs_sel, ht_sel
                    opt.step(); opt.zero_grad(set_to_none=True)
                    if res["cuda"]:
                        torch.cuda.synchronize()
                    ts.append(time.time() - t0)
                m["stage2"] = {"seq": a.seq, "bsz": a.bsz, "lora_r": a.lora_r, "grad_checkpoint": a.grad_checkpoint,
                               "lora_parameters": sum(p.numel() for p in mq.lora_parameters()),
                               "first_step_s": round(ts[0], 2), "step_s": round(min(ts), 3),
                               "tokens_per_s": round(a.seq * a.bsz / min(ts)), "kl": round(kl, 4),
                               "gpu_gb_peak": gb(), "gpu_gb_reserved": gb(True)}
            except Exception as e:  # noqa: BLE001  (out of memory is a result here, not a failure)
                m["stage2"] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "seq": a.seq, "bsz": a.bsz,
                               "grad_checkpoint": a.grad_checkpoint, "gpu_gb_peak": gb(), "gpu_gb_reserved": gb(True)}
        res["backends"] = official.backends(dev)
        dump()
    (out / "progress.jsonl").write_text(json.dumps({"event": "done", "step": 1}) + "\n")
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
