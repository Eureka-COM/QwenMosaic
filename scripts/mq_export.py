"""Write a trained MosaicQwen as a self-contained model: base weights + trained delta, with
LoRA merged and the original attention layers deleted.

    python scripts/mq_export.py --model Qwen/Qwen3.5-0.8B --delta runs/s2/delta --out export/mosaic-qwen-0.8b

The output directory holds config.json, tokenizer files, model-*.safetensors and
mq_config.json; load it with mmllm.load_standalone(path). The script reloads what it wrote
and checks that (1) no attention parameter is left and (2) the reloaded model gives the same
logits as the model it was exported from.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm.mosaic_qwen import load_mosaic_qwen, load_standalone  # noqa: E402
from mmllm.mq_train import pick_device, pick_dtype  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights-dir", default=None, help="where the model files go (default: --out)")
    ap.add_argument("--store", default=None, help="dir:PATH or github-release:TAG: also push the exported files there")
    ap.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--shard-mb", type=int, default=4000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)      # accepted for the runner; unused
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    mq, tok = load_mosaic_qwen(args.model, args.delta, dtype=dtype, device=device, keep_teacher=False)
    save_dtype = torch.bfloat16 if args.save_dtype == "bf16" else None
    ids = tok("Memory Mosaics are networks of associative memories. 記憶のモザイクは、", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        before = mq.model(ids, use_cache=False).logits.float().cpu()
    wdir = args.weights_dir or args.out
    Path(args.out).mkdir(parents=True, exist_ok=True)
    meta = mq.save_standalone(wdir, dtype=save_dtype, shard_mb=args.shard_mb)
    tok.save_pretrained(wdir)
    report = {"out": wdir, "parameters": meta["parameters"], "memory_parameters": meta["num_memory_parameters"],
              "attention_parameters_left": meta["attention_parameters_left"], "shards": len(meta["shards"])}
    if not args.no_verify:
        del mq
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        mq2 = load_standalone(wdir, dtype=dtype, device=device)
        with torch.no_grad():
            after = mq2.model(ids, use_cache=False).logits.float().cpu()
        report["reload_max_logit_diff"] = round((before - after).abs().max().item(), 5)
        report["reload_top1_agree"] = round((before.argmax(-1) == after.argmax(-1)).float().mean().item(), 4)
        report["attention_parameters_after_reload"] = len(mq2.attention_parameters())
        report["attention_modules_after_reload"] = sum(type(m).__name__ == "Qwen3_5Attention" for m in mq2.model.modules())
    if args.store:
        from mmllm.store import open_store
        try:                      # the export itself is done; a failed upload is reported, not fatal
            store = open_store(args.store)
            report["stored"] = {"where": store.describe(), "files": store.push(wdir)}
        except Exception as e:  # noqa: BLE001
            report["stored"] = {"where": args.store, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    Path(args.out, "export_report.json").write_text(json.dumps(report, indent=1))
    Path(args.out, "progress.jsonl").write_text(json.dumps({"event": "done", "step": 1}) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    if report["attention_parameters_left"] or report.get("attention_modules_after_reload"):
        raise SystemExit("attention weights are still present")


if __name__ == "__main__":
    main()
