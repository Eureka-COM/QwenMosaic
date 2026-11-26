"""Does the conversion fit Qwen3.8-27B? Build layers with the REAL 27B shapes (random
weights, tiny vocabulary), convert them, and check the mechanics; then count what the full
model gains and loses.

    python scripts/mq_port27.py --out runs/port27            # 1 attention layer (1.5 GB of RAM)
    python scripts/mq_port27.py --pattern LA --out ...       # + 1 Gated DeltaNet layer (3 GB)

Checks: the teacher path equals the unconverted model, generation with the cache equals the
full forward pass, no attention parameter is left. Counts (per layer, and for the 16
attention layers of the 64-layer model): new memory parameters, removed attention
parameters, numbers stored per token for generation.
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

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "data" / "qwen3.8-27b-config.json"))
    ap.add_argument("--pattern", default="A", help="layers to build: L = Gated DeltaNet, A = attention")
    ap.add_argument("--short-heads", type=int, default=16)
    ap.add_argument("--long-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    full = json.loads(Path(args.config).read_text())["text_config"]
    n_layers_full = full["num_hidden_layers"]
    n_attn_full = sum(t == "full_attention" for t in full["layer_types"])
    types = [{"L": "linear_attention", "A": "full_attention"}[c] for c in args.pattern]
    small = dict(full, num_hidden_layers=len(types), layer_types=types, vocab_size=1024, tie_word_embeddings=True,
                 bos_token_id=1, eos_token_id=1, pad_token_id=0)
    small.pop("dtype", None); small.pop("model_type", None)
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(**small)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    attn_layer = types.index("full_attention")
    attn_params = sum(p.numel() for p in model.model.layers[attn_layer].self_attn.parameters())
    ids = torch.randint(10, 1000, (1, args.seq))
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits
    t0 = time.time()
    mq = convert_to_mosaic_qwen(model, MQConfig.from_upstream_mmv2(
        n_short_heads=args.short_heads, n_long_heads=args.long_heads, head_dim=args.head_dim))
    convert_s = time.time() - t0
    layer = mq.layers[attn_layer]
    mem_params = sum(p.numel() for m in layer.memories() for p in m.parameters())
    with torch.no_grad():
        with mq.teacher():
            teacher_diff = (model(ids, use_cache=False).logits - ref).abs().max().item()
        t0 = time.time()
        full_logits = model(ids, use_cache=False).logits
        fwd_s = time.time() - t0
        P = args.seq - 24
        out = model(ids[:, :P], use_cache=True)
        pkv, parts = out.past_key_values, [out.logits]
        for t in range(P, args.seq):
            out = model(ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values; parts.append(out.logits)
        cache_diff = (torch.cat(parts, 1) - full_logits).abs().max().item()
    H, d = cfg.num_key_value_heads, cfg.head_dim
    kv_orig = 2 * H * d
    kv_long = 2 * args.long_heads * args.head_dim
    total_full = 27_781_427_952          # safetensors parameter count of Qwen/Qwen3.8-27B (with the vision tower)
    rep = {
        "config": args.config, "pattern": args.pattern, "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size, "heads": [args.short_heads, args.long_heads, args.head_dim],
        "upstream_commit": official.upstream_info()["commit"],
        "checks": {"teacher_equals_original_max_diff": teacher_diff, "cache_vs_full_max_diff": cache_diff,
                   "attention_parameters_in_student": len(mq.attention_parameters()),
                   "ffn_shared_with_original": layer.block.feed_forward.w1.weight is mq.state.orig[attn_layer].mlp.gate_proj.weight},
        "per_layer": {"memory_parameters": mem_params, "removed_attention_parameters": attn_params},
        "full_27b": {"attention_layers": n_attn_full, "layers": n_layers_full,
                     "memory_parameters": mem_params * n_attn_full,
                     "removed_attention_parameters": attn_params * n_attn_full,
                     "parameters_before": total_full,
                     "parameters_after": total_full - attn_params * n_attn_full + mem_params * n_attn_full,
                     "memory_gb_fp32": round(mem_params * n_attn_full * 4 / 2**30, 2),
                     "memory_gb_bf16": round(mem_params * n_attn_full * 2 / 2**30, 2),
                     "adam_state_gb_fp32": round(mem_params * n_attn_full * 8 / 2**30, 2)},
        "per_token_cache_numbers": {"original_attention_kv": kv_orig, "mosaics_long": kv_long,
                                    "mosaics_short_window_only": 2 * args.short_heads * args.head_dim,
                                    "ratio_long_vs_original": round(kv_long / kv_orig, 2)},
        "timing_cpu": {"convert_s": round(convert_s, 1), f"forward_{args.seq}_tokens_s": round(fwd_s, 2)},
    }
    ok = (teacher_diff == 0.0 and cache_diff < 1e-3 and rep["checks"]["attention_parameters_in_student"] == 0
          and rep["checks"]["ffn_shared_with_original"])
    rep["status"] = "ok" if ok else "FAILED"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "port27.json").write_text(json.dumps(rep, indent=1))
    (out_dir / "progress.jsonl").write_text(json.dumps({"event": "done", "step": 1, "status": rep["status"]}) + "\n")
    print(json.dumps(rep, indent=1))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
