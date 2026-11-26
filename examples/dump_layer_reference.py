#!/usr/bin/env python3
"""置き換えた 1 層の入出力を、別の実装（たとえば専用のカーネル）の答え合わせ用に保存する。

GPU もネットワークも使わない。乱数で初期化した小さい Qwen3.5 型のモデル（3 層の Gated DeltaNet と
1 層の注意を 2 回くり返す 8 層）を作り、注意の層を公式 Memory Mosaics の層に置き換え、
置き換えた最初の層について次を 1 つの safetensors に書き出す。

    x            層への入力（残差の流れ）          (B, T, dim)
    h            attention_norm(x)                  (B, T, dim)
    short        短い記憶の出力                     (B, T, dim)
    long         長い記憶の出力                     (B, T, dim)
    memory_sum   short + long（= memory_residual）  (B, T, dim)
    block_out    層の出力（FFN まで足したもの）     (B, T, dim)
    params.*     その層の重み（記憶・正規化・FFN）

計算は CPU の参照の経路（密な注意の計算と、漏れ平均の並列スキャン）で行う。
どちらも公式の計算と一致することを tests/test_official_vendor.py が確かめている。

使い方:
    python examples/dump_layer_reference.py --out reference.safetensors
    python examples/dump_layer_reference.py --out ref.safetensors --seq-len 512 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmllm import official  # noqa: E402
from mmllm.mosaic_qwen import convert_to_mosaic_qwen  # noqa: E402


def tiny_qwen35(seed: int):
    """tests/conftest.py の tiny_qwen35 と同じ形の小さいモデル。"""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        eos_token_id=7, pad_token_id=0, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    )
    return Qwen3_5ForCausalLM(cfg).float().eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seq-len", type=int, default=384, help="window 256・skip 64 より長くすると、両方の記憶が働く")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heads", type=int, default=2, help="短い記憶・長い記憶それぞれのヘッドの数（小さいモデル向け）")
    ap.add_argument("--head-dim", type=int, default=16)
    a = ap.parse_args()

    official.set_backends(attention="dense", leaky="scan")
    from mmllm.mosaic_qwen import MQConfig

    model = tiny_qwen35(a.seed)
    mq = MQConfig.from_upstream_mmv2(n_short_heads=a.heads, n_long_heads=a.heads, head_dim=a.head_dim)
    handle = convert_to_mosaic_qwen(model, mq, keep_teacher=False)
    mq = handle.cfg                                  # convert_to_mosaic_qwen は写しに置き換えた層を書き込む
    handle.fixed_skip()                              # 学習中だけの乱数の skip を止め、決まった skip で計算する
    idx = mq.layers[0]
    layer = handle.layers[idx]
    blk = layer.block

    g = torch.Generator().manual_seed(a.seed + 1)
    x = torch.randn(a.batch, a.seq_len, model.config.hidden_size, generator=g)
    st = layer._st
    B, T, _ = x.shape
    with torch.no_grad():
        masks, pos = st.masks(B, T, x.device), st.positions(B, T, x.device)
        h = blk.attention_norm(x)
        short, _ = blk._read(blk.short_memory, h, masks.get("short"), pos, None)
        long, _ = blk._read(blk.long_memory, h, masks.get("long"), pos, None)
        memory_sum = layer.memory_residual(x)
        block_out = layer(x)

    assert torch.allclose(short + long, memory_sum, atol=1e-5), "short + long が memory_residual と合わない"
    with torch.no_grad():
        y = x + memory_sum
        again = y + blk.feed_forward(blk.ffn_norm(y))
    assert torch.allclose(again, block_out, atol=1e-5), "x + 記憶 + FFN の組み立てが層の出力と合わない"
    tensors = {"x": x, "h": h, "short": short, "long": long, "memory_sum": memory_sum, "block_out": block_out}
    for name, p in blk.named_parameters():
        tensors[f"params.{name}"] = p.detach().clone().contiguous()
    meta = {
        "layer_index": str(idx),
        "mq_config": json.dumps({k: v for k, v in vars(mq).items()}, ensure_ascii=False, default=str),
        "upstream_commit": official.upstream_info()["commit"],
        "backends": json.dumps(official.backends("cpu"), ensure_ascii=False),
        "torch": torch.__version__,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(a.out), metadata=meta)
    print(json.dumps({"out": str(a.out), "layer": idx, "tensors": {k: list(v.shape) for k, v in tensors.items()
                                                                  if not k.startswith("params.")},
                      "params": sum(1 for k in tensors if k.startswith("params."))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
