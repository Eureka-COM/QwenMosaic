---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
- en
- ja
pipeline_tag: text-generation
tags:
- memory-mosaics
- attention-replacement
- distillation
- qwen3.8
datasets:
- HuggingFaceTB/cosmopedia-100k
- HuggingFaceH4/ultrachat_200k
- wikimedia/wikipedia
- codeparrot/codeparrot-clean-valid
---

# QwenMosaic-27B-delta

The trained part of Qwen/Qwen3.8-27B in which **all 16 softmax-attention layers (3, 7, ..., 63) are replaced by official Memory Mosaics blocks** (facebookresearch/MemoryMosaics, Memory Mosaics v2 configuration, 16 short + 16 long heads).
This repository holds only the delta: the new memories (1.01B parameters), the norms of the converted blocks and LoRA (rank 32) on the other layers, in bf16 (about 2.4 GB). It is applied to the original Qwen/Qwen3.8-27B weights, which you download from Qwen's repository.
With the delta applied, the model contains no attention layer and no attention weights.

Code, training records and the full evaluation: **[Eureka-COM/QwenMosaic](https://github.com/Eureka-COM/QwenMosaic)**.

> This is a research artifact, distilled with 60M tokens. It is **weaker than Qwen3.8-27B** on general benchmarks (MMLU 77.1% vs 84.3%, with a large drop in mathematics). It is published to make the results checkable and the converted architecture usable for further work.

## How it was made

| Step | What is trained | Tokens | Sequence length |
|---|---|---|---|
| Stage 1: layer fit | the new memories, each fitted to the output of the original attention of its layer | 30M | 2,048 |
| Stage 2: distillation | memories, norms and LoRA (rank 32), KL to the original model's next-token distribution | 30M | 2,048 |

These are the weights at the end of stage 2 (`mq-27b-s2` in the research records).
A stage 3 (8M tokens at 8,192) was also run; it left KL and perplexity almost unchanged and made recall slightly worse, so the stage 2 weights are the ones published here.
Stages 1 and 2 follow the procedure of RADLADS (arXiv:2505.03005).
Training data: cosmopedia-100k, ultrachat_200k, Japanese Wikipedia and codeparrot-clean, mixed 4:2:2:1.

## Evaluation (against the original Qwen3.8-27B)

| | Value |
|---|---|
| KL(original ‖ converted), next token (held-out, 16 x 2,048 tokens) | 0.165 |
| Perplexity ratio (converted / original) | 0.891 |
| Top-1 agreement with the original | 0.844 |
| MMLU, 1,000 sampled test questions, zero-shot | 77.1% (original 84.3%) |
| MMLU, 5 mathematics subjects (77 questions) | 57.1% (original 83.1%) |
| Recall of a 5-digit code 32 to 6,000 tokens back, filler without conversations | 10/10 at every distance (original 10/10) |
| The same inside a chat turn, 6,000 tokens back | 8/8 (original 8/8) |

- The perplexity ratio below 1 does **not** mean the converted model is better. The original model puts almost no probability on the correct token at about 12% of the positions, where the converted model's broader distribution loses less; at the other positions the original model has the lower loss and the higher accuracy.
- GSM8K was attempted but is not a valid measurement: most answers were cut by the 256-token generation limit, for the original model as well.

Sources (GitHub repository): `research/docs/results/mq-27b-s2-2026-11-12.md`, `mq-27b-s2-final-2026-11-13.md`, `mq-27b-recall-probe-2026-11-12.md`, and `research/runs/2026-11-13_0016_mq-27b-s2-final/`.

## Usage

The Memory Mosaics blocks are not part of Hugging Face Transformers. Use the code of the GitHub repository (transformers >= 5.2, < 5.23). Inference in bf16 was run on one 80 GB GPU.

```bash
git clone https://github.com/Eureka-COM/QwenMosaic && cd QwenMosaic && pip install -e .
hf download Eureka-COM/QwenMosaic-27B-delta --local-dir QwenMosaic-27B-delta
```

```python
import torch
from mmllm.mosaic_qwen import load_mosaic_qwen

mq, tok = load_mosaic_qwen("Qwen/Qwen3.8-27B", delta="QwenMosaic-27B-delta",
                           dtype=torch.bfloat16, device="cuda", device_map={"": 0})
text = tok.apply_chat_template([{"role": "user", "content": "What is a memory mosaic?"}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
out = mq.model.generate(ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

`scripts/mq_export.py` in the GitHub repository writes a self-contained model (LoRA merged, attention layers deleted) from the base model and this delta.
Padding inside a batch is not masked out by the memories: generate one prompt at a time, or batch prompts of equal length.

## License and attribution

Apache License 2.0.

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache License 2.0, Copyright 2026 Alibaba Cloud
- Memory Mosaics code: [facebookresearch/MemoryMosaics](https://github.com/facebookresearch/MemoryMosaics) (commit c2ca0e8), Apache License 2.0, Copyright (c) Meta Platforms, Inc. and affiliates
- Training data and their licenses as stated on their cards: cosmopedia-100k (apache-2.0), ultrachat_200k (mit), wikimedia/wikipedia (cc-by-sa-3.0, gfdl), codeparrot-clean (no license on the dataset card; each row carries the license of its source file). Check them for your use.

```bibtex
@inproceedings{zhang2025memory,
  title={Memory mosaics},
  author={Zhang, Jianyu and Nolte, Niklas and Sadhukhan, Ranajoy and Chen, Beidi and Bottou, L{\'e}on},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={36412--36433},
  year={2025}
}
@article{zhang2026memory,
  title={Memory Mosaics at scale},
  author={Zhang, Jianyu and Bottou, L{\'e}on},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={38929--38956},
  year={2026}
}
```

## 日本語の要約

Qwen3.8-27B の注意の 16 層を、Meta の公式 Memory Mosaics の層（v2 の構成、短い記憶 16 ヘッドと長い記憶 16 ヘッド）に置き換え、元のモデルを教師に 6,000 万語で蒸留したときの差分です。
元の Qwen3.8-27B の重みに足して使います。元のモデルとの KL は 0.165 で、6,000 語前に置いた番号も取り出せます。
一方で MMLU は元の 84.3% に対して 77.1% で、数学では大きく下がります。研究の成果物として、結果を確かめられるように公開しています。
