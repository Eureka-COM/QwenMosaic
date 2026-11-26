---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
language:
- en
- ja
pipeline_tag: text-generation
tags:
- memory-mosaics
- attention-replacement
- distillation
- qwen3.5
datasets:
- HuggingFaceTB/cosmopedia-100k
- HuggingFaceH4/ultrachat_200k
- wikimedia/wikipedia
- codeparrot/codeparrot-clean-valid
---

# QwenMosaic-0.8B

Qwen/Qwen3.5-0.8B in which **all 6 softmax-attention layers are replaced by official Memory Mosaics blocks** (facebookresearch/MemoryMosaics, Memory Mosaics v2 configuration, 8 short + 8 long heads).
The converted model contains no attention layer and no attention weights (no q/k/v/o projections).
The other 18 layers (Gated DeltaNet + MLP) are Qwen's.

Code, training records and the full evaluation: **[Eureka-COM/QwenMosaic](https://github.com/Eureka-COM/QwenMosaic)**.

> This is a research artifact. It was distilled from Qwen3.5-0.8B with about 24M tokens. It is not a better model than Qwen3.5-0.8B; it shows that the attention layers can be replaced by Memory Mosaics and stay close to the original.

## How it was made

| Step | What is trained | Tokens | Sequence length |
|---|---|---|---|
| Stage 1: layer fit | the new memories, each fitted to the output of the original attention of its layer | 10M | 2,048 |
| Stage 2: distillation | memories, norms and LoRA (rank 32), KL to the original model's next-token distribution | 10M | 2,048 |
| Stage 3: long sequences | same as stage 2 | 4M | 8,192 |

The persistent memory (FFN) and the norms of each converted block start from the layer's own Qwen weights.
Stages 1 and 2 follow the procedure of RADLADS (arXiv:2505.03005).
Training data: cosmopedia-100k, ultrachat_200k, Japanese Wikipedia and codeparrot-clean, mixed 4:2:2:1.
LoRA is merged into the weights in this export; the file is bf16.

## Evaluation (against the original Qwen3.5-0.8B)

Held-out documents of the training sources, 16 rows x 2,048 tokens (32,752 positions).

| | Value |
|---|---|
| KL(original ‖ converted), next token | 0.144 |
| Perplexity ratio (converted / original) | 1.103 |
| Top-1 agreement with the original | 0.828 |
| Recall of a 5-digit code placed 32 / 512 / 1,500 / 6,000 tokens before the question | 20/20 at every distance (the original: 20/20) |

Writing the model in bf16 and reading it back changes the perplexity ratio by 0.0001.
Source: `research/docs/results/mq-d1-gpu-2026-11-12.md` and `research/runs/2026-11-11_2322_mq-dev-0p8b-rest/` in the GitHub repository.

## Usage

The Memory Mosaics blocks are not part of Hugging Face Transformers. Load the model with the code of the GitHub repository (transformers >= 5.2, < 5.23):

```bash
git clone https://github.com/Eureka-COM/QwenMosaic && cd QwenMosaic && pip install -e .
hf download Eureka-COM/QwenMosaic-0.8B --local-dir QwenMosaic-0.8B
```

```python
import torch
from mmllm.mosaic_qwen import load_standalone
from transformers import AutoTokenizer

mq = load_standalone("QwenMosaic-0.8B", dtype=torch.float32, device="cpu")
tok = AutoTokenizer.from_pretrained("QwenMosaic-0.8B")
text = tok.apply_chat_template([{"role": "user", "content": "What is a memory mosaic?"}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids
out = mq.model.generate(ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

Padding inside a batch is not masked out by the memories: generate one prompt at a time, or batch prompts of equal length.

## License and attribution

Apache License 2.0.

- Base model: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), Apache License 2.0, Copyright 2026 Alibaba Cloud
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

Qwen3.5-0.8B の注意の 6 層を、Meta の公式 Memory Mosaics の層（v2 の構成、短い記憶 8 ヘッドと長い記憶 8 ヘッド）に置き換え、元のモデルを教師に約 2,400 万語で蒸留したモデルです。
元のモデルとの KL は 0.144、PPL 比は 1.103 で、6,000 語前に置いた番号も 20 問中 20 問を取り出しました。
研究の成果物で、元のモデルより良いモデルではありません。動かすには GitHub の Eureka-COM/QwenMosaic のコードが要ります。
