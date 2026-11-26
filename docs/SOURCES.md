# 出典とライセンス

このリポジトリが使っているコード・モデル・データ・論文と、それぞれのライセンスを並べます。
ライセンスは 2026-11-26 に配布元の原文で確かめました。確かめた原文の写しは、作業した PC の調査フォルダ（`~/research/qwenmosaic-release/sources/`）に保存してあり、再配布はしていません。

## このリポジトリに含むもの

| もの | 出典 | ライセンス | 確かめたこと |
|---|---|---|---|
| `third_party/MemoryMosaics/` | [facebookresearch/MemoryMosaics](https://github.com/facebookresearch/MemoryMosaics)（commit c2ca0e8） | Apache License 2.0。Copyright (c) Meta Platforms, Inc. and affiliates | 配布元の LICENSE.txt は apache.org の Apache License 2.0 の本文と完全に一致した（SHA-256 が同じ）。中の Python ファイル 30 個は、どれも先頭に Meta の著作権表示がある |
| それ以外のコードと文書 | Eureka-COM/mosaic-qwen を整理したもの | Apache License 2.0（[LICENSE](../LICENSE)） | |

公式のコードは書き換えていません。2026-11-26 に配布元の commit c2ca0e8（この日の時点で main の最新）を取り直し、同梱した 49 ファイルが 1 バイトも違わないことを確かめました。日常の確かめとしては、`tests/test_official_vendor.py` が、`third_party/MemoryMosaics/UPSTREAM.json` に記録した 1 ファイルずつのハッシュと照合します。
CPU など、公式の部品が動かない場所で使う代わりの計算は `src/mmllm/official.py` にだけ置いています。

## 元にするモデル（含まない。実行するときに Hugging Face から取る）

| モデル | 使い方 | ライセンス（モデルカードの記載） |
|---|---|---|
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.8B の変換・学習の土台と教師 | `license: apache-2.0`。LICENSE は Apache License 2.0、Copyright 2026 Alibaba Cloud |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B の変換・学習の土台と教師 | `license: apache-2.0`。LICENSE は Apache License 2.0、Copyright 2026 Alibaba Cloud |

研究の記録（`research/`）と一部の台本は、ほかに次のモデルを使っています。どれもこのリポジトリには含みません。

| モデル | 使った所 | ライセンス（モデルカードの記載） |
|---|---|---|
| [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 品質が止まる理由の切り分けで、教師を替えた比べ（`research/docs/results/mq-diag-0p8b-2026-11-13.md`） | `license: apache-2.0` |
| [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 同上 | `license: apache-2.0` |
| [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) の tokenizer | 公式の Memory Mosaics を最初から学習する台本（`src/mmllm/scratch.py`、`scripts/scratch_*.py`）。公式の README が、この tokenizer を使うと書いている | `license: apache-2.0` |

## 学習と評価に使ったデータ（含まない）

研究の記録の実行計画（`research/colab/plans/*.json`）が取得していたデータです。

| データ | 使い方 | ライセンス（データセットカードの記載） |
|---|---|---|
| [HuggingFaceTB/cosmopedia-100k](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k) | 学習と評価の文書 | `license: apache-2.0` |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 学習と評価の会話 | `license: mit` |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)（20231101.ja、20231101.en） | 日本語の文書、学習していない種類の文としての英語 | `cc-by-sa-3.0` と `gfdl` |
| [codeparrot/codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid)、[codeparrot/codeparrot-clean-train](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train) | 学習と評価のコード | データセットカードにライセンスの記載がない。各行の `license` の列に、元のファイルのライセンスが入っている |
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | 27B のベンチマーク | `license: mit` |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 27B のベンチマーク（測定として成り立たなかった。記録を参照） | `license: mit` |

データを取って使うときは、それぞれのライセンスに従ってください。学習した重みを配るときは、使ったデータのライセンスの条件も確かめる必要があります。

## 論文

| 論文 | このリポジトリとの関係 |
|---|---|
| Zhang, Nolte, Sadhukhan, Chen, Bottou. Memory Mosaics. ICLR 2025. [arXiv:2405.06394](https://arxiv.org/abs/2405.06394) | Memory Mosaics の元の論文 |
| Zhang, Bottou. Memory Mosaics at scale. NeurIPS 2025. [arXiv:2507.03285](https://arxiv.org/abs/2507.03285) | 置き換えた層の設定（Memory Mosaics v2、`configs/mosaics/memory_mosaics_v2.yaml`）の論文 |
| Zhang, Bottou. Open-world AI: a Tale of Three Requirements. [arXiv:2610.20782](https://arxiv.org/abs/2610.20782) | 同梱した公式コード（c2ca0e8）の版が対応する論文。Memory Mosaics+ と Two-stage Memory Mosaics+ はこの論文のもの。このリポジトリの本線は使っていない |
| Goldstein, Alcaide, Lu, Cheah. RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale. COLM 2025. [arXiv:2505.03005](https://arxiv.org/abs/2505.03005) | 段 1（層ごとに元の注意の出力に合わせる）と段 2（全体の蒸留）の手順の元。コードは使っていない |

論文の著者と題は、arXiv の PDF の 1 ページ目で確かめました。会議名は、PDF の 1 ページ目の表示（ICLR 2025、COLM 2025）と、公式の Memory Mosaics の README（NeurIPS 2025）によります。
