# QwenMosaic

学習済みの Qwen3.5 / Qwen3.8 の注意の層を、Meta FAIR の公式 Memory Mosaics の層に置き換える道具と、その研究の記録です。
置き換えたあとのモデルには、元の注意の層とその重み（q・k・v・o の射影）が残りません。
置き換えた層は過去の入力から作った鍵と値を、ガウスカーネルの回帰で読み出します（重みの付け方は softmax です）。

このリポジトリは Eureka-COM/mosaic-qwen（2026-11-11〜11-14 の研究）を、ほかの人や別の実装へ渡せる形に整理したものです。
公式の Memory Mosaics のコードは無改変のまま `third_party/MemoryMosaics/` に入っています。

## できること

- Hugging Face Transformers の Qwen3.5 / Qwen3.8 のモデルを読み込み、注意の層（`full_attention`）をすべて公式の `MosaicBlock` に置き換えます。`replace="all"` を指定すると Gated DeltaNet の層も含めた全層を置き換えます
- 置き換えた層の FFN と正規化は元の層の重みから始めます。新しく学習するのは短い記憶と長い記憶です
- 元のモデルを教師にして、3 段の蒸留（層合わせ、全体の蒸留、長い列）で記憶を学習します
- 学習した差分（記憶・正規化・LoRA）の保存と読み込み、注意の重みを含まない単体のモデルへの書き出し、`model.generate` による生成ができます
- 置き換えた 1 層の入出力と重みを、別の実装の答え合わせ用に書き出せます（`examples/dump_layer_reference.py`）

## CPU だけで試す

```bash
pip install -e ".[test]"
python -m pytest                                   # 本体と記録の道具の試験。GPU もネットワークも使わない
python examples/dump_layer_reference.py --out reference.safetensors
python tools/verify_results.py                     # docs/RESULTS.md の数字を生データと照合する
```

transformers は 5.22 までを使ってください。5.24.1 では生成のキャッシュと Qwen3・Llama 型の変換の試験が落ちます（[docs/REPRODUCE.md](docs/REPRODUCE.md)）。
`official.py` は `third_party/MemoryMosaics/` をリポジトリの中から読むので、`pip install -e .`（編集可能なインストール）で入れます。

本物の Qwen3.5-0.8B で変換する最小の例です。

```python
import torch
from transformers import AutoModelForCausalLM
from mmllm.mosaic_qwen import convert_to_mosaic_qwen

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32)
mq = convert_to_mosaic_qwen(model)          # 注意の 6 層を置き換える。設定は公式の Memory Mosaics v2 の値
print(mq.cfg.layers, mq.num_memory_parameters())
with mq.teacher():                          # 元のモデルとして動かす（学習中の教師）
    ...
mq.drop_teacher()                           # 元の注意の層を捨てる
```

## 結果のあらまし

数字はすべて `research/` の実行記録から取り、`tools/verify_results.py` で生データと照合しています。詳しくは [docs/RESULTS.md](docs/RESULTS.md) にあります。表の「語」はトークンのことです。

| 置き換えたもの | 教師との KL | PPL 比 | 想起（6,000 語前） | 判定（計画の基準） |
|---|---|---|---|---|
| Qwen3.5-0.8B の注意 6 層 | 0.144 | 1.103 | 20/20 | 合格 |
| Qwen3.8-27B の注意 16 層 | 0.166 | 0.897 | 0/20（評価の作りによる。下を参照） | 数値の基準は満たした |
| Qwen3.5-0.8B の全 24 層 | 0.492 | 1.530 | 19/20 | 保留 |

- 27B の想起の 0/20 は評価の詰め物と質問の入れ方で起きていました。会話を含まない詰め物では 6,000 語前でも 10 問中 10 問を取り出しました
- 27B の段 2 の重みは MMLU 1,000 問で元の 84.3% に対して 77.1% でした。数学 5 科目での差が大きく出ています
- 全 24 層の置き換えは、記録の範囲では合格の線に届いていません。いちばん長い学習（1.6 億語）でも KL は 0.408（合格の線は 0.30）で、学習していない種類の文（英語 Wikipedia）では KL 0.577 まで開きます。品質が止まる理由の切り分けも記録にあります

## 中身

| 場所 | 中身 |
|---|---|
| `src/mmllm/` | 変換（`mosaic_qwen.py`）、公式コードの読み込みと CPU 用の参照の計算（`official.py`）、学習とデータ |
| `third_party/MemoryMosaics/` | 公式リポジトリ（commit c2ca0e8）の無改変の写し。`UPSTREAM.json` のハッシュと照合してから使う |
| `scripts/` | 学習（段 1〜3）、評価、書き出し、測定の台本 |
| `examples/` | 置き換えた 1 層の参照データの書き出し |
| `tests/` | 本体の試験 |
| `tools/verify_results.py` | 結果の数字と生データの照合 |
| `docs/` | 層の仕組み（[ARCHITECTURE](docs/ARCHITECTURE.md)）、再現の手順（[REPRODUCE](docs/REPRODUCE.md)）、結果（[RESULTS](docs/RESULTS.md)）、出典とライセンス（[SOURCES](docs/SOURCES.md)） |
| `research/` | 元の研究の記録。前提・計画・結果の文書と、実行ごとの生データ（[research/README.md](research/README.md)） |

## ライセンスと出典

このリポジトリは Apache License 2.0 です（[LICENSE](LICENSE)、[NOTICE](NOTICE)）。

- 公式の Memory Mosaics のコード（facebookresearch/MemoryMosaics）は Apache License 2.0 で、Copyright は Meta Platforms, Inc. and affiliates です
- 元にするモデルの Qwen/Qwen3.5-0.8B と Qwen/Qwen3.8-27B は Apache License 2.0（Copyright 2026 Alibaba Cloud）です。重みはこのリポジトリに含めず、実行するときに Hugging Face から取得します
- 段 1・段 2 の学習の手順は RADLADS（arXiv:2505.03005）の手順に沿っています。そのコードは含めていません

論文と原典の一覧は [docs/SOURCES.md](docs/SOURCES.md) にあります。

公式リポジトリの README に載っている引用の書き方をそのまま写します。

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
