# 再現の手順

研究の記録（`research/`）の実行を、このリポジトリのコードで繰り返すための手順です。
1 回ごとの正確な設定は実行計画のファイル（`research/colab/plans/<計画名>.json`）に引数ごと残っています。この文書は流れと置き場所だけを書きます。

## 1. 環境

| もの | 版 |
|---|---|
| Python | 3.10 以上（研究の記録の実行環境は 3.12.3 と 3.13.x。この整理の試験は 3.13） |
| PyTorch | 2.5 以上 |
| transformers | 5.2 以上 5.23 未満。5.22.2 では試験がすべて通る。5.24.1 では、生成のキャッシュと、Qwen3・Llama 型の変換の試験の 8 件が落ちる |
| GPU で速く回すとき | CUDA のツールキット（`nvcc`）と `ninja`。公式の漏れ平均の CUDA カーネルを、初めて使うときに組み立てる。無ければ並列のスキャンに切り替わる |
| 任意 | `flash-linear-attention`、`causal-conv1d`（Qwen の Gated DeltaNet の層が速くなる） |

```bash
pip install -e ".[test]"
python -m pytest
```

`python -m pytest` は本体の試験（`tests/`）と、研究の記録に付いていた運用の道具の試験（`research/tests/`）を流します。GPU もネットワークも使いません。
`MMLLM_MODEL` に Qwen3.5-0.8B の置き場を入れると、本物の重みを使う試験も流れます。

研究の記録に付いていた、実行の待ち行列が止まり方を誤らないかの試験は、台本として動かします。

```bash
python research/tests/test_runner_fail_closed.py --runner research/colab/runner.py
```

## 2. データ

学習と評価の台本は、`--data PATH[:FIELD][@WEIGHT]` で文書の置き場を受け取ります（`src/mmllm/mq_data.py`）。
PATH は parquet・jsonl・jsonl.gz・json.gz のファイルか、それらのフォルダです。FIELD は文書の列（既定は `text`）で、会話の列（`messages`）はモデルの会話の書式に直して使います。

研究の記録の実行は次の 4 つを 4:2:2:1 の重みで混ぜていました（データの出典とライセンスは [SOURCES.md](SOURCES.md)）。

```
--data data/cosmopedia-100k@4 --data data/ultrachat:messages@2 --data data/wikipedia-ja@2 --data data/codeparrot:content@1
```

各データの最後の文書（既定では 64 件。ただし文書の数の 1 割まで）は学習に使わず、評価の行はそこから作ります（`src/mmllm/mq_data.py`）。

## 3. 学習の 3 段

| 段 | 台本 | 何を合わせるか | 学習するもの |
|---|---|---|---|
| 段 1（層合わせ） | `scripts/mq_stage1.py` | 置き換えた各層の記憶の出力を、元の注意の出力に 1 層ずつ合わせる。モデル全体の逆伝播はしない | 記憶と正規化 |
| 段 2（全体の蒸留） | `scripts/mq_stage2.py` | 置き換えたモデル全体の次の語の分布を、元のモデルに合わせる（KL） | 記憶・正規化・LoRA（既定） |
| 段 3（長い列） | `scripts/mq_stage2.py` を長い `--seq-len` で | 段 2 と同じ | 段 2 と同じ |

段 1 と段 2 の考え方は RADLADS（arXiv:2505.03005）の手順に沿っています。

```bash
python scripts/mq_stage1.py --model Qwen/Qwen3.5-0.8B --data ... --seq-len 2048 --bsz 4 --tokens 20e6 --out runs/s1
python scripts/mq_stage2.py --model Qwen/Qwen3.5-0.8B --delta-in runs/s1/delta --data ... --seq-len 2048 --bsz 4 --tokens 20e6 --out runs/s2
python scripts/mq_eval.py   --model Qwen/Qwen3.5-0.8B --delta runs/s2/delta --data ... --out runs/eval
python scripts/mq_export.py --model Qwen/Qwen3.5-0.8B --delta runs/s2/delta --out export/mosaic-qwen-0.8b
```

- どの台本も途中経過を `--out` の `progress.jsonl` に 1 行ずつ書きます。止まったら `--resume <前の --out>` で続きから始めます
- 学習したものは `delta/`（記憶・正規化・LoRA の差分）に入ります。`mmllm.load_mosaic_qwen(base_model, delta)` で読み込めます
- `mq_export.py` は LoRA を足し込み、元の注意の層を消した単体のモデルを書き、読み直して注意の重みが 0 個であることと出力が同じことを確かめます

ほかの台本の役目は各ファイルの先頭の説明にあります（`mq_recall_probe.py`：想起の切り分け、`mq_bench.py`：MMLU と GSM8K、`mq_stage1_control.py`：層合わせの対照、`scratch_*.py`：公式の Memory Mosaics を最初から学習する比べ）。

## 4. 必要な GPU の目安

どれも記録にある実測です。

| 規模 | 段 | GPU | 実測 |
|---|---|---|---|
| 0.8B、注意 6 層、8+8 ヘッド | 段 1〜3 | A100 80GB | 段 1・2 は各 1,000 万語、段 3 は 400 万語。比べの 4+4 ヘッドと並べて、2 回の起動で計 78 分（`research/docs/results/mq-d1-gpu-2026-11-12.md`） |
| 27B、注意 16 層、16+16 ヘッド | 段 1 | A100 80GB | 毎秒約 2,680 語、最大 69.1GB |
| 27B | 段 2 | A100 80GB | 勾配チェックポイントが必須。勾配を積み上げると A100 では足りず、RTX PRO 6000 Blackwell（95GB）で毎秒 1,104 語 |
| 27B | 段 3（列 8,192 語） | RTX PRO 6000 Blackwell（95GB） | 毎秒 1,022 語、最大 73.5GiB |

数字の出どころはそれぞれ `research/docs/results/` の P27・段 1・段 2・段 3 の記録です。

## 5. 研究の記録の動かし方

記録の実行は計画のファイルを `research/colab/runner.py` に渡して動かしていました。Colab（`start_cell_mq.py`）、Modal（`modal_mq.py`）、ローカルの GPU（`local_mq.py`、手順は `research/docs/local-5090.md`）の 3 つの入り口があります。
この道具は結果を GitHub の `results/<日時>_<計画名>` ブランチへ、重みを Release へ送る作りで、元のリポジトリ（Eureka-COM/mosaic-qwen）の置き場とトークンの名前を前提にしています。
このリポジトリで使うときは、送り先とトークンを自分の環境に合わせて直す必要があります。詳しくは [research/README.md](../research/README.md) にあります。
