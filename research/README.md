# 研究の記録

このフォルダは元のリポジトリ Eureka-COM/mosaic-qwen の研究（2026-11-11〜11-14）の記録です。
文書と実行の生データは、書かれたときのまま写しています。数字や判断を、あとから書き換えてはいません。
いま読める形のまとめは [../docs/RESULTS.md](../docs/RESULTS.md) にあります。

## 中身

| 場所 | 中身 |
|---|---|
| `docs/PREMISES.md` | 研究の前提（第 2 版・改 1）。`docs/PREMISES.sha256` に版ごとのハッシュと、変えたときの研究者の指示がある |
| `docs/plan-mosaic-qwen.md` | 本線の計画。判定の基準と費用の見積もり |
| `docs/plan-scratch.md` | 公式の Memory Mosaics v2 と Transformer を最初から学習して比べる計画。記録にあるのは準備の実行（`runs/2026-11-14_*_mq-scratch-setup`）までで、本番の学習の記録はない |
| `docs/results/` | 実験ごとの結果の文書。`docs/results/data/` は文書が引く数字の元 |
| `docs/reading/` | Memory Mosaics の論文 3 本を原文で読んだ記録 |
| `docs/local-5090.md` | ローカルの RTX 5090（WSL2）で計画を動かす手順 |
| `runs/<日時>_<計画名>/` | 実行ごとの生データ（ログ、途中経過、評価の数字）。下の「ブランチとフォルダ」を参照 |
| `colab/` | 実行の道具（計画を受けて台本を順に動かし、結果を送る runner）と、実行計画（`colab/plans/*.json`） |
| `tools/` | 前提の固定（`lock_premises.py`）、ローカルでの起動、分析の台本 |
| `tests/` | `colab/` の道具の試験 |
| `README-mosaic-qwen.md`、`CLAUDE-mosaic-qwen.md`、`NOTICE-mosaic-qwen` | 元のリポジトリの README・作業の決まり・NOTICE |

## ブランチとフォルダ

元のリポジトリでは実行ごとの結果を `results/<日時>_<計画名>` のブランチに置いていました。
ここでは、そのブランチの中身を `runs/<日時>_<計画名>/` のフォルダに移しています。記録の文書に `results/2026-11-12_1447_mq-27b-s3` のようなブランチ名が出てきたら、`runs/2026-11-12_1447_mq-27b-s3/` を見てください。

- `runs/2026-11-11_1758_mq-cpu/` だけはさらに前のリポジトリ Eureka-COM/memory-mosaics-LLM のブランチから写しました（`docs/results/mq-cpu-2026-11-12.md` がそう書いています）
- 重みのファイル（`*.safetensors`）と、Qwen の `tokenizer.json` は写していません。写さなかったファイルの一覧は `runs/EXCLUDED-weights.txt` にあります

## 学習した重み

学習した重みはこのリポジトリに入れていません。元のリポジトリ（Eureka-COM/mosaic-qwen、非公開）の Release に、次の名前で置いてあります。

| Release | 中身（記録の文書による） |
|---|---|
| `mq-0p8b-dev` | 0.8B・注意 6 層の書き出したモデル（D1） |
| `mq-27b-s1`、`mq-27b-s2`、`mq-27b-s3` | 27B の各段の終わりの差分（記憶・正規化・LoRA） |
| `mq-all-0p8b-s1`〜`-s3`、`mq-all-0p8b-dev` | 0.8B・全 24 層の手順 1 |
| `mq-all-0p8b-docs-s1`、`-s2`、`mq-all-0p8b-docs-dev` | 0.8B・全 24 層の手順 2 |
| `mq-diag-0p8b-*` | 品質の切り分けの診断 |

## 移したときに変えたところ

記録の文書と生データは変えていません。動かす道具は、置き場が変わったぶんだけ次のように直しました。

- `colab/make_cell.py`：包むコードの置き場を、`research/` とリポジトリの最上位の両方から探すようにしました。包みの中の並びは元と同じです
- `tests/test_mq_scripts.py`（リポジトリの最上位の試験）：実行計画の置き場を `research/colab/plans/` に直しました
- `tests/test_premises.py`・`tests/test_colab.py`・`tests/test_runner_fail_closed.py` は、元の `tests/` からここへ移しました

`colab/` の道具は結果を元のリポジトリのブランチと Release へ送る作りで、トークンの名前（`MM_GITHUB_TOKEN` など）も元の環境のものです。このリポジトリで動かすときは、送り先とトークンを直してください。

## 前提

元のリポジトリでは `docs/PREMISES.md` を研究者の指示なしに変えない決まりで、計画は前提のハッシュが合わないと起動しませんでした（`colab/runner.py` の `check_premises`）。
ここに写した前提も研究の記録として、書き換えずに残しています。
