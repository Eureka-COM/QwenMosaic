# mosaic-qwen

## 最初に読む
- **前提は固定**：`docs/PREMISES.md`（第 2 版・確定）。研究者（リポジトリの持ち主）の明示の指示なしに、前提を変えない・読み替えない・足さない。前提に無いことを決める必要が出たら、決めずに研究者に聞く
- `docs/PREMISES.md` を編集すると試験（`tests/test_premises.py`）が落ち、本線の計画（`colab/plans/mq-*.json`）は runner が起動しなくなる。直すのではなく、編集を戻す。変えるのは研究者の指示があるときだけ（`tools/lock_premises.py --researcher-instruction "<原文>"`）
- 本線の計画：`docs/plan-mosaic-qwen.md`（構成・判定の基準・費用）。判定の基準は結果を見る前に決めてある。結果を見てから動かさない
- 計画の文書（`docs/plan-*.md`）は、冒頭に「前提との照合」の節を置く

## 本線
Qwen3.5 / Qwen3.8 の注意の層を**すべて**、公式 Memory Mosaics（`third_party/MemoryMosaics/`、無改変）の層に置き換える。
- 元の注意を残した形、一部の層だけ置き換えた形は、完成形にしない
- `third_party/MemoryMosaics/` のファイルは編集しない（ハッシュを照合している）。公式の部品が動かない場所の代わりの計算は `src/mmllm/official.py` にだけ置く
- 以前の線（元の注意を窓付きで残した組み込み、checks12〜24）は Eureka-COM/memory-mosaics-LLM にある。本線の結果としては使わず、そのコードをここへ持ち込まない

## 進め方
- 走らせる場所（前提 3、2026-11-13 の改 1）：今回の規模の作業（Qwen3.5-0.8B の試行）は研究者のローカルの RTX 5090（WSL2。手順は `docs/local-5090.md`、起動は研究者が行う）。大きいもの（27B など）は Colab か Modal。境目の作業は研究者に聞く
- GPU を使う前に、損益の分かれ目まで詰める。GPU のメモリに空きがあれば並列に回す
- 長く走るものは `progress.jsonl` で途中経過を追えるようにする
- 実施したものは、すべて GitHub に残す（結果は `results/<日時>_<名前>` ブランチ、27B の重みは Release）
- 論文の著者への問い合わせはしない
- 試験：`python -m pytest tests/test_premises.py tests/test_official_vendor.py tests/test_mosaic_qwen.py tests/test_mq_scripts.py`
