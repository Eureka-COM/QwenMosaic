# mosaic-qwen

学習済みの LLM（Qwen3.5 / Qwen3.8 系：Gated DeltaNet 3 層＋注意 1 層の繰り返し）の**注意の層をすべて、
Meta FAIR の公式 Memory Mosaics の層に置き換えて**、27B 級のモデルを作る研究リポジトリ。

## 本線（2026-11-12〜）：MosaicQwen

- 前提：[docs/PREMISES.md](docs/PREMISES.md)（第 2 版）　計画・費用・判定の基準：[docs/plan-mosaic-qwen.md](docs/plan-mosaic-qwen.md)
- CPU で確かめた結果：[docs/results/mq-cpu-2026-11-12.md](docs/results/mq-cpu-2026-11-12.md)
- GPU で 0.8B の全手順を通した結果（D1、判定：合格）：[docs/results/mq-d1-gpu-2026-11-12.md](docs/results/mq-d1-gpu-2026-11-12.md)
- 本物の 27B での実測（P27：速さ・メモリ・費用の計算し直し）：[docs/results/mq-p27-2026-11-12.md](docs/results/mq-p27-2026-11-12.md)
- 27B 段 1（層合わせ、3,000 万トークン）の結果：[docs/results/mq-27b-s1-2026-11-12.md](docs/results/mq-27b-s1-2026-11-12.md)
- 27B 段 2（全体蒸留、3,000 万トークン）の結果：[docs/results/mq-27b-s2-2026-11-12.md](docs/results/mq-27b-s2-2026-11-12.md)
- 27B 段 3（長い列、800 万トークン）と最後の評価：[docs/results/mq-27b-s3-2026-11-12.md](docs/results/mq-27b-s3-2026-11-12.md)（KL 0.166、6,000 トークン前の想起は 0/20）
- 教師（元の 27B）がどの位置で外しているか：[docs/results/mq-teacher-positions-2026-11-12.md](docs/results/mq-teacher-positions-2026-11-12.md)
- 27B の想起 0/20 の見直し（0.8B とのやり方の差、GPU なしで確かめたこと）：[docs/results/mq-27b-recall-review-2026-11-12.md](docs/results/mq-27b-recall-review-2026-11-12.md)
- 27B の想起の切り分け（会話なしの詰め物・会話の形では 6,000 トークン前を想起できる）：[docs/results/mq-27b-recall-probe-2026-11-12.md](docs/results/mq-27b-recall-probe-2026-11-12.md)
- 27B 段 2 の重みの最後の評価・MMLU・書き出しの確認（KL 0.165。MMLU は元の 27B 84.3% に対し 77.1%、数学で差が大きい）：[docs/results/mq-27b-s2-final-2026-11-13.md](docs/results/mq-27b-s2-final-2026-11-13.md)
- 0.8B の全 24 層を置き換えた結果（§10 手順 1、判定：保留。PPL 比 1.530・KL 0.492、想起は 6,000 トークン前で 19/20）：[docs/results/mq-all-0p8b-2026-11-13.md](docs/results/mq-all-0p8b-2026-11-13.md)
- 0.8B の全 24 層・論文に寄せた学習（§10 手順 2、判定：出なかった。PPL 比 1.900・KL 0.796 で品質の条件を満たさず。文脈から学ぶ量 G は教師 0.244 に対し 0.412 だが、出発点の損失が悪い）：[docs/results/mq-all-0p8b-docs-2026-11-13.md](docs/results/mq-all-0p8b-docs-2026-11-13.md)
- 品質が止まる理由の切り分け（§10 の診断。診断 1：元と同じ型の層を初期化し直しても層合わせの誤差は 0.25 で止まる＝壁は記憶の限界ではなく手順・学習量の側。診断 2：先生を Qwen3-0.6B-Base・MiniCPM5-1B に替えても 0.29・0.22 でほぼ同じ。3〜8：学習量はゆっくり効く、全部の重みは少し効く、蒸留なしは PPL 1.379 だが慣れと未分離、元の重みからの初期値・少しずつの入れ替え・学習率は効かない。夜間の長い学習（全部の重み、KL + 次トークン、1.6 億）は PPL 比 1.153・KL 0.408 で「大きく近づく」、ただし学習していない英語 Wikipedia では 1.718・KL 0.577）：[docs/results/mq-diag-0p8b-2026-11-13.md](docs/results/mq-diag-0p8b-2026-11-13.md)
- Memory Mosaics の論文 3 本を原文で読んだ記録と、今の構成・学習との照合：[docs/reading/memory-mosaics-papers-2026-11-13.md](docs/reading/memory-mosaics-papers-2026-11-13.md)

```
元の注意層                          置き換え後（公式 MosaicBlock、Memory Mosaics v2 の構成）
  x += attention(norm(x))             h  = attention_norm(x)
  x += mlp(norm(x))                   x += short(h) + long(h)          公式 MosaicsMemory × 2（直近 256／64 以上前）
                                      x += feed_forward(ffn_norm(x))   持続の記憶＝その層の Qwen の MLP の重みで開始
```

| 場所 | 中身 |
|---|---|
| `third_party/MemoryMosaics/` | 公式リポジトリ（commit c2ca0e8）の無改変の写し。`UPSTREAM.json` のハッシュと照合してから使う |
| `src/mmllm/official.py` | 公式コードの読み込み。CPU など公式の部品が動かない場所での代わりの計算（一致を試験済み） |
| `src/mmllm/mosaic_qwen.py` | 変換（`convert_to_mosaic_qwen`）、教師モード、生成キャッシュ、LoRA、保存と読み込み |
| `scripts/mq_stage1.py` | 段 1：層ごとに、元の注意の出力を新しい記憶に真似させる（モデル全体の逆伝播なし） |
| `scripts/mq_stage2.py` | 段 2・段 3：置き換えたモデル全体を、元のモデルの出力に合わせる（記憶＋LoRA） |
| `scripts/mq_eval.py` / `mq_export.py` | 元のモデルとの比較（KL・PPL 比・想起・生成）／注意を含まない単体モデルの書き出し |
| `scripts/mq_env.py` / `mq_port27.py` | GPU の事前点検／27B の実寸の層での変換確認 |
| `colab/plans/mq-*.json` | Colab・Modal 用の計画（`colab/start_cell_mq.py`、`colab/modal_mq.py`） |

```python
from mmllm import load_mosaic_qwen
mq, tok = load_mosaic_qwen("Qwen/Qwen3.5-0.8B", "runs/s2/delta")     # 元の重み＋学習した差分。注意の重みは読み込み後に捨てる
ids = tok("Memory Mosaics are", return_tensors="pt").input_ids
print(tok.decode(mq.model.generate(ids, max_new_tokens=32)[0]))
assert mq.attention_parameters() == []                                # 注意のパラメータは 1 つも残っていない
```

試験：`python -m pytest tests/test_official_vendor.py tests/test_mosaic_qwen.py tests/test_mq_scripts.py`（CPU、約 2 分）

## 経緯
- 2026-11-12 に Eureka-COM/memory-mosaics-LLM（ブランチ feature/mosaic-qwen、commit 368d184）から、本線だけを取り出して作ったリポジトリ
- それ以前の線（元の注意を窓付きで残し、横に Mosaics の記憶を足す組み込み。checks12〜24）の記録と結果は、元のリポジトリに残してある。本線の結果としては使わない
- CPU 検証（docs/results/mq-cpu-2026-11-12.md）の生データも、元のリポジトリの `results/2026-11-11_1758_mq-cpu` ブランチにある
