# クラウドの作業場所（Claude の Linux の環境）で走らせたもの

2026-11-11〜13 に、GPU の計画とは別に、Claude の作業場所（CPU）で行った確認・分析の出力（`/home/claude/runs`）。GPU で走らせた計画の結果は、それぞれの `results/<日時>_<計画名>` ブランチにある。CPU の検証のまとめ（cpu-s1・cpu-s2・cpu-eval・cpu-export・port27・env-cpu）は Eureka-COM/memory-mosaics-llm の `results/2026-11-11_1758_mq-cpu` にもある。

## 含めていないもの（40 MB を超えるファイル）

最適化の状態（optim.pt）、書き出した単体モデル、差分の重み。どれも元の重みと同じブランチの設定・差分から作り直せる。d1-s3-r10-delta は Release の重みを手元へ取ってきた写し。

- runs/cpu-export-weights/model-00000.safetensors
- runs/cpu-s1/delta/delta-00000.safetensors
- runs/cpu-s1/optim.pt
- runs/cpu-s2/delta/delta-00000.safetensors
- runs/d1-s3-r10-delta/delta-00000.safetensors
- runs/s1-smoke/delta/delta-00000.safetensors
- runs/s1-smoke/optim.pt

## フォルダ

- `bench08`：bench.json gsm_items.json mmlu_items.json progress.jsonl result.json
- `causality`：NOTE.md cpu_causality.json cpu_stdout_tail.log g4_summary.txt progress.jsonl
- `causality-cpu`：causality.json
- `cpu-eval`：metrics.json progress.jsonl recall.json result.json samples.json
- `cpu-export`：export_report.json progress.jsonl
- `cpu-export-weights`：chat_template.jinja config.json mq_config.json tokenizer.json tokenizer_config.json
- `cpu-s1`：baseline.json delta progress.jsonl result.json run.json
- `cpu-s2`：delta progress.jsonl result.json run.json
- `d1-s3-r10-delta`：delta-00001.safetensors mq_config.json
- `env-all-cpu`：env.json
- `env-cpu`：env.json progress.jsonl
- `evalrows`：row8.py rows.json rows.py trainrows.py
- `evals3`：metrics.json recall.json result.json samples.json
- `flexcheck`：cpu.json emul27.py flex_vs_dense.py recall_ctx.json recall_ctx.py …
- `port27`：port27.json progress.jsonl
- `pos08`：log.txt positions_d1s3.json report
- `pos27`：positions_s2.json positions_s3.json s2 s3
- `probe08`：probe.json progress.jsonl result.json
- `probe27`：probe.json progress.jsonl result.json
- `roundtrip-0p8b`：NOTE.md progress.jsonl roundtrip.json stdout.log
- `s1-smoke`：delta progress.jsonl result.json run.json
- `scalars`：all_scalars.json compare.py fetch.py scalars_27b_s1.json scalars_27b_s2.json …
- `tpos`：an dump.json pos.json teacher_positions.json
- `tpos27`：an split.json split.py teacher_positions.json

記録：`bench08.log` `causality-cpu.log` `cpu-s1.log` `cpu-s2.log` `env-all-cpu.log` `probe08.log` `roundtrip-0p8b.log` `roundtrip.sh` `s1-smoke.log`
