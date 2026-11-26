# mq-cpu

MosaicQwen（docs/plan-mosaic-qwen.md）の CPU 検証：試験、27B の実寸の層での変換確認（port27）、本物の Qwen3.5-0.8B での段1（150 ステップ）・段2（40 ステップ）・評価・書き出し。まとめは feature/mosaic-qwen の docs/results/mq-cpu-2026-11-12.md。書き出した単体モデル（1.5GB）は含めていない（元の重み＋ s2/delta から作り直せる）。

- machine: vm (Linux-6.18.44-fc-v378-x86_64-with-glibc2.39)
- code commit: 91135a0
