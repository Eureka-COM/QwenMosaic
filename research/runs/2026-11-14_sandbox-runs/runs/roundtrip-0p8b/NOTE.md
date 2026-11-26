# mq-roundtrip-0p8b

D1 の段3（8+8 ヘッド）の重み（results/2026-11-11_2322_mq-dev-0p8b-rest の s3-r10/delta）を、メモリ上のまま／fp32 で書き出して読み直し／bf16 で書き出して読み直し、の 3 通りで同じ 1,530 位置を fp32 の計算で比べた（CPU）。scripts/mq_roundtrip.py

- machine: vm (Linux-6.18.44-fc-v378-x86_64-with-glibc2.39)
- code commit: 168eae7
