#!/usr/bin/env python3
"""docs/RESULTS.md に載せた数字を、実行の生データ（research/runs/ と research/docs/results/data/）から
読み直して照合する。1 つでも合わなければ 1 で終わる。

    python tools/verify_results.py

数字は記録どおりの桁で丸めて比べる（記録の文書が丸めた値を書いているため）。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research" / "runs"
DATA = ROOT / "research" / "docs" / "results" / "data"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def recall_counts(p: Path) -> dict:
    """recall.json（1 問ずつの記録）から、距離ごとの 生徒の正解数/問題数 を数える。"""
    c = defaultdict(lambda: [0, 0])
    for x in load(p)["cases"]:
        c[x["distance"]][0] += bool(x["student_ok"])
        c[x["distance"]][1] += 1
    return {k: f"{v[0]}/{v[1]}" for k, v in sorted(c.items())}


def recall_summary(p: Path) -> dict:
    """recall_summary.json（距離ごとの割合）から、距離ごとの 生徒の正解数/問題数 を出す。"""
    d = load(p)
    d = d.get("summary", d)
    return {int(k): f"{round(v['student'] * v['n'])}/{v['n']}" for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))}


# (RESULTS.md での呼び名, 生データ, 取り出し方, 記録の値)
CHECKS = [
    ("0.8B 注意 6 層・段 3 のあと KL", RUNS / "2026-11-11_2322_mq-dev-0p8b-rest/eval-s3-r10/metrics.json",
     lambda d: round(d["kl"], 3), 0.144),
    ("0.8B 注意 6 層・段 3 のあと PPL 比", RUNS / "2026-11-11_2322_mq-dev-0p8b-rest/eval-s3-r10/metrics.json",
     lambda d: round(d["ppl_ratio"], 3), 1.103),
    ("0.8B 注意 6 層・想起", RUNS / "2026-11-11_2322_mq-dev-0p8b-rest/eval-s3-r10/recall.json",
     None, {32: "20/20", 512: "20/20", 1500: "20/20", 6000: "20/20"}),
    ("27B 注意 16 層・段 3 のあと KL", RUNS / "2026-11-12_1447_mq-27b-s3/eval-s3/metrics.json",
     lambda d: round(d["kl"], 3), 0.166),
    ("27B 注意 16 層・段 3 のあと PPL 比", RUNS / "2026-11-12_1447_mq-27b-s3/eval-s3/metrics.json",
     lambda d: round(d["ppl_ratio"], 3), 0.897),
    ("27B 注意 16 層・段 3 のあと top-1 の一致", RUNS / "2026-11-12_1447_mq-27b-s3/eval-s3/metrics.json",
     lambda d: round(d["top1"], 3), 0.840),
    ("27B 注意 16 層・評価の想起", RUNS / "2026-11-12_1447_mq-27b-s3/eval-s3/recall.json",
     None, {32: "17/20", 512: "17/20", 1500: "12/20", 6000: "0/20"}),
    ("27B 段 2 の重み・KL", RUNS / "2026-11-13_0016_mq-27b-s2-final/eval-s2/metrics.json",
     lambda d: round(d["kl"], 3), 0.165),
    ("27B 段 2 の重み・MMLU（元）", RUNS / "2026-11-13_0016_mq-27b-s2-final/bench-s2/result.json",
     lambda d: d["mmlu"]["teacher"], 0.843),
    ("27B 段 2 の重み・MMLU（置き換え後）", RUNS / "2026-11-13_0016_mq-27b-s2-final/bench-s2/result.json",
     lambda d: d["mmlu"]["student"], 0.771),
    ("27B 段 2 の重み・MMLU の問題数", RUNS / "2026-11-13_0016_mq-27b-s2-final/bench-s2/result.json",
     lambda d: d["mmlu"]["n"], 1000),
    ("0.8B 全 24 層・段 3 のあと KL", DATA / "mq-all-0p8b/eval-s3-all.metrics.json",
     lambda d: round(d["kl"], 3), 0.492),
    ("0.8B 全 24 層・段 3 のあと PPL 比", DATA / "mq-all-0p8b/eval-s3-all.metrics.json",
     lambda d: round(d["ppl_ratio"], 3), 1.530),
    ("0.8B 全 24 層・想起", DATA / "mq-all-0p8b/eval-s3-all.recall_summary.json",
     None, {32: "20/20", 512: "20/20", 1500: "20/20", 6000: "19/20"}),
    ("0.8B 全 24 層・論文に寄せた学習 KL", DATA / "mq-all-0p8b-docs/eval-s2-docs.metrics.json",
     lambda d: round(d["kl"], 3), 0.796),
    ("0.8B 全 24 層・論文に寄せた学習 PPL 比", DATA / "mq-all-0p8b-docs/eval-s2-docs.metrics.json",
     lambda d: round(d["ppl_ratio"], 3), 1.900),
    ("0.8B 全 24 層・長い学習 KL", DATA / "mq-diag-0p8b/evalc-long-all.metrics.json",
     lambda d: round(d["kl"], 3), 0.408),
    ("0.8B 全 24 層・長い学習 PPL 比", DATA / "mq-diag-0p8b/evalc-long-all.metrics.json",
     lambda d: round(d["ppl_ratio"], 3), 1.153),
    ("0.8B 全 24 層・長い学習 英語 Wikipedia の KL", DATA / "mq-diag-0p8b/ood-long.metrics.json",
     lambda d: round(d["kl"], 3), 0.577),
    ("0.8B 全 24 層・長い学習 英語 Wikipedia の PPL 比", DATA / "mq-diag-0p8b/ood-long.metrics.json",
     lambda d: round(d["ppl_ratio"], 3), 1.718),
]


def main() -> int:
    bad = 0
    for name, path, get, want in CHECKS:
        if not path.exists():
            print(f"なし  {name}: {path.relative_to(ROOT)} がない")
            bad += 1
            continue
        if get is None:
            got = recall_counts(path) if path.name == "recall.json" else recall_summary(path)
            got = {k: got.get(k) for k in want}
        else:
            got = get(load(path))
        ok = got == want
        bad += not ok
        print(f"{'一致' if ok else '違う'}  {name}: 記録 {want} / 生データ {got}  ({path.relative_to(ROOT)})")
    print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} 件が一致")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
