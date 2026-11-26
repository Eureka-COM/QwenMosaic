"""Section 10, step 2: apply the criteria fixed before the results (docs/plan-mosaic-qwen.md) to a results tree.

    python tools/analysis-2026-11-13/step2_verdict.py <dir with context/, eval-s2-docs/, evalc-s2-docs/>

Prints the numbers the criteria use and the verdict, once with the evaluation rows fixed beforehand
(eval-s2-docs: the rows of step 1) and once with the rows no training read (evalc-s2-docs). Changes nothing.
"""
import json
import sys
from pathlib import Path

PPL_MAX, KL_MAX, SHARE, G_MIN = 1.60, 0.60, 0.7, 0.02


def verdict(ctx: dict, metrics: dict, student: str = "docs", step1: str = "all") -> dict:
    m, pairs = ctx["models"], ctx["pairs"]
    g_t, g_s, g_1 = m["teacher"]["G"]["mean"], m[student]["G"]["mean"], m[step1]["G"]["mean"]
    over_1, over_t = pairs[f"{student} - {step1}"], pairs[f"{student} - teacher"]
    checks = {
        "G(teacher) >= 0.02": g_t >= G_MIN,
        "G(step 2) >= 0.7 x G(teacher)": g_s >= SHARE * g_t,
        "G(step 2) > G(step 1), interval excludes 0": over_1["lo"] > 0,
        "PPL ratio <= 1.60 and KL <= 0.60": metrics["ppl_ratio"] <= PPL_MAX and metrics["kl"] <= KL_MAX,
        "G(step 2) > G(teacher), interval excludes 0": over_t["lo"] > 0,
    }
    first4 = list(checks.values())[:4]
    if not checks["G(teacher) >= 0.02"]:
        v = "測れない"
    elif all(first4):
        v = "教師を上回った" if checks["G(step 2) > G(teacher), interval excludes 0"] else "性質が出た"
    else:
        v = "出なかった"
    return {"verdict": v, "checks": checks,
            "G": {"teacher": g_t, "step1": g_1, "step2": g_s, "step2/teacher": round(g_s / g_t, 4) if g_t else None,
                  "step2 - step1": over_1, "step2 - teacher": over_t},
            "ppl_ratio": metrics["ppl_ratio"], "kl": metrics["kl"]}


def main() -> None:
    root = Path(sys.argv[1])
    ctx = json.loads((root / "context" / "context.json").read_text())
    for name in ("eval-s2-docs", "evalc-s2-docs"):
        f = root / name / "metrics.json"
        if f.exists():
            print(name, json.dumps(verdict(ctx, json.loads(f.read_text())), ensure_ascii=False, indent=1))
    for label, d in ctx["models"].items():
        print(label, "loss", d["loss"], "G", d["G"], "unrelated", d["G_unrelated"], "by source", d["G_by_source"])


if __name__ == "__main__":
    main()
