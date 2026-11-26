"""Compare two runs of scripts/scratch_train.py checkpoint by checkpoint (docs/plan-scratch.md).

    python scripts/scratch_compare.py --mm runs/main-mm --tf runs/main-tf [--out DIR]

For every checkpoint both runs have reached (same step): the held-out loss difference
delta = loss(MM) - loss(TF), and on the same long documents, paired per document, with bootstrap
95% intervals:
    E = early(MM) - early(TF)      the gap at positions 50-100 of a document
    L = late(MM)  - late(TF)       the gap at positions 1000-2000
    D = E - L                      how much more the gap closes (or widens) along the document:
                                   D > 0 = Memory Mosaics gains more from the earlier context
Then the rules of the plan: the stop rule at 25% / 50% and the final verdict at 100%.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm.mq_train import write_json  # noqa: E402
from mmllm.scratch import bootstrap_mean  # noqa: E402

STOP_GAP = 0.30          # nats: Memory Mosaics this much worse at 25% ...
NEAR = 0.05              # nats: "as good as" for the overall loss


def evals(run: Path) -> dict:
    return {e["step"]: e for e in (json.loads(l) for l in (run / "evals.jsonl").read_text().splitlines())}


def compare(mm: dict, tf: dict, total_steps: int) -> dict:
    out = {"checkpoints": []}
    for step in sorted(set(mm) & set(tf)):
        a, b = mm[step], tf[step]
        E = [x - y for x, y in zip(a["early_per_doc"], b["early_per_doc"])]
        L = [x - y for x, y in zip(a["late_per_doc"], b["late_per_doc"])]
        D = [e - l for e, l in zip(E, L)]
        out["checkpoints"].append({
            "step": step, "share": round(step / total_steps, 3), "tokens": a["tokens"],
            "loss": {"mm": a["heldout_loss"], "tf": b["heldout_loss"], "delta": round(a["heldout_loss"] - b["heldout_loss"], 4)},
            "icl_score": {"mm": a["icl_score"], "tf": b["icl_score"]},
            "E": bootstrap_mean(E), "L": bootstrap_mean(L), "D": bootstrap_mean(D),
            "position": {"mm": a["position"], "tf": b["position"]},
            "recall": {d: {"mm": a["recall"].get(d, {}).get("acc"), "tf": b["recall"].get(d, {}).get("acc")} for d in a["recall"]}})
    cps = {round(c["share"], 2): c for c in out["checkpoints"]}
    rule = {}
    if 0.25 in cps:
        g25 = cps[0.25]["loss"]["delta"]
        rule["gap_25"] = g25
        if g25 > STOP_GAP and 0.5 in cps:
            g50 = cps[0.5]["loss"]["delta"]
            rule["gap_50"] = g50
            rule["stop"] = g50 > g25
        elif g25 > STOP_GAP:
            rule["stop"] = "decide at 50%"
        else:
            rule["stop"] = False
    out["stop_rule"] = rule
    if 1.0 in cps:
        c = cps[1.0]
        D, L, delta = c["D"], c["L"], c["loss"]["delta"]
        if L["hi"] < 0:
            v = "文脈の奥で上回った"
        elif D["lo"] > 0:
            v = "文脈からより多く学ぶ"
        elif D["hi"] < 0:
            v = "Transformer のほうが文脈から学ぶ"
        else:
            v = "差なし"
        out["verdict"] = {"verdict": v, "overall_worse": delta > 0.10, "delta": delta, "D": D, "L": L}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    mm, tf = Path(args.mm), Path(args.tf)
    total = json.loads((mm / "run.json").read_text())["total_steps"]
    res = compare(evals(mm), evals(tf), total)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.out) / "compare.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "checkpoints"}, ensure_ascii=False))
    for c in res["checkpoints"]:
        print(c["share"], c["loss"], "D", c["D"], "L", c["L"], "recall", c["recall"])


if __name__ == "__main__":
    main()
