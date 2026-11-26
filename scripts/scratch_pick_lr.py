"""Pick the learning rate of the short runs with the lowest held-out loss at their end, and write
it where the main runs read it (--lr-from).

    python scripts/scratch_pick_lr.py --run /content/out/lr-3e-4 --run /content/out/lr-6e-4 ... --out /content/out/lr-pick

The upstream practice (Memory Mosaics at scale, App. C): the hyperparameters are tuned for the
Transformer and used for Memory Mosaics without tuning again; the runs given here are Transformer runs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm.mq_train import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True, help="the run's folder")
    ap.add_argument("--write", default=None, help="also write lr.json here (where the main runs read it)")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    rows = []
    for r in args.run:
        evals = [json.loads(l) for l in (Path(r) / "evals.jsonl").read_text().splitlines()]
        lr = json.loads((Path(r) / "run.json").read_text())["args"]["lr"]
        last = evals[-1]
        rows.append({"run": r, "lr": lr, "tokens": last["tokens"], "heldout_loss": last["heldout_loss"]})
    finite = [x for x in rows if x["heldout_loss"] == x["heldout_loss"]]
    best = min(finite, key=lambda x: x["heldout_loss"])
    choice = {"lr": best["lr"], "runs": rows, "rule": "lowest held-out loss at the end"}
    for d in {args.out, args.write} - {None}:
        Path(d).mkdir(parents=True, exist_ok=True)
        write_json(Path(d) / "lr.json", choice)
    print(json.dumps({"lr": best["lr"], "runs": rows}))


if __name__ == "__main__":
    main()
