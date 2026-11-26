"""Copy a checkpoint from a store (mmllm/store.py) to a local directory.

    python scripts/mq_fetch.py --store github-release:mq-27b-s1 --dest /content/prev/s1/delta --out runs/fetch
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmllm.store import open_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--require-tag", default=None, help="fail unless the checkpoint was saved with this tag "
                    "(done = the stage ran to its end; periodic/budget = it was still going)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = open_store(a.store)
    ok = store.pull(a.dest)
    files = sorted(p.name for p in Path(a.dest).iterdir()) if ok else []
    meta = json.loads((Path(a.dest) / "mq_config.json").read_text()).get("extra", {}) if "mq_config.json" in files else {}
    rep = {"store": store.describe(), "found": ok, "files": files, "checkpoint": meta}
    (out / "fetch.json").write_text(json.dumps(rep, indent=1))
    (out / "progress.jsonl").write_text(json.dumps({"event": "done" if ok else "missing", "step": 1}) + "\n")
    print(json.dumps(rep))
    if not ok or "mq_config.json" not in files:
        raise SystemExit(f"nothing to fetch in {store.describe()}")
    if a.require_tag and meta.get("tag") != a.require_tag:
        raise SystemExit(f"the checkpoint in {store.describe()} is tagged {meta.get('tag')!r} (step {meta.get('step')}), "
                         f"not {a.require_tag!r}: the stage before this one has not finished")


if __name__ == "__main__":
    main()
