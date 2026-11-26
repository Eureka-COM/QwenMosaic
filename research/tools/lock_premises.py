"""Record the SHA-256 of docs/PREMISES.md (docs/PREMISES.sha256) and re-stamp the main-line plans.

The premises are fixed: this is run only after the researcher has explicitly told us to change
them, and their instruction is recorded with the new hash.

    python tools/lock_premises.py --researcher-instruction "<the instruction, verbatim>"
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def premises_sha(root: Path = ROOT) -> str:
    return hashlib.sha256((root / "docs" / "PREMISES.md").read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--researcher-instruction", required=True,
                    help="the researcher's own words that allow this version of the premises")
    a = ap.parse_args()
    if len(a.researcher_instruction.strip()) < 8:
        sys.exit("give the researcher's instruction verbatim")
    sha = premises_sha()
    lock = ROOT / "docs" / "PREMISES.sha256"
    history = json.loads(lock.read_text())["history"] if lock.exists() else []
    if history and history[-1]["sha256"] == sha:
        print("unchanged", sha)
    else:
        history.append({"sha256": sha, "date": datetime.date.today().isoformat(),
                        "researcher_instruction": a.researcher_instruction})
        lock.write_text(json.dumps({"sha256": sha, "history": history}, ensure_ascii=False, indent=1) + "\n")
        print("locked", sha)
    subprocess.run([sys.executable, str(ROOT / "colab" / "plans" / "make_mq.py")], check=True)


if __name__ == "__main__":
    main()
