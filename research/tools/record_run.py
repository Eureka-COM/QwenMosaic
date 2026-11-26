"""Send a run made outside Colab (this machine, a local PC) to GitHub, like the Colab runs:
branch results/<YYYY-MM-DD_HHMM>_<name> of this repository, everything in the folder.

    python tools/record_run.py <run folder> <name> [--note "what this was"]
A NOTE.md with the note, the machine and the code commit is added to the folder first.
"""
import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab"))
import saver  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("name"); ap.add_argument("--note", default="")
    a = ap.parse_args()
    folder = Path(a.folder)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (folder / "NOTE.md").write_text(
        f"# {a.name}\n\n{a.note}\n\n- machine: {platform.node()} ({platform.platform()})\n- code commit: {commit}\n")
    s = saver.Saver("git", a.name, base=tempfile.mkdtemp())
    ok = s.save(folder, message=f"{a.name} (local run)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
