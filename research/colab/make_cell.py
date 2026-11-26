"""Build one self-contained Colab cell: this repository's code (src/, scripts/, colab/) packed
inside the cell, plus an experiment plan. The cell unpacks the code to /content/mmllm,
records the code's git commit, and calls colab.runner.run_all(PLAN).

    python colab/make_cell.py colab/plans/e1_long_types.json > /tmp/cell.py
The code snapshot is also saved with the results (out/code.tar.gz), so every result can be
reproduced with the exact code that produced it.
"""

import base64
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # research/ (QwenMosaic へ移したあとの置き場)
REPO = ROOT.parent                                  # リポジトリの最上位：src/・scripts/・third_party/ はここにある
INCLUDE = ["src", "scripts", "colab", "third_party", "docs/PREMISES.md", "docs/PREMISES.sha256", "requirements.txt", "NOTICE"]
# 包みの中の並びは移す前（mosaic-qwen の最上位）と同じにする。各名前を、ある方の場所から取る


def pack() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE:
            p = ROOT / name if (ROOT / name).exists() else REPO / name
            tar.add(p, arcname=name, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    plan["code_commit"] = commit + ("-dirty" if dirty else "")
    blob = base64.b64encode(pack()).decode()
    print(f'''# mosaic-qwen: {plan["name"]}  (code {plan["code_commit"]})
# このセルは colab/make_cell.py が作ったもの。中にコード一式（圧縮）と実験計画が入っている。
import base64, io, json, os, sys, tarfile, shutil
CODE_B64 = "{blob}"
PLAN = json.loads({json.dumps(json.dumps(plan, ensure_ascii=False))})
shutil.rmtree("/content/mmllm", ignore_errors=True)
data = base64.b64decode(CODE_B64)
tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall("/content/mmllm", filter="data")
os.makedirs("/content/out", exist_ok=True)
open("/content/out/code.tar.gz", "wb").write(data)
sys.path[:0] = ["/content/mmllm/colab", "/content/mmllm/src"]
import importlib, runner, saver; importlib.reload(saver); importlib.reload(runner)
print("code", PLAN["code_commit"], "->", "/content/mmllm", flush=True)
runner.run_all(PLAN)''')


if __name__ == "__main__":
    main()
