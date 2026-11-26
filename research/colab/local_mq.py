"""Run a MosaicQwen plan (colab/plans/mq-*.json) on a local Linux machine with an NVIDIA GPU.
On Windows: inside WSL2 (Ubuntu). See docs/local-5090.md.

    export MM_GITHUB_TOKEN=...      # fine-grained token, Contents: Read and write on Eureka-COM/mosaic-qwen
    python colab/local_mq.py --check                                    # what is there, what is missing
    python colab/local_mq.py --plan colab/plans/mq-all-0p8b-env.json    # speed and memory, no training
    python colab/local_mq.py --plan colab/plans/mq-all-0p8b.json

Same runner as on Colab and Modal (colab/runner.py): the code is cloned to /content/mmllm at the
given commit, results go to a results/<date>_<plan> branch every few minutes, checkpoints to the
store the plan names; launching a plan again continues from that store. Everything lives under
/content, as on Colab (the plans name their paths there):

    sudo mkdir -p /content && sudo chown "$USER" /content

Stop a running plan with  touch /content/STOP  (results are still saved); remove the file before
the next launch. The token is read from the environment and never printed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

REPO = "Eureka-COM/mosaic-qwen"
BASE = "/content"


def _out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def token():
    for k in ("MM_GITHUB_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    return None


def can_write(tok: str) -> int:
    """HTTP status of creating a blob in the repository (201 = this token can write there)."""
    rq = urllib.request.Request(f"https://api.github.com/repos/{REPO}/git/blobs", method="POST",
                                data=json.dumps({"content": "token check", "encoding": "utf-8"}).encode(),
                                headers={"Authorization": "Bearer " + tok, "Accept": "application/vnd.github+json",
                                         "Content-Type": "application/json", "User-Agent": "mosaic-qwen"})
    try:
        return urllib.request.urlopen(rq, timeout=30).status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return 0


def check(need_token: bool = True) -> list[str]:
    """Print what this machine has; return the problems that stop a run."""
    stop = []
    print("OS            :", sys.platform, "(WSL)" if "microsoft" in _out(["uname", "-r"]).lower() else "")
    if not sys.platform.startswith("linux"):
        stop.append("Linux で実行してください（Windows では WSL2 の Ubuntu の中で）")
    gpu = _out(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    print("GPU           :", gpu or "見つかりません")
    if not gpu:
        stop.append("nvidia-smi で GPU が見えません（WSL2 なら Windows 側の NVIDIA ドライバを確認）")
    try:
        import torch
        print("torch         :", torch.__version__, "CUDA", torch.version.cuda, "使える" if torch.cuda.is_available() else "使えない")
        if not torch.cuda.is_available():
            stop.append("torch から GPU が使えません（CUDA 版の torch を入れてください）")
    except Exception as e:  # noqa: BLE001
        print("torch         : ありません", repr(e)[:80])
        stop.append("torch がありません（docs/local-5090.md の手順 2）")
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
    nvcc = shutil.which("nvcc") or (os.path.join(cuda_home, "bin", "nvcc") if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")) else None)
    tools = {"nvcc": nvcc, "ninja": shutil.which("ninja"), "g++": shutil.which("g++"), "git": shutil.which("git")}
    for k, v in tools.items():
        print(f"{k:14}:", v or "ありません")
    if not tools["git"]:
        stop.append("git がありません")
    if not (tools["nvcc"] and tools["ninja"] and tools["g++"]):
        print("  → 公式の漏れ平均の CUDA カーネルをビルドできず、遅い代わりの計算になります（動きはします）。"
              "docs/local-5090.md の手順 1 で入れられます")
    ok = os.path.isdir(BASE) and os.access(BASE, os.W_OK)
    print(f"{BASE:14}:", "書き込み可" if ok else "ありません／書けません")
    if not ok:
        stop.append(f'{BASE} を作ってください:  sudo mkdir -p {BASE} && sudo chown "$USER" {BASE}')
    else:
        free = shutil.disk_usage(BASE).free / 2**30
        print("空き容量      :", f"{free:.0f} GB")
        if free < 15:
            stop.append("空き容量が 15 GB 未満です")
    tok = token()
    if not tok:
        print("トークン      : ありません（環境変数 MM_GITHUB_TOKEN）")
        if need_token:
            stop.append("export MM_GITHUB_TOKEN=... でトークンを設定してください（値は表示・記録されません）")
    else:
        code = can_write(tok)
        print("トークン      :", "書き込み可" if code == 201 else f"書き込み不可（HTTP {code}）")
        if code != 201:
            stop.append(f"このトークンでは {REPO} に書き込めません（Repository access にこのリポジトリ、Contents を Read and write に）")
    return stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, help="colab/plans/mq-*.json")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--check", action="store_true", help="only report what this machine has")
    a = ap.parse_args()
    stop = check()
    if stop:
        print("\n止まる理由:")
        for s in stop:
            print(" -", s)
        raise SystemExit(1)
    if a.check or not a.plan:
        print("\n準備はできています。--plan で計画を指定してください")
        return
    if os.path.exists(os.path.join(BASE, "STOP")):
        raise SystemExit(f"{BASE}/STOP が残っています。消してから起動してください")

    tok = token()
    ask = os.path.join(tempfile.mkdtemp(), "askpass.sh")
    with open(ask, "w") as f:
        f.write('#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; *) echo "$_MM_TOKEN";; esac\n')
    os.chmod(ask, 0o700)
    env = dict(os.environ, GIT_ASKPASS=ask, _MM_TOKEN=tok, GIT_TERMINAL_PROMPT="0")
    os.environ["MM_GITHUB_TOKEN"] = tok
    code = os.path.join(BASE, "mmllm")
    shutil.rmtree(code, ignore_errors=True)
    r = subprocess.run(["git", "clone", "-q", "--single-branch", "--branch", a.branch,
                        f"https://github.com/{REPO}.git", code], env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("リポジトリを取得できません:\n" + r.stderr.replace(tok, "***")[-400:])
    subprocess.run(["git", "checkout", "-q", a.commit], cwd=code, check=True)
    c = _out(["git", "-C", code, "rev-parse", "--short", "HEAD"])
    plan = json.load(open(os.path.join(code, a.plan)))
    plan["code_commit"], plan["platform"] = c, "local"
    plan["auto_stop"] = False                       # nothing to hand back: the process just ends
    print("code", c, "plan", plan["name"], flush=True)
    sys.path[:0] = [os.path.join(code, "colab"), os.path.join(code, "src")]
    import runner  # noqa: E402
    runner.run_all(plan)


if __name__ == "__main__":
    main()
