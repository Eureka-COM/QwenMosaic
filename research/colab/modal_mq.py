"""Run a MosaicQwen plan (colab/plans/mq-*.json) on a Modal GPU.

    modal run --detach colab/modal_mq.py --plan colab/plans/mq-dev-0p8b.json --gpu H100
    modal run --detach colab/modal_mq.py --plan colab/plans/mq-27b-s1.json   --gpu H200 --hours 3

Same runner as on Colab (colab/runner.py): results go to a results/<date>_<plan> branch.
Notes:
  - the image has the CUDA toolkit, ninja and a C compiler, which the official Memory
    Mosaics code needs to build its leaky-average kernel and to compile FlexAttention
    (without them the run still works, on the slower fallbacks: scripts/mq_env.py reports which)
  - H200 (141 GB) and B200 (192 GB) for the 27B stages
  - a Volume at /vol for checkpoints (use `--store dir:/vol/<name>` in a plan instead of a
    GitHub release if you prefer)
  - `--hours` sets the hard time limit of the GPU call (default 3): the cost ceiling
The GitHub token comes from the Modal secret "mm-colab-results" (key MM_GITHUB_TOKEN).
"""
import os

import modal

BRANCH_DEFAULT = "main"
PIP = ["torch", "transformers>=5.2,<6", "pyarrow", "flash-linear-attention", "datasets", "huggingface_hub",
       "accelerate", "safetensors", "ninja", "pyyaml", "hf_xet"]
if os.environ.get("MM_MODAL_IMAGE") == "slim":      # no CUDA toolkit: fallbacks only
    image = modal.Image.debian_slim(python_version="3.12").apt_install("git", "build-essential").pip_install(*PIP)
else:
    image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
             .apt_install("git", "build-essential")
             .pip_install(*PIP)
             .env({"CUDA_HOME": "/usr/local/cuda"}))
app = modal.App("mosaic-qwen", image=image)
SECRET = modal.Secret.from_name("mm-colab-results")
VOLUME = modal.Volume.from_name("mm-mosaic-qwen", create_if_missing=True)
MAX_HOURS = 12


def _token():
    for k in ("MM_GITHUB_TOKEN", "mosaic-qwen-colab", "mm-colab-results", "GITHUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    cands = [v for v in os.environ.values() if v.startswith(("github_pat_", "ghp_", "gho_"))]
    return cands[0] if len(cands) == 1 else None


def _run(plan_file, branch, commit):
    import json
    import shutil
    import subprocess
    import sys
    import tempfile
    tok = _token()
    if not tok:
        raise SystemExit("GitHub トークンが見つかりません（Modal のシークレット mm-colab-results のキーを MM_GITHUB_TOKEN に）")
    os.makedirs("/content", exist_ok=True)
    ask = os.path.join(tempfile.mkdtemp(), "askpass.sh")
    open(ask, "w").write('#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; *) echo "$_MM_TOKEN";; esac\n')
    os.chmod(ask, 0o700)
    env = dict(os.environ, GIT_ASKPASS=ask, _MM_TOKEN=tok, GIT_TERMINAL_PROMPT="0")
    os.environ["MM_GITHUB_TOKEN"] = tok
    shutil.rmtree("/content/mmllm", ignore_errors=True)
    r = subprocess.run(["git", "clone", "-q", "--single-branch", "--branch", branch,
                        "https://github.com/Eureka-COM/mosaic-qwen.git", "/content/mmllm"],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("リポジトリを取得できません:\n" + r.stderr.replace(tok, "***")[-400:])
    subprocess.run(["git", "checkout", "-q", commit], cwd="/content/mmllm", check=True)
    c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd="/content/mmllm", capture_output=True, text=True).stdout.strip()
    plan = json.load(open("/content/mmllm/" + plan_file)); plan["code_commit"] = c; plan["platform"] = "modal"
    plan["auto_stop"] = False                       # the Modal call ends by itself
    print("code", c, "plan", plan["name"], flush=True)
    sys.path[:0] = ["/content/mmllm/colab", "/content/mmllm/src"]
    import runner  # noqa: E402
    try:
        runner.run_all(plan)
    finally:
        VOLUME.commit()


# one function per GPU type (Modal fixes the GPU at definition time); `timeout` is the outer hard limit,
# the watchdog below enforces the --hours given at launch
_OPTS = dict(secrets=[SECRET], volumes={"/vol": VOLUME}, timeout=MAX_HOURS * 3600, memory=32768)


@app.function(gpu="L4", **_OPTS)
def run_l4(plan_file: str, branch: str, commit: str):
    _run(plan_file, branch, commit)


@app.function(gpu="A100-80GB", **_OPTS)
def run_a100(plan_file: str, branch: str, commit: str):
    _run(plan_file, branch, commit)


@app.function(gpu="H100", **_OPTS)
def run_h100(plan_file: str, branch: str, commit: str):
    _run(plan_file, branch, commit)


@app.function(gpu="H200", **_OPTS)
def run_h200(plan_file: str, branch: str, commit: str):
    _run(plan_file, branch, commit)


@app.function(gpu="B200", **_OPTS)
def run_b200(plan_file: str, branch: str, commit: str):
    _run(plan_file, branch, commit)


@app.function(timeout=(MAX_HOURS + 1) * 3600)
def watchdog(call_id: str, minutes: float):
    """On a CPU container: cancel the GPU call if it is still running after `minutes` (the cost ceiling,
    independent of the machine that launched it)."""
    import time
    time.sleep(minutes * 60)
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        print("already finished", call_id)
    except TimeoutError:
        modal.FunctionCall.from_id(call_id).cancel()
        print("cancelled", call_id)
    except Exception as e:  # noqa: BLE001
        print("finished with error or gone:", call_id, type(e).__name__)


@app.local_entrypoint()
def main(plan: str = "colab/plans/mq-dev-0p8b.json", gpu: str = "H100", branch: str = BRANCH_DEFAULT,
         commit: str = "HEAD", hours: float = 3.0):
    runners = {"L4": run_l4, "A100": run_a100, "A100-80GB": run_a100, "H100": run_h100, "H200": run_h200, "B200": run_b200}
    assert gpu in runners, f"gpu must be one of {sorted(runners)}"
    assert 0 < hours <= MAX_HOURS
    call = runners[gpu].spawn(plan, branch, commit)
    watchdog.spawn(call.object_id, hours * 60)
    print("spawned", gpu, plan, "call id", call.object_id, f"(cancelled after {hours} h if still running)", flush=True)
