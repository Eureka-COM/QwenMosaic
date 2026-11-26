# mosaic-qwen: Colab 起動セル
#   このリポジトリを Colab のシークレット（MM_GITHUB_TOKEN か mosaic-qwen-colab）で取得し、計画 PLAN_FILE を実行する。
#   結果は results/<日時>_<計画名> ブランチへ。終わるとランタイムは自動で止まる。途中で止めるには  touch /content/STOP
#   計画: colab/plans/mq-dev-0p8b.json（0.8B で全手順）／mq-27b-probe.json（27B の速さとメモリ）／mq-27b-s1.json … s3.json
PLAN_FILE = "colab/plans/mq-dev-0p8b.json"
BRANCH = "main"
COMMIT = "HEAD"
REQUIRE_GPU = ""    # 例 "RTX PRO 6000"（G4）。割り当てられた GPU の名前にこの文字が無ければ、何も取得せずにすぐランタイムを返す
import json, os, shutil, subprocess, sys, tempfile
if REQUIRE_GPU:
    _g = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print("GPU:", _g or "なし", flush=True)
    if REQUIRE_GPU not in _g:
        print(REQUIRE_GPU, "ではないので、ランタイムを返します", flush=True)
        from google.colab import runtime
        runtime.unassign()
        raise SystemExit
from google.colab import userdata
def _get(name):
    try:
        return userdata.get(name)
    except Exception:
        return None
# シークレットを順に試し、このリポジトリに書き込めるトークンを使う（書けないまま GPU を回して結果を失わないため。値は表示しない）
import urllib.request, urllib.error
def _write_status(tok):
    rq = urllib.request.Request("https://api.github.com/repos/Eureka-COM/mosaic-qwen/git/blobs", method="POST",
                                data=json.dumps({"content": "token check", "encoding": "utf-8"}).encode(),
                                headers={"Authorization": "Bearer " + tok, "Accept": "application/vnd.github+json",
                                         "Content-Type": "application/json", "User-Agent": "mosaic-qwen"})
    try:
        return urllib.request.urlopen(rq, timeout=30).status
    except urllib.error.HTTPError as e:
        return e.code
_tok, _seen = None, {}
for _n in ("MM_GITHUB_TOKEN", "mosaic-qwen-colab", "mm-colab-results", "GITHUB_TOKEN"):
    _t = _get(_n)
    if not _t:
        _seen[_n] = "なし"
        continue
    _c = _write_status(_t.strip())
    _seen[_n] = "書き込み可" if _c == 201 else f"書き込み不可（HTTP {_c}）"
    if _c == 201:
        _tok = _t.strip()
        break
print("token check:", _seen, flush=True)
if not _tok:
    raise SystemExit("Eureka-COM/mosaic-qwen に書き込めるトークンがありません。トークンの Repository access に mosaic-qwen、"
                     "Permissions の Contents に Read and write を付け、シークレット（MM_GITHUB_TOKEN か mosaic-qwen-colab）のノートブックからのアクセスをオンにしてください")
os.environ["MM_GITHUB_TOKEN"] = _tok        # 保存役（colab/saver.py）も、確かめたこのトークンを使う
_ask = os.path.join(tempfile.mkdtemp(), "askpass.sh")
open(_ask, "w").write('#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; *) echo "$_MM_TOKEN";; esac\n')
os.chmod(_ask, 0o700)
_env = dict(os.environ, GIT_ASKPASS=_ask, _MM_TOKEN=_tok, GIT_TERMINAL_PROMPT="0")
shutil.rmtree("/content/mmllm", ignore_errors=True)
_r = subprocess.run(["git", "clone", "-q", "--single-branch", "--branch", BRANCH, "https://github.com/Eureka-COM/mosaic-qwen.git",
                     "/content/mmllm"], env=_env, capture_output=True, text=True)
if _r.returncode:
    raise SystemExit("リポジトリを取得できません（トークンにこのリポジトリの権限が無い可能性）:\n" + _r.stderr.replace(_tok, "***")[-400:])
subprocess.run(["git", "checkout", "-q", COMMIT], cwd="/content/mmllm", check=True)
_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd="/content/mmllm", capture_output=True, text=True).stdout.strip()
PLAN = json.load(open("/content/mmllm/" + PLAN_FILE)); PLAN["code_commit"] = _commit
print("code", _commit, "plan", PLAN["name"], flush=True)
sys.path[:0] = ["/content/mmllm/colab", "/content/mmllm/src"]
import importlib, runner, saver; importlib.reload(saver); importlib.reload(runner)
runner.run_all(PLAN)
