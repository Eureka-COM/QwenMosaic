# mosaic-qwen: Modal 起動セル（Colab の CPU ランタイムから Modal の GPU に計画を送る）
#   1 回目：Modal へのログイン用のリンクが出るので、Modal にログインしたブラウザで開いて承認する
#   GPU 側の GitHub トークンは Modal のシークレット mm-colab-results から読む（colab/modal_mq.py）
#   --detach なので、送ったあとはこの Colab を止めても Modal 側は最後まで動く。結果は results/<日時>_<計画名> ブランチ
PLAN_FILE = "colab/plans/mq-dev-0p8b.json"
GPU = "H100"          # L4 / A100 / H100 / H200 / B200
BRANCH = "main"
HOURS = 3             # この時間を過ぎたら GPU 側を止める（費用の上限）
import os, re, shutil, subprocess, sys, tempfile, time
from IPython.display import display, HTML
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal"], check=True)
from google.colab import userdata
def _get(name):
    try:
        return userdata.get(name)
    except Exception:
        return None
_tok = _get("MM_GITHUB_TOKEN") or _get("mosaic-qwen-colab") or _get("mm-colab-results") or _get("GITHUB_TOKEN")
_ask = os.path.join(tempfile.mkdtemp(), "askpass.sh")
open(_ask, "w").write('#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; *) echo "$_MM_TOKEN";; esac\n')
os.chmod(_ask, 0o700)
_env = dict(os.environ, GIT_ASKPASS=_ask, _MM_TOKEN=_tok or "", GIT_TERMINAL_PROMPT="0")
shutil.rmtree("/content/mmllm", ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--branch", BRANCH, "https://github.com/Eureka-COM/mosaic-qwen.git", "/content/mmllm"],
               env=_env, check=True)
if not os.path.exists(os.path.expanduser("~/.modal.toml")):
    p = subprocess.Popen(["modal", "token", "new"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, BROWSER="true", PYTHONUNBUFFERED="1"))
    shown = False
    for line in p.stdout:
        m = re.search(r"https://modal\.com/token-flow/\S+", line)
        if m and not shown:
            display(HTML(f'<a href="{m.group(0)}" target="_blank" style="font-size:20px">Modal のログインを承認する（クリック）</a>'))
            shown = True
        print(line, end="", flush=True)
    p.wait()
    print("modal token new:", p.returncode)
r = subprocess.run(["modal", "run", "--detach", "colab/modal_mq.py", "--plan", PLAN_FILE, "--gpu", GPU, "--branch", BRANCH, "--hours", str(HOURS)],
                   cwd="/content/mmllm", capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-3000:])
