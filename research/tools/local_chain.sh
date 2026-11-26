#!/usr/bin/env bash
# mosaic-qwen: 計画を順に動かす（WSL2 の Ubuntu、ふだんのユーザーで実行）。
#   bash local_chain.sh [--after-current] 計画 [optional:計画] ...
#   --after-current   いま動いている計画（local_mq.py）が終わるのを待ってから始める
#   optional:計画     その計画に失敗した実行があっても、次の計画へ進む（付けない計画は、失敗したらそこで止まる）
# このウィンドウが学習そのもの。閉じると止まる（待っている間に閉じれば、何も始まらない）。
# 経過は ~/mq-logs/<起動の日時>.log（~/mq-run.log が最新を指す）と、GitHub の results ブランチ（数分おき）に残る。
# トークンは ~/.mm_github_token（自分で入れたもの、モード 600）から学習の処理へ渡すだけで、表示も記録もしない。
set -o pipefail
WAIT=0
if [ "${1:-}" = "--after-current" ]; then WAIT=1; shift; fi
PLANS=("$@")
[ ${#PLANS[@]} -eq 0 ] && { echo "計画を指定してください"; read -rp "Enter で閉じます"; exit 1; }
source ~/mq-venv/bin/activate
if [ -x /usr/local/cuda-13.0/bin/nvcc ]; then export CUDA_HOME=/usr/local/cuda-13.0; export PATH="$CUDA_HOME/bin:$PATH"; fi
if [ ! -s ~/.mm_github_token ]; then echo "トークンがありません（~/.mm_github_token）"; read -rp "Enter で閉じます"; exit 1; fi
if [ $WAIT = 1 ]; then
  echo "=== $(date -u +%H:%M:%S) UTC  いまの計画が終わるのを待っています（このウィンドウを閉じれば、次は始まりません）"
  quiet=0
  while [ $quiet -lt 2 ]; do
    if pgrep -f "local_mq.py --plan" > /dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
    sleep 60
  done
fi
MM_GITHUB_TOKEN="$(cat ~/.mm_github_token)"; export MM_GITHUB_TOKEN
cd ~
mkdir -p ~/mq-logs
LOG=~/mq-logs/$(date +%Y%m%d-%H%M%S).log
ln -sf "$LOG" ~/mq-run.log
{
  echo "=== mosaic-qwen: ${PLANS[*]}"
  curl -fsSL -H "Authorization: Bearer $MM_GITHUB_TOKEN" -H "Accept: application/vnd.github.raw" \
    https://api.github.com/repos/Eureka-COM/mosaic-qwen/contents/colab/local_mq.py -o ~/local_mq.py \
    || { echo "=== 起動用のスクリプトを取得できません（トークンを確かめてください）"; exit 1; }
  rm -f /content/STOP
  for item in "${PLANS[@]}"; do
    plan="${item#optional:}"; optional=0; [ "$plan" != "$item" ] && optional=1
    [ -e /content/STOP ] && { echo "=== /content/STOP があるので、ここで止めます"; exit 1; }
    echo "=== $(date -u +%H:%M:%S) UTC  $plan"
    if ! python ~/local_mq.py --plan "$plan"; then
      echo "=== 起動に失敗: $plan"
      [ $optional = 1 ] && { echo "=== （失敗しても進む計画なので、次へ進みます）"; continue; }
      exit 1
    fi
    if ! python - <<'PY'
import json, sys
s = json.load(open("/content/out/status.json"))
bad = [k for k, v in s.get("runs", {}).items() if not str(v).startswith("ok")]
print("runs:", s.get("runs"))
sys.exit(1 if (s.get("error") or bad or not s.get("runs")) else 0)
PY
    then
      if [ $optional = 1 ]; then echo "=== この計画に失敗した実行があります（失敗しても進む計画なので、次へ進みます）"
      else echo "=== この計画に失敗した実行があるので、次へは進みません"; exit 1; fi
    fi
  done
  echo "=== すべて終了 $(date -u +%H:%M:%S) UTC"
} 2>&1 | tee "$LOG"
echo
read -rp "終わりました。Enter で閉じます"
