#!/usr/bin/env bash
# mosaic-qwen: ローカルの RTX 5090 で計画を動かす（WSL2 の Ubuntu の中で実行する）
#
#   Windows の PowerShell から:   wsl bash /mnt/c/Users/<ユーザー名>/Downloads/local_setup.sh
#   計画を指定するとき:           wsl bash .../local_setup.sh colab/plans/mq-all-0p8b-docs-env.json
#
# することは 6 つ。何度実行してもよい（済んでいるものは飛ばす）。
#   1. GPU が見えるか確かめる
#   2. ビルドの道具と /content を用意する（sudo のパスワードを聞かれる）
#   3. Python の環境（~/mq-venv）を作る。Windows 側の Python には触らない
#   4. CUDA の開発用の道具（nvcc）を入れる。失敗しても先へ進む（代わりの計算で動く）
#   5. GitHub のトークンを聞く（貼り付けて Enter。画面にもファイルにも残さない）
#   6. 計画を順に動かす。計画を指定しなければ、速さの計測 → 0.8B の学習の続き
# 動いている間、このウィンドウは開いたままにする（閉じると止まる。止まっても、もう一度実行すれば
# 保存済みの重みの続きから進む）。経過は ~/mq-run.log と GitHub の results ブランチに残る。
set -eo pipefail
say() { printf '\n== %s\n' "$*"; }
PLANS=("$@")
[ ${#PLANS[@]} -eq 0 ] && PLANS=(colab/plans/mq-all-0p8b-env.json colab/plans/mq-all-0p8b.json)
SUDO=sudo; [ "$(id -u)" = 0 ] && SUDO=""

if [ "$(uname -s)" != "Linux" ]; then
  echo "Linux（WSL2 の Ubuntu）の中で実行してください。PowerShell からは:  wsl bash <このファイルのパス>"; exit 1
fi

say "1/6 GPU の確認"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then
  echo "GPU が見えません。Windows 側の NVIDIA ドライバを新しくしてから、もう一度実行してください"; exit 1
fi

say "2/6 道具と /content（パスワードを聞かれたら WSL のユーザーのパスワード）"
$SUDO apt-get update -y -q
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -q git build-essential ninja-build python3-venv python3-dev wget curl
$SUDO mkdir -p /content && $SUDO chown "$(id -un)" /content

say "3/6 Python の環境（~/mq-venv）"
[ -x ~/mq-venv/bin/python ] || python3 -m venv ~/mq-venv
# shellcheck disable=SC1090
source ~/mq-venv/bin/activate
pip install -q -U pip
pip install -q torch "transformers>=5.2,<6" pyarrow flash-linear-attention datasets huggingface_hub accelerate safetensors ninja
CUDA_VER=$(python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch から GPU が使えません"
print(torch.version.cuda or "")
PY
)
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"

say "4/6 CUDA の開発用の道具（nvcc）。失敗しても先へ進みます"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  PKG="cuda-toolkit-$(echo "$CUDA_VER" | cut -d. -f1,2 | tr . -)"
  ( cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
      && $SUDO dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && $SUDO apt-get update -y -q \
      && $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$PKG" ) \
    || echo "  nvcc を入れられませんでした（$PKG）。漏れ平均は代わりの計算で動きます"
fi
if [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME=/usr/local/cuda; export PATH="$CUDA_HOME/bin:$PATH"; fi
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -1 || true

say "5/6 GitHub のトークン（貼り付けて Enter。画面には出ません）"
if [ -z "${MM_GITHUB_TOKEN:-}" ]; then read -rs MM_GITHUB_TOKEN; fi
export MM_GITHUB_TOKEN
curl -fsSL -H "Authorization: Bearer $MM_GITHUB_TOKEN" -H "Accept: application/vnd.github.raw" \
  https://api.github.com/repos/Eureka-COM/mosaic-qwen/contents/colab/local_mq.py -o ~/local_mq.py \
  || { echo "起動用のスクリプトを取得できません。トークン（Eureka-COM/mosaic-qwen の Contents: Read and write）を確かめてください"; exit 1; }
python ~/local_mq.py --check

say "6/6 計画を動かします: ${PLANS[*]}"
echo "このウィンドウは開いたままにしてください（PC をスリープさせない）。経過は ~/mq-run.log にも残ります"
rm -f /content/STOP
cd ~
{
  for plan in "${PLANS[@]}"; do
    echo "=== $(date -u +%H:%M:%S) UTC  $plan"
    python ~/local_mq.py --plan "$plan" || { echo "=== 起動に失敗: $plan"; exit 1; }
    python - <<'PY' || { echo "=== この計画に失敗した実行があるので、次へは進みません"; exit 1; }
import json, sys
s = json.load(open("/content/out/status.json"))
bad = [k for k, v in s.get("runs", {}).items() if not str(v).startswith("ok")]
sys.exit(1 if (s.get("error") or bad or not s.get("runs")) else 0)
PY
  done
  echo "=== すべて終了 $(date -u +%H:%M:%S) UTC"
} 2>&1 | tee ~/mq-run.log
