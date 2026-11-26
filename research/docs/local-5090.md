# ローカルの RTX 5090 で計画を動かす（WSL2）

前提 3（改 1、2026-11-13）：今回の規模の作業（Qwen3.5-0.8B の試行）はローカルの RTX 5090 で行う。大きいもの（27B など）は Colab か Modal。

Colab と同じ runner（`colab/runner.py`）を、PC の WSL2（Ubuntu）の中で動かす。結果は数分おきに GitHub の `results/<日時>_<計画名>` ブランチへ、途中の重みは計画が指定する Release へ送られる。同じ計画をもう一度起動すると、Release に残った重みの続きから進む。

- 公式の Memory Mosaics のコードは Linux 向け（CUDA のカーネルのビルドと、コンパイルした注意）。Windows のままではなく WSL2 の中で動かす
- GitHub のトークンは自分で設定する。値はどこにも表示・記録されない
- 動かしている間、PC をスリープさせない（スリープすると WSL ごと止まる）

## いちばん短い手順

`tools/local_setup.sh` が、下の 1〜3 をまとめて行う（GPU の確認、道具と Python の環境、`nvcc`、トークンの入力、計画の起動）。Windows 側の Python には触らない。

1. `tools/local_setup.sh` を Windows の「ダウンロード」フォルダに置く
2. PowerShell で `wsl -l -v` を実行し、Ubuntu が入っていることを確かめる（無ければ `wsl --install -d Ubuntu-24.04`）
3. PowerShell で次を実行する（`<ユーザー名>` は Windows のユーザー名）

```powershell
wsl bash /mnt/c/Users/<ユーザー名>/Downloads/local_setup.sh
```

- 計画を指定しなければ、速さの計測（`mq-all-0p8b-env`）→ 0.8B の学習の続き（`mq-all-0p8b`）の順に動く。別の計画は、後ろにパスを並べる（例：`... local_setup.sh colab/plans/mq-all-0p8b-docs-env.json`）
- 動いている間、そのウィンドウは開いたままにする（閉じると止まる。もう一度実行すれば、保存済みの重みの続きから進む）
- 下の 1〜3 は、同じことを手で行う場合の手順。**どのコマンドも PowerShell ではなく、WSL2 の Ubuntu の中（`wsl` と打って入った先）で実行する**

## 1. 一度だけ：道具を入れる

WSL2 の Ubuntu のターミナルで実行する。

```bash
sudo apt-get update && sudo apt-get install -y git build-essential ninja-build python3-venv python3-dev wget curl
sudo mkdir -p /content && sudo chown "$USER" /content
python3 -m venv ~/mq-venv && source ~/mq-venv/bin/activate
pip install -U pip
pip install torch "transformers>=5.2,<6" pyarrow flash-linear-attention datasets huggingface_hub accelerate safetensors ninja
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

最後の行が `RTX 5090` を表示すれば、GPU は使えている。

公式の漏れ平均の CUDA カーネルをビルドするには、CUDA の開発用の道具（`nvcc`）が要る。上の表示の CUDA が 13.x なら `cuda-toolkit-13-0`、12.x なら `cuda-toolkit-12-8` を入れる。

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

これを入れなくても動く（漏れ平均は `src/mmllm/official.py` の代わりの計算になる）。どちらで動いたかは、結果の `env.json` と各実行の `run.json` の `backends` に残る。

## 2. 毎回：トークンを設定して、起動用のスクリプトを取る

```bash
source ~/mq-venv/bin/activate
read -rs MM_GITHUB_TOKEN && export MM_GITHUB_TOKEN      # 貼り付けて Enter（画面には出ない）
curl -fsSL -H "Authorization: Bearer $MM_GITHUB_TOKEN" -H "Accept: application/vnd.github.raw" \
  https://api.github.com/repos/Eureka-COM/mosaic-qwen/contents/colab/local_mq.py -o ~/local_mq.py
python ~/local_mq.py --check
```

トークンは Colab のシークレットに入れたものと同じ種類（Eureka-COM/mosaic-qwen の Contents が Read and write）。`--check` は、GPU・道具・`/content`・トークンの書き込みを確かめて、足りないものを表示する。

## 3. 計画を動かす

ターミナルを閉じても止まらないように、裏で動かしてログを見る。

```bash
cd ~ && nohup python ~/local_mq.py --plan colab/plans/mq-all-0p8b-env.json > ~/mq-run.log 2>&1 &
tail -f ~/mq-run.log
```

`tail -f` は Ctrl+C で抜けてよい（学習は止まらない）。止めたいときは `touch /content/STOP`（結果は保存される）。次の起動の前に `rm /content/STOP`。

| 計画 | 内容 | 目安 |
|---|---|---|
| `colab/plans/mq-all-0p8b-env.json` | 速さとメモリの計測（学習なし）。最初にこれを動かす | 数分 |
| `colab/plans/mq-all-0p8b.json` | §10 手順 1 の学習の続き（段 2 の途中から） | 計測の値で見積もる |
| `colab/plans/mq-all-0p8b-docs.json` | §10 手順 2（関連文書の行、記憶 3 つ） | 約 3.3 時間 |
| `colab/plans/mq-diag-0p8b-layers.json` | §10 診断 1（層合わせの対照） | 約 25 分 |
| `colab/plans/mq-diag-teachers.json` | §10 診断 2（別のモデルでの層合わせ） | 約 50 分 |
| `colab/plans/mq-diag-0p8b-train.json` | §10 診断 3〜5（蒸留を続ける／全部の重み／蒸留なし） | 約 3.6 時間 |

## 計画を続けて動かす

`tools/local_chain.sh` は、計画を順に動かす。`--after-current` を付けると、いま動いている計画が終わるのを待ってから始める。計画の前に `optional:` を付けると、その計画に失敗した実行があっても次へ進む（付けない計画は、失敗したらそこで止まる）。トークンは `~/.mm_github_token`（自分で入れたファイル、モード 600）から読む。

```bash
bash local_chain.sh --after-current colab/plans/mq-diag-0p8b-layers.json \
  optional:colab/plans/mq-diag-teachers.json colab/plans/mq-diag-0p8b-train.json
```

- 止める：そのウィンドウを閉じるか、`touch /content/STOP`（動いている計画は結果を保存して止まり、次の計画は始まらない）
- 同時に動かせる計画は 1 つ（新しい計画は `/content/out` を使い直す）
