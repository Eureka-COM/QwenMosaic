# 置き換えた層の仕組み

この文書は、置き換えた 1 層が何を計算するかを、別の実装（専用のカーネルなど）を作る人に向けて書きます。
計算の本体は公式の Memory Mosaics のコード（`third_party/MemoryMosaics/model/mosaics/`）で、このリポジトリはそれを Qwen の層の差し込み口につなぐだけです。
式の細部の拠り所は、公式のコードと論文（[SOURCES.md](SOURCES.md)）です。

## 1. 何を何に置き換えるか

Qwen3.5 / Qwen3.8 の言語モデルは、Gated DeltaNet の層 3 つと softmax の注意の層 1 つのくり返しです。
`convert_to_mosaic_qwen`（`src/mmllm/mosaic_qwen.py`）は、注意の層（`layer_types` が `full_attention` の層）を、公式の `MosaicBlock` 1 つずつに置き換えます。

| モデル | 層の数 | 置き換える層 |
|---|---|---|
| Qwen3.5-0.8B | 24 | 注意の 6 層（3, 7, 11, 15, 19, 23） |
| Qwen3.8-27B | 64 | 注意の 16 層（3, 7, …, 63） |

`MQConfig(replace="all")` にすると、Gated DeltaNet の層も含めた全層を置き換えます。

```
元の層                                  置き換えた層（公式 MosaicBlock、layout "flat"）
  x += attention(norm(x))                 h  = attention_norm(x)
  x += mlp(norm(x))                       x += short(h) + long(h)
                                          x += feed_forward(ffn_norm(x))
```

- `feed_forward` は、元の層の MLP の重みそのものです（写しではなく同じ Parameter を共有します）。公式の FeedForward は `w2(silu(w1 x) * w3 x)` で、Qwen の MLP の `gate_proj`・`up_proj`・`down_proj` がそのまま当てはまります。活性化が SiLU でない、またはバイアスがある MLP は変換を断ります
- `attention_norm` と `ffn_norm` は元の層の 2 つの RMSNorm の重みで始めます。Qwen3.5 の RMSNorm は `1 + weight` を掛ける形なので、`_qwen_norm_weight` で公式の形（`weight` を掛ける）に直してから写します
- 元の `q_proj`・`k_proj`・`v_proj`・`o_proj`・`q_norm`・`k_norm` は取り除きます。学習の間は教師として横に残し、`drop_teacher()` か書き出しのときに捨てます

## 2. 記憶 1 つの計算

`short` と `long` はどちらも公式の `MosaicsMemory`（`model/mosaics/memory.py`）です。違うのは、どの位置を読めるかだけです。
公式のコードの冒頭の説明を、記号を変えずに訳します。

各ヘッドで、位置 T の問い合わせ q_T は記憶を `sum_i softmax_i(q_T · k_i) v_i` として読みます。

- 鍵 `k_T = phi(x_T, x_{T-1}, ...)`：射影した入力の漏れ平均（leaky average）を、長さ 1 にそろえたもの
- 問い合わせ `q_T = beta(n) * k_T`：鍵そのものに、帯域幅 `beta(n)` を掛けたもの
- 帯域幅 `beta(n) = b1 * n^a + b0`：記憶の大きさ n が増えるほど鋭くなる
- 値 `v_T = psi(x_{T+1}, x_T)`：次の入力と今の入力の射影を混ぜ、長さ 1 にそろえたもの。値が 1 つ先の入力を持つので、問い合わせは自分の位置を読まない

位置の符号化（RoPE など）は使いません。位置の情報は漏れ平均と、読める範囲の決め方から入ります。

### 読める範囲（マスク）

`model/common/mask.py` の `memory_mask` が決めます。

| 記憶 | 読める位置 | n（帯域幅の記憶の大きさ） |
|---|---|---|
| short | 直前の `window` 語（既定 256） | 文書の中の位置。`window` で頭打ち |
| long | `skip` 語（既定 64）以上前のすべて。学習の間は `skip` に 0〜`random_skip`（既定 192）の乱数を足す | `skip` より後ろにある語の数 |

`MQConfig.attn_bias_type` が `"causal"`（このリポジトリの既定）なら、1 行に詰めた複数の文書をまたいで読みます。
`"doc_causal"`（公式の Memory Mosaics v2 の設定ファイルの値）なら、今の文書の中だけを読みます。研究の記録の主な実行は `"causal"` で、`"doc_causal"` は全層置き換えの手順 2 で使いました。

### 重み

1 つの記憶の重みです。ヘッドの数を H、ヘッドの幅を d、モデルの幅を dim と書きます。

| 名前 | 形 | 役目 |
|---|---|---|
| `proj_kf` | (H·(d+2), dim) | 鍵の射影と、ヘッドごとの減衰のロジットと入力のゲートのロジット（漏れ平均の鍵のとき。そうでなければ (H·d, dim)） |
| `recurrence` | | 漏れ平均（`model/mosaics/leaky_average.py`） |
| `proj_v` | (H·d + H, dim) | 値の射影と、ヘッドごとの読み出しのゲートのロジット（`memory_gate` が真のとき。そうでなければ (H·d, dim)） |
| `proj_out` | (dim, H·d) | 出力の射影 |
| `value_interp`、`bandwidth_bias`、`bandwidth_scale`、`bandwidth_exponent`、`value_scale` | (H,) | 値の混ぜ方、帯域幅の b0・b1・a、出力の大きさ |

形は `third_party/MemoryMosaics/model/mosaics/memory.py` の `MosaicsMemory.__init__` によります。

### 既定の設定

`MQConfig.from_upstream_mmv2()` は、同梱した `configs/mosaics/memory_mosaics_v2.yaml` から記憶の設定を読みます。

| 項目 | 値 |
|---|---|
| ヘッドの数（short・long） | 16・16（研究の記録では、0.8B は 8+8、27B は 16+16 で学習） |
| ヘッドの幅 | 128 |
| 鍵 | 漏れ平均（`leaky_average`） |
| `window` / `skip` / `random_skip` | 256 / 64 / 192 |
| 記憶のゲート | あり |
| 置き方（layout） | `flat`（short と long を同じ入力に並べて足す） |

## 3. 2 つの計算の部品と CPU の代わりの計算

公式のコードには、どこでも動くとは限らない部品が 2 つあります。`src/mmllm/official.py` は、公式のファイルを書き換えずに、それぞれの前に切り替えを置きます。

| 部品 | 公式の実装 | CPU などで使う代わりの計算 | 一致の確認 |
|---|---|---|---|
| 記憶の読み出し（softmax の重み付き和） | PyTorch の FlexAttention（`torch.compile`） | `dense_attention`：同じ式を、マスクを当てた密な行列で計算する。何も読めない行は 0 | `tests/test_official_vendor.py` |
| 漏れ平均 `y_t = omega_t * y_{t-1} + u_t` | JIT で作る CUDA のカーネル（nvcc と ninja が要る） | `leaky_avg_scan`：同じ漸化式を、log2(T) 段の並列のスキャンで計算する | `tests/test_official_vendor.py` |

切り替えは環境変数 `MQ_ATTN_BACKEND`（auto / official / dense）と `MQ_LEAKY_BACKEND`（auto / kernel / scan / loop）で選べます。
新しいハードウェア向けにカーネルを作るときも、この 2 つの部品が入れ替える単位になります。

## 4. 生成のときのキャッシュ

生成では、各記憶が公式のキャッシュ（`init_cache`：鍵と値を (B, 長さ, H, d) で持つ）に加えて、漏れ平均の状態（`_recurrence_state`）と、1 つ前の値（`_prev_v`）を次の語へ持ち越します。
`MosaicQwenLayer._forward_cached` が、これらを Hugging Face のキャッシュの入れ物に入れて、`model.generate` で使えるようにしています。

- 1 行の中の詰め物（padding）は記憶から隠しません。長さのそろった問いだけを 1 つのバッチにするか、1 行ずつ生成してください
- キャッシュを使った生成が、全体を 1 回で計算した結果と一致することを、`tests/test_mosaic_qwen.py` が確かめています（transformers 5.22.2 で通り、5.24.1 では落ちます）

## 5. 別の実装の答え合わせ

`examples/dump_layer_reference.py` は、小さいモデルを変換し、置き換えた 1 層の次の値を 1 つのファイルに書き出します。

| 名前 | 中身 |
|---|---|
| `x` | 層への入力 |
| `h` | `attention_norm(x)` |
| `short`、`long` | 2 つの記憶の出力 |
| `memory_sum` | `short + long` |
| `block_out` | 層の出力（`x + memory_sum` に FFN を足したもの） |
| `params.*` | その層の重み |

計算は CPU の代わりの計算（密な読み出しと並列のスキャン）で行います。別の実装の出力を、このファイルの値と比べてください。
長い記憶は先頭の `skip` 語では何も読めないので、`long` の先頭 64 語は 0 になります（`tests/test_examples.py` が確かめています）。
