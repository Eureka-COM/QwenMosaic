"""Writes the MosaicQwen plans (docs/plan-mosaic-qwen.md). Run from anywhere:

    python colab/plans/make_mq.py

mq-dev-0p8b    Qwen3.5-0.8B, the whole recipe at small scale: preflight, stage 1 (layer-wise),
               stage 2 (whole model, memories + LoRA), stage 3 (long rows), evaluation after
               every stage, export. Two memory sizes side by side (4+4 and 8+8 heads: the
               width ratios of 16+16 and 40+40 heads on the 27B).
mq-dev-0p8b-rest  the runs of mq-dev-0p8b that did not happen in its first launch (see dev_rest).
mq-27b-probe   Qwen3.8-27B with the real weights: load, convert, speed and memory of a
               stage-1 step and of a stage-2 step (LoRA + gradient checkpointing). No training.
mq-27b-s1      27B stage 1. Checkpoints go to a store (a GitHub release or a Modal Volume);
mq-27b-s2      27B stage 2. launching the same plan again continues from the store until the
mq-27b-s3      27B stage 3. token target is reached (one launch = one time-boxed session).
mq-27b-recall-probe  27B, no training: the recall question asked four ways on the teacher and on the
               weights of stages 1, 2 and 3 (docs/results/mq-27b-recall-review-2026-11-12.md, section 6).
mq-27b-s2-final  27B with the stage-2 weights (the ones carried forward, plan section 9): the evaluation,
               MMLU and GSM8K against the original 27B, and the export check. No training.
mq-all-0p8b-env  plan section 10: Qwen3.5-0.8B with EVERY layer replaced (--replace all): speed and memory of a
               stage-1 step and of a stage-2 step at 2,048 and 8,192 tokens. No training.
mq-all-0p8b-env-l4  the same measurement on an L4 (24 GB): is the cheaper, slower card cheaper per token for the 0.8B?
mq-all-0p8b    plan section 10, step 1: the recipe of mq-dev-0p8b (8+8 heads) with every layer replaced: stage 1,
               stage 2, stage 3, the evaluation after each (the numbers the section's criteria are read on), export.
mq-all-0p8b-roundtrip  the final model of step 1 written out and read back (fp32 and bf16) against the model in
               memory, on held-out positions: what the export costs (scripts/mq_roundtrip.py). No training.
mq-all-0p8b-docs-env  plan section 10, step 2, before any training: the clusters of the real data (how long, how
               they look, how many evaluation cases the fresh files give) and the speed and memory of a step with
               three memories under "doc_causal" at 16,384 tokens.
mq-all-0p8b-docs  plan section 10, step 2: three memories, rows of related documents, the next-token loss mixed in;
               the evaluation of step 1, the same on rows the training never read, and the context-growth evaluation
               against the teacher and the model of step 1. (Written by all_docs() once the preflight has run.)
mq-27b-all-probe  plan section 10, step 3: Qwen3.8-27B with all 64 layers replaced, on a B200 (Modal): speed and
               memory of a stage-1 and a stage-2 step at 16+16 and at 40+40 heads. No training.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PIP = ["transformers>=5.2,<6", "pyarrow", "flash-linear-attention", "datasets", "safetensors", "ninja", "accelerate"]
DATA = [
    {"repo": "HuggingFaceTB/cosmopedia-100k", "file": "data/train-00000-of-00002.parquet", "dir": "/content/data/cosmopedia-100k"},
    {"repo": "HuggingFaceH4/ultrachat_200k", "file": "data/train_sft-00000-of-00003-a3ecf92756993583.parquet", "dir": "/content/data/ultrachat"},
    {"repo": "wikimedia/wikipedia", "file": "20231101.ja/train-00014-of-00015.parquet", "dir": "/content/data/wikipedia-ja"},
    {"repo": "codeparrot/codeparrot-clean-valid", "file": "file-000000000054.json.gz", "dir": "/content/data/codeparrot"},
]
# English text 4 : chat 2 : Japanese 2 : code 1
MIX = ["--data", "/content/data/cosmopedia-100k@4", "--data", "/content/data/ultrachat:messages@2",
       "--data", "/content/data/wikipedia-ja@2", "--data", "/content/data/codeparrot:content@1", "--docs", "60000"]
OUT = "/content/out"
# What a stage-2 process holds on the GPU at seq 2048 x bsz 8 x 512 positions (0.8B): 33 GB
# allocated at the peak, 41 GB held (measured, results/2026-11-11_2240_mq-dev-0p8b; most of it
# is the vocabulary projection of the sampled positions). The first estimate (18) let two
# stage-2 runs and two evaluations start together on an 80 GB card, and one ran out of memory.
S2_GB = 42


# Files no training plan reads: the documents of the context-growth evaluation (scripts/mq_context.py).
CTX_DATA = [
    {"repo": "HuggingFaceTB/cosmopedia-100k", "file": "data/train-00001-of-00002.parquet", "dir": "/content/ctx/cosmopedia-100k"},
    {"repo": "HuggingFaceH4/ultrachat_200k", "file": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", "dir": "/content/ctx/ultrachat"},
    {"repo": "wikimedia/wikipedia", "file": "20231101.ja/train-00013-of-00015.parquet", "dir": "/content/ctx/wikipedia-ja"},
    {"repo": "codeparrot/codeparrot-clean-train", "file": "file-000000000001.json.gz", "dir": "/content/ctx/codeparrot"},
]
CTX_MIX = ["--data", "/content/ctx/cosmopedia-100k@4", "--data", "/content/ctx/ultrachat:messages@2",
           "--data", "/content/ctx/wikipedia-ja@2", "--data", "/content/ctx/codeparrot:content@1"]
# Rows of related documents (section 10, step 2): 40,000 documents per source, clustered once per session.
DOCS_MIX = [*MIX[:-2], "--docs", "40000"]
CLUSTERS = ["--rows", "cluster", "--cluster-size", "60", "--holdout-frac", "0.05", "--cluster-cache", "/content/cache/clusters"]
DOCS_MODEL = ["--short-heads", "8", "--long-heads", "8", "--replace", "all", "--bias", "doc_causal", "--cluster-heads", "8"]

# Every layer replaced (section 10), measured on an A100 of 40 GB (results/2026-11-13_0327_mq-all-0p8b-env).
# stage 2 at 2,048 x 8, gradient checkpointing. The preflight said 23.2 GB peak (29.0 held), but scripts/mq_env.py
# then projected every position onto the vocabulary at once; the training (128 positions at a time) peaked at
# 6.7 GB (results/2026-11-13_0338_mq-all-0p8b; 6.71 GB with the corrected mq_env.py on the L4, 13.0 held).
# The first launch used 30 (so no evaluation ran beside stage 2); 14 lets one run beside it on 32 GB and more.
ALL_S2_GB = 14
ALL_S3 = {"bsz": "2", "accum": "1", "gpu_gb": 18}   # stage 3 at 8,192 x 2, 1,024 positions: 13.9 GB peak, 8,590 tokens/s


# Where the plans of the 0.8B with every layer replaced may run: rented cards, and the researcher's local
# RTX 5090 (premise 3 as amended on 2026-11-13: work of this size runs there; larger work is rented).
SMALL_GPUS = ["A100", "H100", "H200", "B200", "RTX PRO 6000", "RTX 5090"]


PREMISES_SHA = json.loads((HERE.parents[1] / "docs" / "PREMISES.sha256").read_text())["sha256"]


def write(name, plan):
    plan = {**plan, "premises_sha": PREMISES_SHA}      # the runner refuses a plan made for other premises
    (HERE / f"{name}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n")
    print(name, len(plan["runs"]), "runs,", plan["max_minutes"], "min")


def dev(tokens_s1="10e6", tokens_s2="10e6", tokens_s3="4e6"):
    M = "/content/models/Qwen3.5-0.8B"
    arms = {"r10": ["--short-heads", "8", "--long-heads", "8"], "r05": ["--short-heads", "4", "--long-heads", "4"]}
    runs = [{"name": "env", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 15, "no_common": True,
             "args": ["--model", M, "--seq", "2048", "--bsz", "8", "--short-heads", "8", "--long-heads", "8", "--stage2",
                      "--grad-checkpoint"],
             "output_files": ["env.json"]}]
    for arm, heads in arms.items():
        s1, s2, s3 = f"s1-{arm}", f"s2-{arm}", f"s3-{arm}"
        runs.append({"name": s1, "script": "scripts/mq_stage1.py", "after": ["env"], "gpu_gb": 14, "max_minutes": 80,
                     "args": ["--model", M, *MIX, *heads, "--seq-len", "2048", "--bsz", "8", "--tokens", tokens_s1,
                              "--lr", "1e-3", "--warmup", "50", "--eval-every", "150", "--eval-rows", "8",
                              "--no-optim-save"],
                     "output_files": ["delta/mq_config.json", "result.json"]})
        runs.append({"name": s2, "script": "scripts/mq_stage2.py", "after": [s1], "gpu_gb": S2_GB, "max_minutes": 110,
                     "args": ["--model", M, *MIX, "--delta-in", f"{OUT}/{s1}/delta", "--train", "mem+lora", "--lora-r", "32",
                              "--grad-checkpoint", "--seq-len", "2048", "--bsz", "8", "--positions", "512", "--tokens", tokens_s2,
                              "--lr", "1e-4", "--warmup", "30", "--eval-every", "150", "--eval-rows", "8",
                              "--no-optim-save"],
                     "output_files": ["delta/mq_config.json", "result.json"]})
        runs.append({"name": f"eval-{s1}", "script": "scripts/mq_eval.py", "after": [s1], "gpu_gb": 8, "max_minutes": 25,
                     "args": ["--model", M, *MIX[:-2], "--docs", "8000", "--delta", f"{OUT}/{s1}/delta", "--eval-rows", "16",
                              "--recall-distances", "32,512,1500,6000", "--recall-n", "20"],
                     "output_files": ["metrics.json", "recall.json"]})
        runs.append({"name": f"eval-{s2}", "script": "scripts/mq_eval.py", "after": [s2], "gpu_gb": 8, "max_minutes": 25,
                     "args": ["--model", M, *MIX[:-2], "--docs", "8000", "--delta", f"{OUT}/{s2}/delta", "--eval-rows", "16",
                              "--recall-distances", "32,512,1500,6000", "--recall-n", "20"],
                     "output_files": ["metrics.json", "recall.json"]})
        if arm == "r10":      # the long-row stage and the export only for the main arm
            runs.append({"name": s3, "script": "scripts/mq_stage2.py", "after": [s2], "gpu_gb": S2_GB, "max_minutes": 60,
                         "args": ["--model", M, *MIX, "--delta-in", f"{OUT}/{s2}/delta", "--train", "mem+lora", "--lora-r", "32",
                                  "--grad-checkpoint", "--seq-len", "8192", "--max-doc", "8192", "--bsz", "2", "--positions", "1024",
                                  "--tokens", tokens_s3, "--lr", "5e-5", "--warmup", "20", "--eval-every", "100",
                                  "--eval-rows", "4", "--no-optim-save"],
                         "output_files": ["delta/mq_config.json", "result.json"]})
            runs.append({"name": f"eval-{s3}", "script": "scripts/mq_eval.py", "after": [s3], "gpu_gb": 8, "max_minutes": 25,
                         "args": ["--model", M, *MIX[:-2], "--docs", "8000", "--delta", f"{OUT}/{s3}/delta", "--eval-rows", "16",
                                  "--recall-distances", "32,512,1500,6000", "--recall-n", "20"],
                         "output_files": ["metrics.json", "recall.json"]})
            runs.append({"name": "export-r10", "script": "scripts/mq_export.py", "after": [s3], "gpu_gb": 8, "max_minutes": 15,
                         "args": ["--model", M, "--delta", f"{OUT}/{s3}/delta", "--weights-dir", "/content/export/mosaic-qwen-0.8b",
                                  "--shard-mb", "1900", "--store", "github-release:mq-0p8b-dev"],
                         "output_files": ["export_report.json"]})
    for r in runs:
        r.setdefault("no_common", True)
    write("mq-dev-0p8b", {
        "name": "mq-dev-0p8b",
        "purpose": "docs/plan-mosaic-qwen.md D1：Qwen3.5-0.8B の注意6層を公式 Memory Mosaics の層に丸ごと置き換え、27B と同じ手順（段1 層合わせ→段2 全体蒸留(記憶+LoRA)→段3 長い列）を小さく通す。記憶の幅 2 通り（8+8・4+4 ヘッド）を並列",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 300, "save_every_min": 10,
        "max_parallel": 4, "stagger_sec": 45, "gpu_frac": 0.9,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


def dev_rest(prev="results/2026-11-11_2240_mq-dev-0p8b"):
    """The part of mq-dev-0p8b that did not run (8+8 heads: stage 2, stage 3, evaluations,
    export), same settings, starting from the stage-1 weights saved by that run."""
    M = "/content/models/Qwen3.5-0.8B"
    s1 = f"/content/prev/{prev}/s1-r10/delta"
    ev = lambda d: ["--model", M, *MIX[:-2], "--docs", "8000", "--delta", d, "--eval-rows", "16",
                    "--recall-distances", "32,512,1500,6000", "--recall-n", "20"]
    runs = [
        {"name": "s2-r10", "script": "scripts/mq_stage2.py", "gpu_gb": S2_GB, "max_minutes": 110,
         "args": ["--model", M, *MIX, "--delta-in", s1, "--train", "mem+lora", "--lora-r", "32",
                  "--grad-checkpoint", "--seq-len", "2048", "--bsz", "8", "--positions", "512", "--tokens", "10e6",
                  "--lr", "1e-4", "--warmup", "30", "--eval-every", "150", "--eval-rows", "8", "--no-optim-save"],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "eval-s2-r10", "script": "scripts/mq_eval.py", "after": ["s2-r10"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(f"{OUT}/s2-r10/delta"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "s3-r10", "script": "scripts/mq_stage2.py", "after": ["s2-r10"], "gpu_gb": S2_GB, "max_minutes": 60,
         "args": ["--model", M, *MIX, "--delta-in", f"{OUT}/s2-r10/delta", "--train", "mem+lora", "--lora-r", "32",
                  "--grad-checkpoint", "--seq-len", "8192", "--max-doc", "8192", "--bsz", "2", "--positions", "1024",
                  "--tokens", "4e6", "--lr", "5e-5", "--warmup", "20", "--eval-every", "100", "--eval-rows", "4",
                  "--no-optim-save"],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "eval-s3-r10", "script": "scripts/mq_eval.py", "after": ["s3-r10"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(f"{OUT}/s3-r10/delta"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "export-r10", "script": "scripts/mq_export.py", "after": ["s3-r10"], "gpu_gb": 8, "max_minutes": 15,
         "args": ["--model", M, "--delta", f"{OUT}/s3-r10/delta", "--weights-dir", "/content/export/mosaic-qwen-0.8b",
                  "--shard-mb", "1900", "--store", "github-release:mq-0p8b-dev"],
         "output_files": ["export_report.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-dev-0p8b-rest", {
        "name": "mq-dev-0p8b-rest",
        "purpose": f"docs/plan-mosaic-qwen.md D1 の続き：{prev} でメモリ不足のため走らなかった 8+8 ヘッドの段2・段3・評価・書き出しを、同じ設定で、その実行が残した段1 の重みから行う",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 150, "save_every_min": 10,
        "max_parallel": 2, "stagger_sec": 45, "gpu_frac": 0.9,
        "setup": {"results": [prev], "pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


def probe27():
    M = "/content/models/Qwen3.8-27B"
    runs = [
        {"name": "env27-s1", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 25,
         "args": ["--model", M, "--seq", "2048", "--bsz", "4", "--steps", "3"], "output_files": ["env.json"]},
        {"name": "env27-s2", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 25,
         "args": ["--model", M, "--seq", "2048", "--bsz", "1", "--steps", "3", "--stage2", "--grad-checkpoint", "--lora-r", "32"],
         "output_files": ["env.json"]},
        {"name": "env27-s2-nockpt", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 20,
         "args": ["--model", M, "--seq", "2048", "--bsz", "1", "--steps", "2", "--stage2", "--lora-r", "32"],
         "output_files": ["env.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-27b-probe", {
        "name": "mq-27b-probe",
        "purpose": "docs/plan-mosaic-qwen.md P27：Qwen3.8-27B の本物の重みで、16 層の置き換え・教師一致・段1 と段2 の 1 ステップの速さと最大メモリを測る（学習はしない）。27B の費用見積もりを実測に置き換える",
        "save_mode": "token", "auto_stop": True, "max_minutes": 100, "save_every_min": 5, "max_parallel": 1,
        "require_gpu": ["A100", "H100", "H200", "B200", "RTX PRO 6000"],
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.8-27B"]},
        "runs": runs})


def stages27(store="github-release:mq-27b", tokens=("30e6", "30e6", "8e6")):
    M = "/content/models/Qwen3.8-27B"
    common = ["--model", M, *MIX, "--store", None, "--shard-mb", "1900", "--no-optim-save", "--save-every-min", "30"]

    def args(tag, extra):
        a = list(common); a[a.index("--store") + 1] = f"{store}-{tag}"
        return a + extra

    plans = {
        "mq-27b-s1": ("scripts/mq_stage1.py", args("s1", ["--seq-len", "2048", "--bsz", "4", "--tokens", tokens[0], "--lr", "1e-3",
                                                          "--warmup", "100", "--eval-every", "250", "--eval-rows", "4"]),
                      "段1（層合わせ）：元の 27B を 1 回流すだけで 16 層の記憶を同時に学習"),
        "mq-27b-s2": ("scripts/mq_stage2.py", args("s2", ["--delta-in", "/content/prev/s1/delta", "--train", "mem+lora", "--lora-r", "32",
                                                          "--grad-checkpoint", "--seq-len", "2048", "--bsz", "1", "--accum", "4",
                                                          "--positions", "512", "--tokens", tokens[1], "--lr", "1e-4",
                                                          "--warmup", "30", "--eval-every", "200", "--eval-rows", "4"]),
                      "段2（全体蒸留）：記憶＋LoRA を、元のモデルの出力に合わせて学習"),
        "mq-27b-s3": ("scripts/mq_stage2.py", args("s3", ["--delta-in", "/content/prev/s2/delta", "--train", "mem+lora", "--lora-r", "32",
                                                          "--grad-checkpoint", "--seq-len", "8192", "--max-doc", "8192", "--bsz", "1",
                                                          "--positions", "1024", "--tokens", tokens[2], "--lr", "5e-5",
                                                          "--warmup", "20", "--eval-every", "100", "--eval-rows", "2"]),
                      "段3（長い列）：8k トークンの列で段2 を続ける"),
    }
    for name, (script, a, what) in plans.items():
        prev = {"mq-27b-s2": "s1", "mq-27b-s3": "s2"}.get(name)
        runs = []
        if prev:   # the previous stage's final checkpoint, from its store (never one of a stage still going)
            runs.append({"name": f"fetch-{prev}", "script": "scripts/mq_fetch.py", "wait": True, "max_minutes": 15, "no_common": True,
                         "args": ["--store", f"{store}-{prev}", "--dest", f"/content/prev/{prev}/delta", "--require-tag", "done"]})
        # One launch: stage 2 is about 12 hours of GPU at the measured 720 tokens/s; relaunching every
        # 165 minutes costs 10 minutes of setup each time and a new draw for a GPU (Colab handed out an
        # L4 three times on 2026-11-12). The weights go to the store every 30 minutes and a relaunch
        # resumes from there (done for stage 1), so a lost session costs at most that.
        limit = {"mq-27b-s2": 705, "mq-27b-s3": 240}.get(name, 165)
        runs.append({"name": name.split("-")[-1], "script": script, "no_common": True, "max_minutes": limit,
                     **({"after": [f"fetch-{prev}"]} if prev else {}), "args": a,
                     "output_files": ["delta/mq_config.json", "result.json"]})
        tail = 0
        if name == "mq-27b-s3":    # the numbers the plan's criteria are judged on, in the same launch (same GPU)
            tail = 40
            runs.append({"name": "eval-s3", "script": "scripts/mq_eval.py", "no_common": True, "max_minutes": 35, "after": ["s3"],
                         "args": ["--model", M, *MIX[:-2], "--docs", "8000", "--delta", f"{OUT}/s3/delta", "--eval-rows", "16",
                                  "--recall-distances", "32,512,1500,6000", "--recall-n", "20"],
                         "output_files": ["metrics.json", "recall.json"]})
        write(name, {
            "name": name,
            "purpose": f"docs/plan-mosaic-qwen.md：Qwen3.8-27B {what}。重みは {store}-* に置き、同じ計画をもう一度起動すると続きから進む（1 回の起動は {limit} 分まで）",
            "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": limit + (15 if prev else 10) + tail,
            "save_every_min": 10,
            "max_parallel": 1, "require_gpu": ["A100", "H100", "H200", "B200", "RTX PRO 6000"],
            "setup": {"pip": PIP, "models": ["Qwen/Qwen3.8-27B"], "data": DATA},
            "runs": runs})


def recall_probe27(store="github-release:mq-27b"):
    M = "/content/models/Qwen3.8-27B"
    runs = [{"name": f"fetch-{t}", "script": "scripts/mq_fetch.py", "wait": True, "max_minutes": 10, "no_common": True,
             "args": ["--store", f"{store}-{t}", "--dest", f"/content/prev/{t}/delta", "--require-tag", "done"]}
            for t in ("s1", "s2", "s3")]
    runs.append({"name": "probe", "script": "scripts/mq_recall_probe.py", "no_common": True, "max_minutes": 30,
                 "after": ["fetch-s1", "fetch-s2", "fetch-s3"],
                 "args": ["--model", M, *[x for t in ("s1", "s2", "s3") for x in ("--delta", f"{t}=/content/prev/{t}/delta")],
                          *MIX[:-2], "--docs", "8000", "--max-minutes", "27"],
                 "output_files": ["probe.json", "result.json"]})
    write("mq-27b-recall-probe", {
        "name": "mq-27b-recall-probe",
        "purpose": "docs/plan-mosaic-qwen.md 追記（想起の切り分け）：Qwen3.8-27B の教師と、段1・段2・段3 の重みで、想起の問いを 4 通り"
                   "（評価と同じ詰め物／会話を含まない詰め物／会話の形／生成の経路）に分けて、答えの位置の確率を測る。学習はしない",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 40, "save_every_min": 5,
        "max_parallel": 1, "require_gpu": ["A100", "H100", "H200", "B200", "RTX PRO 6000"],
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.8-27B"], "data": DATA},
        "runs": runs})


BENCH_DATA = [
    {"repo": "cais/mmlu", "file": "all/test-00000-of-00001.parquet", "dir": "/content/data/mmlu"},
    {"repo": "openai/gsm8k", "file": "main/test-00000-of-00001.parquet", "dir": "/content/data/gsm8k"},
]


def final27(stage="s2", store="github-release:mq-27b"):
    M, D = "/content/models/Qwen3.8-27B", f"/content/prev/{stage}/delta"
    runs = [
        {"name": f"fetch-{stage}", "script": "scripts/mq_fetch.py", "wait": True, "max_minutes": 10, "no_common": True,
         "args": ["--store", f"{store}-{stage}", "--dest", D, "--require-tag", "done"]},
        {"name": f"eval-{stage}", "script": "scripts/mq_eval.py", "no_common": True, "max_minutes": 20, "after": [f"fetch-{stage}"],
         "args": ["--model", M, *MIX[:-2], "--docs", "8000", "--delta", D, "--eval-rows", "16",
                  "--recall-distances", "32,512,1500,6000", "--recall-n", "20"],
         "output_files": ["metrics.json", "recall.json", "samples.json"]},
        {"name": f"bench-{stage}", "script": "scripts/mq_bench.py", "no_common": True, "max_minutes": 45, "after": [f"eval-{stage}"],
         "args": ["--model", M, "--delta", D, "--mmlu", "/content/data/mmlu/all/test-00000-of-00001.parquet",
                  "--gsm", "/content/data/gsm8k/main/test-00000-of-00001.parquet", "--n-mmlu", "1000", "--n-gsm", "100",
                  "--gsm-max-new", "256", "--gsm-minutes", "22", "--max-minutes", "42"],
         "output_files": ["bench.json", "result.json"]},
        # the export check (no attention parameter left, the reloaded model gives the same output). The 54 GB
        # are written to the VM's disk only: uploading them waits until the weights are final.
        {"name": f"export-{stage}", "script": "scripts/mq_export.py", "no_common": True, "max_minutes": 30, "after": [f"bench-{stage}"],
         "args": ["--model", M, "--delta", D, "--weights-dir", "/content/export/mosaic-qwen-27b", "--shard-mb", "1900"],
         "output_files": ["export_report.json"]},
    ]
    write(f"mq-27b-{stage}-final", {
        "name": f"mq-27b-{stage}-final",
        "purpose": f"docs/plan-mosaic-qwen.md §9：次へ進める重み（段 {stage[1:]}）で、最後の評価、一般のベンチマーク（MMLU・GSM8K を元の 27B と比較）、"
                   "書き出しの確認を 1 回の起動で行う。学習はしない",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 115, "save_every_min": 5,
        "max_parallel": 1, "require_gpu": ["A100", "H100", "H200", "B200", "RTX PRO 6000"],
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.8-27B"], "data": DATA + BENCH_DATA},
        "runs": runs})


def all_env():
    """Section 10, before any training: what a stage-1 and a stage-2 step cost when all 24 layers of the
    0.8B are Mosaics blocks (152M memory parameters instead of 38M). The token budgets and batch sizes of
    the training plan are set from these numbers."""
    M = "/content/models/Qwen3.5-0.8B"
    heads = ["--short-heads", "8", "--long-heads", "8", "--replace", "all"]
    runs = [
        {"name": "env-all-2k", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 12,
         "args": ["--model", M, "--seq", "2048", "--bsz", "8", *heads, "--steps", "4", "--stage2", "--grad-checkpoint"],
         "output_files": ["env.json"]},
        {"name": "env-all-2k-nockpt", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 8,
         "args": ["--model", M, "--seq", "2048", "--bsz", "8", *heads, "--steps", "4", "--stage2"],
         "output_files": ["env.json"]},
        {"name": "env-all-8k", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 10,
         "args": ["--model", M, "--seq", "8192", "--bsz", "2", *heads, "--steps", "4", "--stage2", "--grad-checkpoint",
                  "--positions", "1024"],
         "output_files": ["env.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-all-0p8b-env", {
        "name": "mq-all-0p8b-env",
        "purpose": "docs/plan-mosaic-qwen.md §10：Qwen3.5-0.8B の全 24 層を公式 Memory Mosaics の層に置き換えたときの、"
                   "段1 と段2 の 1 ステップの速さと最大メモリを測る（系列長 2,048 と 8,192）。学習はしない。学習量とバッチの大きさをこの実測で決める",
        "save_mode": "token", "auto_stop": True, "max_minutes": 35, "save_every_min": 5, "max_parallel": 1,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"]},
        "runs": runs})


def all_env_l4():
    """The training of the 0.8B with every layer replaced peaks at 6.7 GB, so an L4 holds it. Colab charges
    by the hour per kind of card, so what matters is tokens per compute unit: this measures the speed of a
    stage-1 and a stage-2 step on the L4 (the unit rate is read from Colab's resource panel)."""
    M = "/content/models/Qwen3.5-0.8B"
    heads = ["--short-heads", "8", "--long-heads", "8", "--replace", "all"]
    runs = [
        {"name": "env-all-2k", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 14,
         "args": ["--model", M, "--seq", "2048", "--bsz", "8", *heads, "--steps", "4", "--stage2", "--grad-checkpoint"],
         "output_files": ["env.json"]},
        {"name": "env-all-8k", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 12,
         "args": ["--model", M, "--seq", "8192", "--bsz", "2", *heads, "--steps", "4", "--stage2", "--grad-checkpoint",
                  "--positions", "1024"],
         "output_files": ["env.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-all-0p8b-env-l4", {
        "name": "mq-all-0p8b-env-l4",
        "purpose": "docs/plan-mosaic-qwen.md §10：0.8B の全層置き換えの学習を L4（24GB）で回したときの速さを測る。"
                   "メモリは足りる（学習の最大 6.7GB）ので、1 ユニットあたりのトークン数を A100 と比べて安い方を使う。学習はしない",
        "save_mode": "token", "auto_stop": True, "max_minutes": 32, "save_every_min": 5, "max_parallel": 1,
        "require_gpu": ["L4"],
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"]},
        "runs": runs})


def all_main(tokens=("20e6", "40e6", "8e6"), store="github-release:mq-all-0p8b"):
    """Section 10, step 1. Same recipe as mq-dev-0p8b's main arm (8+8 heads, same data mix, same learning
    rates, same evaluation), with all 24 layers replaced and the token budgets of section 10.

    Measured on the A100 of 40 GB (results/2026-11-13_0327_mq-all-0p8b-env): stage 1 at 2,048 x 8 runs at
    14,950 tokens/s and peaks at 11.4 GB; stage 2 with gradient checkpointing at 9,540 tokens/s and 23.2 GB
    (29.0 GB held); without checkpointing it does not fit in 40 GB. Stage 3 at 8,192 x 2: 8,590 tokens/s, 13.9 GB.
    At these speeds the three stages are 22 + 70 + 16 minutes of training; with the evaluations and the setup
    about 2.3 hours (about 20 Colab units on an A100).

    Order: stage 2 starts as soon as stage 1 ends (the two do not fit beside an evaluation on 40 GB); the
    evaluations of stages 1 and 2 run beside stage 3 (18 + 8 + 8 GB).

    The memories are 152M parameters (about 310 MB per save). They go to release assets (`store`-s1/-s2/-s3,
    one shard, replaced at every save); in a results branch every save would add its size to the repository
    for good. A shard over 95 MB is left out of the branch by colab/saver.py. Launching the plan again
    continues each stage from its store."""
    M = "/content/models/Qwen3.5-0.8B"
    heads = ["--short-heads", "8", "--long-heads", "8", "--replace", "all"]
    keep = lambda tag: ["--store", f"{store}-{tag}", "--shard-mb", "1900", "--no-optim-save", "--save-every-min", "15"]
    ev = lambda d: ["--model", M, *MIX[:-2], "--docs", "8000", "--delta", d, "--eval-rows", "16",
                    "--recall-distances", "32,512,1500,6000", "--recall-n", "20"]
    runs = [
        {"name": "s1-all", "script": "scripts/mq_stage1.py", "gpu_gb": 14, "max_minutes": 50,
         "args": ["--model", M, *MIX, *heads, "--seq-len", "2048", "--bsz", "8", "--tokens", tokens[0],
                  "--lr", "1e-3", "--warmup", "50", "--eval-every", "150", "--eval-rows", "8", *keep("s1")],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "s2-all", "script": "scripts/mq_stage2.py", "after": ["s1-all"], "gpu_gb": ALL_S2_GB, "max_minutes": 115,
         "args": ["--model", M, *MIX, "--delta-in", f"{OUT}/s1-all/delta", "--train", "mem+lora", "--lora-r", "32",
                  "--grad-checkpoint", "--seq-len", "2048", "--bsz", "8", "--positions", "512", "--tokens", tokens[1],
                  "--lr", "1e-4", "--warmup", "30", "--eval-every", "150", "--eval-rows", "8", *keep("s2")],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "eval-s1-all", "script": "scripts/mq_eval.py", "after": ["s1-all"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(f"{OUT}/s1-all/delta"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "eval-s2-all", "script": "scripts/mq_eval.py", "after": ["s2-all"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(f"{OUT}/s2-all/delta"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "s3-all", "script": "scripts/mq_stage2.py", "after": ["s2-all"], "gpu_gb": ALL_S3["gpu_gb"], "max_minutes": 60,
         "args": ["--model", M, *MIX, "--delta-in", f"{OUT}/s2-all/delta", "--train", "mem+lora", "--lora-r", "32",
                  "--grad-checkpoint", "--seq-len", "8192", "--max-doc", "8192", "--bsz", ALL_S3["bsz"],
                  "--accum", ALL_S3["accum"], "--positions", "1024", "--tokens", tokens[2], "--lr", "5e-5",
                  "--warmup", "20", "--eval-every", "100", "--eval-rows", "4", *keep("s3")],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "eval-s3-all", "script": "scripts/mq_eval.py", "after": ["s3-all"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(f"{OUT}/s3-all/delta"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "export-all", "script": "scripts/mq_export.py", "after": ["s3-all"], "gpu_gb": 8, "max_minutes": 15,
         "args": ["--model", M, "--delta", f"{OUT}/s3-all/delta", "--weights-dir", "/content/export/mosaic-qwen-0.8b-all",
                  "--shard-mb", "1900", "--store", f"{store}-dev"],
         "output_files": ["export_report.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-all-0p8b", {
        "name": "mq-all-0p8b",
        "purpose": "docs/plan-mosaic-qwen.md §10 手順 1：Qwen3.5-0.8B の全 24 層を公式 Memory Mosaics の層に置き換え、D1 と同じ手順"
                   "（段1 層合わせ→段2 全体蒸留(記憶+LoRA)→段3 長い列）と同じ評価で、品質がどこまで戻るかを見る。"
                   f"学習量は {tokens[0]} / {tokens[1]} / {tokens[2]} トークン。重みは {store}-* の Release に置く",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 230, "save_every_min": 10,
        "max_parallel": 3, "stagger_sec": 30, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


def all_roundtrip(store="github-release:mq-all-0p8b-s3"):
    """The export check of step 1 compared one prompt: after writing the standalone model in bf16 and reading
    it back, the first choice differed at 1 of 18 positions (largest logit difference 0.92). This measures the
    same thing properly: 6 held-out rows of 256 tokens, fp32 arithmetic, the model in memory against the
    standalone model saved in fp32 and in bf16 (as done for D1, results/2026-11-12_0032_mq-roundtrip-0p8b)."""
    M = "/content/models/Qwen3.5-0.8B"
    D = "/content/prev/all-s3/delta"
    runs = [
        {"name": "fetch-all-s3", "script": "scripts/mq_fetch.py", "wait": True, "max_minutes": 15,
         "args": ["--store", store, "--dest", D, "--require-tag", "done"]},
        {"name": "roundtrip", "script": "scripts/mq_roundtrip.py", "after": ["fetch-all-s3"], "gpu_gb": 16, "max_minutes": 25,
         "args": ["--model", M, "--delta", D, *MIX[:-2], "--docs", "400", "--rows", "6", "--seq-len", "256",
                  "--work", "/content/roundtrip-work"],
         "output_files": ["roundtrip.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-all-0p8b-roundtrip", {
        "name": "mq-all-0p8b-roundtrip",
        "purpose": "docs/plan-mosaic-qwen.md §10 手順 1 の動作の条件：書き出したモデル（fp32・bf16）を読み直し、"
                   "メモリ上のモデルと同じ出力になるかを、学習に使わない位置で測る。学習はしない",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 45, "save_every_min": 5, "max_parallel": 1,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


def all_docs_env():
    """Section 10, step 2, before any training. (1) The clusters on the real data: how long the clustering
    takes, what the rows look like, how many cases the fresh files give the context-growth evaluation.
    (2) What a stage-1 and a stage-2 step cost with three memories under "doc_causal": one row of 16,384
    tokens (the setting of the plan) and two rows of 8,192 (the same number of tokens, for comparison)."""
    M = "/content/models/Qwen3.5-0.8B"
    runs = [
        {"name": "clusters", "script": "scripts/mq_cluster_check.py", "wait": True, "max_minutes": 25,
         "args": ["--model", M, *DOCS_MIX, "--seq-len", "16384", "--max-doc", "8192", "--cluster-size", "60",
                  "--holdout-frac", "0.05", "--cluster-cache", "/content/cache/clusters",
                  *[x if x != "--data" else "--fresh-data" for x in CTX_MIX], "--fresh-docs", "6000"],
         "output_files": ["clusters.json"]},
        {"name": "env-docs-16k", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 12,
         "args": ["--model", M, "--seq", "16384", "--bsz", "1", *DOCS_MODEL, "--steps", "4", "--stage2", "--grad-checkpoint",
                  "--positions", "2048"],
         "output_files": ["env.json"]},
        {"name": "env-docs-8k", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 10,
         "args": ["--model", M, "--seq", "8192", "--bsz", "2", *DOCS_MODEL, "--steps", "4", "--stage2", "--grad-checkpoint",
                  "--positions", "1024"],
         "output_files": ["env.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-all-0p8b-docs-env", {
        "name": "mq-all-0p8b-docs-env",
        "purpose": "docs/plan-mosaic-qwen.md §10 手順 2 の前：本物のデータで文書のまとまりを作って中身と所要時間を見る。"
                   "記憶 3 つ・文書ごとの区切り（doc_causal）での 1 ステップの速さと最大メモリを、16,384 トークンで測る。学習はしない",
        "save_mode": "token", "auto_stop": True, "max_minutes": 50, "save_every_min": 5, "max_parallel": 1,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA + CTX_DATA},
        "runs": runs})


# Step 2, from the preflight (results/<date>_mq-all-0p8b-docs-env); None until it has run.
# Local RTX 5090 (results/2026-11-13_1542_mq-all-0p8b-docs-env), 16,384 x 1, three memories, doc_causal:
# stage 1 at 9,364 tokens/s peaks at 12.8 GB (14.3 held); stage 2 at 6,649 tokens/s at 8.7 GB (14.5 held).
DOCS_GB = {"s1": 15, "s2": 15}


def all_docs(tokens=("20e6", "48e6"), store="github-release:mq-all-0p8b-docs", step1_store="github-release:mq-all-0p8b-s3"):
    """Section 10, step 2 (structure, data, training, evaluation and criteria are fixed there).

    Evaluations: `eval-*` on the rows of step 1 (the documents D1 was read on), `evalc-*` on rows the training
    never read (--docs 60000: the held-out documents of the training of step 1), for this model and for the
    final model of step 1 (fetched from its store); `context`: the context-growth evaluation on files no
    training read, teacher / step 1 / step 2 / step 2 without its cluster memory."""
    if DOCS_GB is None:
        return
    M = "/content/models/Qwen3.5-0.8B"
    keep = lambda tag: ["--store", f"{store}-{tag}", "--shard-mb", "1900", "--no-optim-save", "--save-every-min", "15"]
    rows = ["--seq-len", "16384", "--max-doc", "8192", "--bsz", "1", *CLUSTERS]
    ev = lambda d, docs: ["--model", M, *MIX[:-2], "--docs", docs, "--delta", d, "--eval-rows", "16",
                          "--recall-distances", "32,512,1500,6000", "--recall-n", "20"]
    S1, S2, A1 = f"{OUT}/s1-docs/delta", f"{OUT}/s2-docs/delta", "/content/prev/all-s3/delta"
    runs = [
        {"name": "fetch-all-s3", "script": "scripts/mq_fetch.py", "max_minutes": 15,
         "args": ["--store", step1_store, "--dest", A1, "--require-tag", "done"]},
        {"name": "s1-docs", "script": "scripts/mq_stage1.py", "gpu_gb": DOCS_GB["s1"], "max_minutes": 75,
         "args": ["--model", M, *DOCS_MIX, *DOCS_MODEL, *rows, "--tokens", tokens[0], "--lr", "1e-3", "--warmup", "50",
                  "--eval-every", "150", "--eval-rows", "2", *keep("s1")],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "s2-docs", "script": "scripts/mq_stage2.py", "after": ["s1-docs"], "gpu_gb": DOCS_GB["s2"], "max_minutes": 170,
         "args": ["--model", M, *DOCS_MIX, "--delta-in", S1, "--train", "mem+lora", "--lora-r", "32", "--grad-checkpoint",
                  *rows, "--positions", "2048", "--tokens", tokens[1], "--lr", "1e-4", "--warmup", "30", "--ce-weight", "0.5",
                  "--eval-every", "150", "--eval-rows", "2", *keep("s2")],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "evalc-s3-all", "script": "scripts/mq_eval.py", "after": ["fetch-all-s3"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(A1, "60000"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "eval-s1-docs", "script": "scripts/mq_eval.py", "after": ["s1-docs"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(S1, "8000"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "eval-s2-docs", "script": "scripts/mq_eval.py", "after": ["s2-docs"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(S2, "8000"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "evalc-s2-docs", "script": "scripts/mq_eval.py", "after": ["s2-docs"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev(S2, "60000"), "output_files": ["metrics.json", "recall.json"]},
        {"name": "context", "script": "scripts/mq_context.py", "after": ["s2-docs", "fetch-all-s3"], "gpu_gb": 16, "max_minutes": 45,
         # 40,000 documents per source (as the training rows: about 667 clusters of 60). With 6,000 the clusters
         # of the preflight were coarse (results/2026-11-13_1536_mq-all-0p8b-docs-env: a designer next to a car).
         "args": ["--model", M, *CTX_MIX, "--docs", "40000", "--student", f"all={A1}", "--student", f"docs={S2}",
                  "--seq-len", "16384", "--n-per-source", "20", "--cluster-cache", "/content/cache/clusters"],
         "output_files": ["context.json"]},
        {"name": "export-docs", "script": "scripts/mq_export.py", "after": ["s2-docs"], "gpu_gb": 8, "max_minutes": 15,
         "args": ["--model", M, "--delta", S2, "--weights-dir", "/content/export/mosaic-qwen-0.8b-docs",
                  "--shard-mb", "1900", "--store", f"{store}-dev"],
         "output_files": ["export_report.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-all-0p8b-docs", {
        "name": "mq-all-0p8b-docs",
        "purpose": "docs/plan-mosaic-qwen.md §10 手順 2：全 24 層の置き換えに、関連文書用の記憶を足し、記憶を文書ごとに区切り（doc_causal）、"
                   "関連文書を並べた 16,384 トークンの行で学習する（段1 2,000 万 → 段2 4,800 万トークン、KL + 0.5×次トークン損失）。"
                   "手順 1 と同じ評価、学習に使わない文書での評価、文脈を増やす評価で、手順 1 のモデル・教師と比べる",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 290, "save_every_min": 10,
        "max_parallel": 3, "stagger_sec": 30, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA + CTX_DATA},
        "runs": runs})


def probe27_all():
    """Section 10, step 3: what training the 27B with all 64 layers replaced would cost, measured instead of
    estimated. Each run loads the model once and measures a stage-1 step, then a stage-2 step (LoRA r32,
    gradient checkpointing). 16+16 heads is the width of the current 27B (4.06B memory parameters over 64
    layers); 40+40 is the width ratio 1.0 of the 0.8B runs (about 10B). Out of memory is a result here."""
    M = "/content/models/Qwen3.8-27B"
    runs = [
        {"name": f"env27-all-{h}", "script": "scripts/mq_env.py", "wait": True, "max_minutes": 14,
         "args": ["--model", M, "--seq", "2048", "--bsz", "2", "--short-heads", str(h), "--long-heads", str(h),
                  "--replace", "all", "--steps", "3", "--stage2", "--grad-checkpoint", "--lora-r", "32"],
         "output_files": ["env.json"]}
        for h in (16, 40)]
    for r in runs:
        r["no_common"] = True
    write("mq-27b-all-probe", {
        "name": "mq-27b-all-probe",
        "purpose": "docs/plan-mosaic-qwen.md §10 手順 3：Qwen3.8-27B の全 64 層を公式 Memory Mosaics の層に置き換えたときの、"
                   "段1 と段2 の 1 ステップの速さと最大メモリを B200 で測る（16+16 ヘッドと 40+40 ヘッド）。学習はしない",
        "save_mode": "token", "auto_stop": True, "max_minutes": 40, "save_every_min": 5, "max_parallel": 1,
        "require_gpu": ["B200", "B300", "H200"],
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.8-27B"]},
        "runs": runs})


# ── Section 10, "why does the quality stop there": diagnostics on the step-1 structure ──
# (8+8 heads, every layer replaced, rows of 2,048 x 8 from the training mix of step 1)

FIT_GB = 20        # one layer-matching run at a time on a 32 GB card (not measured with the controls beside the memories)
ALL_TRAIN_GB = 24  # stage 2 with every weight trained: weights, gradients and Adam state in fp32 plus a separate
                   # teacher (estimate: 3.4 x 4 + 1.7 + memories 2.4 + activations 4 = about 22 GB). Runs alone.


def _fit(model, heads, tokens):
    return ["--model", f"/content/models/{model}", *MIX, "--short-heads", heads, "--long-heads", heads, "--replace", "all",
            "--seq-len", "2048", "--bsz", "8", "--tokens", tokens, "--lr", "1e-3", "--control-lr", "1e-3,3e-4",
            "--warmup", "50", "--eval-every", "150", "--eval-rows", "8"]


def diag_layers(tokens="10e6"):
    """Diagnostic 1: beside the memories, a newly initialised mixer of the original's type is matched to
    every layer of Qwen3.5-0.8B on the same batches (scripts/mq_stage1_control.py). The settings of the
    memories are those of stage 1 of step 1 (learning rate 1e-3, 2,048 x 8, the same data), at half its
    tokens. The control is trained at two learning rates (1e-3, as the memories, and 3e-4)."""
    runs = [{"name": "fit-qwen35", "script": "scripts/mq_stage1_control.py", "gpu_gb": FIT_GB, "max_minutes": 70,
             "args": _fit("Qwen3.5-0.8B", "8", tokens), "output_files": ["result.json", "run.json"], "no_common": True}]
    write("mq-diag-0p8b-layers", {
        "name": "mq-diag-0p8b-layers",
        "purpose": "docs/plan-mosaic-qwen.md §10 診断 1：Qwen3.5-0.8B の全 24 層で、記憶の層合わせ（段1 と同じ設定）の横に、"
                   "元と同じ型で重みを新しく初期化した層（対照）を同じ行・同じ損失で合わせる。層合わせの誤差が 0.25 前後で止まるのが、"
                   f"記憶の側の限界か、手順・学習量の側かを分ける。{tokens} トークン。モデルは保存しない（計測のみ）",
        "save_mode": "token", "auto_stop": True, "max_minutes": 90, "save_every_min": 10, "max_parallel": 1,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


TEACHERS = [   # (run name, Hugging Face id, heads per memory: heads x 128 = the model's width)
    ("fit-qwen3", "Qwen/Qwen3-0.6B-Base", "8"),          # 28 layers, width 1,024, softmax attention in every layer
    ("fit-minicpm5", "openbmb/MiniCPM5-1B", "12"),       # 24 layers, width 1,536, Llama type
]


def diag_teachers(tokens="10e6"):
    """Diagnostic 2: the same measurement on two other models of about the same size, with softmax
    attention in every layer. The two runs do not depend on each other: if one model cannot be converted
    (its error is the result), the other still runs."""
    runs = [{"name": name, "script": "scripts/mq_stage1_control.py", "gpu_gb": FIT_GB, "max_minutes": 80,
             "args": _fit(repo.split("/")[-1], heads, tokens), "output_files": ["result.json", "run.json"], "no_common": True}
            for name, repo, heads in TEACHERS]
    write("mq-diag-teachers", {
        "name": "mq-diag-teachers",
        "purpose": "docs/plan-mosaic-qwen.md §10 診断 2：同じ規模の別のモデル（Qwen3-0.6B-Base、MiniCPM5-1B。どちらも全層が softmax 注意）で、"
                   "全層を公式 Memory Mosaics の層に置き換えたときの層合わせの誤差を、診断 1 と同じ設定・同じデータ・同じ対照つきで測る。"
                   f"先生（蒸留の元）によって記憶の合わせやすさが変わるかを見る。各 {tokens} トークン。モデルは保存しない（計測のみ）",
        "save_mode": "token", "auto_stop": True, "max_minutes": 180, "save_every_min": 10, "max_parallel": 1,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": [repo for _, repo, _ in TEACHERS], "data": DATA},
        "runs": runs})


def diag_train(tokens="40e6", step1="github-release:mq-all-0p8b", store="github-release:mq-diag-0p8b"):
    """Diagnostics 3-5: three more stage-2 runs on the model of step 1, each of the size of its stage 2
    (40M tokens, 2,048 x 8, 512 positions, the same data and seed, so the same rows in the same order).

      cont-kl   continues the final model of step 1 (memories + LoRA, KL): is it the amount of training?
      all-kl    from the stage-1 result of step 1, every weight trained (KL): is it the low-rank limit?
      all-ce    the same, on the next token alone (no distillation): is it the distillation?

    Every result is evaluated on the 16 rows no training read (--docs 60000), as the stage-2 result of
    step 1 is (evalc-s2-all; the final model of step 1 on these rows: results/2026-11-13_1546, evalc-s3-all).
    The runs that train every weight run alone (ALL_TRAIN_GB); the others fit beside each other."""
    M = "/content/models/Qwen3.5-0.8B"
    S1, S2, S3 = "/content/prev/all-s1/delta", "/content/prev/all-s2/delta", "/content/prev/all-s3/delta"
    keep = lambda tag, every: ["--store", f"{store}-{tag}", "--shard-mb", "1900", "--no-optim-save", "--save-every-min", every]
    common = ["--model", M, *MIX, "--grad-checkpoint", "--seq-len", "2048", "--bsz", "8", "--positions", "512",
              "--tokens", tokens, "--lr", "1e-4", "--warmup", "30", "--eval-every", "150", "--eval-rows", "8"]
    ev = lambda d: ["--model", M, *MIX[:-2], "--docs", "60000", "--delta", d, "--eval-rows", "16",
                    "--recall-distances", "32,512,1500,6000", "--recall-n", "20", "--no-samples"]
    fetch = lambda tag, dest: {"name": f"fetch-all-{tag}", "script": "scripts/mq_fetch.py", "max_minutes": 15,
                               "args": ["--store", f"{step1}-{tag}", "--dest", dest, "--require-tag", "done"]}
    train = lambda name, gb, args: {"name": name, "script": "scripts/mq_stage2.py", "gpu_gb": gb, "max_minutes": 130,
                                    "args": args, "output_files": ["delta/mq_config.json", "result.json"]}
    evalc = lambda name, after, d: {"name": name, "script": "scripts/mq_eval.py", "after": [after], "gpu_gb": 8,
                                    "max_minutes": 25, "args": ev(d), "output_files": ["metrics.json", "recall.json"]}
    runs = [
        fetch("s1", S1), fetch("s2", S2), fetch("s3", S3),
        evalc("evalc-s2-all", "fetch-all-s2", S2),
        {**train("cont-kl", 8, [*common, "--delta-in", S3, "--train", "mem+lora", "--lora-r", "32", *keep("cont", "15")]),
         "after": ["fetch-all-s3"]},
        evalc("evalc-cont-kl", "cont-kl", f"{OUT}/cont-kl/delta"),
        {**train("all-kl", ALL_TRAIN_GB, [*common, "--delta-in", S1, "--train", "all", "--lr-base", "1e-5", *keep("allkl", "40")]),
         "after": ["fetch-all-s1"]},
        evalc("evalc-all-kl", "all-kl", f"{OUT}/all-kl/delta"),
        {**train("all-ce", ALL_TRAIN_GB, [*common, "--delta-in", S1, "--train", "all", "--lr-base", "1e-5",
                                          "--kl-weight", "0", "--ce-weight", "1", *keep("allce", "40")]),
         "after": ["fetch-all-s1"]},
        evalc("evalc-all-ce", "all-ce", f"{OUT}/all-ce/delta"),
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-diag-0p8b-train", {
        "name": "mq-diag-0p8b-train",
        "purpose": "docs/plan-mosaic-qwen.md §10 診断 3〜5：手順 1 のモデル（全 24 層置き換え、8+8 ヘッド）で、段2 と同じ大きさ"
                   f"（{tokens} トークン、同じ行）の学習を 3 通り行う。cont-kl＝手順 1 の最終モデルから同じ蒸留を続ける（学習量）、"
                   "all-kl＝段1 の結果から全部の重みを動かして蒸留する（LoRA の制約）、all-ce＝同じく全部の重みを動かし、"
                   "蒸留なしで次トークンの損失だけで学習する（蒸留かどうか）。どれも学習に使っていない 16 行で、手順 1 の段2 の結果と比べる",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 330, "save_every_min": 10,
        "max_parallel": 3, "stagger_sec": 30, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


INIT_GB = 13       # stage 1 at 2,048 x 8, every layer: 11.4 GB at the peak (A100), 12.5 GB with three memories (5090)


def diag_init(tokens="10e6"):
    """Diagnostic 6: stage 1 of step 1 again (same settings and data as diagnostic 1's memories), with the
    memories started from the original mixers' weights instead of random values. Two runs fit beside each
    other on the 32 GB card (2 x 13 GB estimated, 11.4 measured each); the third starts when one ends."""
    M = "/content/models/Qwen3.5-0.8B"
    base = ["--model", M, *MIX, "--short-heads", "8", "--long-heads", "8", "--replace", "all", "--seq-len", "2048",
            "--bsz", "8", "--tokens", tokens, "--warmup", "50", "--eval-every", "150", "--eval-rows", "8",
            "--no-optim-save", "--shard-mb", "1900"]
    arms = [("init-ov", "teacher-ov", "1e-3"), ("init-ovk", "teacher-ovk", "1e-3"), ("init-ov-lr3e4", "teacher-ov", "3e-4")]
    runs = [{"name": name, "script": "scripts/mq_stage1.py", "gpu_gb": INIT_GB, "max_minutes": 45, "no_common": True,
             "args": [*base, "--init", init, "--lr", lr], "output_files": ["delta/mq_config.json", "result.json", "run.json"]}
            for name, init, lr in arms]
    write("mq-diag-0p8b-init", {
        "name": "mq-diag-0p8b-init",
        "purpose": "docs/plan-mosaic-qwen.md §10 診断 6：記憶の初期値を、公式の乱数ではなく元の層の重みから作る（値と出力＝元の層の "
                   "出力側の写像の低ランク近似、-ovk は鍵も）。公式の層とコードは変えず、初期値だけを変える。段1 と同じ設定の層合わせを "
                   f"{tokens} トークン行い、乱数から始めた診断 1 の記憶と比べる。計測のみ（重みは Release に置かない）",
        "save_mode": "token", "auto_stop": True, "max_minutes": 80, "save_every_min": 10,
        "max_parallel": 2, "stagger_sec": 30, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


def diag_more(tokens_fit="10e6", tokens="40e6", step1="github-release:mq-all-0p8b", store="github-release:mq-diag-0p8b"):
    """Diagnostics 7 and 8.
    7 (fit-lr2e3, fit-lr3e3): stage 1 of step 1 with a higher learning rate (the control of diagnostic 1 did
       better at the higher of its two rates). Same data, 10M tokens; compared with 1e-3 (0.276).
    8 (prog-kl): stage 2 of step 1 again (same start, data, tokens, memories + LoRA, KL), with the layers going
       from original to Mosaics over the first 60% of the steps instead of all at once. What is evaluated and
       saved has every layer replaced. Compared with the stage-2 result of step 1 on the same 16 rows.
    Memory: stage 1 is 11.4 GB measured (13 estimated), prog-kl about 10 GB estimated (14 declared: the original
    mixers run with gradients through them); two runs at a time, at most 27 GB of estimates on the 32 GB card."""
    M = "/content/models/Qwen3.5-0.8B"
    S1 = "/content/prev/all-s1/delta"
    fit = lambda lr: ["--model", M, *MIX, "--short-heads", "8", "--long-heads", "8", "--replace", "all", "--seq-len", "2048",
                      "--bsz", "8", "--tokens", tokens_fit, "--warmup", "50", "--eval-every", "150", "--eval-rows", "8",
                      "--no-optim-save", "--shard-mb", "1900", "--lr", lr]
    ev = ["--model", M, *MIX[:-2], "--docs", "60000", "--delta", f"{OUT}/prog-kl/delta", "--eval-rows", "16",
          "--recall-distances", "32,512,1500,6000", "--recall-n", "20", "--no-samples"]
    runs = [
        {"name": "fetch-all-s1", "script": "scripts/mq_fetch.py", "max_minutes": 15,
         "args": ["--store", f"{step1}-s1", "--dest", S1, "--require-tag", "done"]},
        {"name": "prog-kl", "script": "scripts/mq_stage2.py", "after": ["fetch-all-s1"], "gpu_gb": 14, "max_minutes": 150,
         "args": ["--model", M, *MIX, "--delta-in", S1, "--train", "mem+lora", "--lora-r", "32", "--grad-checkpoint",
                  "--seq-len", "2048", "--bsz", "8", "--positions", "512", "--tokens", tokens, "--lr", "1e-4", "--warmup", "30",
                  "--eval-every", "150", "--eval-rows", "8", "--progressive", "0.6",
                  "--store", f"{store}-prog", "--shard-mb", "1900", "--no-optim-save", "--save-every-min", "15"],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "fit-lr2e3", "script": "scripts/mq_stage1.py", "gpu_gb": INIT_GB, "max_minutes": 45, "args": fit("2e-3"),
         "output_files": ["delta/mq_config.json", "result.json", "run.json"]},
        {"name": "fit-lr3e3", "script": "scripts/mq_stage1.py", "gpu_gb": INIT_GB, "max_minutes": 45, "args": fit("3e-3"),
         "output_files": ["delta/mq_config.json", "result.json", "run.json"]},
        {"name": "evalc-prog-kl", "script": "scripts/mq_eval.py", "after": ["prog-kl"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev, "output_files": ["metrics.json", "recall.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-diag-0p8b-more", {
        "name": "mq-diag-0p8b-more",
        "purpose": "docs/plan-mosaic-qwen.md §10 診断 7・8：(7) 段1 の層合わせを高い学習率（2e-3、3e-3）で行う。(8) 段2 を、全層を一度に "
                   "置き換える代わりに、学習の前半 6 割をかけて層を元の層から Memory Mosaics の層へ少しずつ入れ替えながら行う"
                   "（評価と保存は全層を置き換えたモデル）。手順 1 の段1・段2 と同じデータ・同じ学習量で比べる",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 200, "save_every_min": 10,
        "max_parallel": 2, "stagger_sec": 30, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


OOD_DATA = [{"repo": "wikimedia/wikipedia", "file": "20231101.en/train-00000-of-00041.parquet", "dir": "/content/ood/wikipedia-en"}]


def diag_fix(tokens="10e6", step1="github-release:mq-all-0p8b", store="github-release:mq-diag-0p8b"):
    """Diagnostic 6b: the teacher-weight start again with the output size fitted by least squares (the first
    attempt started 6-7 times too large). Diagnostic 5b: the stage-2 results (LoRA + KL, all weights + KL,
    all weights + next token only) on a kind of text none of them was trained on (English Wikipedia).
    Memory: two stage-1 runs side by side are 23 GB measured (diagnostic 6); evaluations are about 5 GB."""
    M = "/content/models/Qwen3.5-0.8B"
    base = ["--model", M, *MIX, "--short-heads", "8", "--long-heads", "8", "--replace", "all", "--seq-len", "2048",
            "--bsz", "8", "--tokens", tokens, "--warmup", "50", "--eval-every", "150", "--eval-rows", "8",
            "--no-optim-save", "--shard-mb", "1900", "--lr", "1e-3", "--init-gain"]
    runs = [{"name": name, "script": "scripts/mq_stage1.py", "gpu_gb": INIT_GB, "max_minutes": 45,
             "args": [*base, "--init", init], "output_files": ["delta/mq_config.json", "result.json", "run.json"]}
            for name, init in (("init2-ov", "teacher-ov"), ("init2-ovk", "teacher-ovk"))]
    # Diagnostic 8, continued: stopped at the researcher's request at 27M of 40M tokens (2026-11-13 15:48 UTC);
    # the run picks up from its last save in the store (18.1M tokens) with the same settings.
    S1 = "/content/prev/all-s1/delta"
    runs += [
        {"name": "fetch-all-s1", "script": "scripts/mq_fetch.py", "max_minutes": 15,
         "args": ["--store", f"{step1}-s1", "--dest", S1, "--require-tag", "done"]},
        {"name": "prog-kl", "script": "scripts/mq_stage2.py", "after": ["fetch-all-s1"], "gpu_gb": 14, "max_minutes": 90,
         "args": ["--model", M, *MIX, "--delta-in", S1, "--train", "mem+lora", "--lora-r", "32", "--grad-checkpoint",
                  "--seq-len", "2048", "--bsz", "8", "--positions", "512", "--tokens", "40e6", "--lr", "1e-4", "--warmup", "30",
                  "--eval-every", "150", "--eval-rows", "8", "--progressive", "0.6",
                  "--store", f"{store}-prog", "--shard-mb", "1900", "--no-optim-save", "--save-every-min", "15"],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "evalc-prog-kl", "script": "scripts/mq_eval.py", "after": ["prog-kl"], "gpu_gb": 8, "max_minutes": 25,
         "args": ["--model", M, *MIX[:-2], "--docs", "60000", "--delta", f"{OUT}/prog-kl/delta", "--eval-rows", "16",
                  "--recall-distances", "32,512,1500,6000", "--recall-n", "20", "--no-samples"],
         "output_files": ["metrics.json", "recall.json"]},
    ]
    runs = runs[-3:] + runs[:-3]          # the longest first
    # (5b, the evaluations on English Wikipedia, moved to mq-diag-0p8b-ood: the download of that file stalled
    #  at 134 MB on 2026-11-13 17:42 UTC and held up the whole plan in its setup)
    for r in runs:
        r["no_common"] = True
    write("mq-diag-0p8b-fix", {
        "name": "mq-diag-0p8b-fix",
        "purpose": "docs/plan-mosaic-qwen.md §10 診断 6b と診断 8 の続き：(6b) 元の層の重みからの初期値を、出力の大きさを最小二乗で合わせ直して測り直す。"
                   "(8) 研究者の指示で止めた少しずつの入れ替え"
                   "（2,736 万/4,000 万トークン）を、最後の保存（1,810 万）から同じ設定で続ける",
        "max_minutes": 100,
        "save_mode": "token", "auto_stop": True, "export_token": True, "save_every_min": 10,
        "max_parallel": 2, "stagger_sec": 30, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


def diag_ood(step1="github-release:mq-all-0p8b", store="github-release:mq-diag-0p8b"):
    """Diagnostic 5b: the stage-2 variants (and the overnight run) on English Wikipedia, which none of them
    was trained on. Evaluations only, about 5 GB each, two at a time."""
    M = "/content/models/Qwen3.5-0.8B"
    runs = []
    for tag, rel in (("s2", f"{step1}-s2"), ("allkl", f"{store}-allkl"), ("allce", f"{store}-allce"), ("long", f"{store}-long")):
        dest = f"/content/prev/{tag}/delta"
        runs.append({"name": f"fetch-{tag}", "script": "scripts/mq_fetch.py", "max_minutes": 20,
                     "args": ["--store", rel, "--dest", dest, "--require-tag", "done"]})
        runs.append({"name": f"ood-{tag}", "script": "scripts/mq_eval.py", "after": [f"fetch-{tag}"], "gpu_gb": 8, "max_minutes": 25,
                     "args": ["--model", M, "--data", "/content/ood/wikipedia-en", "--docs", "4000", "--delta", dest,
                              "--eval-rows", "16", "--recall-distances", "32", "--recall-n", "2", "--no-samples"],
                     "output_files": ["metrics.json"]})
    for r in runs:
        r["no_common"] = True
    write("mq-diag-0p8b-ood", {
        "name": "mq-diag-0p8b-ood",
        "purpose": "docs/plan-mosaic-qwen.md §10 診断 5b：段2 の 3 通り（記憶+LoRA の蒸留、全部の重みの蒸留、全部の重みで蒸留なし）と夜間の長い学習を、"
                   "どれも学習していない種類の文章（英語 Wikipedia）で比べる。蒸留なしの PPL の良さがデータへの慣れかどうかを見る",
        "save_mode": "token", "auto_stop": True, "max_minutes": 60, "save_every_min": 10,
        "max_parallel": 2, "stagger_sec": 20, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": OOD_DATA},
        "runs": runs})


def diag_long(tokens="160e6", step1="github-release:mq-all-0p8b", store="github-release:mq-diag-0p8b"):
    """The overnight run: what diagnostics 3-5 found to help, together and for longer. From the stage-1 result
    of step 1: every weight trained (diagnostic 4), KL plus half the next-token loss (diagnostic 5 found the
    next-token loss lowers perplexity; the KL keeps the model close to the teacher), 160M tokens (4 x stage 2
    of step 1; diagnostic 3). Alone on the card: 16.6 GB measured for this kind of run."""
    M = "/content/models/Qwen3.5-0.8B"
    S1 = "/content/prev/all-s1/delta"
    ev = ["--model", M, *MIX[:-2], "--docs", "60000", "--delta", f"{OUT}/long-all/delta", "--eval-rows", "16",
          "--recall-distances", "32,512,1500,6000", "--recall-n", "20", "--no-samples"]
    runs = [
        {"name": "fetch-all-s1", "script": "scripts/mq_fetch.py", "max_minutes": 15,
         "args": ["--store", f"{step1}-s1", "--dest", S1, "--require-tag", "done"]},
        {"name": "long-all", "script": "scripts/mq_stage2.py", "after": ["fetch-all-s1"], "gpu_gb": ALL_TRAIN_GB, "max_minutes": 330,
         "args": ["--model", M, *MIX, "--delta-in", S1, "--train", "all", "--lr-base", "1e-5", "--grad-checkpoint",
                  "--seq-len", "2048", "--bsz", "8", "--positions", "512", "--tokens", tokens, "--lr", "1e-4", "--warmup", "30",
                  "--ce-weight", "0.5", "--eval-every", "300", "--eval-rows", "8",
                  "--store", f"{store}-long", "--shard-mb", "1900", "--no-optim-save", "--save-every-min", "20"],
         "output_files": ["delta/mq_config.json", "result.json"]},
        {"name": "evalc-long-all", "script": "scripts/mq_eval.py", "after": ["long-all"], "gpu_gb": 8, "max_minutes": 25,
         "args": ev, "output_files": ["metrics.json", "recall.json"]},
    ]
    for r in runs:
        r["no_common"] = True
    write("mq-diag-0p8b-long", {
        "name": "mq-diag-0p8b-long",
        "purpose": "docs/plan-mosaic-qwen.md §10 夜間の長い学習：診断 3〜5 で効いたものを合わせる。手順 1 の段1 の結果から、全部の重みを動かし、"
                   f"損失は KL + 0.5 × 次トークンの損失、学習量は {tokens} トークン（手順 1 の段2 の 4 倍）。"
                   "全層を公式 Memory Mosaics の層に置き換えた 0.8B が、この組み合わせで合格の線にどこまで近づくかを見る",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 380, "save_every_min": 10,
        "max_parallel": 1, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": ["Qwen/Qwen3.5-0.8B"], "data": DATA},
        "runs": runs})


# ── Training the upstream models from scratch (docs/plan-scratch.md) ──

SCRATCH_DATA = "/content/scratch-data"         # outside /content/out: kept from one plan to the next
SCRATCH_LR = "/content/scratch-lr"
FINEWEB = "HuggingFaceFW/fineweb-edu"
FINEWEB_FILES = ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet", "sample/10BT/002_00000.parquet"]
SCRATCH_MODEL = ["--dim", "512", "--heads", "4", "--seq-len", "4096", "--bsz", "4", "--accum", "4"]
SCRATCH_GB = 8                                  # one model of width 512 x 24 layers with recomputed activations (estimate)
SCRATCH_TOKENS = "1.2e9"                        # 10 x the parameters of the Mosaics model (127M); the setup's schedule
# The main run: 4e8 each (the researcher, 2026-11-14: the 1.2e9 run would take about 21 h; results within the day)
SCRATCH_MAIN_TOKENS = "4e8"


def scratch_setup():
    """Data (fineweb-edu, upstream tokenizer), the speed of the two models side by side, and the
    learning rate picked on the Transformer (the upstream practice)."""
    fd = "/content/data/fineweb-edu"
    prep = ["--tokenizer", "mistralai/Mistral-7B-v0.1", *sum((["--fetch", f"{FINEWEB}:{f}"] for f in FINEWEB_FILES), []),
            "--fetch-dir", fd, "--train", f"{fd}/{FINEWEB_FILES[0]}", "--train", f"{fd}/{FINEWEB_FILES[1]}",
            "--eval", f"{fd}/{FINEWEB_FILES[2]}", "--train-tokens", "1.5e9", "--data-out", SCRATCH_DATA]
    common = ["--data", SCRATCH_DATA, *SCRATCH_MODEL, "--duty", "0.7", "--save-every-min", "1000"]
    runs = [{"name": "prep", "script": "scripts/scratch_prep.py", "max_minutes": 180, "args": prep,
             "output_files": ["meta.json"]}]
    for fam, tag in (("mosaics_v2", "mm"), ("transformer", "tf")):
        runs.append({"name": f"speed-{tag}", "script": "scripts/scratch_train.py", "after": ["prep"], "gpu_gb": SCRATCH_GB,
                     "max_minutes": 40, "args": ["--family", fam, *common, "--tokens", SCRATCH_TOKENS, "--stop-after", "2e7",
                                                 "--eval-at", "0", "--lr", "6e-4"],
                     "output_files": ["result.json", "run.json", "evals.jsonl"]})
    lrs = ["3e-4", "6e-4", "1.2e-3"]
    for lr in lrs:
        runs.append({"name": f"lr-{lr}", "script": "scripts/scratch_train.py", "after": ["speed-mm", "speed-tf"],
                     "gpu_gb": SCRATCH_GB, "max_minutes": 90,
                     "args": ["--family", "transformer", *common, "--tokens", "6e7", "--eval-at", "1.0", "--lr", lr],
                     "output_files": ["result.json", "run.json", "evals.jsonl"]})
    runs.append({"name": "pick-lr", "script": "scripts/scratch_pick_lr.py", "after": [f"lr-{lr}" for lr in lrs],
                 "max_minutes": 5, "args": [*sum((["--run", f"{OUT}/lr-{lr}"] for lr in lrs), []), "--write", SCRATCH_LR],
                 "output_files": ["lr.json"]})
    for r in runs:
        r["no_common"] = True
    write("mq-scratch-setup", {
        "name": "mq-scratch-setup",
        "purpose": "docs/plan-scratch.md 準備：学習の文章（fineweb-edu）を公式のトークナイザで用意し、Memory Mosaics v2 と Transformer "
                   "（どちらも公式の設定を幅 512・4 ヘッドに縮めたもの、24 層）を並べたときの速さと VRAM を測り、学習率を Transformer で決める"
                   "（3e-4・6e-4・1.2e-3 を各 6,000 万トークン、学習に使わない行の損失がいちばん低いもの）",
        "save_mode": "token", "auto_stop": True, "max_minutes": 420, "save_every_min": 10,
        "max_parallel": 2, "stagger_sec": 30, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": [], "data": []},
        "runs": runs})


def scratch_main(tokens=SCRATCH_MAIN_TOKENS):
    """The two models side by side on the same rows: checkpoints at 10/25/50/75/100%."""
    runs = []
    for fam, tag in (("mosaics_v2", "mm"), ("transformer", "tf")):
        runs.append({"name": f"main-{tag}", "script": "scripts/scratch_train.py", "gpu_gb": SCRATCH_GB, "max_minutes": 1500,
                     "args": ["--family", fam, "--data", SCRATCH_DATA, *SCRATCH_MODEL, "--tokens", tokens,
                              "--lr-from", f"{SCRATCH_LR}/lr.json", "--duty", "0.7", "--gpu-lock", "/content/gpu.lock",
                              "--eval-at", "0.1,0.25,0.5,0.75,1.0",
                              "--store", f"github-release:mq-scratch-{tag}", "--save-every-min", "30"],
                     "output_files": ["result.json", "run.json", "evals.jsonl"]})
    for r in runs:
        r["no_common"] = True
    write("mq-scratch-main", {
        "name": "mq-scratch-main",
        "purpose": "docs/plan-scratch.md 本番：Memory Mosaics v2 と Transformer を、同じ行・同じ順・同じ設定で最初から並べて学習する"
                   f"（{tokens} トークンずつ、GPU の使用は 7 割に抑える）。1・2.5・5・7.5・10 割の時点で、学習に使わない文章で損失・文書の中での"
                   "損失の下がり方・想起を測る",
        "save_mode": "token", "auto_stop": True, "export_token": True, "max_minutes": 1560, "save_every_min": 10,
        "max_parallel": 2, "stagger_sec": 20, "gpu_frac": 0.92,
        "require_gpu": SMALL_GPUS,
        "setup": {"pip": PIP, "models": [], "data": []},
        "runs": runs})


if __name__ == "__main__":
    scratch_setup()
    scratch_main()
    diag_ood()
    diag_long()
    diag_fix()
    diag_more()
    diag_init()
    diag_layers()
    diag_teachers()
    diag_train()
    all_env()
    all_env_l4()
    all_main()
    all_roundtrip()
    all_docs_env()
    all_docs()
    probe27_all()
    dev()
    dev_rest()
    probe27()
    stages27()
    recall_probe27()
    final27()
