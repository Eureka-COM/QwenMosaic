"""The MosaicQwen stage scripts end to end on a tiny random model: stage 1 -> stage 2 ->
export -> reload, with resume, gradient checkpointing and the train-everything mode."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import tiny_qwen35

ROOT = Path(__file__).resolve().parents[1]
CONV = ["--short-heads", "2", "--long-heads", "2", "--head-dim", "16", "--window", "8", "--skip", "4", "--random-skip", "4"]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """A saved tiny Qwen3.5 text model, a matching tokenizer and a small text corpus."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    d = tmp_path_factory.mktemp("tiny")
    rng = random.Random(0)
    words = ["memory", "mosaic", "key", "value", "the", "a", "of", "reads", "writes", "token", "layer", "model",
             "long", "short", "window", "past", "next", "state", "kernel", "norm", "gate", "head", "is", "and"]
    docs = [" ".join(rng.choice(words) for _ in range(rng.randrange(40, 160))) for _ in range(300)]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<|endoftext|>", "<pad>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<|endoftext|>", pad_token="<pad>")
    model = tiny_qwen35()
    model.config.eos_token_id = tok.eos_token_id
    model.save_pretrained(d / "model")
    tok.save_pretrained(d / "model")
    with (d / "docs.jsonl").open("w") as f:
        for t in docs:
            f.write(json.dumps({"text": t}) + "\n")
    # documents about six topics (four of the words each), for the rows of related documents
    for name, seed, n in (("topics.jsonl", 1, 360), ("topics_fresh.jsonl", 2, 240)):
        r = random.Random(seed)
        with (d / name).open("w") as f:
            for i in range(n):
                own = words[4 * (i % 6): 4 * (i % 6) + 4]
                text = " ".join(r.choice(own) if r.random() < 0.85 else r.choice(words) for _ in range(r.randrange(20, 60)))
                f.write(json.dumps({"text": text, "topic": i % 6}) + "\n")
    return d


def _run(script, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def _events(out):
    return [json.loads(line) for line in (Path(out) / "progress.jsonl").read_text().splitlines()]


def test_pipeline(tiny_dir, tmp_path):
    model, data = tiny_dir / "model", tiny_dir / "docs.jsonl"
    common = ["--model", model, "--data", data, "--seq-len", 48, "--bsz", 2, "--eval-rows", 2, "--log-every", 1,
              "--device", "cpu"]

    # stage 1: 6 of 12 steps (stopped by --steps), then resumed to the end
    s1 = tmp_path / "s1"
    _run("mq_stage1.py", *common, *CONV, "--steps", 12, "--eval-every", 6, "--lr", "3e-3", "--warmup", 2, "--out", s1)
    ev = _events(s1)
    assert ev[0]["event"] == "start" and ev[-1]["event"] == "done" and ev[-1]["step"] == 12
    evals = [e for e in ev if e["event"] == "eval"]
    assert evals[-1]["rel_mse_mean"] < evals[0]["rel_mse_mean"]            # the memories learned
    steps = [e for e in ev if e["event"] == "step"]
    assert steps[-1]["loss"] < steps[0]["loss"]
    res = json.loads((s1 / "result.json").read_text())
    assert res["status"] == "done" and res["step"] == 12
    meta = json.loads((s1 / "delta" / "mq_config.json").read_text())
    assert meta["attention_parameters_left"] == 0 and meta["mq_config"]["layers"] == [3, 7]
    run = json.loads((s1 / "run.json").read_text())
    assert run["attention_parameters_in_student"] == 0
    assert run["backends"]["upstream_commit"].startswith("c2ca0e8")

    s1b = tmp_path / "s1b"
    _run("mq_stage1.py", *common, *CONV, "--steps", 16, "--eval-every", 100, "--resume", s1, "--out", s1b)
    evb = _events(s1b)
    assert evb[0]["resumed_from"] == str(s1) and evb[-1]["step"] == 16
    assert [e["step"] for e in evb if e["event"] == "step"][0] == 13

    # stage 2: memories + LoRA, shared teacher
    s2 = tmp_path / "s2"
    _run("mq_stage2.py", *common, "--delta-in", s1 / "delta", "--steps", 6, "--eval-every", 3, "--positions", 16,
         "--lora-r", 2, "--lr", "1e-3", "--out", s2)
    ev2 = _events(s2)
    assert ev2[-1]["event"] == "done"
    meta2 = json.loads((s2 / "delta" / "mq_config.json").read_text())
    assert meta2["lora"] and meta2["extra"]["train"] == "mem+lora"
    e2 = [e for e in ev2 if e["event"] == "eval"]
    assert e2[0]["kl"] > 0 and all(torch.isfinite(torch.tensor(e["kl"])) for e in e2)

    # stage 2 with gradient checkpointing, memories only
    s2c = tmp_path / "s2c"
    _run("mq_stage2.py", *common, "--delta-in", s1 / "delta", "--steps", 2, "--eval-every", 100, "--positions", 16,
         "--train", "mem", "--grad-checkpoint", "--out", s2c)
    assert _events(s2c)[-1]["event"] == "done"

    # stage 2 training every weight, with a separate teacher and the hidden-state term
    s2a = tmp_path / "s2a"
    _run("mq_stage2.py", *common, "--delta-in", s1 / "delta", "--steps", 2, "--eval-every", 100, "--positions", 16,
         "--train", "all", "--hidden-weight", "0.1", "--out", s2a)
    assert json.loads((s2a / "run.json").read_text())["teacher"] == "separate copy"

    # evaluation and export of the stage-2 model
    ev_out = tmp_path / "eval"
    _run("mq_eval.py", "--model", model, "--delta", s2 / "delta", "--data", data, "--seq-len", 48, "--eval-rows", 2,
         "--recall-distances", "8", "--recall-n", 1, "--no-samples", "--device", "cpu", "--out", ev_out)
    m = json.loads((ev_out / "metrics.json").read_text())
    assert m["kl"] >= 0 and "rel_mse" in m
    ex = tmp_path / "export"
    rep = json.loads(_run("mq_export.py", "--model", model, "--delta", s2 / "delta", "--out", ex, "--save-dtype", "fp32",
                          "--device", "cpu").strip().splitlines()[-1])
    assert rep["attention_parameters_left"] == 0 and rep["attention_parameters_after_reload"] == 0
    assert rep["attention_modules_after_reload"] == 0 and rep["reload_max_logit_diff"] < 1e-3
    keys = set()
    from safetensors import safe_open
    for f in ex.glob("model-*.safetensors"):
        with safe_open(str(f), "pt") as sf:
            keys |= set(sf.keys())
    assert not any("self_attn" in k or ".q_proj" in k or ".k_proj" in k for k in keys)
    assert any("short_memory.proj_kf" in k for k in keys) and any("long_memory.proj_out" in k for k in keys)
    assert not any("lora_" in k for k in keys)                              # merged


def test_pipeline_with_every_layer_replaced(tiny_dir, tmp_path):
    """--replace all: every layer (the Gated DeltaNet ones included) becomes a Mosaics block, and the
    same stage scripts, evaluation and export run on it."""
    model, data = tiny_dir / "model", tiny_dir / "docs.jsonl"
    common = ["--model", model, "--data", data, "--seq-len", 48, "--bsz", 2, "--eval-rows", 2, "--log-every", 1,
              "--device", "cpu"]
    s1 = tmp_path / "s1"
    _run("mq_stage1.py", *common, *CONV, "--replace", "all", "--steps", 12, "--eval-every", 6, "--lr", "3e-3",
         "--warmup", 2, "--out", s1)
    meta = json.loads((s1 / "delta" / "mq_config.json").read_text())
    n_layers = json.loads((model / "config.json").read_text())
    n_layers = (n_layers.get("text_config") or n_layers)["num_hidden_layers"]
    assert meta["mq_config"]["replace"] == "all" and meta["mq_config"]["layers"] == list(range(n_layers))
    evals = [e for e in _events(s1) if e["event"] == "eval"]
    assert evals[-1]["rel_mse_mean"] < evals[0]["rel_mse_mean"]
    assert set(map(int, evals[-1]["rel_mse"])) == set(range(n_layers))      # every layer is matched to its original

    s2 = tmp_path / "s2"
    _run("mq_stage2.py", *common, "--delta-in", s1 / "delta", "--steps", 4, "--eval-every", 2, "--positions", 16,
         "--lora-r", 2, "--lr", "1e-3", "--grad-checkpoint", "--ce-weight", "0.5", "--out", s2)
    e2 = [e for e in _events(s2) if e["event"] == "eval"]
    assert e2 and all(torch.isfinite(torch.tensor(e["kl"])) for e in e2)

    ev_out = tmp_path / "eval"
    _run("mq_eval.py", "--model", model, "--delta", s2 / "delta", "--data", data, "--seq-len", 48, "--eval-rows", 2,
         "--recall-distances", "8", "--recall-n", 1, "--no-samples", "--device", "cpu", "--out", ev_out)
    assert json.loads((ev_out / "metrics.json").read_text())["kl"] >= 0
    ex = tmp_path / "export"
    rep = json.loads(_run("mq_export.py", "--model", model, "--delta", s2 / "delta", "--out", ex, "--save-dtype", "fp32",
                          "--device", "cpu").strip().splitlines()[-1])
    assert rep["attention_parameters_left"] == 0 and rep["attention_modules_after_reload"] == 0
    assert rep["reload_max_logit_diff"] < 1e-3
    from safetensors import safe_open
    keys = set()
    for f in ex.glob("model-*.safetensors"):
        with safe_open(str(f), "pt") as sf:
            keys |= set(sf.keys())
    assert not any("self_attn" in k or "linear_attn" in k for k in keys)    # neither attention nor Gated DeltaNet is left


def test_cluster_rows_hold_related_documents_and_keep_whole_clusters_out_of_training(tiny_dir):
    from transformers import AutoTokenizer
    from mmllm.mq_data import ClusterRowPacker, cluster_documents, context_row, load_source
    tok = AutoTokenizer.from_pretrained(tiny_dir / "model")
    src = load_source(str(tiny_dir / "topics.jsonl"))
    topic = [i % 6 for i in range(len(src.docs))]
    assign = cluster_documents(tok, src, cluster_size=60, seed=0)["assign"].tolist()
    assert assign == cluster_documents(tok, src, cluster_size=60, seed=0)["assign"].tolist()       # deterministic
    by = {}
    for c, t in zip(assign, topic):
        by.setdefault(c, []).append(t)
    purity = sum(max(ts.count(t) for t in set(ts)) for ts in by.values()) / len(topic)
    assert len(by) == 6 and purity > 0.9                                   # the clusters are the topics

    pk = ClusterRowPacker(tok, [src], seq_len=256, seed=0, cluster_size=20, holdout_frac=0.2)
    gr = pk.groups[src.name]
    assert gr["eval"] and gr["train"] and not set(gr["eval"]) & set(gr["train"])
    held_docs = {i for c in gr["eval"] for i in gr["members"][c]}
    assert 0.15 * len(src.docs) <= len(held_docs) <= 0.35 * len(src.docs)
    text_of = {tuple(pk._encode(src, d)): i for i, d in enumerate(src.docs)}

    def docs_of(row):                       # the complete documents of a row (the last one may be cut)
        out, cur = [], []
        for t in row:
            if t == pk.sep_id:
                out.append(text_of[tuple(cur)]); cur = []
            else:
                cur.append(t)
        return out

    rows = pk.batch(12)
    assert rows.shape == (12, 256)
    for row in rows.tolist():
        ds = docs_of(row)
        assert len(ds) >= 3 and not set(ds) & held_docs                    # training rows never read held-out clusters
        topics = [topic[i] for i in ds]
        assert max(topics.count(t) for t in set(topics)) / len(topics) > 0.8    # one topic per row
    for row in pk.eval_rows(4).tolist():
        assert set(docs_of(row)) <= held_docs
    assert torch.equal(pk.eval_rows(4), pk.eval_rows(4))
    d = pk.describe()
    assert d["rows"] == "cluster" and d["sources"][0]["held_out_clusters"] == len(gr["eval"])

    cases = pk.context_cases(3, max_tokens=256, target_tokens=(8, 24), context_doc_tokens=40, min_context_docs=2)
    assert len(cases) == 3
    for c in cases:
        assert 8 <= len(c["target"]) <= 24 and len(c["related"]) >= 2 and c["doc"] in held_docs
        assert all(d[-1] == pk.sep_id for d in c["related"] + c["unrelated"])
        row0, start0 = context_row(c, 0)
        row2, start2 = context_row(c, 2)
        full, start = context_row(c, len(c["related"]))
        assert (row0, start0) == (c["target"], 0) and len(full) <= 256
        assert row2 == c["related"][1] + c["related"][0] + c["target"] and row2[start2:] == c["target"]
        assert full[-len(row2):] == row2                                    # more context is added in front
        cl = gr["members"][c["cluster"]]
        rel = [text_of[tuple(d[:-1])] for d in c["related"] if tuple(d[:-1]) in text_of]
        assert set(rel) <= set(cl) and c["doc"] not in rel
        unrel = [text_of[tuple(d[:-1])] for d in c["unrelated"] if tuple(d[:-1]) in text_of]
        assert not set(unrel) & set(cl)


def test_pipeline_with_documents_and_a_cluster_memory(tiny_dir, tmp_path):
    """Plan section 10, step 2: rows of related documents, memories confined to the document, a cluster
    memory for the earlier documents, the next-token loss mixed in; then the context-growth evaluation."""
    model, data, fresh = tiny_dir / "model", tiny_dir / "topics.jsonl", tiny_dir / "topics_fresh.jsonl"
    common = ["--model", model, "--data", data, "--seq-len", 128, "--bsz", 1, "--eval-rows", 2, "--log-every", 1,
              "--device", "cpu", "--rows", "cluster", "--cluster-size", 30, "--holdout-frac", "0.2"]
    s1 = tmp_path / "s1"
    _run("mq_stage1.py", *common, *CONV, "--replace", "all", "--bias", "doc_causal", "--cluster-heads", 2,
         "--steps", 10, "--eval-every", 5, "--lr", "3e-3", "--warmup", 2, "--out", s1)
    meta = json.loads((s1 / "delta" / "mq_config.json").read_text())
    cfg = meta["mq_config"]
    assert (cfg["attn_bias_type"], cfg["n_cluster_heads"], cfg["replace"]) == ("doc_causal", 2, "all")
    from transformers import AutoTokenizer
    assert cfg["eos_token_id"] == AutoTokenizer.from_pretrained(model).convert_tokens_to_ids("<|endoftext|>")
    run = json.loads((s1 / "run.json").read_text())
    assert run["data"]["rows"] == "cluster" and run["data"]["sources"][0]["held_out_clusters"] >= 1
    evals = [e for e in _events(s1) if e["event"] == "eval"]
    assert evals[-1]["rel_mse_mean"] < evals[0]["rel_mse_mean"]

    s2 = tmp_path / "s2"
    _run("mq_stage2.py", *common, "--delta-in", s1 / "delta", "--steps", 4, "--eval-every", 2, "--positions", 32,
         "--lora-r", 2, "--lr", "1e-3", "--grad-checkpoint", "--ce-weight", "0.5", "--out", s2)
    e2 = [e for e in _events(s2) if e["event"] == "eval"]
    assert e2 and all(torch.isfinite(torch.tensor(e["kl"])) for e in e2)
    assert json.loads((s2 / "delta" / "mq_config.json").read_text())["mq_config"]["n_cluster_heads"] == 2

    # a second student without documents or cluster memory (step 1's kind), for the comparison
    plain = tmp_path / "plain"
    _run("mq_stage1.py", "--model", model, "--data", data, "--seq-len", 128, "--bsz", 1, "--eval-rows", 2, "--device", "cpu",
         *CONV, "--replace", "all", "--steps", 4, "--eval-every", 100, "--out", plain)

    ctx = tmp_path / "ctx"
    _run("mq_context.py", "--model", model, "--data", fresh, "--docs", 240, "--student", f"docs={s2 / 'delta'}",
         "--student", f"all={plain / 'delta'}", "--seq-len", 256, "--n-per-source", 6, "--ks", "0,1,2", "--min-context-docs", 2,
         "--target-min", 8, "--target-max", 24, "--context-doc-tokens", 40, "--cluster-size", 40, "--pad-to", 64,
         "--device", "cpu", "--out", ctx)
    res = json.loads((ctx / "context.json").read_text())
    assert res["n_cases"] == 6 and res["ks"] == ["0", "1", "2", "max", "unrelated"]
    assert set(res["models"]) == {"teacher", "docs", "docs/no-cluster", "all"}
    for m in res["models"].values():
        assert set(m["loss"]) == set(res["ks"]) and m["G"]["lo"] <= m["G"]["mean"] <= m["G"]["hi"]
    assert "docs - all" in res["pairs"] and "G_over_teacher" in res["models"]["docs"]
    c = res["cases"][0]["loss"]
    # without its cluster memory the student cannot see the documents before the target: the context changes
    # nothing but the leaky-average key of the first tokens; with it the context changes the loss
    assert abs(c["docs/no-cluster"]["max"] - c["docs/no-cluster"]["unrelated"]) < abs(c["docs"]["max"] - c["docs"]["0"]) + 1e-3
    assert c["docs"]["max"] != c["docs"]["0"]

    # the usual evaluation and the export run on this model as on the others
    ev_out = tmp_path / "eval"
    _run("mq_eval.py", "--model", model, "--delta", s2 / "delta", "--data", tiny_dir / "docs.jsonl", "--seq-len", 48,
         "--eval-rows", 2, "--recall-distances", "8", "--recall-n", 1, "--no-samples", "--device", "cpu", "--out", ev_out)
    assert json.loads((ev_out / "metrics.json").read_text())["kl"] >= 0
    ex = tmp_path / "export"
    rep = json.loads(_run("mq_export.py", "--model", model, "--delta", s2 / "delta", "--out", ex, "--save-dtype", "fp32",
                          "--device", "cpu").strip().splitlines()[-1])
    assert rep["attention_parameters_left"] == 0 and rep["attention_modules_after_reload"] == 0
    assert rep["reload_max_logit_diff"] < 1e-3


def test_vocab_loss_backward_is_the_same_gradient():
    """The loss over the vocabulary computed a chunk of positions at a time (27B: the card has no
    room for positions x vocabulary tensors) gives the gradient of the loss computed at once."""
    import torch.nn.functional as F
    from mmllm.mq_train import vocab_loss_backward
    torch.manual_seed(0)
    n, d, v = 37, 12, 50
    x, ht = torch.randn(n, d), torch.randn(n, d)
    target = torch.randint(0, v, (n,))
    for ce_w, with_extra, chunk, kl_w in ((0.0, False, 8, 1.0), (0.3, False, 5, 1.0), (0.0, True, 37, 1.0),
                                          (0.2, True, 100, 1.0), (1.0, False, 8, 0.0), (0.5, True, 16, 0.25)):
        body = torch.nn.Linear(d, d)                       # stands for the model below the head
        head_s, head_t = torch.nn.Linear(d, v, bias=False), torch.nn.Linear(d, v, bias=False)
        scale = 0.25

        def extra_of(h):
            return h.pow(2).mean() * 0.1 if with_extra else None

        # at once
        h = body(x)
        t_logp = F.log_softmax(head_t(ht), -1).detach()
        s_logp = F.log_softmax(head_s(h), -1)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
        ce = F.nll_loss(s_logp, target)
        loss = (kl_w * kl + ce_w * ce) * scale
        if with_extra:
            loss = loss + extra_of(h)
        loss.backward()
        ref = [p.grad.clone() for p in (*body.parameters(), *head_s.parameters())]
        for p in (*body.parameters(), *head_s.parameters()):
            p.grad = None
        # in chunks
        h = body(x)
        kl2, ce2 = vocab_loss_backward(head_s, head_t, h, ht, target, ce_weight=ce_w, kl_weight=kl_w, scale=scale,
                                       chunk=chunk, extra=extra_of(h))
        got = [p.grad for p in (*body.parameters(), *head_s.parameters())]
        assert abs(kl2 - kl.item()) < 1e-5 and abs(ce2 - ce.item()) < 1e-5
        for a, b in zip(ref, got):
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (ce_w, with_extra, chunk, (a - b).abs().max())
        assert all(p.grad is None for p in head_t.parameters())
    with pytest.raises(ValueError):
        vocab_loss_backward(head_s, head_t, body(x), ht, target, ce_weight=0.0, kl_weight=0.0)


def test_stage2_on_the_text_alone_with_every_weight_trained(tiny_dir, tmp_path):
    """--kl-weight 0 --ce-weight 1 --train all: no distillation; the converted model is trained on the
    next token with all its weights, and the KL to the original is still measured."""
    model, data = tiny_dir / "model", tiny_dir / "docs.jsonl"
    common = ["--model", model, "--data", data, "--seq-len", 48, "--bsz", 2, "--eval-rows", 2, "--log-every", 1,
              "--device", "cpu"]
    s1 = tmp_path / "s1"
    _run("mq_stage1.py", *common, *CONV, "--replace", "all", "--steps", 4, "--eval-every", 4, "--out", s1)
    s2 = tmp_path / "s2"
    _run("mq_stage2.py", *common, "--delta-in", s1 / "delta", "--steps", 6, "--eval-every", 3, "--positions", 16,
         "--train", "all", "--kl-weight", "0", "--ce-weight", "1", "--lr", "1e-3", "--out", s2)
    ev = _events(s2)
    steps = [e for e in ev if e["event"] == "step"]
    assert steps and all(e["kl"] == e["kl"] and e["ce"] > 0 for e in steps)
    assert steps[-1]["ce"] < steps[0]["ce"]
    run = json.loads((s2 / "run.json").read_text())
    assert run["teacher"] == "separate copy"
    # Evaluation: the delta carries base weights, so the teacher must be a separate copy of the base model.
    # Its loss is then the same as in the evaluation of a delta that left the base alone.
    assert json.loads((s2 / "delta" / "mq_config.json").read_text())["base_tensors"] > 0
    assert json.loads((s1 / "delta" / "mq_config.json").read_text())["base_tensors"] == 0
    ev = ["--model", model, "--data", data, "--seq-len", 48, "--eval-rows", 2, "--recall-distances", "8", "--recall-n", 1,
          "--no-samples", "--device", "cpu"]
    _run("mq_eval.py", *ev, "--delta", s1 / "delta", "--out", tmp_path / "e1")
    _run("mq_eval.py", *ev, "--delta", s2 / "delta", "--out", tmp_path / "e2")
    m1 = json.loads((tmp_path / "e1" / "metrics.json").read_text())
    m2 = json.loads((tmp_path / "e2" / "metrics.json").read_text())
    assert (m1["teacher"], m2["teacher"]) == ("shared weights", "separate copy")
    assert abs(m1["nll_teacher"] - m2["nll_teacher"]) < 1e-4
    assert abs(m1["nll_student"] - m2["nll_student"]) > 1e-3        # the student did change


def test_chat_documents_without_a_chat_template_become_plain_text(tiny_dir):
    """A base model's tokenizer may have no chat template: chat data is then written out as plain turns."""
    from transformers import AutoTokenizer
    from mmllm.mq_data import RowPacker, Source
    tok = AutoTokenizer.from_pretrained(tiny_dir / "model")
    assert not getattr(tok, "chat_template", None)
    turns = [{"role": "user", "content": "memory key"}, {"role": "assistant", "content": "value token"}]
    src = Source(name="chat", docs=[turns] * 40, weight=1.0, is_chat=True)
    rows = RowPacker(tok, [src], seq_len=32).batch(2)
    assert rows.shape == (2, 32)
    text = tok.decode(rows[0])
    assert "memory" in text and "value" in text


def test_every_script_a_plan_names_accepts_what_the_runner_adds():
    """research/colab/runner.py appends --out and --max-minutes to every run (a plan of 2026-11-13 named a
    script without --max-minutes and its run stopped at the argument parser)."""
    scripts = set()
    for f in (ROOT / "research" / "colab" / "plans").glob("mq-*.json"):
        for run in json.loads(f.read_text())["runs"]:
            scripts.add(run["script"])
    assert scripts
    for s in sorted(scripts):
        src = (ROOT / s).read_text()
        assert '"--out"' in src and '"--max-minutes"' in src, s


@pytest.mark.parametrize("which", ["qwen35", "qwen3"])
def test_stage1_control_trains_a_fresh_mixer_beside_the_memories(tiny_dir, tmp_path, which):
    """scripts/mq_stage1_control.py: for every layer, the memories and a newly initialised mixer of the
    original's type are trained on the same batches towards the same target; both errors are reported
    per layer and by the type of the original layer. Also on a model with attention in every layer."""
    model = tiny_dir / "model"
    if which == "qwen3":
        from conftest import tiny_attention_only
        from transformers import AutoTokenizer
        model = tmp_path / "qwen3"
        tok = AutoTokenizer.from_pretrained(tiny_dir / "model")
        m = tiny_attention_only("qwen3")
        m.config.eos_token_id = tok.eos_token_id
        m.save_pretrained(model)
        tok.save_pretrained(model)
    out = tmp_path / "fit"
    _run("mq_stage1_control.py", "--model", model, "--data", tiny_dir / "docs.jsonl", *CONV, "--replace", "all",
         "--seq-len", 48, "--bsz", 2, "--tokens", 48 * 2 * 30, "--eval-every", 10, "--eval-rows", 2, "--log-every", 5,
         "--lr", "3e-3", "--control-lr", "3e-3,1e-5", "--warmup", 2, "--device", "cpu", "--out", out)
    res = json.loads((out / "result.json").read_text())
    n_layers = json.loads((model / "config.json").read_text())["num_hidden_layers"]
    assert res["status"] == "done" and res["step"] == 30
    # a control per learning rate, from the same new weights; "the control" is the one that ends lowest
    assert set(res["controls"]) == {"0.003", "1e-05"} and res["control_lr"] == "0.003"
    assert res["control"] == res["controls"]["0.003"]
    assert res["controls"]["0.003"]["mean"] < res["controls"]["1e-05"]["mean"]
    for k in ("mm", "control"):
        assert set(map(int, res[k]["layers"])) == set(range(n_layers))
        assert all(v == v and v >= 0 for v in res[k]["layers"].values())
    kinds = set(res["kinds"].values())
    assert kinds == ({"linear_attention", "full_attention"} if which == "qwen35" else {"full_attention"})
    assert all(k in res["mm"] and k in res["control"] for k in kinds)
    evals = [e for e in _events(out) if e["event"] == "eval"]
    assert evals[0]["step"] == 0 and evals[-1]["mm"] < evals[0]["mm"] and evals[-1]["control"] < evals[0]["control"]
    # the control starts from new weights: before training it does not reproduce the original mixer
    assert evals[0]["control"] > 0.5
    run = json.loads((out / "run.json").read_text())
    assert all(v > 0 for v in run["control_parameters_per_layer"].values())


def test_memories_started_from_the_original_mixers_weights(tiny_dir, tmp_path):
    """--init teacher-ov / teacher-ovk: values and output (and keys) come from the original mixer; every
    layer of both types is filled, the run trains as before, and before training the error is not worse
    than with upstream's random start."""
    model, data = tiny_dir / "model", tiny_dir / "docs.jsonl"
    common = ["--model", model, "--data", data, "--seq-len", 48, "--bsz", 2, "--eval-rows", 2, "--log-every", 1,
              "--device", "cpu", *CONV, "--replace", "all", "--steps", 6, "--eval-every", 3, "--no-optim-save"]
    first = {}
    for init in ("upstream", "teacher-ov", "teacher-ovk"):
        out = tmp_path / init
        _run("mq_stage1.py", *common, "--init", init, "--out", out)
        ev = [e for e in _events(out) if e["event"] == "eval"]
        first[init] = ev[0]["rel_mse_mean"]
        assert all(e["rel_mse_mean"] == e["rel_mse_mean"] for e in ev)
        run = json.loads((out / "run.json").read_text())
        assert run["init"] == init
        if init != "upstream":
            rep = run["init_report"]
            assert len(rep["layers"]) == 8 and all(m["filled"] > 0 for l in rep["layers"].values() for m in l["memories"].values())
            assert len(rep["value_scale"]) == 8
    # with the fitted gain the start is never worse than silence (error 1), whatever the values' size
    out = tmp_path / "gain"
    _run("mq_stage1.py", *common, "--init", "teacher-ov", "--init-gain", "--out", out)
    ev = [e for e in _events(out) if e["event"] == "eval"]
    assert ev[0]["rel_mse_mean"] <= 1.0 + 1e-3
    assert len(json.loads((out / "run.json").read_text())["init_report"]["gain"]) == 8


def test_low_rank_factors_reproduce_the_heads_output_map():
    from mmllm.teacher_init import _key_dirs, _low_rank
    torch.manual_seed(0)
    o, v = torch.randn(40, 8), torch.randn(8, 40)
    a, b = _low_rank(o, v, 8)
    assert torch.allclose(a @ b, o @ v, atol=1e-4)
    a, b = _low_rank(o, v, 3)
    u, s, wh = torch.linalg.svd(o @ v)
    assert torch.allclose(a @ b, (u[:, :3] * s[:3]) @ wh[:3], atol=1e-4)
    q, k = torch.randn(8, 40), torch.randn(8, 40)
    kd = _key_dirs(q, k, 16)
    sym = (q.T @ k + k.T @ q) / 2
    want = torch.linalg.eigh(sym)
    best = (want[1][:, want[0] > 1e-6] * want[0][want[0] > 1e-6]) @ want[1][:, want[0] > 1e-6].T
    assert torch.allclose(kd.T @ kd, best, atol=1e-3)


def test_progressive_replacement_ends_with_every_layer_replaced(tiny_dir, tmp_path):
    """--progressive: early steps run some layers as the original, the share falls to zero, and what is
    evaluated and saved is the fully converted model (no original layer in it)."""
    model, data = tiny_dir / "model", tiny_dir / "docs.jsonl"
    common = ["--model", model, "--data", data, "--seq-len", 48, "--bsz", 2, "--eval-rows", 2, "--log-every", 1,
              "--device", "cpu"]
    s1 = tmp_path / "s1"
    _run("mq_stage1.py", *common, *CONV, "--replace", "all", "--steps", 4, "--eval-every", 4, "--out", s1)
    s2 = tmp_path / "s2"
    _run("mq_stage2.py", *common, "--delta-in", s1 / "delta", "--steps", 12, "--eval-every", 6, "--positions", 16,
         "--lora-r", 2, "--grad-checkpoint", "--progressive", "0.5", "--out", s2)
    steps = [e for e in _events(s2) if e["event"] == "step"]
    kept = [e["orig_layers"] for e in steps]
    assert kept[0] > 4 and all(k == 0 for k in kept[6:]) and all(e["kl"] == e["kl"] for e in steps)
    meta = json.loads((s2 / "delta" / "mq_config.json").read_text())
    assert meta["attention_parameters_left"] == 0 and meta["base_tensors"] == 0
    ex = json.loads(_run("mq_export.py", "--model", model, "--delta", s2 / "delta", "--out", tmp_path / "ex",
                         "--save-dtype", "fp32", "--device", "cpu").strip().splitlines()[-1])
    assert ex["attention_modules_after_reload"] == 0 and ex["reload_max_logit_diff"] < 1e-3


def test_keep_original_runs_those_layers_as_the_original():
    from conftest import tiny_qwen35
    from mmllm.mosaic_qwen import MQConfig, convert_to_mosaic_qwen
    m = tiny_qwen35()
    mq = convert_to_mosaic_qwen(m, MQConfig(replace="all", n_short_heads=2, n_long_heads=2, head_dim=16, window=8, skip=4,
                                            random_skip=4))
    ids = torch.randint(8, 300, (1, 20))
    with torch.no_grad():
        with mq.teacher():
            t = m(input_ids=ids, use_cache=False).logits
        s = m(input_ids=ids, use_cache=False).logits
        mq.keep_original(range(8))
        allo = m(input_ids=ids, use_cache=False).logits
        mq.keep_original([0, 1, 2])
        part = m(input_ids=ids, use_cache=False).logits
        mq.keep_original(())
        back = m(input_ids=ids, use_cache=False).logits
    assert torch.allclose(allo, t, atol=1e-5) and torch.equal(back, s)
    assert not torch.allclose(part, t, atol=1e-3) and not torch.allclose(part, s, atol=1e-3)


def test_a_stopped_stage2_run_saves_before_it_ends(tiny_dir, tmp_path):
    """The runner stops a run with SIGTERM (/content/STOP). Stage 2 then ends at the next step and saves
    its weights, so a later run resumes from there instead of from the last periodic save."""
    import signal
    import subprocess
    import time
    model, data = tiny_dir / "model", tiny_dir / "docs.jsonl"
    common = ["--model", model, "--data", data, "--seq-len", 48, "--bsz", 2, "--eval-rows", 2, "--log-every", 1,
              "--device", "cpu"]
    s1 = tmp_path / "s1"
    _run("mq_stage1.py", *common, *CONV, "--replace", "all", "--steps", 2, "--eval-every", 2, "--out", s1)
    s2 = tmp_path / "s2"
    cmd = [sys.executable, str(ROOT / "scripts" / "mq_stage2.py"), *map(str, common), "--delta-in", str(s1 / "delta"),
           "--steps", "100000", "--eval-every", "100000", "--positions", "16", "--lora-r", "2", "--out", str(s2)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 240
    while time.time() < deadline:
        f = s2 / "progress.jsonl"
        if f.exists() and sum(1 for l in f.read_text().splitlines() if '"event": "step"' in l) >= 3:
            break
        time.sleep(1)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=240)
    assert proc.returncode == 0, out[-2000:]
    res = json.loads((s2 / "result.json").read_text())
    assert res["status"] == "budget" and 1 <= res["step"] < 100000
    meta = json.loads((s2 / "delta" / "mq_config.json").read_text())
    assert meta["extra"]["step"] == res["step"]
