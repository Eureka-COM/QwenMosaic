"""Training the upstream models from scratch (scripts/scratch_prep.py, scratch_train.py, scratch_pick_lr.py)."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORDS = ["memory", "mosaic", "key", "value", "the", "a", "of", "reads", "writes", "token", "layer", "model", "long",
         "short", "window", "past", "next", "state", "kernel", "norm", "gate", "head", "is", "and", "code", "access",
         "for", "The", "Arlen", "Bexley", "Corwin", "Dalia"] + list("0123456789.")


def _run(script, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def _events(out):
    return [json.loads(l) for l in (Path(out) / "progress.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    d = tmp_path_factory.mktemp("scratch")
    rng = random.Random(0)
    docs = lambda n, lo, hi: [" ".join(rng.choice(WORDS) for _ in range(rng.randrange(lo, hi))) for _ in range(n)]  # noqa: E731
    train_a, train_b, ev = docs(150, 20, 120), docs(150, 20, 120), docs(40, 300, 400)
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(train_a + ev, trainers.BpeTrainer(vocab_size=200, special_tokens=["<unk>", "<s>", "</s>"]))
    assert tk.token_to_id("<s>") == 1 and tk.token_to_id("</s>") == 2
    tk.save(str(d / "tok.json"))
    for name, texts in (("a", train_a), ("b", train_b), ("ev", ev)):
        pq.write_table(pa.table({"text": texts}), d / f"{name}.parquet")
    out = d / "data"
    _run("scratch_prep.py", "--tokenizer", d / "tok.json", "--train", d / "a.parquet", "--train", d / "b.parquet",
         "--eval", d / "ev.parquet", "--train-tokens", 15000, "--heldout-tokens", 3000, "--long-docs", 4, "--long-len", 256,
         "--filler-tokens", 3000, "--out", d / "run", "--data-out", out, "--out", d / "run2")   # the runner adds --out last
    assert (d / "run2" / "progress.jsonl").exists() and (d / "run2" / "meta.json").exists()
    return out


def test_prep_writes_the_stream_and_the_held_out_material(data_dir):
    import numpy as np
    meta = json.loads((data_dir / "meta.json").read_text())
    train = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
    assert meta["train_tokens"] == len(train) and 15000 <= len(train) < 15000 + 400
    assert train[0] == 1 and (train == 2).sum() == meta["train_docs"]             # every document: BOS ... EOS
    longd = np.load(data_dir / "long_docs.npy")
    assert longd.shape == (4, 257) and (longd[:, 0] == 1).all() and not (longd[:, 1:] == 2).any()
    filler = np.load(data_dir / "filler.npy")
    assert not ((filler == 1) | (filler == 2)).any()
    assert (data_dir / "tokenizer.json").exists()


def test_rows_come_in_the_same_order_for_every_model(data_dir):
    from mmllm.scratch import TokenFile
    a, b = TokenFile(data_dir / "train.bin", 64), TokenFile(data_dir / "train.bin", 64)
    assert torch.equal(a.batch(3, 4, seed=0), b.batch(3, 4, seed=0))
    assert not torch.equal(a.batch(3, 4, seed=0), a.batch(3, 4, seed=1))
    seen = torch.cat([a.batch(s, 4, 0)[:, 0] for s in range(a.n_rows // 4)])
    assert seen.numel() == (a.n_rows // 4) * 4


def test_chunked_loss_matches_the_plain_loss():
    from mmllm.scratch import chunked_ce
    torch.manual_seed(0)
    h = torch.randn(2, 37, 16, requires_grad=True)
    w = torch.randn(50, 16, requires_grad=True)
    y = torch.randint(0, 50, (2, 37))
    a = chunked_ce(h, w, y, chunk=10)
    b = torch.nn.functional.cross_entropy((h @ w.t()).reshape(-1, 50), y.reshape(-1))
    assert torch.allclose(a, b, atol=1e-5)
    ga = torch.autograd.grad(a, (h, w)); gb = torch.autograd.grad(b, (h, w))
    assert all(torch.allclose(x, z, atol=1e-5) for x, z in zip(ga, gb))
    per = chunked_ce(h, w, y, chunk=7, reduction="none")
    assert per.shape == (2, 37)


TINY = ["--dim", 32, "--heads", 2, "--head-dim", 16, "--layers", 2, "--seq-len", 256, "--bsz", 2, "--accum", 2,
        "--eval-rows", 2, "--pos-docs", 2, "--pos-len", 200, "--icl-early", "10,20", "--icl-late", "100,200",
        "--recall-distances", "16,40", "--recall-n", 2, "--log-every", 2, "--device", "cpu", "--lr", "2e-3"]


@pytest.mark.parametrize("family", ["mosaics_v2", "transformer"])
def test_training_from_scratch_checks_itself_on_the_way(data_dir, tmp_path, family):
    out = tmp_path / family
    steps = 8
    _run("scratch_train.py", "--family", family, "--data", data_dir, *TINY, "--tokens", 256 * 4 * steps,
         "--eval-at", "0,0.5,1", "--out", out)
    res = json.loads((out / "result.json").read_text())
    assert res["status"] == "done" and res["step"] == steps
    evals = [json.loads(l) for l in (out / "evals.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [0, 4, 8]
    e = evals[-1]
    assert set(e["position"]) == {"1-16", "17-64", "65-256"} and len(e["icl_per_doc"]) == 2
    assert set(e["recall"]) == {"16", "40"} and all(r["n"] == 2 for r in e["recall"].values())
    assert evals[-1]["heldout_loss"] < evals[0]["heldout_loss"]
    run = json.loads((out / "run.json").read_text())
    assert run["settings"]["attn_bias_type"] == "doc_causal"
    if family == "mosaics_v2":
        s = run["settings"]
        assert (s["n_short_heads"], s["n_long_heads"], s["short_mem_window_size"], s["long_mem_skip_size"],
                s["random_long_mem_skip_size"], s["layout"]) == (2, 2, 256, 64, 192, "flat")
    else:
        assert run["settings"]["n_heads"] == 2 and run["settings"]["position_encoding"] == "rope"


def test_a_stopped_run_resumes_from_its_store(data_dir, tmp_path):
    store = tmp_path / "store"
    args = ["--family", "mosaics_v2", "--data", data_dir, *TINY, "--tokens", 256 * 4 * 8, "--eval-at", "0,0.5,1",
            "--store", f"dir:{store}"]
    _run("scratch_train.py", *args, "--stop-after", 256 * 4 * 5, "--out", tmp_path / "a")
    r = json.loads((tmp_path / "a" / "result.json").read_text())
    assert r["status"] == "stop_after" and r["step"] == 5 and (store / "model.pt").exists()
    _run("scratch_train.py", *args, "--out", tmp_path / "b")
    ev = _events(tmp_path / "b")
    assert ev[0]["event"] == "start" and ev[0]["resumed"] and ev[0]["step"] == 5
    evals = [json.loads(l) for l in (tmp_path / "b" / "evals.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [8]                     # the evaluations at 0 and 4 are not repeated
    assert json.loads((tmp_path / "b" / "result.json").read_text())["status"] == "done"


def test_a_failed_upload_to_the_store_does_not_end_the_run(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MQ_STORE_RETRY_SEC", "0")
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("x")                                      # the store cannot make its folder here
    out = tmp_path / "run"
    _run("scratch_train.py", "--family", "transformer", "--data", data_dir, *TINY, "--tokens", 256 * 4 * 4,
         "--eval-at", "1", "--store", f"dir:{blocked}", "--out", out)
    assert json.loads((out / "result.json").read_text())["status"] == "done"
    ev = _events(out)
    failed = [e for e in ev if e["event"] == "store_failed"]
    assert len(failed) >= 3 and failed[0]["attempt"] == 1
    saved = [e for e in ev if e["event"] == "saved"]
    assert saved and not saved[-1]["stored"] and (out / "ckpt" / "model.pt").exists()


def test_two_runs_take_turns_on_the_gpu_and_both_finish(data_dir, tmp_path):
    lock = tmp_path / "gpu.lock"
    procs = []
    for fam in ("mosaics_v2", "transformer"):
        cmd = [sys.executable, str(ROOT / "scripts" / "scratch_train.py"), "--family", fam, "--data", str(data_dir),
               *map(str, TINY), "--tokens", str(256 * 4 * 6), "--eval-at", "0,1", "--duty", "0.5", "--log-every", "1",
               "--gpu-lock", str(lock), "--out", str(tmp_path / fam)]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    for p in procs:
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, err[-2000:]
    for fam in ("mosaics_v2", "transformer"):
        assert json.loads((tmp_path / fam / "result.json").read_text())["status"] == "done"
        steps = [e for e in _events(tmp_path / fam) if e["event"] == "step"]
        assert len(steps) == 6 and all("turn_wait_share" in e for e in steps)


def test_the_learning_rate_with_the_lowest_held_out_loss_is_picked(tmp_path):
    for lr, loss in (("0.0003", 5.1), ("0.0006", 4.8), ("0.0012", float("nan"))):
        d = tmp_path / lr
        d.mkdir()
        (d / "run.json").write_text(json.dumps({"args": {"lr": float(lr)}}))
        (d / "evals.jsonl").write_text(json.dumps({"tokens": 100, "heldout_loss": loss}) + "\n")
    _run("scratch_pick_lr.py", *sum((["--run", tmp_path / lr] for lr in ("0.0003", "0.0006", "0.0012")), []),
         "--write", tmp_path / "shared", "--out", tmp_path / "pick")
    assert json.loads((tmp_path / "pick" / "lr.json").read_text())["lr"] == 0.0006
    assert json.loads((tmp_path / "shared" / "lr.json").read_text())["lr"] == 0.0006


def test_recall_cases_cue_is_a_prefix_of_the_fact(data_dir):
    from tokenizers import Tokenizer
    from mmllm.scratch import recall_cases
    tk = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    enc = lambda s: tk.encode(s, add_special_tokens=False).ids                     # noqa: E731
    cs = recall_cases(enc, list(range(3, 150)) * 10, [16, 40], 3)
    for c in cs:
        k = c["answer_len"]
        ids = c["ids"]
        assert ids[0] == 1 and k >= 5
        tail = ids[-k:]
        # the answer also appears right after the same cue earlier in the document
        s = " ".join(map(str, ids[:-k]))
        assert " ".join(map(str, tail)) in s


def test_compare_applies_the_rules_of_the_plan():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmp", ROOT / "scripts" / "scratch_compare.py")
    cmp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmp)

    def ev(step, loss, early, late):
        return {"step": step, "tokens": step * 10, "heldout_loss": loss, "icl_score": 0, "position": {}, "recall": {},
                "early_per_doc": early, "late_per_doc": late}
    n = 40
    mm = {s: ev(s, l, [5.0] * n, [4.0 + 0.01 * (i % 3) for i in range(n)]) for s, l in ((25, 4.6), (50, 4.5), (100, 4.0))}
    tf = {s: ev(s, l, [5.0] * n, [4.5 + 0.01 * (i % 3) for i in range(n)]) for s, l in ((25, 4.2), (50, 4.0), (100, 3.95))}
    r = cmp.compare(mm, tf, 100)
    assert r["stop_rule"]["gap_25"] == 0.4 and r["stop_rule"]["stop"] is True        # 0.40 > 0.30 and 0.50 > 0.40
    assert r["verdict"]["verdict"] == "文脈の奥で上回った"                            # late gap -0.5 on every document
    tf2 = {s: ev(s, l, [5.0] * n, [4.0 + 0.01 * (i % 3) for i in range(n)]) for s, l in ((25, 4.5), (50, 4.4), (100, 4.0))}
    r = cmp.compare(mm, tf2, 100)
    assert r["stop_rule"]["stop"] is False and r["verdict"]["verdict"] == "差なし"
