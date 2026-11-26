"""The Colab machinery, run locally: cell -> unpack -> run -> save to a git remote."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab"))


def test_saver_token_mode_pushes_everything(tmp_path, monkeypatch):
    import saver
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token-value")
    monkeypatch.setattr(saver, "CODE_REPO", str(bare))
    s = saver.Saver("token", "unit-test", base=str(tmp_path))
    out = tmp_path / "out"; (out / "run").mkdir(parents=True)
    (out / "run" / "progress.jsonl").write_text('{"step": 1}\n')
    (out / "run" / "memory.pt").write_bytes(b"x" * 10)
    assert s.save(out)
    (out / "run" / "progress.jsonl").write_text('{"step": 1}\n{"step": 2}\n')
    assert s.save(out, quiet=True)                       # repeated (partial) saves
    files = subprocess.run(["git", "--git-dir", str(bare), "ls-tree", "-r", "--name-only", s.branch],
                           capture_output=True, text=True).stdout.split()
    assert s.branch.startswith("results/") and s.branch.endswith("_unit-test")
    assert f"{s.rel}/run/progress.jsonl" in files
    assert any(f.endswith("run/memory.pt") for f in files)      # weights are saved too
    log = subprocess.run(["git", "--git-dir", str(bare), "log", "--oneline", s.branch],
                         capture_output=True, text=True).stdout
    assert len(log.strip().splitlines()) == 2


def test_token_never_in_error_text(tmp_path, monkeypatch):
    import saver
    monkeypatch.setenv("GITHUB_TOKEN", "secret-abc-123")
    monkeypatch.setattr(saver, "CODE_REPO", "https://example.invalid/none.git")
    s = saver.Saver("token", "t", base=str(tmp_path))
    ok, err = s._push(attempts=1)
    assert not ok and "secret-abc-123" not in err


def test_make_cell_packs_the_code(tmp_path):
    plan = {"name": "x", "runs": []}
    p = tmp_path / "plan.json"; p.write_text(json.dumps(plan))
    cell = subprocess.run([sys.executable, str(ROOT / "colab/make_cell.py"), str(p)],
                          capture_output=True, text=True, check=True).stdout
    assert "CODE_B64" in cell and "runner.run_all(PLAN)" in cell
    compile(cell, "cell", "exec")


def test_runner_parallel_queue(tmp_path, monkeypatch):
    """max_parallel runs overlap; a "wait" run runs alone first; statuses and saves recorded."""
    import runner
    code = tmp_path / "code"; (code / "scripts").mkdir(parents=True)
    (code / "scripts" / "job.py").write_text(
        "import sys, time, json, pathlib\n"
        "a = sys.argv; out = pathlib.Path(a[a.index('--out') + 1]); d = float(a[a.index('--sec') + 1])\n"
        "t0 = time.time(); time.sleep(d)\n"
        "(out / 'progress.jsonl').write_text(json.dumps({'start': t0, 'end': time.time()}) + '\\n')\n"
        "sys.exit(int(a[a.index('--rc') + 1]))\n")
    monkeypatch.setattr(runner, "CODE", code)
    monkeypatch.setattr(runner, "OUT", tmp_path / "out")
    monkeypatch.setattr(runner, "STOP", tmp_path / "STOP")
    (tmp_path / "out").mkdir()
    saves = []
    class FakeSaver:
        def save(self, src, message=None, quiet=False):
            saves.append(message)
    plan = {"name": "p", "max_parallel": 2, "stagger_sec": 0, "poll_sec": 0.2, "runs": [
        {"name": "base", "script": "scripts/job.py", "args": ["--sec", "0.5", "--rc", "0"], "wait": True},
        {"name": "a", "script": "scripts/job.py", "args": ["--sec", "1.5", "--rc", "0"]},
        {"name": "b", "script": "scripts/job.py", "args": ["--sec", "1.5", "--rc", "3"]},
        {"name": "c", "script": "scripts/job.py", "args": ["--sec", "0.3", "--rc", "0"]}]}
    status = {"runs": {}}
    runner._run_queue(plan, FakeSaver(), status, lambda: None, __import__("time").time(), budget=10)
    t = {n: json.loads((tmp_path / "out" / n / "progress.jsonl").read_text()) for n in ("base", "a", "b", "c")}
    assert t["base"]["end"] <= min(t["a"]["start"], t["b"]["start"])      # base alone, first
    assert t["a"]["start"] < t["b"]["end"] and t["b"]["start"] < t["a"]["end"]  # a and b overlap
    assert t["c"]["start"] >= min(t["a"]["end"], t["b"]["end"]) - 0.05     # c waits for a free slot
    assert status["runs"]["b"].startswith("failed (exit 3)") and status["runs"]["a"].startswith("ok")
    assert len(saves) == 4


def test_saver_git_mode(tmp_path, monkeypatch):
    import saver
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    monkeypatch.setattr(saver, "CODE_REPO", str(bare))
    s = saver.Saver("git", "local-x", base=str(tmp_path))
    out = tmp_path / "o"; out.mkdir(); (out / "a.txt").write_text("1")
    assert s.save(out)
    files = subprocess.run(["git", "--git-dir", str(bare), "ls-tree", "-r", "--name-only", s.branch],
                           capture_output=True, text=True).stdout.split()
    assert f"{s.rel}/a.txt" in files


def test_runner_after_dependency(tmp_path, monkeypatch):
    """A run with "after" waits for the named runs; a later independent run may start first."""
    import runner
    code = tmp_path / "code"; (code / "scripts").mkdir(parents=True)
    (code / "scripts" / "job.py").write_text(
        "import sys, time, json, pathlib\n"
        "a = sys.argv; out = pathlib.Path(a[a.index('--out') + 1]); d = float(a[a.index('--sec') + 1])\n"
        "t0 = time.time(); time.sleep(d)\n"
        "(out / 'progress.jsonl').write_text(json.dumps({'start': t0, 'end': time.time(), 'argv': a}) + '\\n')\n")
    monkeypatch.setattr(runner, "CODE", code)
    monkeypatch.setattr(runner, "OUT", tmp_path / "out")
    monkeypatch.setattr(runner, "STOP", tmp_path / "STOP")
    (tmp_path / "out").mkdir()
    class FakeSaver:
        def save(self, src, message=None, quiet=False):
            pass
    plan = {"name": "p", "max_parallel": 3, "stagger_sec": 0, "poll_sec": 0.1, "common_args": ["--common", "1"], "runs": [
        {"name": "train", "script": "scripts/job.py", "args": ["--sec", "1.0"]},
        {"name": "eval", "script": "scripts/job.py", "args": ["--sec", "0.2"], "after": ["train"], "no_common": True},
        {"name": "other", "script": "scripts/job.py", "args": ["--sec", "0.2"]}]}
    runner._run_queue(plan, FakeSaver(), {"runs": {}}, lambda: None, __import__("time").time(), budget=10)
    t = {n: json.loads((tmp_path / "out" / n / "progress.jsonl").read_text()) for n in ("train", "eval", "other")}
    assert t["eval"]["start"] >= t["train"]["end"] - 0.05
    assert t["other"]["start"] < t["train"]["end"]
    assert "--common" not in t["eval"]["argv"] and "--common" in t["train"]["argv"]


def test_fetch_results(tmp_path, monkeypatch):
    import runner
    bare = tmp_path / "remote.git"; subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    w = tmp_path / "w"; w.mkdir()
    subprocess.run("git init -q && git checkout -q -b results/x && mkdir -p results/x/run && echo 1 > results/x/run/memory.pt "
                   "&& git add -A && git -c user.email=a@b -c user.name=a commit -qm r && git push -q " + str(bare) + " results/x",
                   shell=True, cwd=w, check=True)
    code = tmp_path / "code"; subprocess.run(["git", "clone", "-q", str(bare), str(code)], check=True)
    monkeypatch.setattr(runner, "CODE", code)
    runner.fetch_results(["results/x"], dest=str(tmp_path / "prev"))
    assert (tmp_path / "prev" / "results/x/run/memory.pt").read_text().strip() == "1"


def test_runner_memory_budget(tmp_path, monkeypatch):
    """With gpu_gb estimates, the running sum stays within the GPU's memory."""
    import runner
    code = tmp_path / "code"; (code / "scripts").mkdir(parents=True)
    (code / "scripts" / "job.py").write_text(
        "import sys, time, json, pathlib\n"
        "a = sys.argv; out = pathlib.Path(a[a.index('--out') + 1]); t0 = time.time(); time.sleep(0.4)\n"
        "(out / 'progress.jsonl').write_text(json.dumps({'start': t0, 'end': time.time()}) + '\\n')\n")
    monkeypatch.setattr(runner, "CODE", code); monkeypatch.setattr(runner, "OUT", tmp_path / "out")
    monkeypatch.setattr(runner, "STOP", tmp_path / "STOP"); monkeypatch.setattr(runner, "_gpu_used_gb", lambda: (0, 40.0))
    (tmp_path / "out").mkdir()
    class FakeSaver:
        def save(self, src, message=None, quiet=False):
            pass
    runs = [{"name": f"t{i}", "script": "scripts/job.py", "args": [], "gpu_gb": 26} for i in range(2)]
    runs += [{"name": f"d{i}", "script": "scripts/job.py", "args": [], "gpu_gb": 5} for i in range(2)]
    plan = {"name": "p", "max_parallel": 8, "stagger_sec": 0, "poll_sec": 0.05, "runs": runs}
    runner._run_queue(plan, FakeSaver(), {"runs": {}}, lambda: None, __import__("time").time(), budget=10)
    t = {n: json.loads((tmp_path / "out" / n / "progress.jsonl").read_text()) for n in ("t0", "t1", "d0", "d1")}
    assert t["t1"]["start"] >= t["t0"]["end"] - 0.05          # two trainings (52 GB) never together on 40 GB
    assert t["d0"]["start"] < t["t0"]["end"]                   # a diagnosis fits next to a training


def test_local_launcher_check_reports_what_is_missing_and_never_prints_the_token(tmp_path):
    """colab/local_mq.py --check on a machine without a GPU or a token: it says why it stops and exits 1."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "colab" / "local_mq.py"
    env = {k: v for k, v in os.environ.items() if k not in ("MM_GITHUB_TOKEN", "GITHUB_TOKEN")}
    r = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, env=env)
    assert r.returncode == 1
    assert "トークン" in r.stdout and "止まる理由" in r.stdout and "MM_GITHUB_TOKEN" in r.stdout
    assert "GPU" in r.stdout
