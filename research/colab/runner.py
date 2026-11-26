"""Run a list of experiments on Colab, save progress to GitHub as it goes, stop the runtime.

The cell built by colab/make_cell.py unpacks this repository into /content/mmllm and calls
run_all(PLAN). PLAN = {"name": ..., "save_mode": "rln"|"token", "auto_stop": bool,
"max_minutes": total budget, "setup": {...}, "runs": [{"name", "script", "args": [...]}, ...]}.

- Each run is a subprocess; its stdout goes to /content/out/<run>/stdout.log and its own
  progress.jsonl. One line per finished/failed run is printed to the cell.
- Every `save_every_min` minutes (and after each run) /content/out is saved to GitHub.
- Creating /content/STOP stops the current run and skips the rest (results are still saved).
- The total budget (max_minutes) is enforced across all runs.
- With auto_stop the runtime is released at the end, also after an error.
"""

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

OUT = Path("/content/out")
STOP = Path("/content/STOP")
CODE = Path("/content/mmllm")


def _log(msg):
    print(f"[mm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_results(branches, env=None, dest="/content/prev"):
    """Copy earlier results branches (e.g. trained weights) of this repository to dest/<path>."""
    for b in branches:
        r = subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", b], cwd=str(CODE), env=env,
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"results branch {b} を取得できません: {r.stderr[-300:]}")
        Path(dest).mkdir(parents=True, exist_ok=True)
        subprocess.run(f"git archive FETCH_HEAD | tar -x -C {dest}", shell=True, cwd=str(CODE), env=env, check=True)
        _log(f"results ready: {b} -> {dest}")


def setup(cfg, env=None):
    """Install packages and fetch the model, data and earlier results (idempotent)."""
    if cfg.get("results"):
        fetch_results(cfg["results"], env)
    pkgs = cfg.get("pip", [])
    if pkgs:
        _log("pip install " + " ".join(pkgs))
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
        if r.returncode != 0:
            _log("pip の失敗（続行）: " + r.stderr[-500:])
    from huggingface_hub import hf_hub_download, snapshot_download
    for m in cfg.get("models", []):
        snapshot_download(m, local_dir=f"/content/models/{m.split('/')[-1]}",
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.jinja"])
        _log(f"model ready: {m}")
    for d in cfg.get("data", []):
        hf_hub_download(d["repo"], d["file"], repo_type="dataset", local_dir=d["dir"])
        _log(f"data ready: {d['repo']}/{d['file']}")


def environment():
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import fla  # noqa: F401
        info["flash_linear_attention"] = True
    except Exception:
        info["flash_linear_attention"] = False
    return info


def _tail_progress(run_dir):
    p = run_dir / "progress.jsonl"
    if not p.exists():
        return None
    lines = p.read_text().strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def _validate_dependencies(runs):
    """Validate the local run graph before starting any process."""
    names = [r["name"] for r in runs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate run names")
    graph = {r["name"]: list(r.get("after", [])) for r in runs}
    for name, deps in graph.items():
        unknown = set(deps) - set(names)
        if unknown:
            raise ValueError(f"{name}: unknown dependencies {sorted(unknown)}")
    visiting, visited = set(), set()
    def visit(name):
        if name in visiting:
            raise ValueError(f"dependency cycle involving {name}")
        if name in visited:
            return
        visiting.add(name)
        for dep in graph[name]:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
    for name in names:
        visit(name)


def _dependency_state(run, ended, succeeded):
    deps = run.get("after", [])
    if any(n in ended and n not in succeeded for n in deps):
        return "blocked"
    return "ready" if all(n in succeeded for n in deps) else "pending"


def _missing_files(paths, base):
    """Explicit required_files/output_files must be nonempty regular files."""
    missing = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        try:
            good = path.is_file() and path.stat().st_size > 0
        except OSError:
            good = False
        if not good:
            missing.append(str(path))
    return missing


def _completion_issues(run):
    """Optional artifact/step requirements; no model-specific defaults or argument inference."""
    issues = []
    missing = _missing_files(run.get("output_files", []), run["dir"])
    if missing:
        issues.append("missing or empty outputs: " + ", ".join(missing))
    minimum = run.get("min_completed_step")
    if minimum is not None:
        try:
            tail = _tail_progress(run["dir"])
            complete = (tail is not None and tail.get("event") == "done"
                        and isinstance(tail.get("step"), (int, float))
                        and tail["step"] >= minimum)
        except (OSError, ValueError, TypeError):
            complete = False
        if not complete:
            issues.append(f"no completed step >= {minimum}")
    return issues


_TOKEN = {"value": None}     # set by run_all from the saver (token mode)


def _launch(plan, run, t0, budget):
    name = run["name"]
    left = budget - (time.time() - t0) / 60
    rdir = OUT / name; rdir.mkdir(parents=True, exist_ok=True)
    common = [] if run.get("no_common") else plan.get("common_args", [])
    args = [sys.executable, str(CODE / run["script"]), *common, *run["args"], "--out", str(rdir),
            "--max-minutes", str(round(min(run.get("max_minutes", left), left - 2), 1))]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if plan.get("export_token") and _TOKEN.get("value"):      # runs that push checkpoints to a release (mmllm/store.py)
        env["MM_GITHUB_TOKEN"] = _TOKEN["value"]
    log = (rdir / "stdout.log").open("w")
    proc = subprocess.Popen(args, cwd=str(CODE), stdout=log, stderr=subprocess.STDOUT, env=env)
    _log(f"{name}: start ({round(left)} min left)")
    return {"name": name, "proc": proc, "log": log, "dir": rdir, "t": time.time(), "note": time.time(),
            "wait": bool(run.get("wait")), "gpu_gb": run.get("gpu_gb", 0),
            "max_min": min(run.get("max_minutes", left), left),
            "output_files": run.get("output_files", []),
            "min_completed_step": run.get("min_completed_step")}


def _gpu_used_gb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        used, total = (float(x) for x in r.stdout.strip().split("\n")[0].split(","))
        return round(used / 1024, 1), round(total / 1024, 1)
    except Exception:
        return None, None


def _run_queue(plan, saver, status, write_status, t0, budget):
    """Run plan["runs"]: up to plan["max_parallel"] at once (default 1), in the listed order.
    A run with "wait": true starts only when nothing else is running (e.g. a quick baseline).
    A run with "after": [names] starts only when those runs succeeded (e.g. an evaluation of
    their weights); later runs may start before it. "no_common": true skips common_args."""
    _validate_dependencies(plan["runs"])
    pending = list(plan["runs"])
    running = []
    ended = set()
    succeeded = set()
    max_par = plan.get("max_parallel", 1)
    # memory budget: runs may declare "gpu_gb" (estimated peak); the sum of running estimates
    # stays below plan["gpu_frac"] (default 0.92) of the GPU's memory, whatever GPU was assigned
    cap = None
    if any("gpu_gb" in r for r in plan["runs"]):
        _, total = _gpu_used_gb()
        if total:
            cap = total * plan.get("gpu_frac", 0.92)
            _log(f"GPU メモリ {total}GB: 見積もりの合計 {round(cap, 1)}GB まで同時に回す")

    def fits(r):
        if cap is None:
            return True
        used = sum(x.get("gpu_gb", 0) for x in running)
        return not running or used + r.get("gpu_gb", 0) <= cap

    def ready(r):
        return _dependency_state(r, ended, succeeded) == "ready"

    while pending or running:
        # Failed/skipped dependencies block descendants; a failed run is never "ready" input.
        for run in list(pending):
            if _dependency_state(run, ended, succeeded) == "blocked":
                pending.remove(run)
                ended.add(run["name"])
                status["runs"][run["name"]] = "skipped (dependency not successful)"
                _log(f"{run['name']}: skipped (dependency not successful)")
                write_status()
        while pending and len(running) < max_par and not any(r["wait"] for r in running):
            cand = next((r for r in pending if ready(r) and fits(r)), None)
            if cand is None or (cand.get("wait") and running):
                break
            pending.remove(cand)
            run = cand
            if STOP.exists():
                status["runs"][run["name"]] = "skipped (STOP)"; ended.add(run["name"]); continue
            if budget - (time.time() - t0) / 60 < 3:
                status["runs"][run["name"]] = "skipped (budget)"; ended.add(run["name"]); continue
            missing = _missing_files(run.get("required_files", []), CODE)
            if missing:
                status["runs"][run["name"]] = "skipped (required files missing or empty)"
                ended.add(run["name"])
                _log(f"{run['name']}: required files missing or empty: {missing}")
                write_status()
                continue
            running.append(_launch(plan, run, t0, budget))
            status["runs"][run["name"]] = "running"; write_status()
            if pending and len(running) < max_par:
                time.sleep(plan.get("stagger_sec", 60))   # stagger memory peaks
            if run.get("wait"):
                break
        time.sleep(plan.get("poll_sec", 15))
        if STOP.exists():
            for r in running:
                if r["proc"].poll() is None:
                    r["proc"].terminate()
        over_budget = (time.time() - t0) / 60 > budget + 5
        for r in list(running):
            lim = r.get("max_min")
            if r["proc"].poll() is None and (over_budget or (lim and (time.time() - r["t"]) / 60 > lim + 5)):
                _log(f"{r['name']}: 時間の上限を過ぎても終わらないので止めます")   # hard stop: scripts honour
                r["proc"].kill()                                              # --max-minutes, this is the backstop
        for r in list(running):
            if time.time() - r["note"] > 600 and r["proc"].poll() is None:
                r["note"] = time.time()
                tail = _tail_progress(r["dir"])
                used, total = _gpu_used_gb()
                if tail:
                    _log(f"{r['name']}: step {tail.get('step')} {tail.get('event')} gpu {used}/{total}GB "
                         f"{json.dumps({k: tail[k] for k in ('loss', 'kl', 'beyond') if k in tail})}")
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["log"].close()
            running.remove(r); ended.add(r["name"])
            mins = round((time.time() - r["t"]) / 60, 1)
            issues = _completion_issues(r) if rc == 0 else []
            if rc == 0 and not issues:
                succeeded.add(r["name"])
                st = "ok"
            elif rc == 0:
                st = "incomplete (" + "; ".join(issues) + ")"
            else:
                st = "stopped" if STOP.exists() else f"failed (exit {rc})"
            status["runs"][r["name"]] = f"{st} in {mins} min"
            _log(f"{r['name']}: {st} in {mins} min")
            if rc != 0:
                _log("最後の出力:\n" + (r["dir"] / "stdout.log").read_text()[-1500:])
            write_status(); saver.save(OUT, message=f"{plan['name']}: {r['name']} {st}")


def check_premises(plan, code=None):
    """The main line (plans named mq-*) runs only under the fixed premises: the plan's premises_sha must be
    the SHA-256 of docs/PREMISES.md as it is in this checkout, and of the recorded lock. Raises otherwise."""
    import hashlib
    code = Path(code) if code is not None else CODE
    want = plan.get("premises_sha")
    if want is None:
        if str(plan.get("name", "")).startswith("mq-"):
            raise RuntimeError("この計画には premises_sha がありません（本線の計画は前提の照合が必須）")
        return None
    doc, lock = code / "docs" / "PREMISES.md", code / "docs" / "PREMISES.sha256"
    if not doc.is_file() or not lock.is_file():
        raise RuntimeError("docs/PREMISES.md か docs/PREMISES.sha256 がありません：前提を照合できないので起動しません")
    have = hashlib.sha256(doc.read_bytes()).hexdigest()
    locked = json.loads(lock.read_text())["sha256"]
    if not (have == locked == want):
        raise RuntimeError(f"前提が計画と一致しません（文書 {have[:12]}、固定 {locked[:12]}、計画 {str(want)[:12]}）。"
                           "前提は研究者の指示なしに変えない：docs/PREMISES.md を確かめてください")
    return have


def run_all(plan):
    t0 = time.time()
    if OUT.exists() and any(OUT.iterdir()):        # results of an earlier plan in this runtime: keep them aside
        OUT.rename(OUT.parent / f"out_prev_{int(t0)}")
    OUT.mkdir(parents=True, exist_ok=True)
    budget = plan.get("max_minutes", 120)
    saver = None
    status = {"plan": plan["name"], "runs": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    stop_saving = threading.Event()

    def write_status():
        status["elapsed_min"] = round((time.time() - t0) / 60, 1)
        (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))

    def periodic():
        while not stop_saving.wait(plan.get("save_every_min", 10) * 60):
            try:
                write_status(); saver.save(OUT, message=f"{plan['name']}: partial", quiet=True)
            except Exception as e:
                _log(f"途中保存に失敗（続行）: {e}")

    try:
        _validate_dependencies(plan["runs"])
        sha = check_premises(plan)                 # before anything is installed, downloaded or rented
        if sha:
            status["premises_sha"] = sha
            _log(f"前提の照合 OK（{sha[:12]}）")
        from saver import Saver
        saver = Saver(plan.get("save_mode", "rln"), plan["name"])
        _TOKEN["value"] = getattr(saver, "env", {}).get("_MM_TOKEN") or os.environ.get("MM_GITHUB_TOKEN")
        status["environment"] = environment()
        _log("environment " + json.dumps(status["environment"]))
        need = plan.get("require_gpu")
        if need and not any(n in str(status["environment"].get("gpu")) for n in need):
            raise RuntimeError(f"GPU {status['environment'].get('gpu')} は計画の条件 {need} に合わないので止めます")
        setup(plan.get("setup", {}), env=getattr(saver, "env", None))
        status["environment"] = environment()          # again, after the packages are installed
        (OUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        write_status(); saver.save(OUT, message=f"{plan['name']}: start")
        threading.Thread(target=periodic, daemon=True).start()

        _run_queue(plan, saver, status, write_status, t0, budget)
        status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        status["error"] = traceback.format_exc()[-3000:]
        _log("エラー:\n" + status["error"])
    finally:
        stop_saving.set()
        try:
            write_status()
            if saver is not None:
                saver.save(OUT, message=f"{plan['name']}: end")
        except Exception as e:
            _log(f"最後の保存に失敗: {e}")
        _log(f"終了（{round((time.time() - t0) / 60, 1)} 分）")
        if plan.get("auto_stop", True):
            _log("ランタイムを止めます")
            try:
                from google.colab import runtime  # type: ignore
                runtime.unassign()
            except Exception as e:
                _log(f"ランタイムを止められませんでした: {e}")
