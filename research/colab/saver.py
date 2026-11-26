"""Save Colab results to GitHub, and nowhere else.

Backends (pick with save_mode in the plan):

  "rln"    research-lab-notebook (Eureka-COM/research-lab-notebook), written with that
           repository's own deploy key, given as the Colab secret RLN_DEPLOY_KEY.
           Results go to  mosaic-qwen/<YYYY-MM-DD_HHMM>_<run>/  on main.
  "token"  this repository (Eureka-COM/mosaic-qwen), written with the Colab secret
           MM_GITHUB_TOKEN (falls back to GITHUB_TOKEN). Results go to the branch  results/<YYYY-MM-DD_HHMM>_<run>  (one per run).
           The token is handed to git through GIT_ASKPASS, never through the command line
           or a URL, so it cannot appear in a git error message.
  "git"    this repository, with whatever git credentials the machine already has (used for
           runs outside Colab: tools/record_run.py). Branch  results/<YYYY-MM-DD_HHMM>_<run>.

Everything in the run folder is saved, weights included (keep them under 95 MB: GitHub
refuses files over 100 MB; a larger file is skipped with a warning and stays local).

save(src_dir) can be called repeatedly (partial saves while a run is going); every call
commits whatever changed. A failed push is retried; if it still fails, the results stay in
/content and push_pending() sends them later.
"""

import datetime as _dt
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

MAX_BYTES = 95 * 1024 * 1024
_GITHUB_HOSTKEY = "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n"
RLN_REPO = "git@github.com:Eureka-COM/research-lab-notebook.git"
RLN_KEY_FINGERPRINT = "SHA256:5IAfx+Rh84RbpdKy2kpX1HMqdPdFwdUL+2Si3p2PBDM"   # research-lab-notebook deploy key
CODE_REPO = os.environ.get("MM_RESULTS_REPO", "https://github.com/Eureka-COM/mosaic-qwen.git")


def _run(cmd, cwd=None, check=True, env=None):
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env or os.environ.copy())
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} failed:\n{_scrub(r.stderr.strip())}")
    return r


def _scrub(text: str) -> str:
    tok = os.environ.get("_MM_TOKEN_SCRUB", "")
    return text.replace(tok, "***") if tok else text


TOKEN_NAMES = ("MM_GITHUB_TOKEN", "mosaic-qwen-colab", "mm-colab-results", "GITHUB_TOKEN")   # first one present wins


def _secret(name):
    """The environment first (the start cell puts the token it verified there), then the Colab secret."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name)
    except Exception:
        return None




def _normalize_key(text):
    text = text.replace("\r", "").replace("\\n", "\n").strip().strip('"').strip("'").strip()
    head, tail = "-----BEGIN OPENSSH PRIVATE KEY-----", "-----END OPENSSH PRIVATE KEY-----"
    if head in text and tail in text:
        body = "".join(text.split(head, 1)[1].split(tail, 1)[0].split())
        return head + "\n" + "\n".join(body[i:i + 70] for i in range(0, len(body), 70)) + "\n" + tail + "\n"
    return text + "\n"


class Saver:
    def __init__(self, mode: str, run_name: str, base: str = "/content"):
        assert mode in ("rln", "token", "git"), mode
        self.mode, self.run_name, self.base = mode, run_name, Path(base)
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        if mode == "rln":
            self.clone = self.base / "research-lab-notebook"
            self.rel = Path("mosaic-qwen") / f"{stamp}_{run_name}"
            self.branch = "main"
        else:                     # "token" and "git": this repository, one branch per run
            self.clone = self.base / "mm-results"
            self.rel = Path("results") / f"{stamp}_{run_name}"
            self.branch = f"results/{stamp}_{run_name}"
            if self.clone.exists():     # an earlier run in this runtime (it has its own branch): start clean. A save
                shutil.rmtree(self.clone, ignore_errors=True)      # that was interrupted can leave git lock files behind
        self.env = os.environ.copy()
        self._setup()

    # ── setup ──
    def _setup(self):
        if self.mode == "rln":
            key = _secret("RLN_DEPLOY_KEY")
            if not key:
                raise RuntimeError("シークレット RLN_DEPLOY_KEY がありません（または許可されていません）")
            ssh = Path.home() / ".ssh"; ssh.mkdir(mode=0o700, exist_ok=True)
            kp = ssh / "rln_key"; kp.write_text(_normalize_key(key)); kp.chmod(0o600)
            fp = _run(["ssh-keygen", "-l", "-f", str(kp)], check=False).stdout.split()
            if len(fp) < 2 or fp[1] != RLN_KEY_FINGERPRINT:
                kp.unlink()
                raise RuntimeError("RLN_DEPLOY_KEY が research-lab-notebook の鍵ではありません")
            kh = ssh / "rln_known_hosts"; kh.write_text(_GITHUB_HOSTKEY)
            self.env["GIT_SSH_COMMAND"] = (f"ssh -i {kp} -o IdentitiesOnly=yes -o UserKnownHostsFile={kh} "
                                           f"-o StrictHostKeyChecking=yes -o ConnectTimeout=20")
            if not (self.clone / ".git").exists():
                _run(["git", "clone", "-q", "--depth", "1", RLN_REPO, str(self.clone)], env=self.env)
        elif self.mode == "git":
            if not (self.clone / ".git").exists():
                self.clone.mkdir(parents=True, exist_ok=True)
                _run(["git", "init", "-q"], cwd=self.clone)
                _run(["git", "remote", "add", "origin", CODE_REPO], cwd=self.clone)
                _run(["git", "checkout", "-q", "-b", self.branch], cwd=self.clone)
        else:
            tok = next((t for t in map(_secret, TOKEN_NAMES) if t), None)
            if not tok:
                raise RuntimeError("シークレット MM_GITHUB_TOKEN（または GITHUB_TOKEN）がありません（または許可されていません）")
            os.environ["_MM_TOKEN_SCRUB"] = tok
            askpass = Path(tempfile.mkdtemp()) / "askpass.sh"
            askpass.write_text('#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; *) echo "$_MM_TOKEN";; esac\n')
            askpass.chmod(0o700)
            self.env.update({"GIT_ASKPASS": str(askpass), "_MM_TOKEN": tok, "GIT_TERMINAL_PROMPT": "0"})
            if not (self.clone / ".git").exists():
                self.clone.mkdir(parents=True, exist_ok=True)
                _run(["git", "init", "-q"], cwd=self.clone, env=self.env)
                _run(["git", "remote", "add", "origin", CODE_REPO], cwd=self.clone, env=self.env)
                _run(["git", "checkout", "-q", "-b", self.branch], cwd=self.clone, env=self.env)
        _run(["git", "config", "user.name", "mosaic-qwen colab"], cwd=self.clone)
        _run(["git", "config", "user.email", "mosaic-qwen@users.noreply.github.com"], cwd=self.clone)
        print(f"✓ 保存先の準備完了: {self.where()}", flush=True)

    def where(self):
        repo = "research-lab-notebook" if self.mode == "rln" else "mosaic-qwen"
        return f"{repo} ブランチ {self.branch} / {self.rel}"

    # ── save ──
    def _push(self, attempts=5):
        last = ""
        for i in range(1, attempts + 1):
            r = _run(["git", "push", "-q", "origin", f"HEAD:{self.branch}"], cwd=self.clone, check=False, env=self.env)
            if r.returncode == 0:
                return True, ""
            last = _scrub(r.stderr.strip())
            if self.mode == "rln":
                _run(["git", "pull", "-q", "--rebase", "origin", "main"], cwd=self.clone, check=False, env=self.env)
            if i < attempts:
                time.sleep(min(60, 5 * 2 ** (i - 1)))
        return False, last

    def save(self, src, message=None, quiet=False, exclude=()):
        """Copy src into the results folder and push (everything, unless `exclude` patterns)."""
        dest = self.clone / self.rel
        dest.mkdir(parents=True, exist_ok=True)
        src = Path(src)
        ignore = shutil.ignore_patterns(*exclude) if exclude else None
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True, ignore=ignore)
        else:
            shutil.copy2(src, dest / src.name)
        for p in [p for p in dest.rglob("*") if p.is_file() and p.stat().st_size > MAX_BYTES]:
            p.unlink()
            print(f"  ・{p.relative_to(dest)} は大きすぎて GitHub に送れないため除外（元の場所には残っています）")
        _run(["git", "add", "-A", str(self.rel)], cwd=self.clone)
        if _run(["git", "diff", "--cached", "--quiet"], cwd=self.clone, check=False).returncode == 0:
            return True
        _run(["git", "commit", "-qm", message or f"mosaic-qwen: {self.run_name}"], cwd=self.clone)
        ok, err = self._push()
        if not quiet or not ok:
            print(("✓ GitHub に保存: " + self.where()) if ok else
                  f"✗ GitHub への送信に失敗。成果は {dest} に残っています。push_pending() で送り直してください。\n  {err[-300:]}",
                  flush=True)
        return ok

    def push_pending(self):
        ok, err = self._push()
        print("✓ 送信しました" if ok else f"✗ まだ送れません: {err[-300:]}")
        return ok
