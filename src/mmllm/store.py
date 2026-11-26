"""Where large checkpoints go: a directory, or the assets of a GitHub release.

Results (logs, numbers, small weights) are saved by colab/saver.py to results/* branches.
A 27B checkpoint (about 2 GB of memories) does not belong in a branch: every save would add
its full size to the repository for good. A release keeps the files next to the repository
without growing it, and an asset can be replaced.

    store = open_store("dir:/vol/mq/run1")                       # e.g. a Modal Volume
    store = open_store("github-release:mq-27b-s1")               # tag in MM_RESULTS_REPO
    store.push(local_dir)        # upload every file of local_dir (replacing same-named files)
    store.pull(local_dir)        # download everything; False if the store is empty / missing

The GitHub store needs a token with Contents: write in the environment (MM_GITHUB_TOKEN or
GITHUB_TOKEN) and files under 2 GB each (GitHub's limit per asset).
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
ASSET_LIMIT = 2 * 1024**3 - 1


class DirStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def push(self, src) -> list[str]:
        self.path.mkdir(parents=True, exist_ok=True)
        names = []
        for f in sorted(Path(src).iterdir()):
            if f.is_file():
                tmp = self.path / (f.name + ".tmp")
                shutil.copyfile(f, tmp)
                os.replace(tmp, self.path / f.name)
                names.append(f.name)
        for old in self.path.iterdir():                 # shards of an earlier, larger save
            if old.is_file() and old.name not in names and old.name.startswith(("delta-", "model-")):
                old.unlink()
        return names

    def pull(self, dst) -> bool:
        if not self.path.is_dir() or not any(self.path.iterdir()):
            return False
        Path(dst).mkdir(parents=True, exist_ok=True)
        for f in self.path.iterdir():
            if f.is_file() and not f.name.endswith(".tmp"):
                shutil.copyfile(f, Path(dst) / f.name)
        return True

    def describe(self) -> str:
        return f"dir:{self.path}"


class _NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    """An asset download redirects to a signed storage URL, which must not get the API token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            for h in (new.headers, new.unredirected_hdrs):
                h.pop("Authorization", None)
        return new


class GitHubReleaseStore:
    def __init__(self, tag: str, repo: str | None = None, token: str | None = None) -> None:
        self.tag = tag
        repo = repo or os.environ.get("MM_RESULTS_REPO", "https://github.com/Eureka-COM/mosaic-qwen.git")
        self.repo = repo.removeprefix("https://github.com/").removesuffix(".git")
        self.token = token or os.environ.get("MM_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("github-release store: no token in MM_GITHUB_TOKEN / GITHUB_TOKEN")

    def _req(self, method: str, url: str, data=None, headers: dict | None = None):
        h = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
             "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "mosaic-qwen"}
        h.update(headers or {})
        if isinstance(data, dict):
            data = json.dumps(data).encode(); h["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise RuntimeError(f"GitHub {method} {url.split('?')[0]} -> {e.code}: {e.read()[:300]!r}") from None

    def _release(self, create: bool):
        rel = self._req("GET", f"{API}/repos/{self.repo}/releases/tags/{self.tag}")
        if rel is None and create:
            rel = self._req("POST", f"{API}/repos/{self.repo}/releases",
                            {"tag_name": self.tag, "name": self.tag, "prerelease": True,
                             "body": "Checkpoints written by the MosaicQwen training stages (replaced in place)."})
            if rel is None:
                raise RuntimeError("could not create the release (does the token have Contents: write?)")
        return rel

    def push(self, src) -> list[str]:
        rel = self._release(create=True)
        existing = {a["name"]: a["id"] for a in rel.get("assets", [])}
        names = []
        for f in sorted(Path(src).iterdir()):
            if not f.is_file():
                continue
            size = f.stat().st_size
            if size > ASSET_LIMIT:
                raise RuntimeError(f"{f.name}: {size} bytes is over GitHub's 2 GB per asset (use a smaller --shard-mb)")
            if f.name in existing:
                self._req("DELETE", f"{API}/repos/{self.repo}/releases/assets/{existing[f.name]}")
            with f.open("rb") as fh:
                self._req("POST", f"{UPLOADS}/repos/{self.repo}/releases/{rel['id']}/assets?name={f.name}", data=fh,
                          headers={"Content-Type": "application/octet-stream", "Content-Length": str(size)})
            names.append(f.name)
        for name, aid in existing.items():
            if name not in names and name.startswith(("delta-", "model-")):
                self._req("DELETE", f"{API}/repos/{self.repo}/releases/assets/{aid}")
        return names

    def pull(self, dst) -> bool:
        rel = self._release(create=False)
        if rel is None or not rel.get("assets"):
            return False
        Path(dst).mkdir(parents=True, exist_ok=True)
        for a in rel["assets"]:
            req = urllib.request.Request(f"{API}/repos/{self.repo}/releases/assets/{a['id']}",
                                         headers={"Authorization": f"Bearer {self.token}", "Accept": "application/octet-stream",
                                                  "User-Agent": "mosaic-qwen"})
            opener = urllib.request.build_opener(_NoAuthRedirect)
            with opener.open(req, timeout=600) as resp, (Path(dst) / a["name"]).open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
        return True

    def describe(self) -> str:
        return f"github-release:{self.tag} ({self.repo})"


def open_store(spec: str | None):
    if not spec:
        return None
    kind, _, rest = spec.partition(":")
    if kind == "dir":
        return DirStore(rest)
    if kind == "github-release":
        return GitHubReleaseStore(rest)
    raise ValueError(f"unknown store {spec!r} (dir:PATH or github-release:TAG)")
