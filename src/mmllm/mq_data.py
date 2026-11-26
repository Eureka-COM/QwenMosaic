"""Training rows for the MosaicQwen stages: documents packed into fixed-length rows.

A source is given as  PATH[:FIELD][@WEIGHT]
    PATH    a .parquet / .jsonl / .jsonl.gz / .json.gz file, or a directory of them
    FIELD   the column holding the document (default "text"). A column of chat turns
            ("messages": [{"role", "content"}], or ShareGPT "conversations":
            [{"from", "value"}]) is rendered with the tokenizer's chat template.
    WEIGHT  sampling weight of the source (default 1)

Rows are built as upstream's `pretrain` loader does: documents in random order, each
followed by the end-of-document token, cut at the row length. The last `holdout` documents
of every source are never used for training rows; evaluation rows are packed from them.
Everything is a deterministic function of (sources, seed).
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}


@dataclass
class Source:
    name: str
    docs: list
    weight: float
    is_chat: bool


def _read_file(f: Path, field: str, limit: int) -> list:
    out = []
    if f.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2048, columns=[field]):
            out += batch.column(0).to_pylist()
            if len(out) >= limit:
                break
        return out[:limit]
    opener = gzip.open if f.suffix == ".gz" else open
    with opener(f, "rt", encoding="utf-8") as fh:
        head = fh.read(1)
        fh.seek(0)
        if head == "[":
            return [r[field] for r in json.load(fh)][:limit]
        for line in fh:
            if line.strip():
                out.append(json.loads(line)[field])
                if len(out) >= limit:
                    break
    return out


def parse_spec(spec: str) -> tuple[str, str, float]:
    weight = 1.0
    if "@" in spec:
        spec, w = spec.rsplit("@", 1)
        weight = float(w)
    field = "text"
    head, sep, tail = spec.rpartition(":")
    if sep and head and not tail.startswith(("/", "\\")) and len(head) > 1:     # not a Windows drive letter
        spec, field = head, tail
    return spec, field, weight


def load_source(spec: str, limit: int = 200_000) -> Source:
    path, field, weight = parse_spec(spec)
    p = Path(path)
    if p.is_dir():
        files = sorted(x for pat in ("**/*.parquet", "**/*.jsonl", "**/*.jsonl.gz", "**/*.json.gz") for x in p.glob(pat))
    else:
        files = [p]
    if not files:
        raise FileNotFoundError(f"no data file under {path}")
    docs = []
    for f in files:
        docs += _read_file(f, field, limit - len(docs))
        if len(docs) >= limit:
            break
    docs = [d for d in docs if d is not None and len(d) > 0]
    if not docs:
        raise ValueError(f"{spec}: no documents in field {field!r}")
    is_chat = isinstance(docs[0], (list, tuple))
    return Source(f"{p.name}:{field}", docs, weight, is_chat)


def _as_messages(turns) -> list[dict]:
    msgs = []
    for t in turns:
        role = t.get("role") or _ROLE.get(str(t.get("from", "")).lower(), "user")
        content = t.get("content") if t.get("content") is not None else t.get("value", "")
        msgs.append({"role": role, "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
    return msgs


def sep_token_id(tok) -> int:
    """The end-of-document token the packers put after every document."""
    sep_id = tok.convert_tokens_to_ids("<|endoftext|>")
    if sep_id is None or sep_id == tok.unk_token_id:
        sep_id = tok.eos_token_id
    return int(sep_id)


class RowPacker:
    def __init__(self, tok, sources: list[Source], seq_len: int, seed: int = 0, max_doc: int | None = None,
                 holdout: int = 64, sep_id: int | None = None, min_doc_chars: int = 32) -> None:
        self.tok, self.sources, self.seq_len = tok, sources, seq_len
        self.max_doc = max_doc or seq_len
        self.min_doc_chars = min_doc_chars
        self.sep_id = int(sep_id) if sep_id is not None else sep_token_id(tok)
        self.rng = random.Random(seed)
        self.holdout = {}
        for s in sources:
            h = min(holdout, max(1, len(s.docs) // 10))
            self.holdout[s.name] = h
        self.total_w = sum(s.weight for s in sources)
        self.docs_used = 0

    def _encode(self, src: Source, doc) -> list[int]:
        if src.is_chat and getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template(_as_messages(doc), tokenize=False, add_generation_prompt=False)
        elif src.is_chat:        # a base model without a chat template: the turns as plain text
            text = "\n\n".join(f"{m['role']}: {m['content']}" for m in _as_messages(doc))
        else:
            text = doc
        return self.tok(text, add_special_tokens=False).input_ids[: self.max_doc]

    def _pick(self, rng, held_out: bool):
        x = rng.random() * self.total_w
        for s in self.sources:
            x -= s.weight
            if x <= 0:
                break
        n, h = len(s.docs), self.holdout[s.name]
        idx = rng.randrange(n - h, n) if held_out else rng.randrange(0, n - h)
        return s, s.docs[idx]

    def _row(self, rng, held_out: bool) -> list[int]:
        ids: list[int] = []
        tries = 0
        while len(ids) < self.seq_len and tries < 10_000:
            tries += 1
            src, doc = self._pick(rng, held_out)
            if not src.is_chat and len(doc) < self.min_doc_chars:
                continue
            ids += self._encode(src, doc) + [self.sep_id]
            if not held_out:
                self.docs_used += 1
        if len(ids) < self.seq_len:
            raise RuntimeError("could not fill a row: documents too short or too few")
        return ids[: self.seq_len]

    def batch(self, bsz: int) -> torch.Tensor:
        return torch.tensor([self._row(self.rng, False) for _ in range(bsz)], dtype=torch.long)

    def eval_rows(self, n: int, seed: int = 1234) -> torch.Tensor:
        """Fixed rows from the held-out documents (the same for every call with the same seed)."""
        rng = random.Random(seed)
        return torch.tensor([self._row(rng, True) for _ in range(n)], dtype=torch.long)

    def describe(self) -> dict:
        return {"seq_len": self.seq_len, "max_doc": self.max_doc, "sep_id": self.sep_id,
                "sources": [{"name": s.name, "docs": len(s.docs), "weight": s.weight, "chat": s.is_chat,
                             "holdout": self.holdout[s.name]} for s in self.sources]}


# ── rows of related documents (plan section 10, step 2; upstream's `cluster` loader) ──

def _feature_text(src: Source, doc, chars: int) -> str:
    if src.is_chat:
        doc = " ".join(m["content"] for m in _as_messages(doc))
    return doc[:chars]


def cluster_documents(tok, src: Source, cluster_size: int = 60, seed: int = 0, feat_chars: int = 3000,
                      hash_dim: int = 8192, feat_dim: int = 64, iters: int = 25, batch: int = 512) -> dict:
    """Group the documents of one source by content: token counts of the first `feat_chars`
    characters, hashed to `hash_dim` bins, tf-idf weighted, reduced to `feat_dim` dimensions
    (PCA fitted on a sample), unit length, then k-means on the sphere with about
    `cluster_size` documents per cluster. Deterministic for a given seed.

    Returns {"assign": LongTensor [n_docs], "centroids": FloatTensor [k, feat_dim]}."""
    n = len(src.docs)
    k = max(1, round(n / cluster_size))
    g = torch.Generator().manual_seed(seed)
    bins, counts = [], []
    for i in range(0, n, batch):
        texts = [_feature_text(src, d, feat_chars) for d in src.docs[i:i + batch]]
        for ids in tok(texts, add_special_tokens=False).input_ids:
            b, c = torch.unique(torch.tensor(ids, dtype=torch.long) % hash_dim, return_counts=True)
            bins.append(b); counts.append(c.float())
    df = torch.zeros(hash_dim)
    for b in bins:
        df[b] += 1
    idf = torch.log((n + 1) / (df + 1)) + 1

    def dense(rows):
        x = torch.zeros(len(rows), hash_dim)
        for r, i in enumerate(rows):
            x[r, bins[i]] = torch.log1p(counts[i]) * idf[bins[i]]
        return torch.nn.functional.normalize(x, dim=-1)

    sample = torch.randperm(n, generator=g)[: min(n, 8192)].tolist()
    xs = dense(sample)
    mean = xs.mean(0, keepdim=True)
    q = min(feat_dim, hash_dim, max(1, len(sample) - 1))
    # torch.pca_lowrank draws from the global generator: fix it for this call only
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        _, _, basis = torch.pca_lowrank(xs - mean, q=q, center=False, niter=4)
    z = torch.cat([torch.nn.functional.normalize((dense(list(range(i, min(i + 4096, n)))) - mean) @ basis, dim=-1)
                   for i in range(0, n, 4096)])
    cent = z[torch.randperm(n, generator=g)[:k]].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for it in range(iters):
        best, new = (z @ cent.T).max(dim=1)
        if it > 0 and torch.equal(new, assign):
            break
        assign = new
        sums = torch.zeros_like(cent).index_add_(0, assign, z)
        size = torch.bincount(assign, minlength=k)
        empty = (size == 0).nonzero().flatten().tolist()
        if empty:                                   # re-seed an empty cluster with the worst-placed documents
            far = best.argsort()[: len(empty)]
            for c, i in zip(empty, far.tolist()):
                sums[c] = z[i]
        cent = torch.nn.functional.normalize(sums, dim=-1)
    assign = (z @ cent.T).argmax(dim=1)
    return {"assign": assign, "centroids": cent}


class ClusterRowPacker(RowPacker):
    """Rows filled with documents of one cluster (related documents), each followed by the
    end-of-document token. A cluster that runs out is continued with the clusters nearest
    to it. `holdout_frac` of every source's documents, as whole clusters, is kept away from
    training rows; evaluation rows and `context_cases` are built from those clusters."""

    def __init__(self, tok, sources: list[Source], seq_len: int, seed: int = 0, max_doc: int | None = None,
                 sep_id: int | None = None, min_doc_chars: int = 32, cluster_size: int = 60,
                 holdout_frac: float = 0.05, cluster_seed: int = 0, cache_dir: str | None = None) -> None:
        super().__init__(tok, sources, seq_len, seed=seed, max_doc=max_doc, holdout=0, sep_id=sep_id,
                         min_doc_chars=min_doc_chars)
        self.cluster_size, self.holdout_frac = cluster_size, holdout_frac
        self.groups: dict[str, dict] = {}
        for si, s in enumerate(sources):
            res = self._clusters(tok, s, cluster_size, cluster_seed + si, cache_dir)
            assign, cent = res["assign"], res["centroids"]
            k = cent.shape[0]
            members = [[] for _ in range(k)]
            for i, c in enumerate(assign.tolist()):
                if s.is_chat or len(s.docs[i]) >= min_doc_chars:
                    members[c].append(i)
            order = list(range(k))
            random.Random(cluster_seed * 7919 + si).shuffle(order)
            held, n_held, target = set(), 0, holdout_frac * len(s.docs)
            for c in order:
                if n_held >= target or len(held) >= k - 1:
                    break
                if members[c]:
                    held.add(c); n_held += len(members[c])
            near = (cent @ cent.T).argsort(dim=1, descending=True)
            self.groups[s.name] = {
                "members": members, "held": held, "near": near,
                "train": [c for c in range(k) if c not in held and members[c]],
                "eval": [c for c in sorted(held) if members[c]],
            }
        self.holdout = {s.name: sum(len(self.groups[s.name]["members"][c]) for c in self.groups[s.name]["eval"])
                        for s in sources}

    @staticmethod
    def _clusters(tok, s: Source, cluster_size: int, seed: int, cache_dir: str | None) -> dict:
        """cluster_documents, kept on disk between the runs of one session (a minute or two per
        source of 40,000 documents; stage 1, stage 2 and the evaluation need the same clusters)."""
        if not cache_dir:
            return cluster_documents(tok, s, cluster_size=cluster_size, seed=seed)
        import hashlib
        key = "|".join([s.name, str(len(s.docs)), str(cluster_size), str(seed), str(len(tok)),
                        repr(s.docs[0])[:200], repr(s.docs[-1])[:200]])
        f = Path(cache_dir) / f"clusters-{hashlib.sha1(key.encode()).hexdigest()[:16]}.pt"
        if f.exists():
            return torch.load(f)
        res = cluster_documents(tok, s, cluster_size=cluster_size, seed=seed)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        torch.save(res, tmp)
        tmp.replace(f)
        return res

    def _source(self, rng) -> Source:
        x = rng.random() * self.total_w
        for s in self.sources:
            x -= s.weight
            if x <= 0:
                break
        return s

    def _cluster_docs(self, rng, s: Source, held_out: bool):
        """Document indices of one cluster picked with probability proportional to its size,
        in random order, then those of the nearest clusters of the same split."""
        gr = self.groups[s.name]
        pool = gr["eval"] if held_out else gr["train"]
        sizes = [len(gr["members"][c]) for c in pool]
        x, first = rng.random() * sum(sizes), pool[-1]
        for c, n in zip(pool, sizes):
            x -= n
            if x <= 0:
                first = c
                break
        allowed = set(pool)
        for c in gr["near"][first].tolist():       # the cluster itself comes first
            if c in allowed:
                docs = list(gr["members"][c])
                rng.shuffle(docs)
                yield from docs

    def _row(self, rng, held_out: bool) -> list[int]:
        s = self._source(rng)
        ids: list[int] = []
        for i in self._cluster_docs(rng, s, held_out):
            ids += self._encode(s, s.docs[i]) + [self.sep_id]
            if not held_out:
                self.docs_used += 1
            if len(ids) >= self.seq_len:
                return ids[: self.seq_len]
        raise RuntimeError("could not fill a row: documents too short or too few")

    def context_cases(self, n_per_source: int, max_tokens: int, target_tokens: tuple[int, int] = (256, 1024),
                      context_doc_tokens: int = 2048, min_context_docs: int = 8, seed: int = 4321) -> list[dict]:
        """Cases of the context-growth evaluation, from held-out clusters: a target document
        (cut to target_tokens[1] tokens; at least target_tokens[0]) and, nearest first, the
        related documents that may be put before it (same cluster, each cut to
        `context_doc_tokens`), as many as fit in `max_tokens` with the target; and as many
        unrelated ones (other held-out clusters of the same source). Deterministic."""
        rng = random.Random(seed)
        cases = []
        for s in self.sources:
            gr = self.groups[s.name]
            clusters = [c for c in gr["eval"] if len(gr["members"][c]) > min_context_docs]
            rng.shuffle(clusters)
            got, tries = 0, 0
            while got < n_per_source and clusters and tries < 50 * n_per_source:
                c = clusters[tries % len(clusters)]
                tries += 1
                docs = list(gr["members"][c])
                rng.shuffle(docs)
                saved = self.max_doc
                try:
                    self.max_doc = target_tokens[1]
                    target = self._encode(s, s.docs[docs[0]])
                    if len(target) < target_tokens[0]:
                        continue
                    self.max_doc = context_doc_tokens
                    budget = max_tokens - len(target)

                    def take(indices):
                        out, used = [], 0
                        for i in indices:
                            d = self._encode(s, s.docs[i]) + [self.sep_id]
                            if used + len(d) > budget:
                                break
                            out.append(d); used += len(d)
                        return out

                    related = take(docs[1:])
                    if len(related) < min_context_docs:
                        continue
                    others = [i for o in gr["eval"] if o != c for i in gr["members"][o]]
                    rng.shuffle(others)
                    unrelated = take(others)[: len(related)]
                finally:
                    self.max_doc = saved
                cases.append({"source": s.name, "cluster": int(c), "doc": int(docs[0]), "target": target,
                              "related": related, "unrelated": unrelated})
                got += 1
        return cases

    def describe(self) -> dict:
        d = super().describe()
        d["rows"] = "cluster"
        d["cluster_size"], d["holdout_frac"] = self.cluster_size, self.holdout_frac
        for e, s in zip(d["sources"], self.sources):
            gr = self.groups[s.name]
            sizes = sorted(len(m) for m in gr["members"] if m)
            e["clusters"] = len(sizes)
            e["held_out_clusters"] = len(gr["eval"])
            e["cluster_docs_min_median_max"] = [sizes[0], sizes[len(sizes) // 2], sizes[-1]] if sizes else []
        return d


def context_row(case: dict, k: int, unrelated: bool = False) -> tuple[list[int], int]:
    """The row of a context case with `k` documents before the target (the nearest k of the
    list, the farthest first) and where the target starts in it."""
    docs = (case["unrelated"] if unrelated else case["related"])[:k]
    ctx = [t for d in reversed(docs) for t in d]
    return ctx + case["target"], len(ctx)


def make_packer(kind: str, tok, sources, seq_len: int, seed: int = 0, max_doc: int | None = None, **cluster_kw):
    if kind == "cluster":
        return ClusterRowPacker(tok, sources, seq_len, seed=seed, max_doc=max_doc, **cluster_kw)
    assert kind == "random", kind
    return RowPacker(tok, sources, seq_len, seed=seed, max_doc=max_doc)
