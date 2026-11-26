"""Which held-out documents make up the evaluation rows (same picks as RowPacker.eval_rows)."""
import json, random, sys
sys.path.insert(0, "src")
from transformers import AutoTokenizer
from mmllm.mq_data import load_source, RowPacker

tok = AutoTokenizer.from_pretrained("/home/claude/models/Qwen3.5-0.8B")
specs = ["/home/claude/data/cosmopedia-100k@4", "/home/claude/data/ultrachat:messages@2",
         "/home/claude/data/wikipedia-ja@2", "/home/claude/data/codeparrot:content@1"]
docs = int(sys.argv[1])
sources = [load_source(s, docs) for s in specs]
out = {}
for seq_len, max_doc, n in ((8192, 8192, 2), (2048, None, 4)):
    p = RowPacker(tok, sources, seq_len, seed=0, max_doc=max_doc)
    rng = random.Random(1234)
    rows = []
    for _ in range(n):
        ids, parts = [], []
        while len(ids) < seq_len:
            src, doc = p._pick(rng, True)
            if not src.is_chat and len(doc) < p.min_doc_chars:
                continue
            e = p._encode(src, doc)
            start = len(ids)
            ids += e + [p.sep_id]
            text = tok.decode(e[:60])
            parts.append({"source": src.name, "start": start, "tokens_in_row": min(len(ids), seq_len) - start,
                          "doc_tokens": len(e), "head": text[:160].replace("\n", "\\n")})
        rows.append(parts)
    out[str(seq_len)] = rows
    print("== seq", seq_len, p.describe()["sources"])
    for i, parts in enumerate(rows):
        print(" row", i)
        for q in parts:
            print("   ", q["source"], "start", q["start"], "in_row", q["tokens_in_row"], "doc", q["doc_tokens"], "|", q["head"][:110])
json.dump(out, open("/home/claude/runs/evalrows/rows.json", "w"), ensure_ascii=False, indent=1)
