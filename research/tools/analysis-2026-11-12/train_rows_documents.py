"""Which documents make up the first training rows of stage 3 (same picks as RowPacker.batch)."""
import json, sys
sys.path.insert(0, "src")
from transformers import AutoTokenizer
from mmllm.mq_data import load_source, RowPacker

tok = AutoTokenizer.from_pretrained("/home/claude/models/Qwen3.5-0.8B")
specs = ["/home/claude/data/cosmopedia-100k@4", "/home/claude/data/ultrachat:messages@2",
         "/home/claude/data/wikipedia-ja@2", "/home/claude/data/codeparrot:content@1"]
sources = [load_source(s, 60000) for s in specs]
p = RowPacker(tok, sources, 8192, seed=0, max_doc=8192)
ce = {1: None, 2: None, 3: 2.70, 4: 1.97, 5: 3.86, 6: 1.34, 7: 4.26, 8: 3.91, 9: 2.07, 12: 2.26}
orig_pick = p._pick
log = []
def pick(rng, held_out):
    s, d = orig_pick(rng, held_out)
    log.append((s, d))
    return s, d
p._pick = pick
for step in range(1, 13):
    log.clear()
    row = p.batch(1)[0].tolist()
    parts, pos = [], 0
    for s, d in log:
        if not s.is_chat and len(d) < p.min_doc_chars:
            continue
        n = len(p._encode(s, d)) + 1
        parts.append((s.name.split(":")[0], pos, min(n, 8192 - pos)))
        pos += n
    share = {}
    for name, _, n in parts:
        share[name] = share.get(name, 0) + n
    print(step, "ce", ce.get(step), {k: round(v / 8192, 2) for k, v in share.items()}, "| docs", [(a[:4], n) for a, _, n in parts])
