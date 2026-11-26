import sys
sys.path.insert(0, "src")
from transformers import AutoTokenizer
from mmllm.mq_data import load_source, RowPacker
tok = AutoTokenizer.from_pretrained("/home/claude/models/Qwen3.5-0.8B")
specs = ["/home/claude/data/cosmopedia-100k@4", "/home/claude/data/ultrachat:messages@2",
         "/home/claude/data/wikipedia-ja@2", "/home/claude/data/codeparrot:content@1"]
sources = [load_source(s, 60000) for s in specs]
p = RowPacker(tok, sources, 8192, seed=0, max_doc=8192)
orig = p._pick; log = []
def pick(rng, held_out):
    s, d = orig(rng, held_out); log.append((s, d)); return s, d
p._pick = pick
for step in range(1, 9):
    log.clear(); p.batch(1)
    if step in (6, 8):
        print("===== row", step)
        for s, d in log:
            if s.is_chat:
                print("  [chat]", str(d[0])[:150].replace("\n", " "))
            else:
                print("  [%s] %d chars |" % (s.name[:4], len(d)), d[:230].replace("\n", "\\n"), "...", d[-120:].replace("\n", "\\n"))
