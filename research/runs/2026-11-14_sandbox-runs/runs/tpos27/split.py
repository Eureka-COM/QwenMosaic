import json, sys, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from transformers import AutoTokenizer
from mq_teacher_positions import segments
tok = AutoTokenizer.from_pretrained("/home/claude/models/Qwen3.5-0.8B")
d = json.load(open("/home/claude/runs/tpos27/teacher_positions.json"))
sep = d["sep"]; im_end = tok.convert_tokens_to_ids("<|im_end|>"); im_start = tok.convert_tokens_to_ids("<|im_start|>")
tot = {}
def add(key, nll, top, ptop):
    t = tot.setdefault(key, {"n": 0, "nll": 0.0, "bad": 0, "end": 0, "end_sure": 0})
    t["n"] += len(nll); t["nll"] += float(nll.sum()); t["bad"] += int((nll > 8).sum())
    t["end"] += int((top == im_end).sum()); t["end_sure"] += int(((top == im_end) & (ptop > 0.9)).sum())
for name, row in d["rows"].items():
    if "_" in name: continue
    ids = row["ids"]; nll = torch.tensor([0.0] + row["nll"]); top = torch.tensor([0] + row["top"]); ptop = torch.tensor([0.0] + row["ptop"])
    seen_chat = False
    for s, e in segments(ids, sep):
        chat = ids[s] == im_start
        kind = "chat document" if chat else ("plain text after a chat document" if seen_chat else "plain text, no chat document before it")
        idx = torch.arange(max(s, 1), e)
        # the real next token being <|im_end|> is not a miss: leave those targets out of the "predicts the end" count
        add(kind, nll[idx], top[idx], ptop[idx]); add(name + " | " + kind, nll[idx], top[idx], ptop[idx])
        seen_chat = seen_chat or chat
for k, t in tot.items():
    print(f"{k:62s} n {t['n']:6d}  loss {t['nll']/t['n']:6.3f}  bad {100*t['bad']/t['n']:5.1f}%  top=<|im_end|> {100*t['end']/t['n']:5.1f}%  (sure {100*t['end_sure']/t['n']:5.1f}%)")
json.dump(tot, open("/home/claude/runs/tpos27/split.json", "w"), indent=1)
