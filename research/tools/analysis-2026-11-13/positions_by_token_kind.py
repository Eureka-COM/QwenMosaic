#!/usr/bin/env python3
"""Where the stage-2 student differs from the teacher, by the kind of the next token (number / other).
Input: the per-position dump of the evaluation rows (scripts/mq_positions_dump.py), e.g.
results/2026-11-12_1447_mq-27b-s3:s3/positions_s2.json. No GPU. After-the-fact analysis (not a criterion of the plan).
usage: positions_by_token_kind.py <positions.json> <tokenizer dir>"""
import json, sys, re, collections
sys.path.insert(0, "scripts")
from transformers import AutoTokenizer
from mq_positions_report import classify

d = json.load(open(sys.argv[1]))
tok = AutoTokenizer.from_pretrained(sys.argv[2])
num = re.compile(r"\d")
agg = collections.defaultdict(lambda: collections.defaultdict(float))
for name, r in d["rows"].items():
    if not name.startswith("r2048"):
        continue
    ids = r["ids"]
    kinds = classify(ids, tok, d["sep"])
    for i in range(len(ids) - 1):
        k = kinds[i + 1]
        if k not in ("plain text, no conversation before it", "conversation: assistant's turns"):
            continue
        nxt = tok.decode([ids[i + 1]])
        kind = "number" if num.search(nxt) else "other"
        # a number that continues a number (2nd digit onwards) vs. the first digit after something else
        if kind == "number":
            kind = "number, continuing a number" if num.search(tok.decode([ids[i]])) else "number, first token"
        for key in (kind, "all"):
            a = agg[key]
            a["n"] += 1
            a["kl"] += r["kl"][i]
            a["same_top"] += r["top"][i] == r["tops"][i]
            a["teacher_right"] += r["top"][i] == ids[i + 1]
            a["student_right"] += r["tops"][i] == ids[i + 1]
            a["nt"] += r["nt"][i]; a["ns"] += r["ns"][i]
            if r["top"][i] == ids[i + 1]:                      # the teacher is right: does the student follow?
                a["t_right_n"] += 1; a["t_right_student_right"] += r["tops"][i] == ids[i + 1]
out = {}
for key, a in agg.items():
    n = a["n"]
    out[key] = {"positions": int(n), "kl": round(a["kl"] / n, 4), "same_top_as_teacher": round(a["same_top"] / n, 4),
                "teacher_right": round(a["teacher_right"] / n, 4), "student_right": round(a["student_right"] / n, 4),
                "nll_teacher": round(a["nt"] / n, 4), "nll_student": round(a["ns"] / n, 4),
                "student_right_where_teacher_right": round(a["t_right_student_right"] / max(a["t_right_n"], 1), 4)}
print(json.dumps(out, indent=1))
