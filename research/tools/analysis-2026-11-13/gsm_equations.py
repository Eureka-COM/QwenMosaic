#!/usr/bin/env python3
"""The calculations written in the GSM8K solutions of mq_bench.py ("a op b = c" with plain numbers): how many are wrong,
for the teacher and for the student. A rough count: it only sees equations of this simple form on one line. No GPU.
usage: gsm_equations.py <gsm_items.json> [out.json]"""
import json, re, sys

NUM = r"\$?\\?\$?\s*(-?\d[\d,]*(?:\.\d+)?)"
PAT = re.compile(NUM + r"\s*(?:\\text\{[^}]*\})?\s*(\+|-|\\times|×|\*|/|\\div|÷)\s*" + NUM + r"\s*(?:\\text\{[^}]*\})?\s*=\s*" + NUM)


def check(text):
    ok, bad = 0, []
    for m in PAT.finditer(text):
        a, op, b, c = (m.group(1), m.group(2), m.group(3), m.group(4))
        a, b, c = (float(v.replace(",", "")) for v in (a, b, c))
        if op == "+": r = a + b
        elif op == "-": r = a - b
        elif op in ("\\times", "×", "*"): r = a * b
        elif b: r = a / b
        else: continue
        if abs(r - c) < 0.006 or abs(r - c) <= 1e-6 * abs(r):
            ok += 1
        else:
            bad.append(" ".join(m.group(0).split()))
    return ok, bad


items = json.load(open(sys.argv[1]))
out = {"questions": len(items)}
for who in ("teacher", "student"):
    per = [check(x[who]) for x in items]
    out[who] = {"equations_found": sum(o + len(b) for o, b in per), "wrong": sum(len(b) for _, b in per),
                "questions_with_a_wrong_equation": sum(bool(b) for _, b in per),
                "wrong_equations": [{"k": x["k"], "text": t} for x, (_, b) in zip(items, per) for t in b],
                "wrote_the_asked_form": sum("####" in x[who] for x in items), "right": sum(x[who + "_ok"] for x in items)}
print(json.dumps(out, indent=1, ensure_ascii=False))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
