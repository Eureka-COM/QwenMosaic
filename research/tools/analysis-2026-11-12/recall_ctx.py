"""Rebuild the 80 recall prompts of the 27B evaluation exactly and look at what surrounds the question."""
import json, sys, collections
sys.path.insert(0, "/home/claude/runs/flexcheck"); sys.path.insert(0, "/home/claude/mosaic-qwen/scripts"); sys.path.insert(0, "/home/claude/mosaic-qwen/src")
from emul27 import packer
import mq_eval

tok, p = packer(2048, 8000)
filler = p.eval_rows(5, seed=99).reshape(-1).tolist()
filler = [t for t in filler if t != p.sep_id]
cases = mq_eval.recall_cases(tok, filler, [32, 512, 1500, 6000], 20, 0)
rec = json.load(open("/home/claude/runs/evals3/recall.json"))["cases"]
assert len(cases) == len(rec)
match = sum(c["code"] == r["code"] and len(c["ids"]) == r["prompt_tokens"] for c, r in zip(cases, rec))
print("cases rebuilt with the same code and length:", match, "of", len(rec), "| filler tokens", len(filler))
im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")

def state_at(ids, upto):
    """What the model has seen before position `upto`: (state, turns seen, tokens since the last chat marker)."""
    state, turns, last = "no conversation yet", 0, None
    for j in range(upto):
        t = ids[j]
        if t == im_start:
            role = tok.decode(ids[j + 1:j + 2]).strip()
            state, turns, last = "inside a turn: " + role, turns + 1, j
        elif t == im_end:
            state, last = "after a turn ended", j
    return state, turns, (upto - last if last is not None else None)

rows = []
for c, r in zip(cases, rec):
    ids = c["ids"]; d = c["distance"]
    q_len = len(tok(f"\nQuestion: What is the access code for {c['name']}?\nAnswer: The access code for {c['name']} is", add_special_tokens=False).input_ids)
    f_len = len(tok(f"\nNote: The access code for {c['name']} is {c['code']}.\n", add_special_tokens=False).input_ids)
    q_at = len(ids) - q_len
    fact_at = q_at - d - f_len
    s_q, turns, since = state_at(ids, q_at)
    s_f, _, _ = state_at(ids, fact_at)
    rows.append({"distance": d, "teacher_ok": r["teacher_ok"], "student_ok": r["student_ok"], "at_question": s_q, "at_fact": s_f,
                 "turns_before_question": turns, "tokens_since_last_marker": since, "lead": fact_at,
                 "tail": tok.decode(ids[q_at - 14:q_at]).replace("\n", "\\n")[-46:]})
json.dump(rows, open("/home/claude/runs/flexcheck/recall_ctx.json", "w"), ensure_ascii=False, indent=1)
for d in (32, 512, 1500, 6000):
    print("== distance", d)
    tab = collections.defaultdict(lambda: [0, 0, 0])
    for x in rows:
        if x["distance"] == d:
            k = x["at_question"]; tab[k][0] += 1; tab[k][1] += x["teacher_ok"]; tab[k][2] += x["student_ok"]
    for k, (n, t, s) in sorted(tab.items()):
        print(f"   {k:28s} cases {n:2d}  teacher {t:2d}  student {s:2d}")
print("== all distances, by what precedes the question")
tab = collections.defaultdict(lambda: [0, 0, 0])
for x in rows:
    k = x["at_question"]; tab[k][0] += 1; tab[k][1] += x["teacher_ok"]; tab[k][2] += x["student_ok"]
for k, (n, t, s) in sorted(tab.items()):
    print(f"   {k:28s} cases {n:2d}  teacher {t:2d}  student {s:2d}")
print("== cases up to 1500, listed")
for x in rows:
    if x["distance"] <= 1500:
        print(f"   d{x['distance']:5d} T{int(x['teacher_ok'])} S{int(x['student_ok'])} | q: {x['at_question']:26s} turns {x['turns_before_question']:2d} since {str(x['tokens_since_last_marker']):>5s} | fact: {x['at_fact']:26s} | …{x['tail']}")
