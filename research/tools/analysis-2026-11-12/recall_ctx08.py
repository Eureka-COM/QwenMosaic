"""The 80 recall prompts of the 0.8B evaluation (D1), rebuilt, and what precedes the question."""
import json, sys, collections, subprocess
sys.path.insert(0, "/home/claude/mosaic-qwen/scripts"); sys.path.insert(0, "/home/claude/mosaic-qwen/src")
from transformers import AutoTokenizer
from mmllm.mq_data import RowPacker, load_source
import mq_eval
tok = AutoTokenizer.from_pretrained("/home/claude/models/Qwen3.5-0.8B")
src = [load_source("/home/claude/data/" + s, 8000) for s in ("cosmopedia-100k@4", "ultrachat:messages@2", "wikipedia-ja@2", "codeparrot:content@1")]
p = RowPacker(tok, src, 2048, seed=0)
filler = p.eval_rows(5, seed=99).reshape(-1).tolist()
filler = [t for t in filler if t != p.sep_id]
cases = mq_eval.recall_cases(tok, filler, [32, 512, 1500, 6000], 20, 0)
B = "origin/results/2026-11-11_2322_mq-dev-0p8b-rest"
rec = json.loads(subprocess.run(["git", "show", f"{B}:{B[7:]}/eval-s3-r10/recall.json"], capture_output=True, text=True).stdout)["cases"]
print("rebuilt with the same code and length:", sum(c["code"] == r["code"] and len(c["ids"]) == r["prompt_tokens"] for c, r in zip(cases, rec)), "of", len(rec))
im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
def state_at(ids, upto):
    state = "no conversation yet"
    for j in range(upto):
        if ids[j] == im_start: state = "inside a turn: " + tok.decode(ids[j + 1:j + 2]).strip()
        elif ids[j] == im_end: state = "after a turn ended"
    return state
tab = collections.defaultdict(lambda: [0, 0, 0])
for c, r in zip(cases, rec):
    q_len = len(tok(f"\nQuestion: What is the access code for {c['name']}?\nAnswer: The access code for {c['name']} is", add_special_tokens=False).input_ids)
    k = (c["distance"], state_at(c["ids"], len(c["ids"]) - q_len))
    tab[k][0] += 1; tab[k][1] += r["teacher_ok"]; tab[k][2] += r["student_ok"]
for k, (n, t, s) in sorted(tab.items()):
    print(f"   d {k[0]:5d} {k[1]:28s} cases {n:2d}  teacher {t:2d}  student {s:2d}")
print(repr(tok.decode(filler[:0])), "chat sample:", repr(tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], tokenize=False, add_generation_prompt=False)))
