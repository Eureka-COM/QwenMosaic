#!/usr/bin/env python3
"""Per-position loss of the ORIGINAL model on the stage-3 rows, written out raw (analysis: mq_teacher_positions.py --dump).
Runs next to a training job: the model stays in CPU memory, each layer visits the GPU for its own forward pass only."""
import json, os, sys, time
import torch, torch.nn.functional as F
E = os.environ.get
sys.path.insert(0, E("MQ_SRC", "/content/mmllm/src"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmllm.mosaic_qwen import _text_model
from mmllm.mq_data import RowPacker, load_source
M, D, O = E("MQ_MODEL", "/content/models/Qwen3.8-27B"), E("MQ_DATA", "/content/data/"), E("MQ_OUT", "/content/out/s3/teacher_positions.json")
T, cuda = int(E("MQ_SEQ", "8192")), E("MQ_DEV", "cuda") == "cuda"
dev = torch.device("cuda" if cuda else "cpu")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16 if cuda else torch.float32).eval()
text = _text_model(model)
if cuda:
    torch.cuda.set_per_process_memory_fraction(float(E("MQ_FRACTION", "0.12")))
    for n, m in text.named_children():
        if n != "layers":
            m.to(dev)
    model.lm_head.to(dev)
    for l in text.layers:
        l.register_forward_pre_hook(lambda m, a: m.to(dev) and None)
        l.register_forward_hook(lambda m, a, o: m.to("cpu") and None)
src = [load_source(D + s, int(E("MQ_DOCS", "60000"))) for s in ("cosmopedia-100k@4", "ultrachat:messages@2", "wikipedia-ja@2", "codeparrot:content@1")]
pk = RowPacker(tok, src, T, seed=0, max_doc=T)
ev = pk.eval_rows(2).tolist()
rows = {"eval0": ev[0], "eval1": ev[1]}
for i in range(1, 9):
    r = pk.batch(1)[0].tolist()
    if i in (6, 8):
        rows["train%d" % i] = r
def tail(r):
    c = next((i for i in range(T // 2, T - 64) if r[i] == pk.sep_id), T // 2)
    return r[c + 1:]
jobs = [(k, v, False) for k, v in rows.items()] + [(k + "_tail", tail(rows[k]), False) for k in ("eval0", "eval1")] + [("eval1_autocast", rows["eval1"], True)]
out = {"model": M, "sep": pk.sep_id, "rows": {}}
for name, ids, ac in jobs:
    t1 = time.time()
    with torch.no_grad():
        x = torch.tensor([ids], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=ac and cuda):
            h = text(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
        t, nll, top, p = x[0, 1:], [], [], []
        for c in range(0, len(t), 512):
            lp = F.log_softmax(model.lm_head(h[c:c + 512]).float(), -1)
            nll += (-lp.gather(1, t[c:c + 512, None])[:, 0]).tolist()
            v, k = lp.max(-1)
            top += k.tolist(); p += v.exp().tolist()
    out["rows"][name] = {"ids": ids, "nll": [round(a, 2) for a in nll], "top": top, "ptop": [round(a, 3) for a in p]}
    json.dump(out, open(O, "w"))
    print(name, len(ids), "loss %.3f" % (sum(nll) / len(nll)), "bad", sum(a > 8 for a in nll), "%.0fs" % (time.time() - t1), "total %.0fs" % (time.time() - t0),
          "gpu %.1fGiB" % (torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0), flush=True)
print("done", flush=True)
