#!/usr/bin/env python3
"""Teacher and student at every position of the evaluation rows, written out raw (report: mq_positions_report.py).
Runs next to a training job: the model stays in CPU memory, each layer visits the GPU for its own forward pass only."""
import json, os, sys, time
import torch, torch.nn.functional as F
E = os.environ.get
sys.path.insert(0, E("MQ_SRC", "/content/mmllm/src"))
from mmllm.mosaic_qwen import _text_model, load_mosaic_qwen
from mmllm.mq_data import RowPacker, load_source
M, D, O, DELTA = E("MQ_MODEL", "/content/models/Qwen3.8-27B"), E("MQ_DATA", "/content/data/"), E("MQ_OUT"), E("MQ_DELTA")
SETS = json.loads(E("MQ_SETS", '[[2048, 16, 8000, 4], [8192, 2, 60000, 1]]'))      # row length, rows, documents per source, rows per pass
cuda = E("MQ_DEV", "cuda") == "cuda"
dev = torch.device("cuda" if cuda else "cpu")
t0 = time.time()
mq, tok = load_mosaic_qwen(M, DELTA, dtype=torch.bfloat16 if cuda else torch.float32, device="cpu", keep_teacher=True)
model, text = mq.model, _text_model(mq.model)
mq.fixed_skip()
if cuda:
    torch.cuda.set_per_process_memory_fraction(float(E("MQ_FRACTION", "0.12")))
    for n, m in text.named_children():
        if n != "layers":
            m.to(dev)
    model.lm_head.to(dev)
    def move(m, d):                     # a replaced layer keeps its original (the teacher's) outside the module tree
        m.to(d)
        o = m._st.orig.get(m.layer_idx) if hasattr(m, "_st") else None
        if o is not None:
            o.to(d)
    for l in text.layers:
        l.register_forward_pre_hook(lambda m, a: move(m, dev))
        l.register_forward_hook(lambda m, a, o: move(m, "cpu"))
out = {"model": M, "delta": DELTA, "rows": {}}
for T, n, docs, per in SETS:
    src = [load_source(D + s, docs) for s in ("cosmopedia-100k@4", "ultrachat:messages@2", "wikipedia-ja@2", "codeparrot:content@1")]
    pk = RowPacker(tok, src, T, seed=0, max_doc=T)
    out["sep"] = pk.sep_id
    rows = pk.eval_rows(n).tolist()
    for a in range(0, n, per):
        t1 = time.time()
        with torch.no_grad():
            x = torch.tensor(rows[a:a + per], device=dev)
            with mq.teacher():
                ht = text(input_ids=x, use_cache=False).last_hidden_state[:, :-1]
            hs = text(input_ids=x, use_cache=False).last_hidden_state[:, :-1]
            for b in range(x.shape[0]):
                t, o = x[b, 1:], {k: [] for k in ("nt", "ns", "kl", "top", "p", "tops")}
                for c in range(0, T - 1, 256):
                    lt, ls = F.log_softmax(model.lm_head(ht[b, c:c + 256]).float(), -1), F.log_softmax(model.lm_head(hs[b, c:c + 256]).float(), -1)
                    g = t[c:c + 256, None]
                    o["nt"] += (-lt.gather(1, g)[:, 0]).tolist(); o["ns"] += (-ls.gather(1, g)[:, 0]).tolist()
                    o["kl"] += (lt.exp() * (lt - ls)).sum(-1).tolist()
                    v, k = lt.max(-1)
                    o["top"] += k.tolist(); o["p"] += v.exp().tolist(); o["tops"] += ls.argmax(-1).tolist()
                r = {k: [round(q, 3) for q in v] if k in ("nt", "ns", "kl", "p") else v for k, v in o.items()}
                out["rows"]["r%d_%d" % (T, a + b)] = {"ids": rows[a + b], **r}
        json.dump(out, open(O, "w"))
        kl = [q for k, v in out["rows"].items() if k.startswith("r%d_" % T) for q in v["kl"]]
        print(T, a, "kl so far %.4f" % (sum(kl) / len(kl)), "%.0fs" % (time.time() - t1), "total %.0fs" % (time.time() - t0),
              "gpu %.1fGiB" % (torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0), flush=True)
print("done", flush=True)
