import json, math, statistics as st
from safetensors import safe_open
import torch

def local(d):
    cfg = json.load(open(d + "mq_config.json")); out = {}
    for sh in cfg["shards"]:
        with safe_open(d + sh, "pt") as f:
            for k in f.keys():
                if k.endswith(("bandwidth_bias", "bandwidth_scale", "bandwidth_exponent", "value_interp", "value_scale")):
                    out[k] = f.get_tensor(k).float().tolist()
    return out

sets = {"0.8B s3": local("/home/claude/runs/d1-s3-r10-delta/")}
for s in ("s1", "s2", "s3"):
    sets["27B " + s] = json.load(open(f"scalars_27b_{s}.json"))[s]

def layers(d):
    return sorted({int(k.split(".")[1]) for k in d})

def beta(b0, b1, a, n):
    return math.exp(min(b1, 5)) * n ** min(abs(a), 1) + math.exp(min(b0, 5))

for name, d in sets.items():
    print(f"\n== {name}  (long memory; mean over heads, [min..max])")
    print(" layer |  b0=exp(bias)    b1=exp(scale)   exponent a      | beta(n=448)  beta(1436)  beta(5936) | gain=exp(value_scale)  value_interp")
    allb = {448: [], 1436: [], 5936: []}; gains = []; exps = []
    for L in layers(d):
        p = f"layers.{L}.block.long_memory."
        bb, bs, be, vi, vs = (d[p + x] for x in ("bandwidth_bias", "bandwidth_scale", "bandwidth_exponent", "value_interp", "value_scale"))
        H = len(bb)
        betas = {n: [beta(bb[h], bs[h], be[h], n) for h in range(H)] for n in allb}
        for n in allb: allb[n] += betas[n]
        g = [math.exp(v) for v in vs]; gains += g; exps += [abs(x) for x in be]
        f = lambda xs: f"{st.mean(xs):7.2f} [{min(xs):6.2f}..{max(xs):7.2f}]"
        print(f" {L:5d} | {f([math.exp(x) for x in bb])} {f([math.exp(x) for x in bs])} {st.mean([abs(x) for x in be]):5.3f} [{min(abs(x) for x in be):.2f}..{max(abs(x) for x in be):.2f}] | "
              f"{st.mean(betas[448]):8.1f} {st.mean(betas[1436]):10.1f} {st.mean(betas[5936]):10.1f} | {f(g)}  {st.mean(vi):5.2f} [{min(vi):5.2f}..{max(vi):5.2f}]")
    print(f"  all heads: beta median at n=448/1436/5936: {st.median(allb[448]):.1f} / {st.median(allb[1436]):.1f} / {st.median(allb[5936]):.1f};"
          f"  min {min(allb[448]):.1f} / {min(allb[1436]):.1f} / {min(allb[5936]):.1f};  gain median {st.median(gains):.2f} (min {min(gains):.2f}, max {max(gains):.2f});  exponent median {st.median(exps):.3f}")
json.dump({k: v for k, v in sets.items()}, open("all_scalars.json", "w"))
