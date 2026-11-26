"""The official compiled FlexAttention against the dense path, at lengths that are and are not multiples of 128."""
import sys, time, json
sys.path.insert(0, "src")
import torch
from mmllm import official
from mmllm.official import memory_mask

torch.manual_seed(0)
H, d = 2, 16
out = {}
lens = [int(x) for x in sys.argv[1].split(",")]
for T in lens:
    q, k, v = (torch.nn.functional.normalize(torch.randn(1, H, T, d), dim=-1) for _ in range(3))
    q = q * 12.0                                   # queries carry the bandwidth, as in the memories
    tokens, q_lens = torch.zeros(1, T, dtype=torch.long), torch.tensor([T])
    for scope, kw in (("short", {"window": 256}), ("long", {"skip": 64})):
        mask = memory_mask(scope, False, tokens, q_lens, "causal", **kw)
        t0 = time.time()
        with torch.no_grad():
            ref = official.dense_attention(q, k, v, block_mask=mask.block_mask, scale=1)
            try:
                got = official._upstream_flex(q, k, v, block_mask=mask.block_mask, scale=1)
                err = None
            except Exception as e:                 # noqa: BLE001
                got, err = None, f"{type(e).__name__}: {str(e)[:200]}"
        if got is None:
            print(T, scope, "official call failed:", err, flush=True)
            out[f"{T}/{scope}"] = {"error": err}
            continue
        diff = (ref - got).abs().amax(dim=(0, 1, 3))            # per position
        bad = (diff > 1e-3).nonzero()[:, 0]
        res = {"max_diff": float(diff.max()), "positions_off": int(len(bad)),
               "first_off": int(bad[0]) if len(bad) else None, "last_off": int(bad[-1]) if len(bad) else None,
               "diff_at_last_position": float(diff[-1]), "scale": float(ref.abs().mean()), "seconds": round(time.time() - t0, 1)}
        out[f"{T}/{scope}"] = res
        print(T, scope, res, flush=True)
json.dump(out, open("/home/claude/runs/flexcheck/cpu.json", "w"), indent=1)
