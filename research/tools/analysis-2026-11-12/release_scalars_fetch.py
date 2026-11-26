"""Read the small per-head parameters of the memories straight out of the Release shards (range requests)."""
import json, struct, subprocess, sys, re
import torch

REPO = "Eureka-COM/mosaic-qwen"
ASSETS = {"s1": [738196766, 738203104], "s2": [739846433, 739853374], "s3": [740333447, 740338169]}
WANT = re.compile(r"(long|short)_memory\.(bandwidth_bias|bandwidth_scale|bandwidth_exponent|value_interp|value_scale)$|recurrence\.")
DT = {"BF16": (torch.bfloat16, 2), "F32": (torch.float32, 4), "F16": (torch.float16, 2)}

def rng(asset, a, b):
    out = subprocess.run(["gh", "api", "-H", "Accept: application/octet-stream", "-H", f"Range: bytes={a}-{b}",
                          f"repos/{REPO}/releases/assets/{asset}"], capture_output=True, timeout=120)
    assert out.returncode == 0 and len(out.stdout) == b - a + 1, (out.returncode, len(out.stdout), out.stderr[:200])
    return out.stdout

res = {}
for stage in sys.argv[1:]:
    res[stage] = {}
    for asset in ASSETS[stage]:
        n = struct.unpack("<Q", rng(asset, 0, 7))[0]
        head = json.loads(rng(asset, 8, 8 + n - 1))
        names = [k for k in head if k != "__metadata__"]
        want = [k for k in names if WANT.search(k)]
        print(stage, asset, "tensors", len(names), "wanted", len(want), "| sample names:", [k for k in names if "layers.3." in k and "lora" not in k][:30], flush=True)
        # one request spanning all wanted tensors when they sit close together, else one by one
        for k in want:
            m = head[k]; a, b = m["data_offsets"]; dt, sz = DT[m["dtype"]]
            buf = rng(asset, 8 + n + a, 8 + n + b - 1)
            res[stage][k] = torch.frombuffer(bytearray(buf), dtype=dt).float().reshape(m["shape"]).tolist()
    json.dump(res, open("/home/claude/runs/scalars/scalars_27b_%s.json" % "_".join(sys.argv[1:]), "w"))
    print(stage, "read", len(res[stage]), "tensors", flush=True)
