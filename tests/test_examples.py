"""examples/ の台本が CPU で動き、書き出した値が層の計算と合うこと。"""
import subprocess
import sys
from pathlib import Path

from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]


def test_dump_layer_reference(tmp_path):
    out = tmp_path / "ref.safetensors"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "dump_layer_reference.py"), "--out", str(out),
                        "--seq-len", "320"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    t = load_file(str(out))
    for k in ("x", "h", "short", "long", "memory_sum", "block_out"):
        assert t[k].shape == (1, 320, 64), k
    assert (t["short"] + t["long"] - t["memory_sum"]).abs().max() < 1e-5
    # 長い記憶は skip より前しか読めない：先頭の skip（64）語では何も読めず 0 になる
    assert t["long"][:, :64].abs().max() == 0
    assert t["long"][:, 128:].abs().max() > 0
