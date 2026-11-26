"""The 27B tokenizer's chat rendering, emulated with the shared vocabulary (checked against rows recorded on the VM)."""
import json, sys
sys.path.insert(0, "/home/claude/mosaic-qwen/src")
from transformers import AutoTokenizer
from mmllm.mq_data import RowPacker, load_source, _as_messages

SYSTEM = ("<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key "
          "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
          "answer.<|im_end|>\n")
SPECS = ["cosmopedia-100k@4", "ultrachat:messages@2", "wikipedia-ja@2", "codeparrot:content@1"]


def render27(msgs, strip=True) -> str:
    out = SYSTEM
    for m in msgs:
        c = m["content"].strip() if strip else m["content"]
        head = "<think>\n\n</think>\n\n" if m["role"] == "assistant" else ""
        out += f"<|im_start|>{m['role']}\n{head}{c}<|im_end|>\n"
    return out


class Packer27(RowPacker):
    strip = True

    def _encode(self, src, doc):
        text = render27(_as_messages(doc), self.strip) if src.is_chat else doc
        return self.tok(text, add_special_tokens=False).input_ids[: self.max_doc]


def packer(seq, docs, max_doc=None, strip=True):
    tok = AutoTokenizer.from_pretrained("/home/claude/models/Qwen3.5-0.8B")
    src = [load_source("/home/claude/data/" + s, docs) for s in SPECS]
    p = Packer27(tok, src, seq, seed=0, max_doc=max_doc)
    p.strip = strip
    return tok, p


if __name__ == "__main__":
    d = json.load(open("/home/claude/runs/pos27/positions_s3.json"))
    want = [d["rows"]["r2048_%d" % i]["ids"] for i in range(16)]
    for strip in (True, False):
        tok, p = packer(2048, 8000, strip=strip)
        got = p.eval_rows(16).tolist()
        same = [g == w for g, w in zip(got, want)]
        print("strip", strip, "rows identical:", sum(same), "of 16")
        if not all(same):
            i = same.index(False)
            j = next(k for k in range(2048) if got[i][k] != want[i][k])
            print("  first difference: row", i, "pos", j, repr(tok.decode(got[i][max(0, j - 8):j + 6])), "| VM:", repr(tok.decode(want[i][max(0, j - 8):j + 6])))
