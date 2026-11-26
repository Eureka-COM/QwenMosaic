#!/usr/bin/env python3
"""Teacher against student by KIND of position, from a dump of mq_positions_dump.py.

The rows the stages train and evaluate on pack several documents into one row, some of them
conversations written out with the chat template. The original 27B is far off on part of them:
inside the user's turns, and on any plain text that follows a conversation in the same row, it
keeps predicting the end-of-turn token (docs/results/mq-teacher-positions-2026-11-12.md). An
average over all positions therefore mixes positions where the teacher's output means something
with positions where it does not. This report keeps them apart:

    plain text, no conversation before it in the row
    conversation: the assistant's turns (what the model itself would write)
    conversation: the user's turns / the system turn / template tokens
    plain text after a conversation
    the separator between documents

    python scripts/mq_positions_report.py --dump positions_s2.json --tokenizer <model dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

KINDS = ["plain text, no conversation before it", "conversation: assistant's turns", "conversation: user's turns",
         "conversation: system turn", "conversation: template tokens", "plain text after a conversation", "separator"]
MEANINGFUL = KINDS[:2]


def classify(ids: list[int], tok, sep: int) -> list[str | None]:
    """The kind of every target position (None for position 0, which has no prediction)."""
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    think_end = tok.convert_tokens_to_ids("</think>")
    kinds: list[str | None] = [None] * len(ids)
    seen_chat, i, T = False, 0, len(ids)
    while i < T:
        e = i
        while e < T and ids[e] != sep:
            e += 1
        if ids[i] == im_start:                                   # a conversation
            seen_chat = True
            j = i
            while j < e:
                if ids[j] != im_start:                           # between turns: the newline after <|im_end|>
                    kinds[j] = KINDS[4]
                    j += 1
                    continue
                role = tok.decode(ids[j + 1:j + 2]).strip() if j + 1 < e else ""
                end = j + 1
                while end < e and ids[end] != im_end and ids[end] != im_start:
                    end += 1
                body = j + 3                                     # <|im_start|> role \n
                if role == "assistant":                          # skip the (empty) thinking block of the template
                    for q in range(body, min(end, body + 8)):
                        if ids[q] == think_end:
                            body = q + 2
                            break
                for q in range(j, min(body, e)):
                    kinds[q] = KINDS[4]
                kind = {"assistant": KINDS[1], "user": KINDS[2], "system": KINDS[3]}.get(role, KINDS[4])
                for q in range(min(body, e), end):
                    kinds[q] = kind
                if end < e and ids[end] == im_end:               # ending the turn is the speaker's own token
                    kinds[end] = kind
                    end += 1
                j = end
        else:
            for q in range(i, e):
                kinds[q] = KINDS[5] if seen_chat else KINDS[0]
        if e < T:
            kinds[e] = KINDS[6]
        i = e + 1
    kinds[0] = None
    return kinds


def table(rows: dict, tok, sep: int, hard: float) -> dict:
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    acc: dict[str, dict] = {}

    def add(key, r, j):
        a = acc.setdefault(key, {"n": 0, "nt": 0.0, "ns": 0.0, "kl": 0.0, "agree": 0, "bad": 0, "end": 0})
        a["n"] += 1
        a["nt"] += r["nt"][j - 1]; a["ns"] += r["ns"][j - 1]; a["kl"] += r["kl"][j - 1]
        a["agree"] += int(r["top"][j - 1] == r["tops"][j - 1])
        a["bad"] += int(r["nt"][j - 1] > hard)
        a["end"] += int(r["top"][j - 1] == im_end and r["ids"][j] != im_end)

    for r in rows.values():
        kinds = classify(r["ids"], tok, sep)
        for j in range(1, len(r["ids"])):
            add("all positions", r, j)
            add(kinds[j], r, j)
            if kinds[j] in MEANINGFUL:
                add("where the teacher's output means something (the first two)", r, j)
            elif kinds[j] != KINDS[6]:
                add("the other positions (separator left out)", r, j)
    out = {}
    for k, a in acc.items():
        n = a["n"]
        out[k] = {"positions": n, "loss_teacher": round(a["nt"] / n, 4), "loss_student": round(a["ns"] / n, 4),
                  "ppl_ratio": round(math.exp((a["ns"] - a["nt"]) / n), 4), "kl": round(a["kl"] / n, 4),
                  "top1_agreement": round(a["agree"] / n, 4), "teacher_bad_share": round(a["bad"] / n, 4),
                  "teacher_predicts_end_of_turn_wrongly": round(a["end"] / n, 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--hard", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dump = json.loads(Path(args.dump).read_text())
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets: dict[str, dict] = {}
    for name, r in dump["rows"].items():
        sets.setdefault(name.split("_")[0], {})[name] = r
    report = {"model": dump.get("model"), "delta": dump.get("delta"), "sets": {}}
    order = ["all positions", "where the teacher's output means something (the first two)", *KINDS, "the other positions (separator left out)"]
    for s, rows in sets.items():
        t = table(rows, tok, dump["sep"], args.hard)
        report["sets"][s] = {"rows": len(rows), "kinds": t}
        print(f"\n== rows of {s[1:]} tokens: {len(rows)} rows  ({dump.get('delta')})")
        print(f"{'':60s} {'n':>7s} {'teacher':>8s} {'student':>8s} {'PPL':>6s} {'KL':>6s} {'agree':>6s} {'bad':>6s} {'end?':>6s}")
        for k in order:
            if k in t:
                x = t[k]
                print(f"{k:60s} {x['positions']:7d} {x['loss_teacher']:8.3f} {x['loss_student']:8.3f} {x['ppl_ratio']:6.3f} {x['kl']:6.3f} "
                      f"{x['top1_agreement']:6.3f} {x['teacher_bad_share']:6.3f} {x['teacher_predicts_end_of_turn_wrongly']:6.3f}")
    (out / "positions_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
