#!/usr/bin/env python3
"""MMLU of mq_bench.py by subject group (the grouping of the MMLU repository: categories.py), teacher against student.
The interval is the 95% interval of the paired difference. No GPU.
usage: mmlu_by_subject.py <mmlu_items.json> [out.json]"""
import collections, json, math, sys

SUB = {"abstract_algebra": "math", "anatomy": "health", "astronomy": "physics", "business_ethics": "business", "clinical_knowledge": "health",
       "college_biology": "biology", "college_chemistry": "chemistry", "college_computer_science": "computer science", "college_mathematics": "math",
       "college_medicine": "health", "college_physics": "physics", "computer_security": "computer science", "conceptual_physics": "physics",
       "econometrics": "economics", "electrical_engineering": "engineering", "elementary_mathematics": "math", "formal_logic": "philosophy",
       "global_facts": "other", "high_school_biology": "biology", "high_school_chemistry": "chemistry", "high_school_computer_science": "computer science",
       "high_school_european_history": "history", "high_school_geography": "geography", "high_school_government_and_politics": "politics",
       "high_school_macroeconomics": "economics", "high_school_mathematics": "math", "high_school_microeconomics": "economics",
       "high_school_physics": "physics", "high_school_psychology": "psychology", "high_school_statistics": "math", "high_school_us_history": "history",
       "high_school_world_history": "history", "human_aging": "health", "human_sexuality": "culture", "international_law": "law", "jurisprudence": "law",
       "logical_fallacies": "philosophy", "machine_learning": "computer science", "management": "business", "marketing": "business",
       "medical_genetics": "health", "miscellaneous": "other", "moral_disputes": "philosophy", "moral_scenarios": "philosophy", "nutrition": "health",
       "philosophy": "philosophy", "prehistory": "history", "professional_accounting": "other", "professional_law": "law",
       "professional_medicine": "health", "professional_psychology": "psychology", "public_relations": "politics", "security_studies": "politics",
       "sociology": "culture", "us_foreign_policy": "politics", "virology": "health", "world_religions": "philosophy"}
CAT = {"STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"], "humanities": ["history", "philosophy", "law"],
       "social sciences": ["politics", "culture", "economics", "geography", "psychology"], "other (business, health, misc.)": ["other", "business", "health"]}
CAT_OF = {s: c for c, ss in CAT.items() for s in ss}


def table(items, key):
    d = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for x in items:
        t, s = x["teacher"] == x["gold"], x["student"] == x["gold"]
        r = d[key(x)]
        r[0] += 1; r[1] += t; r[2] += s; r[3] += t and not s; r[4] += s and not t
    out = {}
    for k, (n, t, s, a, b) in sorted(d.items(), key=lambda kv: -(kv[1][1] - kv[1][2]) / kv[1][0]):
        se = math.sqrt(max(a + b - (a - b) ** 2 / n, 0)) / n
        out[k] = {"n": n, "teacher": round(100 * t / n, 1), "student": round(100 * s / n, 1), "difference": round(100 * (t - s) / n, 1),
                  "plus_minus_95": round(196 * se, 1), "only_teacher_right": a, "only_student_right": b}
    return out


items = json.load(open(sys.argv[1]))
res = {"all": table(items, lambda x: "all"), "math_or_not": table(items, lambda x: "math" if SUB[x["subject"]] == "math" else "not math"),
       "category": table(items, lambda x: CAT_OF[SUB[x["subject"]]]), "subcategory": table(items, lambda x: SUB[x["subject"]]),
       "subject": table(items, lambda x: x["subject"])}
print(json.dumps({k: v for k, v in res.items() if k != "subject"}, indent=1))
if len(sys.argv) > 2:
    json.dump(res, open(sys.argv[2], "w"), indent=1)
