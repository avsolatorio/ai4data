"""Structure and verification of format-test outputs, per run and test set.

For every run: the share of explanations that state a number or place found in neither the record nor the search
(claims._outside_refs on the plain explanation; comparable across arms). For runs with tags (B, C): output parses,
tl;dr present, grade present and within the verdict's range. For C: claims and references per answer, the share of
references that verify, and the share of answers in which everything verifies.

  python scripts/fmt_structure_report.py runs/fmt-s-A runs/fmt-s-B runs/fmt-s-C --sets realtest_v4 test
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from relevance_explainer.claims import (_GRADE, _outside_refs, from_model, parse_record, QUERY_FIELDS, record_fields,
                                        tldr_of, verify)

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--sets", nargs="+", default=["realtest_v4", "test"])
a = ap.parse_args()


def outside(exp: str, inp: str) -> bool:
    search = "\n".join(f"{k}: {v}" for k, v in parse_record(inp) if k in QUERY_FIELDS)
    record = "\n".join(f"{k}: {v}" for k, v in record_fields(inp).items())
    return bool(_outside_refs(exp, record, search))


def one(path: Path) -> dict:
    st = Counter()
    for line in open(path):
        g = json.loads(line)
        st["n"] += 1
        verdict, _, exp = g["generated"].partition("\n")
        st["outside reference"] += outside(exp, g["input"])
        raw = g.get("raw")
        if not raw or "<|" not in raw:
            continue
        ck = verify(from_model(raw), g["input"])
        st["tagged"] += 1
        st["parses"] += not ck.errors
        st["tl;dr"] += tldr_of(raw) is not None
        st["grade"] += ck.grade is not None
        st["grade in range"] += ck.grade is not None and ck.verdict in _GRADE and \
            _GRADE[ck.verdict][0] <= ck.grade <= _GRADE[ck.verdict][1]
        st["claims"] += len(ck.claims)
        st["claims ok"] += sum(c.ok for c in ck.claims.values())
        st["refs"] += len(ck.refs)
        st["refs ok"] += sum(r.ok for r in ck.refs)
        st["with refs"] += bool(ck.refs)
        st["a ref fails"] += any(not r.ok for r in ck.refs)
        st["all verified"] += ck.all_verified
    return st


def pct(x, n):
    return f"{100 * x / n:.1f}" if n else "-"


print("| run | set | n | outside ref % | parses % | tl;dr % | grade in range % | claims/ans | refs/ans | refs verified % "
      "| answer with a failed ref % | all verified % |")
print("|---" * 12 + "|")
for r in a.runs:
    for s in a.sets:
        p = Path(r) / f"{s}_generations_int8.jsonl"
        if not p.exists():
            continue
        st = one(p)
        n, t = st["n"], st["tagged"]
        print(f"| {Path(r).name} | {s} | {n} | {pct(st['outside reference'], n)} | {pct(st['parses'], t)} | {pct(st['tl;dr'], t)} "
              f"| {pct(st['grade in range'], t)} | {st['claims'] / t if t else 0:.2f} | {st['refs'] / t if t else 0:.2f} "
              f"| {pct(st['refs ok'], st['refs'])} | {pct(st['a ref fails'], t)} | {pct(st['all verified'], t)} |", flush=True)
