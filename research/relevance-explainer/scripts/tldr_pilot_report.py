"""Report on a tl;dr and grade run (synth/tldr_grade.py) and gate the full run on it: pass rate, attempts, failure
reasons, the grade distribution per verdict and examples per language. Exits with status 1 when fewer than
--min-pass of the rows passed, or when one grade value takes more than --max-share of a verdict's rows (a grade that
does not vary carries no information).

  python scripts/tldr_pilot_report.py data/fmt/pilot.jsonl data/v2_g_itemfirst_nosim/train.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

from relevance_explainer.synth.tldr_grade import key

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("rows", nargs="+")
ap.add_argument("--min-pass", type=float, default=0.9)
ap.add_argument("--max-share", type=float, default=0.7)
a = ap.parse_args()

res = {json.loads(line)["key"]: json.loads(line) for line in open(a.results)}
rows = {}
for f in a.rows:
    for line in open(f):
        r = json.loads(line)
        if key(r) in res:
            rows[key(r)] = r
ok = [k for k, v in res.items() if "tldr" in v]
print(f"{len(res)} rows, passed {len(ok)} ({len(ok) / len(res):.1%})")
print("attempts of passed rows:", dict(Counter(res[k]["attempts"] for k in ok)))
reasons = Counter(b.split(":")[0] for v in res.values() for att in v.get("failed", []) for b in att)
print("failure reasons (all attempts of failed rows):", dict(reasons.most_common(8)))
grades = defaultdict(Counter)
for k in ok:
    if k in rows:
        grades[rows[k]["output"].split("\n")[0].strip()][res[k]["grade"]] += 1
flat = False
for v, c in grades.items():
    n = sum(c.values())
    top = c.most_common(1)[0][1] / n
    flat |= top > a.max_share
    print(f"grades for {v} (n {n}): {dict(sorted(c.items()))}; most common share {top:.0%}")
shown = set()
for k in ok:
    r = rows.get(k)
    if r and r["language"] not in shown:
        shown.add(r["language"])
        print(f"\n[{r['language']}] {r['output'].split(chr(10))[0]} grade {res[k]['grade']}: {res[k]['tldr']}")
fail = len(ok) / len(res) < a.min_pass or flat
print("\nGATE", "FAILED" if fail else "PASSED")
sys.exit(1 if fail else 0)
