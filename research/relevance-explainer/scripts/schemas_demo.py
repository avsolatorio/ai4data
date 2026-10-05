"""Convert every NADA demo entry (data/nada_demo.jsonl) to a record and print one model input per schema type.

  python scripts/schemas_demo.py [search]
"""
import json
import sys
from collections import Counter

from relevance_explainer.schemas import to_input, to_record

rows = [json.loads(l) for l in open("data/nada_demo.jsonl")]
recs, fails = [], Counter()
for r in rows:
    try:
        recs.append(to_record(r["detail"], r["row"]))
    except Exception as e:
        fails[f"{r['row']['type']}: {type(e).__name__} {e}"] += 1
print("converted", len(recs), "of", len(rows), dict(fails))
cov = Counter((x["schema"], bool(x["countries"]), bool(x["year_start"])) for x in recs)
print("schema, has countries, has years:", dict(cov))
seen = set()
search = sys.argv[1] if len(sys.argv) > 1 else "household survey data Bangladesh 2020"
for x in recs:
    if x["schema"] in seen:
        continue
    seen.add(x["schema"])
    print(f"\n===== {x['schema']} ({x['id']})")
    print(to_input(x, search, sim=0.40))
