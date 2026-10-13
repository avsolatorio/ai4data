"""Convert explanations to the claim-carrying format (relevance_explainer/claims.py) and report what the conversion
found: claims and references per example, the share of examples with at least one reference, whether every converted
example verifies and gives back its original text, and the numbers and places the original text states outside the
record.

  python scripts/claims_annotate.py data/eval_real/itemfirst_nosim_v4/test.jsonl [out.jsonl] [--limit N]
"""
import argparse
import json
from collections import Counter

from relevance_explainer.claims import annotate, strip_refs, verify

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out", nargs="?")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.src)]
rows = rows[: a.limit] if a.limit else rows
st, fields, reasons, examples = Counter(), Counter(), Counter(), []
out = open(a.out, "w") if a.out else None
for r in rows:
    verdict, _, narrative = r["output"].partition("\n")
    s = annotate(verdict.strip(), narrative.strip(), r["input"])
    ck = verify(s, r["input"])
    st["n"] += 1
    st["claims"] += len(ck.claims)
    st["refs"] += len(ck.refs)
    st["with_ref"] += bool(ck.refs)
    st["round_trip"] += ck.sections.get("exp") == narrative.strip() and strip_refs(s).count("<r") == 0
    st["claims_ok"] += all(c.ok for c in ck.claims.values())
    st["refs_ok"] += all(x.ok for x in ck.refs)
    st["unverified"] += bool(ck.unverified)
    st["flagged"] += bool(ck.flags)
    fields.update(c.field for c in ck.claims.values())
    reasons.update(x.reason for x in ck.refs if not x.ok)
    reasons.update(c.reason for c in ck.claims.values() if not c.ok)
    if (not all(x.ok for x in ck.refs) or ck.unverified) and len(examples) < 6:
        examples.append((s, [(x.claim, x.text, x.reason) for x in ck.refs if not x.ok], ck.unverified))
    if out:
        out.write(json.dumps({**r, "output": s}, ensure_ascii=False) + "\n")
n = st["n"]
print(f"{n} examples: claims {st['claims'] / n:.2f} and references {st['refs'] / n:.2f} per example; "
      f"with a reference {st['with_ref'] / n:.1%}; text recovered {st['round_trip'] / n:.1%}; "
      f"all claims verified {st['claims_ok'] / n:.1%}; all references verified {st['refs_ok'] / n:.1%}; "
      f"unverified numbers or places outside references {st['unverified'] / n:.1%}; consistency flag {st['flagged'] / n:.1%}")
print("claim fields:", dict(fields.most_common(12)))
print("failure reasons:", dict(reasons.most_common(8)))
for s, bad, unv in examples:
    print("\n---\n" + s[:900], "\nfailed references:", bad, "\nunverified:", unv)
