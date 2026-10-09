"""Compare two labellings of the same challenge searches: teacher-Gemma 4 agreement per phenomenon, label distributions
and how the teacher's labels moved.

  python scripts/label_compare.py data/challenge/labeled.jsonl data/challenge/labeled_v4.jsonl
"""
import json
import sys
from collections import Counter, defaultdict

A = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
B = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
ids = [i for i in A if i in B]
by = defaultdict(list)
for i in ids:
    by[A[i]["phenomenon"]].append(i)
S = {"relevant": "R", "partly_relevant": "P", "irrelevant": "I", None: "-"}
print(f"{'phenomenon':17s} {'n':>4s}  agree old  new   teacher labels old -> new          moved")
tot = Counter()
for ph, xs in sorted(by.items()):
    ao = sum(A[i]["gemma4"] == A[i]["verdict"] for i in xs) / len(xs)
    an = sum(B[i]["gemma4"] == B[i]["verdict"] for i in xs) / len(xs)
    do = Counter(S[A[i]["verdict"]] for i in xs)
    dn = Counter(S[B[i]["verdict"]] for i in xs)
    mv = Counter(f"{S[A[i]['verdict']]}>{S[B[i]['verdict']]}" for i in xs if A[i]["verdict"] != B[i]["verdict"])
    tot["n"] += len(xs)
    tot["ao"] += ao * len(xs)
    tot["an"] += an * len(xs)
    fmt = lambda c: " ".join(f"{k}{c[k]}" for k in "RPI")
    print(f"{ph:17s} {len(xs):4d}  {ao:9.2f} {an:4.2f}   {fmt(do):16s} -> {fmt(dn):16s} {dict(mv.most_common(4))}")
print(f"{'all':17s} {tot['n']:4d}  {tot['ao'] / tot['n']:9.2f} {tot['an'] / tot['n']:4.2f}")
