"""Compare multilingual base candidates on held-out bits per byte (last evaluation of each run's log.json).

  python scripts/multibase_compare.py runs/mb-a runs/mb-b ...

Prints parameters, mean bits per byte over all languages, over the explainer's 11 languages, over the smallest third
of languages by planned budget, and per script group, plus the INT8 file size each shape implies.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

OURS = {"eng_Latn", "fra_Latn", "spa_Latn", "por_Latn", "arb_Arab", "cmn_Hani", "rus_Cyrl", "ind_Latn", "hin_Deva",
        "tur_Latn", "swh_Latn"}
plan = json.loads(Path("data/corpus_multi/plan.json").read_text())
small = set(sorted(plan, key=lambda k: plan[k]["chars"])[: len(plan) // 3])
rows = []
for r in sys.argv[1:]:
    p = Path(r) / "log.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    last = d["log"][-1]["heldout_ppl"]
    bpb = {k[:-4]: v for k, v in last.items() if k.endswith("_bpb")}
    script = defaultdict(list)
    for k, v in bpb.items():
        script[k.split("_")[-1]].append(v)
    mean = lambda ks: sum(bpb[k] for k in ks) / max(1, len(ks))  # noqa: E731
    rows.append((Path(r).name, d["params_m"], d["log"][-1]["tokens"], mean(list(bpb)), mean([k for k in bpb if k in OURS]),
                 mean([k for k in bpb if k in small]), {s: sum(v) / len(v) for s, v in script.items()}, len(bpb)))
print(f"{'run':16s} {'params':>7s} {'MB int8':>8s} {'tokens':>7s} {'all':>7s} {'our 11':>7s} {'small':>7s}  (bits per byte, lower is better)")
for name, pm, tok, a, o, s, _, n in rows:
    print(f"{name:16s} {pm:7.2f} {pm * 1.045:8.1f} {tok / 1e9:6.2f}B {a:7.4f} {o:7.4f} {s:7.4f}  ({n} languages)")
scripts = sorted({s for r in rows for s in r[6]}, key=lambda s: -sum(1 for k in plan if k.endswith(s)))
print("\nby script: " + " ".join(f"{s:>6s}" for s in scripts))
for r in rows:
    print(f"{r[0]:16s} " + " ".join(f"{r[6].get(s, float('nan')):6.3f}" for s in scripts))
