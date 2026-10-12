"""Held-out bits per byte at every evaluation of a pretraining run: all languages, the explainer's 11, the smallest third
of the run's corpus plan, and English.

  python scripts/base_trajectory.py runs/base-final data/corpus_multi_final
"""
import json
import sys
from pathlib import Path

OURS = {"eng_Latn", "fra_Latn", "spa_Latn", "por_Latn", "arb_Arab", "cmn_Hani", "rus_Cyrl", "ind_Latn", "hin_Deva",
        "tur_Latn", "swh_Latn"}
run, root = Path(sys.argv[1]), Path(sys.argv[2])
plan = json.loads((root / "plan.json").read_text())
small = set(sorted(plan, key=lambda k: plan[k]["chars"])[: len(plan) // 3])
d = json.loads((run / "log.json").read_text())
print(f"{'step':>7} {'tokens':>7} {'lr':>8} {'all':>7} {'our 11':>7} {'small':>7} {'eng':>7}")
for e in d["log"]:
    if "heldout_ppl" not in e:
        continue
    bpb = {k[:-4]: v for k, v in e["heldout_ppl"].items() if k.endswith("_bpb")}
    mean = lambda ks: sum(bpb[k] for k in ks) / max(1, len(ks))  # noqa: E731
    print(f"{e['step']:>7} {e['tokens'] / 1e9:>6.1f}B {e.get('lr', float('nan')):>8.1e} {mean(list(bpb)):>7.4f} "
          f"{mean([k for k in bpb if k in OURS]):>7.4f} {mean([k for k in bpb if k in small]):>7.4f} {bpb.get('eng_Latn', float('nan')):>7.4f}")
