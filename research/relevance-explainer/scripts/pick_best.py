"""Print the run directory with the highest best-epoch validation verdict accuracy (metrics.json), and each run's value
on stderr. Selection uses validation data only.

  python scripts/pick_best.py runs/a runs/b runs/c
"""
import json
import sys
from pathlib import Path

scores = {}
for r in sys.argv[1:]:
    m = Path(r) / "metrics.json"
    if m.exists():
        log = json.loads(m.read_text())["train_log"]
        scores[r] = max(e["val_verdict_acc"] for e in log)
        print(f"{r}: val verdict accuracy {scores[r]:.4f} (by epoch: {[round(e['val_verdict_acc'], 4) for e in log]})", file=sys.stderr)
print(max(scores, key=scores.get))
