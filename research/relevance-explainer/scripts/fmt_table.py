"""Verdict accuracy (int8) of format-test runs on the version-4 test sets, the Eurostat tables and the held-out test.

  python scripts/fmt_table.py runs/v2g-from-dapt-4ep runs/fmt-A-full runs/fmt-s-A runs/fmt-s-B runs/fmt-s-C
"""
import json
import sys
from pathlib import Path


def get(run: Path, name: str):
    p = run / name
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    m = m.get("int8", m.get("precision", {}).get("int8", m))
    return m.get("verdict_accuracy"), m.get("f1", {}).get("macro")


cols = [("realtest_v4_metrics.json", "realistic"), ("challenge_v4_metrics.json", "challenge"), ("para_v4_metrics.json", "paraphrase"),
        ("schemas_v4_metrics.json", "schemas"), ("tables_metrics.json", "tables"), ("metrics.json", "held-out test")]
print("| run | " + " | ".join(c for _, c in cols) + " |")
print("|---" * (len(cols) + 1) + "|")
for r in sys.argv[1:]:
    cells = []
    for f, _ in cols:
        v = get(Path(r), f)
        cells.append("-" if v is None or v[0] is None else f"{100 * v[0]:.1f} ({100 * (v[1] or 0):.1f})")
    print(f"| {Path(r).name} | " + " | ".join(cells) + " |")
print("cells: verdict accuracy (macro F1), int8")
