"""Render-version-3 copies of the no-score test sets: the same searches, records and labels, with the Understood and
Checks lines from the version-3 understanding (thresholds, exclusions, named periods) and version-2 item lines.
The labels were written against the earlier inputs; the new lines state facts about the same search and record.

  python -m relevance_explainer.synth.eval_v3
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .realistic import item_first, render_v2
from .understand import Understander

# test set -> (source rows with query and record, existing no-score eval dir); sources and eval rows are in the same order
SETS = {
    "real": ("data/real_test/examples_labeled.jsonl", "data/eval_real/itemfirst_nosim"),
    "para": ("data/para_test/examples.jsonl", "data/eval_para/itemfirst_nosim"),
    "challenge": ("data/challenge/labeled.jsonl", "data/eval_challenge/itemfirst_nosim"),
    "schemas": ("data/schemas_eval2/labeled.jsonl", "data/schemas_eval2/itemfirst_nosim"),
}


def main() -> None:
    u = Understander()
    for name, (src, ev) in SETS.items():
        srows = [json.loads(l) for l in Path(src).read_text().splitlines()]
        erows = [json.loads(l) for l in (Path(ev) / "test.jsonl").read_text().splitlines()]
        assert len(srows) == len(erows), (name, len(srows), len(erows))
        out, changed = [], 0
        for s, e in zip(srows, erows):
            assert f"Search: {s['query']}\n" in e["input"], name
            p = {"query": s["query"], "language": s["language"], "record": s["record"], "facet_values": s.get("facet_values") or {},
                 "render_version": 3}
            inp = item_first(render_v2(p, u, u.parse(s["query"], p["facet_values"], level=3), s.get("similarity") or 0.0))
            inp = re.sub(r"\nRetrieval similarity: [-0-9.]+", "", inp)
            changed += inp != e["input"]
            out.append({**e, "input": inp})
        d = Path(ev + "_v3")
        d.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
        for split in ("test", "val"):
            (d / f"{split}.jsonl").write_text(text)
        (d / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
        print(f"{name}: {len(out)} rows, input changed in {changed}")


if __name__ == "__main__":
    main()
