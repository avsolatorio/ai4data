"""Render-version-3 copies of the no-score test sets: the same searches, records and labels, with the Understood and
Checks lines from the version-3 understanding (thresholds, exclusions, named periods) and version-2 item lines.
The labels were written against the earlier inputs; the new lines state facts about the same search and record.

  python -m relevance_explainer.synth.eval_v3
  python -m relevance_explainer.synth.eval_v3 --version 4   # _v4 copies (exclusion wording); the challenge suite's
                                                             # version-4 copy is built from its relabelled file instead
  python -m relevance_explainer.synth.eval_v3 --version 5   # _v5 copies (place filters, plural subnational words);
                                                             # the challenge copy starts from its relabelled _v4 rows
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=3)
    v = ap.parse_args().version
    u = Understander()
    sets = dict(SETS)
    if v >= 5:  # the relabelled challenge rows (labeled_v4.jsonl, eval dir _v4) in the same order
        sets["challenge"] = ("data/challenge/labeled_v4.jsonl", "data/eval_challenge/itemfirst_nosim_v4")
    for name, (src, ev) in sets.items():
        if v == 4 and name == "challenge":  # relabelled with the verdict rules: challenge build --labeled labeled_v4.jsonl
            continue
        srows = [json.loads(l) for l in Path(src).read_text().splitlines()]
        erows = [json.loads(l) for l in (Path(ev) / "test.jsonl").read_text().splitlines()]
        assert len(srows) == len(erows), (name, len(srows), len(erows))
        out, changed = [], 0
        for s, e in zip(srows, erows):
            assert f"Search: {s['query']}\n" in e["input"], name
            p = {"query": s["query"], "language": s["language"], "record": s["record"], "facet_values": s.get("facet_values") or {},
                 "render_version": v}
            inp = item_first(render_v2(p, u, u.parse(s["query"], p["facet_values"], level=v), s.get("similarity") or 0.0))
            inp = re.sub(r"\nRetrieval similarity: [-0-9.]+", "", inp)
            changed += inp != e["input"]
            out.append({**e, "input": inp})
        d = Path(ev.removesuffix("_v4") + f"_v{v}")
        d.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
        for split in ("test", "val"):
            (d / f"{split}.jsonl").write_text(text)
        (d / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
        print(f"{name}: {len(out)} rows, input changed in {changed}")


if __name__ == "__main__":
    main()
