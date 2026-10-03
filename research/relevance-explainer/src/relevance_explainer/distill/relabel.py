"""Relabel and re-narrate training examples under a tightened verdict definition.

The data audit (lab notebook, 2026-10-03 17:03 UTC) found that the teacher often calls an item partly relevant when
it matches a filter (usually the country) but addresses a different topic. This rewrites the verdict definitions so
that topic decides first, and lets the teacher write a new verdict and narrative for each input.

  TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1 python -m relevance_explainer.distill.relabel \
      --data data/old50k --out data/tight50k
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..synth.generate import NARRATE_SYSTEM
from ..synth.llm import parse_json
from .teacher_checks import MODEL, post, strip_similarity, urls

OLD_DEFS = NARRATE_SYSTEM[NARRATE_SYSTEM.index("Verdict definitions:") : NARRATE_SYSTEM.index("Narrative rules:")]
NEW_DEFS = """Verdict definitions (decide the topic first, then the filters):
- relevant: the item addresses what the query asks for and satisfies the filters that can be checked.
- partly_relevant: the item addresses the topic the query asks for (the same or a closely related concept), but covers only part of what was asked (population group, geography, period, frequency, level of detail) or fails a filter.
- irrelevant: the item addresses a different topic from what the query asks for. Matching a filter (country, years, frequency) or sharing only a broad theme does not make an item partly relevant when what it measures or contains differs from what the query asks for.

"""
SYSTEM = NARRATE_SYSTEM.replace(OLD_DEFS, NEW_DEFS)
assert SYSTEM != NARRATE_SYSTEM


def narrate(base: str, text: str) -> tuple[str, str] | None:
    body = {"model": MODEL, "max_tokens": 400, "temperature": 0.2,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": strip_similarity(text)}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        j = parse_json(post(base, body)["choices"][0]["message"]["content"])
        v, n = str(j["verdict"]).strip(), str(j["narrative"]).strip()
    except Exception:
        return None
    return (v, n) if v in ("relevant", "partly_relevant", "irrelevant") and n and "\n" not in n else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=192)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.data, f"{a.split}.jsonl").read_text().splitlines()]
    servers = urls()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda k: narrate(servers[k % len(servers)], rows[k]["input"]), range(len(rows))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    changes, failed = Counter(), 0
    with (out / f"{a.split}.jsonl").open("w") as f:
        for r, x in zip(rows, res):
            r = dict(r)
            if x is None:
                failed += 1  # keep the original label and narrative
            else:
                changes[(r["verdict"], x[0])] += 1
                r["verdict"], r["output"] = x[0], f"{x[0]}\n{x[1]}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("val.jsonl", "test.jsonl"):
        if s != f"{a.split}.jsonl" and Path(a.data, s).exists():
            shutil.copy(Path(a.data, s), out / s)
    stats = {"examples": len(rows), "failed_kept_original": failed,
             "old_to_new": {f"{o}->{n}": c for (o, n), c in sorted(changes.items())}}
    (out / "relabel_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
