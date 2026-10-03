"""Audit training examples with a second model: its own verdict, the faithfulness of the stored narrative, and the
plausibility of the query.

The second model gets the teacher's narration prompt (same verdict definitions, input without the similarity line,
as the teacher saw it) and returns a verdict; separately it grades the stored narrative with the evaluation judge's
rubric, and rates the query.

  LLM_BASE_URL=http://localhost:8002/v1 python -m relevance_explainer.distill.audit --data data/f6 --n 800 \
      --model gemma-4-31b
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..synth import llm
from ..synth.generate import NARRATE_SYSTEM
from ..tiny.judge import judge
from .teacher_checks import strip_similarity

QUERY_SYSTEM = """You review a search that a user typed into a data catalog of development statistics, survey microdata, research papers and humanitarian datasets, with the filters they applied.

Rate:
- plausible (1-5): 5 = a real person could well type this search with these filters; 3 = understandable but unnatural or oddly specific; 1 = not something a person would search.
- coherent_filters: true if the filters fit the query (for example the query does not name a country that contradicts the country filter).

Return only a JSON object: {"plausible": int, "coherent_filters": bool}"""


def own_verdict(model: str, text: str) -> str | None:
    try:
        out, _ = llm.chat(model, NARRATE_SYSTEM, strip_similarity(text), temperature=0.0, max_tokens=300)
        v = str(llm.parse_json(out)["verdict"]).strip()
        return v if v in ("relevant", "partly_relevant", "irrelevant") else None
    except Exception:
        return None


def query_check(model: str, text: str) -> dict | None:
    lines = [l for l in text.split("\n") if l.startswith(("Query:", "Filters:", "Respond in:"))]
    try:
        out, _ = llm.chat(model, QUERY_SYSTEM, "\n".join(lines), temperature=0.0, max_tokens=60)
        j = llm.parse_json(out)
        return {"plausible": int(j["plausible"]), "coherent_filters": bool(j["coherent_filters"])}
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--model", default="gemma-4-31b")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.data, f"{a.split}.jsonl").read_text().splitlines()]
    idx = random.Random(a.seed).sample(range(len(rows)), min(a.n, len(rows)))

    def one(i: int) -> dict:
        r = rows[i]
        return {"i": i, "language": r["language"], "source": r["source"], "stored": r["verdict"],
                "second": own_verdict(a.model, r["input"]), "narrative": judge(a.model, r["input"], r["output"]),
                "query": query_check(a.model, r["input"])}

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, idx))
    out = Path(a.data, f"audit_{a.split}_{a.model}_{a.n}.jsonl")
    with out.open("w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")

    v = [x for x in res if x["second"]]
    print(f"verdicts: {len(v)} parsed; agreement {sum(x['stored'] == x['second'] for x in v) / len(v):.3f}")
    print("stored -> second:", dict(sorted(Counter((x["stored"], x["second"]) for x in v).items())))
    for key in ("source", "language"):
        groups = Counter(x[key] for x in v)
        print(f"agreement by {key}:", {g: round(sum(x["stored"] == x["second"] for x in v if x[key] == g) / c, 3) for g, c in groups.most_common()})
    s = [x["narrative"] for x in res if x["narrative"]]
    print(f"narratives: {len(s)} judged; faithful {sum(t['faithful'] for t in s) / len(s):.3f}; "
          f"faithful<=3 {sum(t['faithful'] <= 3 for t in s) / len(s):.3f}; consistent {sum(t['consistent'] for t in s) / len(s):.3f}; "
          f"language_ok {sum(t['language_ok'] for t in s) / len(s):.3f}")
    q = [x["query"] for x in res if x["query"]]
    print(f"queries: {len(q)} rated; plausible mean {sum(t['plausible'] for t in q) / len(q):.2f}; "
          f"plausible<=2 {sum(t['plausible'] <= 2 for t in q) / len(q):.3f}; coherent filters {sum(t['coherent_filters'] for t in q) / len(q):.3f}")


if __name__ == "__main__":
    main()
