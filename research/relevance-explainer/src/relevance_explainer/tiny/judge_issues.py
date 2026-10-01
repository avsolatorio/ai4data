"""Classify what is wrong with unfaithful narratives, using the teacher as judge.

  TEACHER_URLS=http://localhost:8000/v1 python -m relevance_explainer.tiny.judge_issues --run runs/f1-dpo-d --tag ev --prec int8 --n 800
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..distill.teacher_checks import MODEL, post, urls
from ..synth.llm import parse_json

CATEGORIES = {
    "filter_status": "misstates whether a filter is satisfied compared with the Filter check line, or claims to verify a filter that the input does not allow verifying",
    "item_content": "describes what the item measures or contains incorrectly, or adds content that the metadata does not state",
    "query_relation": "misdescribes what the query asks for or how the item relates to it",
    "numbers_names": "wrong or invented number, year, name or place",
    "garbled": "nonsense words, broken grammar or repetition",
    "verdict_mismatch": "argues against its own verdict line",
    "none": "no problem found",
}

SYSTEM = """You grade a short explanation written for a data-catalog search result. You are given the system input (query, filters, retrieval similarity, item metadata, filter check) and the explanation with its verdict line.

Score faithful from 1 to 5: 5 = every claim is supported by the input; 3 = some unsupported or doubtful detail; 1 = mostly invented or contradicting the input.
Then name the single most damaging problem ("primary_issue"), choosing from:
""" + "\n".join(f'- "{k}": {v}' for k, v in CATEGORIES.items()) + """

Return only a JSON object: {"faithful": int, "primary_issue": "<category>"}"""


SHORT_NOTE = """

This explanation is a single sentence shown next to a separate display of the filter results. Do not penalize it for leaving the filters out or for not mentioning them; judge only the claims it makes."""


def grade(base: str, inp: str, text: str, short: bool = False) -> dict | None:
    body = {"model": MODEL, "max_tokens": 80, "temperature": 0.0,
            "messages": [{"role": "system", "content": SYSTEM + (SHORT_NOTE if short else "")}, {"role": "user", "content": f"INPUT:\n{inp}\n\nEXPLANATION:\n{text}"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        j = parse_json(post(base, body)["choices"][0]["message"]["content"])
        issue = j["primary_issue"] if j["primary_issue"] in CATEGORIES else "none"
        return {"faithful": int(j["faithful"]), "issue": issue}
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="ev")
    ap.add_argument("--prec", default="int8")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--short", action="store_true", help="the narrative is a one-sentence rationale (filters shown separately)")
    ap.add_argument("--out", default="issues.json")
    a = ap.parse_args()

    run = Path(a.run)
    rows = [json.loads(l) for l in (run / f"{a.tag}_generations_{a.prec}.jsonl").read_text().splitlines()]
    rows = random.Random(0).sample(rows, min(a.n, len(rows)))
    servers = urls()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda k: grade(servers[k % len(servers)], rows[k]["input"], rows[k]["generated"], a.short), range(len(rows))))

    scored = [(r, s) for r, s in zip(rows, res) if s]
    out = {"n": len(scored), "mean_faithful": sum(s["faithful"] for _, s in scored) / len(scored),
           "share_faithful_le3": sum(s["faithful"] <= 3 for _, s in scored) / len(scored)}
    for name, sel in (("all", lambda s: True), ("faithful_le3", lambda s: s["faithful"] <= 3)):
        for group, g in (("all languages", lambda r: True), ("English", lambda r: r["language"] == "English"),
                         ("other languages", lambda r: r["language"] != "English")):
            c = Counter(s["issue"] for r, s in scored if sel(s) and g(r))
            tot = sum(c.values())
            out[f"{name} / {group}"] = {"n": tot, **{k: round(v / tot, 3) for k, v in c.most_common()}} if tot else {"n": 0}
    (run / a.out).write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(k, round(v, 3) if isinstance(v, float) else v)


if __name__ == "__main__":
    main()
