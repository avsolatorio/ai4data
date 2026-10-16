"""Effect of a render-version change on the evaluation sets, with labels written on both versions' inputs.

  scripts/teachers.sh up && scripts/teachers.sh gemma-up
  python scripts/render_change_relabel.py label --old 4 --new 5 --out data/relabel_v4_v5.jsonl
  scripts/teachers.sh gemma-down && scripts/teachers.sh down
  python scripts/render_change_relabel.py score data/relabel_v4_v5.jsonl --run runs/final-C-dpo --old 4 --new 5

label: for every row of the four evaluation sets whose input differs between data/eval_*/itemfirst_nosim_v<old> and
_v<new> (built by synth/eval_v3.py), the teacher and Gemma 4 judge the row on its old and its new input (NARRATE_V3,
temperature 0, the teacher's thinking off as in the project's label scripts). score: the run's verdicts on the old
and new inputs (tags <set>_v<old> and <set>_v<new>, tiny/evaluate.py) against the stored labels and each judge's
labels, with McNemar's exact test (notebook 2026-10-16).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb

sys.path.insert(0, "src")
SETS = {"realtest": ("data/real_test/examples_labeled.jsonl", "data/eval_real/itemfirst_nosim"),
        "para": ("data/para_test/examples.jsonl", "data/eval_para/itemfirst_nosim"),
        "challenge": ("data/challenge/labeled_v4.jsonl", "data/eval_challenge/itemfirst_nosim"),
        "schemas": ("data/schemas_eval2/labeled.jsonl", "data/schemas_eval2/itemfirst_nosim")}
VERDICTS = ("relevant", "partly_relevant", "irrelevant")


def label(old: int, new: int, out: str, workers: int) -> None:
    os.environ.setdefault("LLM_EXTRA", '{"chat_template_kwargs":{"enable_thinking":false}}')
    os.environ["LLM_REASONING"] = ""
    from relevance_explainer.synth import llm
    from relevance_explainer.synth.challenge import GEMMA, TEACHER
    from relevance_explainer.synth.realistic import NARRATE_V3, render_v2
    from relevance_explainer.synth.understand import Understander
    u = Understander()
    jobs = []
    for name, (src, ev) in SETS.items():
        srows = [json.loads(l) for l in open(src)]
        a = [json.loads(l) for l in open(f"{ev}_v{old}/test.jsonl")]
        b = [json.loads(l) for l in open(f"{ev}_v{new}/test.jsonl")]
        for i, (s, x, y) in enumerate(zip(srows, a, b)):
            if x["input"] == y["input"]:
                continue
            row = {"set": name, "i": i, "language": s["language"], "query": s["query"], "label": x["output"].split("\n", 1)[0]}
            for v in (old, new):  # the full input the labels were written on (with the similarity line)
                p = {"query": s["query"], "language": s["language"], "record": s["record"],
                     "facet_values": s.get("facet_values") or {}, "render_version": v}
                row[f"input_v{v}"] = render_v2(p, u, u.parse(s["query"], p["facet_values"], level=v), s.get("similarity") or 0.0)
            jobs.append(row)
    print(len(jobs), "changed rows", flush=True)

    def verdict(model, text):
        for _ in range(3):
            try:
                j = llm.parse_json(llm.chat(model[0], NARRATE_V3, text, temperature=0.0, max_tokens=450, base_urls=model[1])[0])
            except Exception:  # noqa: BLE001  a failed call is retried, then left unlabelled
                continue
            if j.get("verdict") in VERDICTS:
                return j["verdict"]
        return None

    def one(r):
        return {**r, **{f"{j}_v{v}": verdict(m, r[f"input_v{v}"]) for j, m in (("teacher", TEACHER), ("gemma", GEMMA))
                        for v in (old, new)}}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs))
    with open(out, "w") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print({k: sum(r[k] is not None for r in rows) for k in rows[0] if k.startswith(("teacher_", "gemma_"))} if rows else {})


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def score(path: str, run: str, old: int, new: int) -> None:
    gen = {(name, v): [json.loads(l)["generated"].split("\n", 1)[0].strip()
                       for l in open(f"{run}/{name}_v{v}_generations_int8.jsonl")] for name in SETS for v in (old, new)}
    rows = [json.loads(l) for l in open(path)]
    refs = [("stored label", "label")] + [(f"{j} on v{v} input", f"{j}_v{v}") for j in ("teacher", "gemma") for v in (old, new)]
    for title, key in refs:
        rs = [r for r in rows if r.get(key)]
        mo = [gen[(r["set"], old)][r["i"]] == r[key] for r in rs]
        mn = [gen[(r["set"], new)][r["i"]] == r[key] for r in rs]
        b, c = sum(y and not x for x, y in zip(mo, mn)), sum(x and not y for x, y in zip(mo, mn))
        print(f"{title:20s} n={len(rs):4d}  on v{old} inputs {sum(mo) / len(rs):.1%}  on v{new} inputs {sum(mn) / len(rs):.1%}  "
              f"v{new} better {b}, worse {c}, McNemar p={mcnemar(b, c):.3f}")
    for j in ("teacher", "gemma"):
        rs = [r for r in rows if r.get(f"{j}_v{old}") and r.get(f"{j}_v{new}")]
        print(f"{j}: label differs between the two inputs on {sum(r[f'{j}_v{old}'] != r[f'{j}_v{new}'] for r in rs)} of {len(rs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label")
    lb.add_argument("--old", type=int, default=4)
    lb.add_argument("--new", type=int, default=5)
    lb.add_argument("--out", required=True)
    lb.add_argument("--workers", type=int, default=32)
    sc = sub.add_parser("score")
    sc.add_argument("labels")
    sc.add_argument("--run", default="runs/final-C-dpo")
    sc.add_argument("--old", type=int, default=4)
    sc.add_argument("--new", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "label":
        label(a.old, a.new, a.out, a.workers)
    else:
        score(a.labels, a.run, a.old, a.new)


if __name__ == "__main__":
    main()
