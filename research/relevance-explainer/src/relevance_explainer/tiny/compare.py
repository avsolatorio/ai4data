"""Paired comparison of two runs judged on the same examples (judge files with per-example "items").

Reports, for each judge, the mean faithfulness and coherence of each run's student narratives, the mean paired
difference with a 95% bootstrap interval, and how many examples got better or worse.

  python -m relevance_explainer.tiny.compare runs/f5-dpo runs/f5-grpo --suffixes "" _gemma4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def student_scores(run: Path, tag: str, prec: str, suffix: str) -> dict[int, dict]:
    items = json.loads((run / f"judge_{tag}_{prec}{suffix}.json").read_text())["items"]
    return {x["k"]: x["score"] for x in items if x["who"] == "student" and x["score"]}


def paired(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"n": len(d), "a": sum(a) / len(a), "b": sum(b) / len(b), "diff": sum(d) / len(d),
            "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]],
            "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tag", default="f2test")
    ap.add_argument("--prec", default="int8")
    ap.add_argument("--suffixes", nargs="+", default=["", "_gemma4"])
    a = ap.parse_args()
    for suf in a.suffixes:
        sa, sb = student_scores(Path(a.a), a.tag, a.prec, suf), student_scores(Path(a.b), a.tag, a.prec, suf)
        keys = sorted(set(sa) & set(sb))
        for metric in ("faithful", "coherent"):
            r = paired([sa[k][metric] for k in keys], [sb[k][metric] for k in keys])
            print(f"judge{suf or '_teacher'} {metric}: {r['a']:.3f} -> {r['b']:.3f}  diff {r['diff']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  better {r['better']} worse {r['worse']} (n {r['n']})")


if __name__ == "__main__":
    main()
