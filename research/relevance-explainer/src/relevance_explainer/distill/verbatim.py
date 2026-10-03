"""Re-narrate training examples so that the item's title is quoted verbatim, for the stored verdict.

In non-English narratives the teacher usually translates the item's title and technical terms, which the student
must then reproduce by translating them itself. This writes a narrative that quotes the exact title from the input
("Result:" line) in double quotes and explains in the requested language. The verdict is the stored one. A new
narrative is kept only if it contains the exact title in quotes and is a single paragraph of reasonable length;
otherwise the original narrative is kept.

  TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1 python -m relevance_explainer.distill.verbatim \
      --data data/f6 --n 50000 --out-new data/vb50k --out-old data/old50k
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..synth.generate import NARRATE_SYSTEM
from ..synth.llm import parse_json
from .teacher_checks import MODEL, post, strip_similarity, urls

EXTRA = """

Additional rule for this narrative: refer to the item by its exact title from the "Result:" line, copied character for character in double quotes and not translated, for example: "Urban population (% of total population)". Write everything else in the requested language. Technical names of variables, indicators or databases that you mention should also be copied as written in the input.

The verdict is given; write the narrative for that verdict. Return only a JSON object: {"narrative": "..."}"""

SYSTEM = NARRATE_SYSTEM.rsplit("\n\nReturn only a JSON object", 1)[0] + EXTRA


def title_of(text: str) -> str:
    m = re.search(r"^Result: (.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def valid(narrative: str, title: str) -> bool:
    n = narrative.strip()
    return bool(n) and "\n" not in n and len(n) <= 700 and bool(title) and f'"{title}"' in n.replace("“", '"').replace("”", '"')


def write(base: str, text: str, verdict: str) -> str | None:
    body = {"model": MODEL, "max_tokens": 350, "temperature": 0.3,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"{strip_similarity(text)}\nVerdict: {verdict}"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        n = parse_json(post(base, body)["choices"][0]["message"]["content"])["narrative"].strip()
    except Exception:
        return None
    n = n.replace("“", '"').replace("”", '"')
    return n if valid(n, title_of(text)) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--out-new", required=True, help="dataset with re-narrated non-English examples")
    ap.add_argument("--out-old", required=True, help="same inputs with the original narratives")
    ap.add_argument("--all-languages", action="store_true", help="also re-narrate English examples")
    ap.add_argument("--workers", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = Path(a.data)
    rows = [json.loads(l) for l in (data / "train.jsonl").read_text().splitlines()]
    rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    todo = [i for i, r in enumerate(rows) if (a.all_languages or r["language"] != "English") and title_of(r["input"])]
    servers = urls()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda k: write(servers[k % len(servers)], rows[todo[k]]["input"], rows[todo[k]]["verdict"]), range(len(todo))))
    new = [dict(r) for r in rows]
    kept = 0
    for i, n in zip(todo, res):
        if n:
            new[i]["output"] = new[i]["verdict"] + "\n" + n
            kept += 1
    for out, rs in ((Path(a.out_new), new), (Path(a.out_old), rows)):
        out.mkdir(parents=True, exist_ok=True)
        with (out / "train.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for s in ("val.jsonl", "test.jsonl"):
            shutil.copy(data / s, out / s)
    stats = {"examples": len(rows), "attempted": len(todo), "rewritten": kept, "kept_original": len(todo) - kept}
    (Path(a.out_new) / "verbatim_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
