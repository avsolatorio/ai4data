"""Check the short explainers of a series against its plan.

A series plan (videos/series/<id>.json) lists the parts and shorts in order.
Each short is a step script videos/<short id>.json with "series" and "next"
fields, and a stage whose end screen names the next short. This script
reports, for every short in the plan, whether its step script exists and, if
it does:

- "series" names the plan and "next" names the following short in the plan
  (the last short of the series has "next": null);
- the end screen of the stage shows the series title and the title of the
  next short;
- the narration has 150 to 230 words;
- no narration line speaks more than one figure (a warning).

    python series.py videos/series/ai-applications.json

Standard library only. Exits with status 1 when an error is found.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORDS = (150, 230)
# A figure: a number with a decimal point or a percent sign, or a number of
# three or more digits that is not a year.
FIGURE = re.compile(r"\b\d+\.\d+\b|\b\d+(?:\.\d+)? ?percent\b|\b\d+%|\b(?!(?:19|20)\d\d\b)\d{1,3}(?:,\d{3})+\b|\b(?!(?:19|20)\d\d\b)\d{3,}\b")


def shorts(plan):
    for part in plan["parts"]:
        for k, short in enumerate(part["shorts"], 1):
            yield part, k, short


def main(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    order = list(shorts(plan))
    errors = warnings = made = 0
    for i, (part, k, short) in enumerate(order):
        sid = short["id"]
        after = order[i + 1][2] if i + 1 < len(order) else None
        script = HERE / "videos" / f"{sid}.json"
        label = f"{part['part']}.{k} {sid}"
        if not script.exists():
            print(f"  -  {label}: not made")
            continue
        made += 1
        problems, notes = [], []
        spec = json.loads(script.read_text(encoding="utf-8"))
        if spec.get("series") != plan["id"]:
            problems.append(f'"series" is {spec.get("series")!r}, expected {plan["id"]!r}')
        expected = after["id"] if after else None
        if spec.get("next") != expected:
            problems.append(f'"next" is {spec.get("next")!r}, expected {expected!r}')
        if spec.get("title") != short["title"]:
            problems.append(f'"title" is {spec.get("title")!r}, the plan has {short["title"]!r}')
        url = next((s["url"] for s in spec["steps"] if s.get("do") == "goto"), None)
        stage = (HERE / url) if url else None
        if not stage or not stage.exists():
            problems.append(f"stage {url!r} not found")
        else:
            html = stage.read_text(encoding="utf-8")
            scenes = re.findall(r'<section class="scene dark"[^>]*>(.*?)</section>', html, re.DOTALL)
            end = scenes[-1] if scenes else ""
            if plan["title"] not in end:
                problems.append("end screen does not show the series title")
            if after and after["title"] not in end:
                problems.append(f"end screen does not name the next short {after['title']!r}")
        says = [s["say"] for s in spec["steps"] if s.get("say")]
        words = sum(len(s.split()) for s in says)
        if not WORDS[0] <= words <= WORDS[1]:
            problems.append(f"{words} words of narration, expected {WORDS[0]} to {WORDS[1]}")
        for s in says:
            found = FIGURE.findall(s)
            if len(found) > 1:
                notes.append(f"{len(found)} figures in one line: {s[:70]}…")
        status = "ERR" if problems else "ok "
        print(f"  {status} {label}: {words} words, {len(says)} lines")
        for p in problems:
            print(f"        error: {p}")
        for n in notes:
            print(f"        warning: {n}")
        errors += len(problems)
        warnings += len(notes)
    print(f"{made} of {len(order)} shorts made; {errors} errors, {warnings} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else HERE / "videos" / "series" / "ai-applications.json"))
