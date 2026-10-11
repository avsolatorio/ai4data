"""Draw the human rating sample of docs/rating_protocol.md from the app's answers (scripts/app_answers.py).

  python scripts/rating_sample.py --languages English French Spanish Arabic --teacher 60 --out docs/rating_sample/study1 \\
      --source realistic runs/<run>/app_answers_realtest.jsonl data/eval_real/itemfirst_nosim_v4 50 \\
      --source hard runs/<run>/app_answers_challenge.jsonl data/eval_challenge/itemfirst_nosim_v4 25

Each --source gives a name, the app answers, the test set they come from, and the number of model answers per
language. Within a source and language the answers are split evenly across three confidence bands; a band with too
few answers gives what it has and the rest goes to the other bands. The teacher items come from the first source.

Writes <out>_sheet.csv for raters (no system, score, source or label) and <out>_key.csv kept by the study lead.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BANDS = ((0.0, 0.5, "low"), (0.5, 0.8, "mid"), (0.8, 1.01, "high"))
QUESTIONS = ["q1_verdict_right", "q2_correct_verdict", "q3_statements_supported", "q4_clear_and_language",
             "q5_helpful", "q6_comment"]


def band(s):
    return next(name for lo, hi, name in BANDS if lo <= s < hi)


def load(answers, data):
    rows = [json.loads(l) for l in (Path(data) / "test.jsonl").open()]
    ans = [json.loads(l) for l in open(answers)]
    return rows, [x for x in ans if x["event"] and x["event"].get("type") == "done" and x["event"].get("score") is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs=4, action="append", required=True, metavar=("NAME", "ANSWERS", "DATA", "PER_LANG"))
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--teacher", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    picked = []  # (system, source, rows, answer, weight)
    first_rows = first_ans = None
    for name, answers, data, per_lang in a.source:
        rows, ans = load(answers, data)
        if first_rows is None:
            first_rows, first_ans, first_name = rows, ans, name
        pop = Counter((x["language"], band(x["event"]["score"])) for x in ans)
        for lang in a.languages:
            pool = defaultdict(list)
            for x in ans:
                if x["language"] == lang:
                    pool[band(x["event"]["score"])].append(x)
            want, take = int(per_lang), {}
            left = [b for _, _, b in BANDS]
            while want > 0 and left:  # even split; a short band gives what it has and the rest moves on
                share = max(1, want // len(left))
                for b in sorted(left, key=lambda b: len(pool[b])):
                    k = min(share, len(pool[b]) - take.get(b, 0), want)
                    take[b] = take.get(b, 0) + k
                    want -= k
                left = [b for b in left if take.get(b, 0) < len(pool[b])]
            for b, k in take.items():
                for x in rng.sample(pool[b], k):
                    # weight: answers of this source, language and band that each sampled answer stands for
                    picked.append(("model", name, rows, x, pop[(lang, b)] / k))
    used = {(n, x["i"]) for _, n, _, x, _ in picked}
    rest = [x for x in first_ans if x["language"] in a.languages and (first_name, x["i"]) not in used]
    for x in rng.sample(rest, min(a.teacher, len(rest))):
        picked.append(("teacher", first_name, first_rows, x, None))
    rng.shuffle(picked)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}_sheet.csv", "w", newline="") as fs, open(f"{out}_key.csv", "w", newline="") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["item", "language", "input_seen_by_the_model", "verdict", "explanation"] + QUESTIONS)
        wk.writerow(["item", "source", "test_row", "system", "answer_confidence", "band", "teacher_label", "teachers_agree",
                     "phenomenon", "weight", "retried"])
        for n, (system, src, rows, x, w) in enumerate(picked, 1):
            r, ev = rows[x["i"]], x["event"]
            if system == "teacher":
                verdict, _, expl = r["output"].partition("\n")
            else:
                verdict, expl = ev["verdict"], ev["narrative"]
            ws.writerow([n, x["language"], r["input"], verdict.strip(), expl.strip()] + [""] * len(QUESTIONS))
            model = system == "model"
            wk.writerow([n, src, x["i"], system, ev["score"] if model else "", band(ev["score"]) if model else "",
                         r["verdict"], r.get("unanimous"), r.get("relation") or r.get("mode") or "",
                         "" if w is None else round(w, 3), ev["stats"].get("retried", 0)])
    print(f"{len(picked)} items -> {out}_sheet.csv, {out}_key.csv")
    for k, v in sorted(Counter((s, src, x["language"], band(x["event"]["score"]) if s == "model" else "-")
                               for s, src, _, x, _ in picked).items()):
        print(" ", k, v)


if __name__ == "__main__":
    main()
