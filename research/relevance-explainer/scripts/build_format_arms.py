"""Build the three output-format arms from one dataset and the teacher's tl;dr and grade (synth/tldr_grade.py).

  A  verdict, explanation (the current format)
  B  verdict, grade, tl;dr, explanation
  C  verdict, grade, claims, tl;dr and explanation with references (claims.py)

Every arm holds exactly the same rows, in the same order: those whose tl;dr and grade passed the teacher checks.
Inputs are sanitized (no "<|" can open a tag). In C, a row that occurs several times gets a different permutation of
claim ids each time, and every C output is verified before it is written (a row whose conversion does not verify is
dropped from all arms). Validation and test come from fixed samples of the source splits; --samples-only writes those
samples for the teacher first.

  python scripts/build_format_arms.py data/v2_g_itemfirst_nosim data/fmt/teacher.jsonl data/fmt --val 20000 --test 3000
"""
import argparse
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from relevance_explainer.claims import _u, annotate, from_model, sanitize_input, to_model, verify
from relevance_explainer.synth.tldr_grade import key

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("teacher"); ap.add_argument("out")
ap.add_argument("--val", type=int, default=20000)
ap.add_argument("--test", type=int, default=3000)
ap.add_argument("--workers", type=int, default=96)
ap.add_argument("--samples-only", action="store_true", help="write the validation and test samples (src_val.jsonl, "
                "src_test.jsonl) for the teacher and stop")
a = ap.parse_args()


def sample(split: str, n: int) -> list[dict]:
    rows = [json.loads(line) for line in open(Path(a.src) / f"{split}.jsonl")]
    return random.Random(0).sample(rows, n) if n and len(rows) > n else rows


def convert(job: tuple) -> tuple[str, list[str] | None]:
    """One row to its three arm lines, or the reason it is dropped."""
    r, t, occ = job
    verdict, _, exp = r["output"].partition("\n")
    verdict, exp = verdict.strip(), exp.strip()
    inp = sanitize_input(r["input"])
    c_attr = annotate(verdict, exp, inp, grade=t["grade"], tldr=t["tldr"], seed=None if occ == 1 else occ)
    ck = verify(c_attr, inp)
    if not (all(c.ok for c in ck.claims.values()) and all(x.ok for x in ck.refs)) or from_model(to_model(c_attr)) != c_attr:
        return "C conversion failed", None
    b_attr = f'<verdict grade="{t["grade"]}">{verdict}</verdict>\n<tldr>{t["tldr"]}</tldr>\n<exp>{exp}</exp>'
    base = {k: v for k, v in r.items() if k not in ("input", "output")}
    return "written", [json.dumps({**base, "input": inp, "output": f"{verdict}\n{exp}"}, ensure_ascii=False),
                       json.dumps({**base, "input": inp, "output": to_model(b_attr), "grade": t["grade"]}, ensure_ascii=False),
                       json.dumps({**base, "input": inp, "output": to_model(c_attr), "grade": t["grade"]}, ensure_ascii=False)]


if __name__ == "__main__":
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.samples_only:
        for split, n in (("val", a.val), ("test", a.test)):
            with open(out / f"src_{split}.jsonl", "w") as f:
                for r in sample(split, n):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        raise SystemExit(0)
    tg = {}
    for path in a.teacher.split(","):  # several teacher files, comma-separated
        for line in open(path):
            d = json.loads(line)
            if "tldr" in d:
                tg[d["key"]] = d
    for arm in "ABC":
        (out / arm).mkdir(parents=True, exist_ok=True)
    stats = Counter()
    _u()  # load the gazetteer once; worker processes share it after the fork
    with Pool(a.workers) as pool:
        for split, n in (("train", 0), ("val", a.val), ("test", a.test)):
            jobs, seen = [], Counter()
            for r in sample(split, n):
                t = tg.get(key(r))
                if t is None:
                    stats[f"{split} no tl;dr"] += 1
                    continue
                seen[key(r)] += 1  # a repeated row gets another permutation of claim ids
                jobs.append((r, t, seen[key(r)]))
            files = [open(out / arm / f"{split}.jsonl", "w") for arm in "ABC"]
            for status, lines in pool.imap(convert, jobs, chunksize=256):  # order kept, so the arms stay aligned
                stats[f"{split} {status}"] += 1
                if lines:
                    for f, x in zip(files, lines):
                        f.write(x + "\n")
            for f in files:
                f.close()
            print(split, {k: v for k, v in stats.items() if k.startswith(split)}, flush=True)
    print(json.dumps(stats, indent=1))
