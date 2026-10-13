"""Teacher-written tl;dr and relevance grade for existing (input, verdict, explanation) rows.

For each unique row the teacher writes one sentence that states the deciding reason for the verdict, in the
explanation's language, and a 0-10 grade of how well the item answers the search. Every answer is checked before it
is kept; a failed answer is retried (at most 3 attempts) with the reasons it failed, and rows that never pass are
recorded as failed so that every format arm can leave out the same rows.

Checks: valid JSON; grade an integer within the verdict's range (claims._GRADE); one sentence of at most 30 words
(60 characters in Chinese) and shorter than the explanation; written in the requested language (target script, or
fastText not confidently naming another language); no number or place that is in neither the explanation, the record
nor the search (claims._outside_refs).

Output: one JSON line per row, keyed by the hash of input and output, appended as answers arrive, so an interrupted
run resumes where it stopped.

  LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' python -m relevance_explainer.synth.tldr_grade \
      --out data/tldr_grade/teacher.jsonl data/v2_g_itemfirst_nosim/train.jsonl [--sample 400] [--workers 96]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import llm
from .realistic import language_ok
from ..claims import _GRADE, _outside_refs, parse_record, QUERY_FIELDS, record_fields

TEACHER = ("qwen3.8-27b", ["http://localhost:8000/v1", "http://localhost:8001/v1"])
SYSTEM = "You summarize search-result explanations for a data catalog and grade how well each result answers its search."
PROMPT = """A user searched a data catalog. Below are the search, the catalog record shown as a result, the verdict given for it, and the explanation of that verdict.

{input}

Verdict: {verdict}
Explanation: {explanation}

Write two things.
1. "tldr": one sentence in {language} that states the deciding reason for the verdict, for a reader who sees only this sentence. At most 25 words{cjk}. Use only facts stated in the explanation; no number, place or name that the explanation does not contain. Do not start with "This item" or a similar phrase; state the reason directly.
2. "grade": an integer from 0 to 10 for how well the item answers the search, using this scale:
   0 unrelated topic; 1-2 related field but a different topic; 3-4 right topic but a required place, period or group is missing;
   5-6 partly answers (some requirements met, others missing or only approximate); 7-8 answers the search with minor gaps;
   9-10 answers every part of the search directly.
   The grade must agree with the verdict: irrelevant 0-4, partly_relevant 3-8, relevant 6-10.
{feedback}Return only a JSON object: {{"tldr": "...", "grade": n}}"""


def key(r: dict) -> str:
    return hashlib.md5((r["input"] + "\0" + r["output"]).encode()).hexdigest()


def _words(s: str) -> int:
    return len(re.findall(r"[一-鿿]", s)) / 2 + len(re.findall(r"[^\W\d_]+|\d+", re.sub(r"[一-鿿]", " ", s)))


def check(ans: dict, r: dict) -> list[str]:
    """Reasons the answer fails; empty when it passes."""
    bad = []
    verdict, _, explanation = r["output"].partition("\n")
    verdict, explanation = verdict.strip(), explanation.strip()
    t, g = ans.get("tldr"), ans.get("grade")
    if not isinstance(t, str) or not t.strip():
        return ["no tldr"]
    t = t.strip()
    if not isinstance(g, int) or isinstance(g, bool):
        bad.append("grade is not an integer")
    elif verdict in _GRADE and not _GRADE[verdict][0] <= g <= _GRADE[verdict][1]:
        bad.append(f"grade {g} is outside the range for {verdict} ({_GRADE[verdict][0]}-{_GRADE[verdict][1]})")
    if _words(t) > 30:
        bad.append("tldr longer than 25 words")
    if len(t) >= len(explanation):
        bad.append("tldr not shorter than the explanation")
    if len(re.findall(r"[.!?。！？](\s|$)", t.rstrip(".!?。！？ ") + " ")) > 0:
        bad.append("tldr is more than one sentence")
    if not language_ok(t, r["language"]):
        bad.append(f"tldr is not in {r['language']}")
    search = "\n".join(f"{k}: {v}" for k, v in parse_record(r["input"]) if k in QUERY_FIELDS)
    record = "\n".join(f"{k}: {v}" for k, v in record_fields(r["input"]).items())
    extra = _outside_refs(t, record + "\n" + explanation, search)
    if extra:
        bad.append("tldr states what the explanation does not: " + ", ".join(extra[:3]))
    return bad


def one(r: dict) -> dict:
    verdict, _, explanation = r["output"].partition("\n")
    feedback, attempts = "", []
    for attempt in range(3):
        prompt = PROMPT.format(input=r["input"], verdict=verdict.strip(), explanation=explanation.strip(),
                               language=r["language"], cjk=" (at most 50 characters)" if r["language"] == "Chinese" else "",
                               feedback=feedback)
        try:
            text, _ = llm.chat(TEACHER[0], SYSTEM, prompt, temperature=0.3 if attempt == 0 else 0.7, max_tokens=200,
                               base_urls=TEACHER[1])
            ans = llm.parse_json(text)
            ans = ans if isinstance(ans, dict) else {}
        except Exception as e:  # network failure or unparsable answer: counts as a failed attempt
            ans, text = {}, f"error: {type(e).__name__}"
        bad = check(ans, r)
        attempts.append(bad)
        if not bad:
            return {"key": key(r), "tldr": ans["tldr"].strip(), "grade": ans["grade"], "attempts": attempt + 1}
        feedback = "Your previous answer was rejected: " + "; ".join(bad) + ". Correct it.\n"
    return {"key": key(r), "failed": attempts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="a random sample of this many unique rows (pilot)")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows, seen = [], set()
    for f in a.files:
        for line in open(f):
            r = json.loads(line)
            if key(r) not in seen:
                seen.add(key(r))
                rows.append(r)
    if a.sample:
        rows = random.Random(a.seed).sample(rows, min(a.sample, len(rows)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["key"] for line in open(out)} if out.exists() else set()
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(rows)} unique rows, {len(done)} done before, {len(todo)} to do", flush=True)
    lock, n, failed = threading.Lock(), 0, 0
    with open(out, "a") as fo, ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed(ex.submit(one, r) for r in todo):
            res = fut.result()
            with lock:
                fo.write(json.dumps(res, ensure_ascii=False) + "\n")
                n += 1
                failed += "failed" in res
                if n % 2000 == 0:
                    fo.flush()
                    print(f"{n}/{len(todo)} done, {failed} failed", flush=True)
    print(f"finished: {n} rows, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
