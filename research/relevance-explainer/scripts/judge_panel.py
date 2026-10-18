"""Open-weights models as judges of the evaluation sets: each model's verdict on the same inputs, and how the models
agree with each other and with the references each set has (notebook 2026-10-18).

  python -m relevance_explainer.synth.models serve <name>        (one server per model, synth/models.py)
  python scripts/judge_panel.py label <name> [--sets challenge,realtest,unseen] [--out data/judge_panel]
  python scripts/judge_panel.py report [--out data/judge_panel]

label: NARRATE_V3 at temperature 0 with the model's request fields (reasoning off, or "low" for gpt-oss), on
  challenge  data/challenge/labeled_v4.jsonl, its stored input (2,244 searches; reference: the verdict the case was
             written for, writer_expected)
  realtest   data/real_test/examples_labeled.jsonl, its stored input (1,495; references: the stored label and the
             three-label majority)
  unseen     evalsets/unseen_catalogs, the 45 supported-language searches rendered as the app renders them (render
             version 5, opendata item type, no similarity line; reference: the set's label, also_ok accepted)
Writes <out>/<name>.jsonl, one row per (set, id) with the verdict, whether the explanation is in the requested
language (synth.realistic.language_ok) and the finish state; rows already present are kept, so a rerun resumes.
report: per set, each model's agreement with the references, Cohen's kappa between models, accuracy by language on
realtest, and the majority of the models present.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "src")
VERDICTS = ("relevant", "partly_relevant", "irrelevant")
UNSEEN = Path("evalsets/unseen_catalogs")


def items(sets: list[str]) -> list[dict]:
    out = []
    if "challenge" in sets:
        for r in map(json.loads, open("data/challenge/labeled_v4.jsonl", encoding="utf-8")):
            out.append({"set": "challenge", "id": r["id"], "language": r["language"], "input": r["input_v2"],
                        "refs": {"designed": r["writer_expected"], "stored": r["verdict"]}})
    if "realtest" in sets:
        for i, r in enumerate(map(json.loads, open("data/real_test/examples_labeled.jsonl", encoding="utf-8"))):
            out.append({"set": "realtest", "id": str(i), "language": r["language"], "input": r["input_v2"],
                        "refs": {"stored": r["verdict"], "majority": r["majority"]}})
    if "unseen" in sets:
        from relevance_explainer import inference
        from relevance_explainer.synth.understand import Understander
        u = Understander()
        recs = {r["id"]: r for r in map(json.loads, open(UNSEEN / "records.jsonl", encoding="utf-8"))}
        for c in csv.DictReader(open(UNSEEN / "cases.tsv", encoding="utf-8"), delimiter="\t"):
            if c["group"] == "unsupported":
                continue
            text = inference.render(recs[c["record"]], c["search"], c["language"], understander=u, render_version=5)
            out.append({"set": "unseen", "id": f"{c['record']}|{c['search']}", "language": c["language"], "input": text,
                        "refs": {"label": c["label"], "also_ok": c["also_ok"]}})
    return out


def _last_verdict_json(text: str) -> dict:
    """The last JSON object with a verdict in the text: a model served without a reasoning parser writes its reasoning
    before the answer (Nemotron with thinking on). Only used when llm.parse_json finds no verdict."""
    dec = json.JSONDecoder()
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "{":
            try:
                j, _ = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(j, dict) and j.get("verdict") in VERDICTS:
                return j
    return {}


def _up(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def label(name: str, sets: list[str], out: Path, workers: int, request: dict | None = None, tag: str | None = None,
          max_tokens: int | None = None, sample: int = 0) -> None:
    from relevance_explainer.synth import llm
    from relevance_explainer.synth.models import MODELS
    from relevance_explainer.synth.realistic import NARRATE_V3, language_ok
    m = MODELS[name]
    request = m["request"] if request is None else request
    os.environ.pop("LLM_EXTRA", None)  # the model's own request fields only
    os.environ.pop("LLM_REASONING", None)
    path = out / f"{tag or name}.jsonl"
    # Rows without a verdict are retried; the report reads the last row of each (set, id).
    done = {(r["set"], r["id"]) for r in map(json.loads, open(path, encoding="utf-8")) if r["verdict"]} if path.exists() else set()
    pool = items(sets)
    if sample:  # the same seeded sample of each set for every model and setting
        import random
        pool = [x for s in sets for x in random.Random(0).sample([y for y in pool if y["set"] == s],
                                                                  min(sample, sum(y["set"] == s for y in pool)))]
    todo = [x for x in pool if (x["set"], x["id"]) not in done]
    urls = [u for u in m["urls"] if _up(u)]  # the teacher has two servers; one may be paused for training
    if not urls:
        sys.exit(f"{name}: no server answers at {m['urls']}")
    print(f"{name}: {len(todo)} to label, {len(done)} already labelled, servers {urls}", flush=True)

    def one(x: dict) -> dict:
        t0, row = time.time(), {"set": x["set"], "id": x["id"], "language": x["language"], "verdict": None}
        for attempt in range(3):
            try:
                text, usage = llm.chat(name, NARRATE_V3, x["input"], temperature=0.0, max_tokens=max_tokens or m.get("max_tokens", 450),
                                       base_urls=urls, extra=request, retries=2)
                try:
                    j = llm.parse_json(text)
                except json.JSONDecodeError:
                    j = {}
                if not isinstance(j, dict) or j.get("verdict") not in VERDICTS:
                    j = _last_verdict_json(text)
            except Exception as e:  # noqa: BLE001  recorded, retried, then left without a verdict
                row["error"] = str(e)[:200]
                continue
            if j.get("verdict") in VERDICTS:
                narrative = str(j.get("narrative") or j.get("explanation") or "")
                row.update(verdict=j["verdict"], language_ok=language_ok(narrative, x["language"]) if narrative else None,
                           tokens=usage.get("completion_tokens"), attempts=attempt + 1)
                row.pop("error", None)
                break
            row["error"] = f"no verdict in {text[:120]!r}"
        row["seconds"] = round(time.time() - t0, 2)
        return row

    out.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for n, row in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 250 == 0:
                f.flush()
                print(f"  {n}/{len(todo)}", flush=True)
    last = {(r["set"], r["id"]): r for r in map(json.loads, open(path, encoding="utf-8"))}
    print(f"{tag or name}: {sum(r['verdict'] is not None for r in last.values())} of {len(last)} with a verdict")


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[v] * cb[v] for v in VERDICTS) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report(out: Path, only: list[str] | None = None) -> None:
    refs = {(x["set"], x["id"]): x for x in items(["challenge", "realtest", "unseen"])}
    judged = {p.stem: {(r["set"], r["id"]): r for r in map(json.loads, open(p, encoding="utf-8"))}
              for p in sorted(out.glob("*.jsonl")) if not only or p.stem in only}
    names = list(judged)
    for s in ("challenge", "realtest", "unseen"):
        keys = [k for k in refs if k[0] == s]
        if not any(k in judged[n] for n in names for k in keys):
            continue
        print(f"\n== {s} ({len(keys)} searches)")
        ref_names = list(refs[keys[0]]["refs"])
        print(f"{'model':24s} {'verdicts':>9s} {'lang ok':>8s} {'s/answer':>8s}  " +
              "  ".join(f"vs {r:>8s}" for r in ref_names if r != "also_ok"))
        votes = defaultdict(list)
        for n in names:
            rows = [judged[n][k] for k in keys if k in judged[n]]
            got = [r for r in rows if r["verdict"]]
            for r in got:
                votes[(r["set"], r["id"])].append(r["verdict"])
            if not rows:
                continue
            lo = [r["language_ok"] for r in got if r.get("language_ok") is not None]
            cells = []
            for rn in ref_names:
                if rn == "also_ok":
                    continue
                if s == "unseen":
                    ok = sum(r["verdict"] in (refs[(s, r["id"])]["refs"]["label"], refs[(s, r["id"])]["refs"]["also_ok"]) for r in got)
                    cells.append(f"lenient {ok}/{len(got)}")
                else:
                    ok = sum(r["verdict"] == refs[(s, r["id"])]["refs"][rn] for r in got)
                    cells.append(f"{ok / len(got):10.1%}")
            print(f"{n:24s} {len(got):5d}/{len(keys):<4d} {sum(lo) / max(len(lo), 1):8.1%} "
                  f"{sum(r['seconds'] for r in rows) / len(rows):8.2f}  " + "  ".join(cells))
        maj = {k: Counter(v).most_common(1)[0][0] for k, v in votes.items() if len(v) == len(names) and len(names) >= 3}
        if maj and s != "unseen":
            for rn in ref_names:
                print(f"  majority of {len(names)} models vs {rn}: {sum(maj[k] == refs[k]['refs'][rn] for k in maj) / len(maj):.1%} "
                      f"(n={len(maj)})")
        print("  Cohen's kappa between models:")
        for i, a in enumerate(names):
            line = []
            for b in names[i + 1:]:
                common = [k for k in keys if judged[a].get(k, {}).get("verdict") and judged[b].get(k, {}).get("verdict")]
                if common:
                    line.append(f"{b} {kappa([judged[a][k]['verdict'] for k in common], [judged[b][k]['verdict'] for k in common]):.2f}")
            if line:
                print(f"    {a:24s} " + ", ".join(line))
        if s == "realtest":
            langs = sorted({refs[k]["language"] for k in keys})
            print("  agreement with the stored label by language:")
            print("    " + " " * 24 + " ".join(f"{l[:5]:>6s}" for l in langs))
            for n in names:
                cells = []
                for l in langs:
                    rs = [judged[n][k] for k in keys if k in judged[n] and judged[n][k]["verdict"] and refs[k]["language"] == l]
                    cells.append(f"{sum(r['verdict'] == refs[(s, r['id'])]['refs']['stored'] for r in rs) / len(rs):6.0%}"
                                 if rs else f"{'-':>6s}")
                print(f"    {n:24s} " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label")
    lb.add_argument("name")
    lb.add_argument("--sets", default="challenge,realtest,unseen")
    lb.add_argument("--out", type=Path, default=Path("data/judge_panel"))
    lb.add_argument("--workers", type=int, default=48)
    lb.add_argument("--request", type=json.loads, help="request fields in place of the model's own (JSON)")
    lb.add_argument("--tag", help="output name in place of the model name, for a run with other settings")
    lb.add_argument("--max-tokens", type=int)
    lb.add_argument("--sample", type=int, default=0, help="a seeded sample of this many searches per set")
    rp = sub.add_parser("report")
    rp.add_argument("--out", type=Path, default=Path("data/judge_panel"))
    rp.add_argument("--models", help="comma-separated output names to include (default: all)")
    a = ap.parse_args()
    if a.cmd == "label":
        label(a.name, a.sets.split(","), a.out, a.workers, a.request, a.tag, a.max_tokens, a.sample)
    else:
        report(a.out, a.models.split(",") if a.models else None)


if __name__ == "__main__":
    main()
