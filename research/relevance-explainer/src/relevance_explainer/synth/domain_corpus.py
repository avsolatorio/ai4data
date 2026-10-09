"""Domain pretraining text from every synthetic set, including rows the training data left out.

Documents (one JSON line {"t": text} each, per language, in data/corpus_domain/raw/<Language>.jsonl, the layout of
tiny/corpus.py so `corpus tokenize` turns them into token shards):
  example   item lines of the record, the search, the explanation (no verdict line: the verdict is what labellers
            disagreed on in the rows that were left out). The full record is written for its first 3 examples, then
            only its title line.
  searches  searches that have no explanation (written, rendered or paired but not narrated), 40 per document
  record    every pool or catalog record not already written in full, in its metadata language

Left out: test material (the searches of every test set; rows whose topic or seed falls in the validation or test
split; the records of the challenge suite, the nine-type set and the held-out half of the Eurostat tables),
explanations in the wrong language, and explanations with a number absent from their record and search.

  python -m relevance_explainer.synth.domain_corpus --out data/corpus_domain
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import generate as G
from .challenge import record_half
from .realistic import _clean_item_lines, language_ok, renderer

LANGS = ["English", "French", "Spanish", "Portuguese", "Arabic", "Chinese", "Russian", "Indonesian", "Hindi", "Turkish", "Swahili"]
TEST_FILES = ["data/real_test/examples_labeled.jsonl", "data/real_test/rendered.jsonl", "data/para_test/examples.jsonl",
              "data/para_test/rendered.jsonl", "data/challenge/searches.jsonl", "data/schemas_eval2/labeled.jsonl",
              "data/estat_test/searches.jsonl", "data/policy_check_real/searches.jsonl"]
# records kept out of all training: the challenge suite, the NADA nine-type set and the Eurostat test half. Records of
# the realistic and paraphrase tests also occur in fine-tuning rows for other searches, so only their searches are excluded
HELD_OUT_RECORDS = ["data/challenge/searches.jsonl", "data/schemas_eval2/labeled.jsonl", "data/estat_test/searches.jsonl"]
# (glob, split key) for files with explanations; key gives the theme that decides the split
EXAMPLE_SOURCES = [
    ("data/real/c*/examples.jsonl", "topic"), ("data/real/t*/examples.jsonl", "topic"), ("data/real_types/c*/examples.jsonl", "topic"),
    ("data/real_para/c*/examples.jsonl", "topic"), ("data/real_meta/m*/examples.jsonl", "topic"), ("data/real_pilot/examples.jsonl", "topic"),
    ("data/v[0-9]/examples.jsonl", "seed"), ("data/f[0-9]/examples.jsonl", "seed"), ("data/s[0-9]/examples.jsonl", "seed"),
    # challenge-style labelled files keep the rows the two labellers disagreed on
    ("data/real_phen/p*/labeled.jsonl", "phen"), ("data/concepts*/labeled.jsonl", "phen"), ("data/real_hardneg/h*/labeled.jsonl", "phen"),
    ("data/real_estat/e*/labeled.jsonl", "phen"),
]
SEARCH_SOURCES = [("data/real/c*/queries.jsonl", "topic"), ("data/real/c*/rendered.jsonl", "topic"), ("data/real_types/c*/rendered.jsonl", "topic"),
                  ("data/real_para/c*/rendered.jsonl", "topic"), ("data/v[0-9]/queries.jsonl", "seed"),
                  ("data/real_phen/p*/searches.jsonl", "phen"), ("data/concepts/searches.jsonl", "phen"),
                  ("data/real_hardneg/h*/searches.jsonl", "phen"), ("data/real_estat/e*/searches.jsonl", "phen")]
POOLS = ["data/pools/table_statcan.jsonl", "data/pools/table_statcan_fr.jsonl", "data/pools/table_ilostat.jsonl",
         "data/pools/table_ilostat_fr.jsonl", "data/pools/table_ilostat_es.jsonl", "data/pools/geospatial_hdx.jsonl",
         "data/pools/script_rrr.jsonl", "data/pools/table_eurostat.jsonl"]


def rows(pattern: str):
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def theme(r: dict, key: str) -> str | None:
    if key == "phen":
        return f"phen-{r.get('phenomenon')}-{r['record']['id']}" if r.get("record") else None
    return r.get(key)


def item_text(rec: dict) -> str:
    try:
        return "\n".join(_clean_item_lines(renderer(rec["kind"]).render(rec), 2))
    except Exception:
        return f"Result: {rec.get('name', '')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus_domain")
    ap.add_argument("--full-per-record", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from ..tiny.numguard import unsupported

    test_q, test_rec = set(), set()
    for f in TEST_FILES:
        for r in rows(f):
            if r.get("query"):
                test_q.add(r["query"].strip())
    for f in HELD_OUT_RECORDS:
        for r in rows(f):
            if r.get("record"):
                test_rec.add(r["record"]["id"])
    for r in rows("data/pools/table_eurostat.jsonl"):
        if record_half(r["id"]) == 1:
            test_rec.add(r["id"])
    print(f"test material: {len(test_q)} searches, {len(test_rec)} records")

    docs = defaultdict(list)
    stats = Counter()
    full = Counter()
    seen_text, seen_q = set(), set()
    allrec = {}
    for pattern, key in EXAMPLE_SOURCES:
        for r in rows(pattern):
            rec, q, narr, lang = r.get("record"), (r.get("query") or "").strip(), (r.get("narrative") or "").strip(), r.get("language")
            if not (rec and q and narr) or lang not in LANGS:
                stats["skip_fields"] += 1
                continue
            t = theme(r, key)
            if not t or G.split_of(t) != "train":
                stats["skip_split"] += 1
                continue
            if q in test_q or rec.get("id") in test_rec:
                stats["skip_test_material"] += 1
                continue
            if narr in seen_text:
                stats["skip_dup"] += 1
                continue
            seen_text.add(narr)
            seen_q.add(q)
            if not language_ok(narr, lang):
                stats["skip_language"] += 1
                continue
            item = item_text(rec)
            orig = r.get("input_v2") or r.get("input") or ""  # the input the narrative was written against
            facets = next((l for l in orig.split("\n") if l.startswith(("Facets: ", "Filters: "))), "")
            if facets in ("Facets: none", "Filters: none"):
                facets = ""
            if unsupported("verdict\n" + narr, "\n".join([orig, item, q])):  # unsupported() skips the first (verdict) line
                stats["skip_number"] += 1
                continue
            allrec.setdefault(rec["id"], rec)
            if full[rec["id"]] < a.full_per_record:
                full[rec["id"]] += 1
                head = item
            else:
                head = f"Result: {rec.get('name', '')}"
            docs[lang].append(f"{head}\nSearch: {q}\n" + (f"{facets}\n" if facets else "") + narr)
            stats["example"] += 1
    searches = defaultdict(list)
    for pattern, key in SEARCH_SOURCES:
        for r in rows(pattern):
            q, lang = (r.get("query") or "").strip(), r.get("language")
            if not q or lang not in LANGS or q in seen_q or q in test_q:
                continue
            t = theme(r, key)
            if not t or G.split_of(t) != "train" or (r.get("record") or {}).get("id") in test_rec:
                continue
            seen_q.add(q)
            if r.get("record"):
                allrec.setdefault(r["record"]["id"], r["record"])
            searches[lang].append(q)
    rng = random.Random(a.seed)
    for lang, qs in searches.items():
        rng.shuffle(qs)
        for i in range(0, len(qs), 40):
            docs[lang].append("\n".join(f"Search: {q}" for q in qs[i:i + 40]))
            stats["search_docs"] += 1
        stats["searches"] += len(qs)
    for f in POOLS:
        for rec in rows(f):
            allrec.setdefault(rec["id"] + ("" if rec.get("meta_language", "English") == "English" else "|" + rec["meta_language"]), rec)
    for rid, rec in allrec.items():
        if rec["id"] in test_rec or full[rid] > 0:
            continue
        lang = rec.get("meta_language") or "English"
        if lang in LANGS:
            docs[lang].append(item_text(rec))
            stats["record"] += 1
    out = Path(a.out) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    chars = {}
    for lang in LANGS:
        ds = docs.get(lang, [])
        rng.shuffle(ds)
        (out / f"{lang}.jsonl").write_text("".join(json.dumps({"t": d}, ensure_ascii=False) + "\n" for d in ds))
        chars[lang] = sum(len(d) for d in ds)
    print(dict(stats))
    print("documents:", {l: len(docs.get(l, [])) for l in LANGS})
    print("characters (M):", {l: round(c / 1e6, 1) for l, c in chars.items()})


if __name__ == "__main__":
    main()
