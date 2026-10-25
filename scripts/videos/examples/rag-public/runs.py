"""Run the retrieval cases of the public RAG example and record them in retrieval_runs.json.

    python runs.py

Every case is a search of the store built by build_store.py, with the
question, k, and filter given below. Each ranked passage is recorded with its
score (cosine similarity), report, edition, PDF page, and text. The figures
that the cases rely on are checked against the chunk text (with page) and,
where an indicator exists, against World Development Indicators values
fetched from api.worldbank.org at run time.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import common as c
from langchain_core.documents import Document

# ----------------------------------------------------------------------------
# The cases
# ----------------------------------------------------------------------------

# 1 and 2. The main question: two parts, both answered in the June 2026 Uganda edition.
MAIN_QUESTION = "How did the conflict in the Middle East affect Uganda, and how much did poverty fall?"
SUB_QUERIES = ["How did the conflict in the Middle East affect Uganda?", "How much did poverty fall in Uganda?"]

# 3. A question about the latest figure, which every edition reports for its own time.
LATEST_QUESTION = "What is the latest inflation rate in Uganda?"
LATEST_REWORDED = "Uganda headline inflation April 2026, percent year-on-year"
SINCE = "2026-01"  # the date filter: editions dated January 2026 or later

# 4. A question the collection can answer and one it cannot (no Rwanda report is in the collection).
INSIDE_QUESTION = "How much did Uganda's economy grow in FY25?"
OUTSIDE_QUESTION = "How much did Rwanda's economy grow in 2025?"

# Labelled examples for setting a score cut-off. "in": the collection holds the answer
# (the figure and page are in CHECKS); "out": it does not.
CALIBRATION = [
    ("in", "How much did Kenya's economy grow in 2025?"),
    ("in", "What share of bank loans in Kenya are non-performing?"),
    ("in", "What is Kenya's policy interest rate?"),
    ("in", "What is Kenya's GDP growth projection for 2026?"),
    ("in", "What share of Uganda's petroleum comes from the Middle East?"),
    ("in", "What is the poverty rate in Uganda?"),
    ("in", INSIDE_QUESTION),
    ("in", LATEST_QUESTION),
    ("out", OUTSIDE_QUESTION),
    ("out", "What was inflation in Rwanda in 2025?"),
    ("out", "What is the unemployment rate in Rwanda?"),
    ("out", "What was Tanzania's GDP growth in 2025?"),
    ("out", "What was inflation in Ethiopia in 2025?"),
    ("out", "What is the unemployment rate in Ghana?"),
    ("out", "How much coffee does Ethiopia export?"),
    ("out", "What is the literacy rate in Peru?"),
    ("out", "How much rice does Thailand export?"),
]

# 5. Retrieval tests without the model: expected reports and the numbers a correct answer needs.
TESTS = [
    {"id": "t1", "category": "two parts", "question": MAIN_QUESTION,
     "expected_documents": ["ueu-2026-06"], "expected_numbers": ["ueu_energy_24", "ueu_poverty_fy25", "ueu_poverty_fy24"]},
    {"id": "t2", "category": "latest figure", "question": LATEST_QUESTION,
     "expected_documents": ["ueu-2026-06"], "expected_numbers": ["ueu_inflation_apr26"]},
    {"id": "t3", "category": "latest figure", "question": "What was Kenya's real GDP growth in 2025?",
     "expected_documents": ["keu-2026-07"], "expected_numbers": ["keu_growth_2025"]},
    {"id": "t4", "category": "single topic", "question": "How did the conflict in the Middle East affect fuel prices in Kenya?",
     "expected_documents": ["keu-2026-07"], "expected_numbers": ["keu_diesel", "keu_petrol"]},
    {"id": "t5", "category": "projection", "question": "What is Kenya's GDP growth projection for 2026?",
     "expected_documents": ["keu-2026-07"], "expected_numbers": ["keu_growth_2026f"]},
    {"id": "t6", "category": "single topic", "question": "What share of bank loans in Kenya are non-performing?",
     "expected_documents": ["keu-2026-07"], "expected_numbers": ["keu_npl"]},
    {"id": "t7", "category": "single topic", "question": "How much did Uganda's economy grow in FY25?",
     "expected_documents": ["ueu-2025-12", "ueu-2026-06"], "expected_numbers": ["ueu_growth_fy25"]},
    {"id": "t8", "category": "outside the collection", "question": OUTSIDE_QUESTION,
     "expected_documents": [], "expected_numbers": []},
]
TEST_K = 4

# Figures the cases rely on: the quote must occur in a chunk of that report and page.
# "wdi" names a World Development Indicators series to compare with, where one exists.
CHECKS = {
    "ueu_energy_24": {"value": 24, "doc": "ueu-2026-06", "page": 22,
                      "quote": "Global energy prices are projected to rise by 24 percent in 2026"},
    "ueu_imports_15": {"value": 15, "doc": "ueu-2026-06", "page": 22,
                       "quote": "and 15 percent of Uganda’s total imports"},
    "ueu_petroleum_87": {"value": 87, "doc": "ueu-2026-06", "page": 22,
                         "quote": "Uganda sources approximately 87 percent of its petroleum from the Middle East"},
    "ueu_fuel_petrol_diesel": {"value": 10.8, "doc": "ueu-2026-06", "page": 20,
                               "quote": "with petrol and diesel prices rising by 8.7 and 10.8 percent respectively"},
    "ueu_poverty_fy25": {"value": 51.5, "doc": "ueu-2026-06", "page": 12,
                         "quote": "The share of Ugandans living below the USD 3.00-per-day line fell to 51.5 percent in FY25, down from 52.9 percent in FY24",
                         "wdi": {"indicator": "SI.POV.DDAY", "country": "UGA",
                                 "note": "WDI holds survey years only; the report's FY24 and FY25 values are estimates from the Macro Poverty Outlook"}},
    "ueu_poverty_fy24": {"value": 52.9, "doc": "ueu-2026-06", "page": 23,
                         "quote": "fallen to 51.5 percent in FY25, down from 52.9 percent in FY24"},
    "ueu_poverty_2019_figure": {"value": 59.8, "doc": "ueu-2025-12", "page": 26,
                                "quote": "56.2% 60.1% 59.8%",
                                "wdi": {"indicator": "SI.POV.DDAY", "country": "UGA", "year": "2019"}},
    "ueu_inflation_apr26": {"value": 3.0, "doc": "ueu-2026-06", "page": 20,
                            "quote": "Headline year-on-year inflation stood at 3.0 percent in April 2026",
                            "wdi": {"indicator": "FP.CPI.TOTL.ZG", "country": "UGA",
                                    "note": "WDI holds annual averages; the report gives a monthly year-on-year rate"}},
    "ueu_inflation_nov23": {"value": 2.6, "doc": "ueu-2023-12", "page": 17,
                            "quote": "The inflation rate began to decline in February and reached 2.6 percent in November"},
    "ueu_inflation_fy23": {"value": 8.8, "doc": "ueu-2023-12", "page": 17,
                           "quote": "Headline inflation increased to 8.8 percent in FY2023"},
    "ueu_growth_fy25": {"value": 6.3, "doc": "ueu-2025-12", "page": 12,
                        "quote": "Real gross domestic product (GDP) growth accelerated to 6.3 percent, up from 6.1 percent in FY24",
                        "wdi": {"indicator": "NY.GDP.MKTP.KD.ZG", "country": "UGA", "year": "2025"}},
    "keu_growth_2025": {"value": 4.6, "doc": "keu-2026-07", "page": 13,
                        "quote": "Real GDP growth moderated to 4.6 percent in 2025, from 4.7 percent in 2024",
                        "wdi": {"indicator": "NY.GDP.MKTP.KD.ZG", "country": "KEN", "year": "2025"}},
    "keu_growth_2025_projected_may25": {"value": 4.5, "doc": "keu-2025-05", "page": 20,
                                        "quote": "Real GDP growth is projected at 4.5 percent in 2025"},
    "keu_growth_2026f": {"value": 4.3, "doc": "keu-2026-07", "page": 21,
                         "quote": "Growth is projected at 4.3 percent in 2026"},
    "keu_diesel": {"value": 17.9, "doc": "keu-2026-07", "page": 9,
                   "quote": "diesel prices increased sharply by 17.9 percent"},
    "keu_petrol": {"value": 10.8, "doc": "keu-2026-07", "page": 9,
                   "quote": "petrol prices by 10.8 percent"},
    "keu_npl": {"value": 15.6, "doc": "keu-2026-07", "page": 10,
                "quote": "the gross non-performing loan to gross loans ratio reached 15.6 percent in March 2026"},
    "keu_policy_rate": {"value": 8.75, "doc": "keu-2026-07", "page": 9,
                        "quote": "reduce the policy rate from 11.25 percent in January 2025 to 8.75 percent in February 2026"},
    "rwanda_growth_table": {"value": 8.6, "doc": "keu-2024-12", "page": 51,
                            "quote": "Rwanda 8.5 9.4 -3.4 10.6 7.8 8.6",
                            "wdi": {"indicator": "NY.GDP.MKTP.KD.ZG", "country": "RWA", "year": "2025",
                                    "note": "The Kenya editions' tables of EAC growth rates end in 2023; the collection holds no Rwanda figure for 2025"}},
    "rwanda_inflation_footnote": {"value": 7.2, "doc": "ueu-2025-02", "page": 23,
                                  "quote": "During the twelve months up to June 2024, average inflation in Kenya, Ethiopia, Rwanda, Tanzania, and Malawi was recorded at 6.2 percent, 26.6 percent, 7.2 percent, 3.1 percent, and 31.6 percent respectively",
                                  "wdi": {"indicator": "FP.CPI.TOTL.ZG", "country": "RWA", "year": "2025",
                                          "note": "The collection holds no Rwanda report; this footnote is its only Rwanda inflation figure, for the twelve months to June 2024"}},
}


# ----------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------


def passage(rank: int, doc: Document, score: float | None) -> dict:
    m = doc.metadata
    out = {"rank": rank}
    if score is not None:
        out["score"] = round(float(score), 4)
    out.update({"citation": c.citation(doc), "id": m["id"], "series": m["series"], "edition": m["edition"],
                "date": m["date"], "page": m["page"], "chunk": m["chunk"], "text": doc.page_content})
    return out


def run(store, name: str, question: str, k: int, since: str | None = None, note: str = "") -> dict:
    where = c.since(since) if since else None
    ranked = c.hits(store, question, k=k, where=where)
    return {"name": name, "question": question, "k": k,
            "filter": None if since is None else {"field": "date", "op": ">=", "value": since,
                                                  "description": f"reports dated {since} or later"},
            "note": note, "passages": [passage(i + 1, d, s) for i, (d, s) in enumerate(ranked)]}


def rank_of(store, question: str, quote: str, doc_id: str, since: str | None = None) -> dict | None:
    """Rank and score, in the full ranking, of the first chunk of `doc_id` that contains `quote`."""
    where = c.since(since) if since else None
    for i, (d, s) in enumerate(c.hits(store, question, k=len(store.store), where=where)):
        if d.metadata["id"] == doc_id and quote in d.page_content:
            return {"rank": i + 1, "score": round(float(s), 4), "citation": c.citation(d)}
    return None


def wdi(indicator: str, country: str) -> tuple[dict, str]:
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&date=2015:2026&per_page=50"
    with urllib.request.urlopen(url, timeout=60) as resp:
        body = json.load(resp)
        date = resp.headers.get("Date", "")
    meta, rows = body[0], body[1] or []
    return {"indicator": indicator, "country": country, "url": url, "lastupdated": meta.get("lastupdated"),
            "values": {r["date"]: r["value"] for r in rows if r["value"] is not None}}, date


def check_figures(chunks: list[dict]) -> tuple[dict, str]:
    out, server_date = {}, ""
    cache: dict[tuple[str, str], dict] = {}
    for key, chk in CHECKS.items():
        found = [ch for ch in chunks if ch["metadata"]["id"] == chk["doc"] and ch["metadata"]["page"] == chk["page"]
                 and " ".join(chk["quote"].split()) in " ".join(ch["text"].split())]
        rec = {**chk, "found_in_text": bool(found),
               "chunks": [f"{ch['metadata']['id']}/p{ch['metadata']['page']}/{ch['metadata']['chunk']}" for ch in found]}
        if "wdi" in chk:
            w = chk["wdi"]
            ck = (w["indicator"], w["country"])
            if ck not in cache:
                cache[ck], server_date = wdi(*ck)
            series = cache[ck]
            rec["wdi"] = {**w, "lastupdated": series["lastupdated"], "url": series["url"],
                          "values": {y: round(v, 2) for y, v in series["values"].items() if y >= "2019"}}
        out[key] = rec
        if not found:
            print(f"CHECK FAILED: {key}: quote not found on {chk['doc']} p. {chk['page']}")
    return out, server_date


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=c.HERE, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def main() -> int:
    store = c.load_store()
    chunks = [json.loads(line) for line in c.CHUNKS_JSONL.open()]
    summary = json.loads((c.WORK / "store_summary.json").read_text())
    runs: dict[str, dict] = {}

    # 1. RAG on the main question, and the ranking behind it
    runs["main_one_search"] = run(store, "main_one_search", MAIN_QUESTION, k=4,
                                  note="The passages that make_rag(k=4) puts in the prompt.")
    ranking = run(store, "main_ranking", MAIN_QUESTION, k=10, note="The ten closest chunks; the prompt receives the first four.")
    for p in ranking["passages"]:
        text = p["text"]
        p["topic"] = "conflict" if "Middle East" in text else "poverty" if "povert" in text.lower() else "other"
    runs["main_ranking"] = ranking
    runs["main_needed_ranks"] = {
        "name": "main_needed_ranks", "question": MAIN_QUESTION,
        "note": "Rank of the chunks that hold the needed figures, in the full ranking of the main question.",
        "ranks": {needle: rank_of(store, MAIN_QUESTION, needle, "ueu-2026-06")
                  for needle in ("rise by 24 percent in 2026", "15 percent of Uganda’s total imports",
                                 "87 percent of its petroleum", "51.5 percent in FY25")},
    }

    # 2. One search per part, merged without duplicates (as in notebook 03)
    merged: list[Document] = []
    for i, q in enumerate(SUB_QUERIES):
        runs[f"main_part_{i + 1}"] = run(store, f"main_part_{i + 1}", q, k=4, note="One search for one part of the main question.")
        for d in store.similarity_search(q, k=4):
            if d not in merged:
                merged.append(d)
    runs["main_merged"] = {"name": "main_merged", "question": MAIN_QUESTION, "sub_queries": SUB_QUERIES, "k_per_query": 4,
                           "note": "The merged passages, in the order of the two searches.",
                           "passages": [passage(i + 1, d, None) for i, d in enumerate(merged)]}

    # 3. The latest figure: no filter, a reworded question, a filter on the date
    runs["latest_no_filter"] = run(store, "latest_no_filter", LATEST_QUESTION, k=2)
    runs["latest_reworded"] = run(store, "latest_reworded", LATEST_REWORDED, k=2,
                                  note="The question reworded with the date and the vocabulary of the June 2026 edition; no filter.")
    runs["latest_filtered"] = run(store, "latest_filtered", LATEST_QUESTION, k=2, since=SINCE)
    runs["latest_needed_ranks"] = {
        "name": "latest_needed_ranks", "question": LATEST_QUESTION,
        "note": "Rank of the chunk with the April 2026 figure, without and with the date filter, and for the reworded question.",
        "no_filter": rank_of(store, LATEST_QUESTION, "3.0 percent in April 2026", "ueu-2026-06"),
        "filtered": rank_of(store, LATEST_QUESTION, "3.0 percent in April 2026", "ueu-2026-06", since=SINCE),
        "reworded": rank_of(store, LATEST_REWORDED, "3.0 percent in April 2026", "ueu-2026-06"),
    }

    # 4. Inside and outside the collection, and a cut-off set from labelled examples
    runs["inside"] = run(store, "inside", INSIDE_QUESTION, k=3)
    runs["outside"] = run(store, "outside", OUTSIDE_QUESTION, k=3)
    labelled = []
    for label, q in CALIBRATION:
        (d, s), = c.hits(store, q, k=1)
        labelled.append({"label": label, "question": q, "top_score": round(float(s), 4), "top_citation": c.citation(d)})
    scores = sorted({x["top_score"] for x in labelled})
    best = None
    for t in scores:
        correct = sum((x["top_score"] >= t) == (x["label"] == "in") for x in labelled)
        if best is None or correct > best[1]:
            best = (t, correct)
    cutoff = best[0]
    for x in labelled:
        x["passes_cutoff"] = x["top_score"] >= cutoff
    runs["cutoff"] = {"name": "cutoff", "examples": labelled, "cutoff": cutoff,
                      "correct": best[1], "total": len(labelled),
                      "note": "The cut-off is the top score that classifies the most labelled examples correctly: "
                              "a question passes when its best chunk scores at or above it."}

    # 5. Retrieval tests without the model
    tests = []
    for since in (None, SINCE):
        for t in TESTS:
            found_docs = store.similarity_search(t["question"], k=TEST_K, filter=c.since(since) if since else None)
            text = " ".join(d.page_content for d in found_docs)
            needed = [{"key": key, "value": CHECKS[key]["value"]} for key in t["expected_numbers"]]
            hits_n = sum(c.numbers_present(text, [n])[0] for n in needed)
            strict_n = sum(c.figure_present(text, n["value"]) for n in needed)
            tests.append({"id": t["id"], "category": t["category"], "question": t["question"], "k": TEST_K,
                          "filter": None if since is None else f"date >= {since}",
                          "expected_documents": t["expected_documents"],
                          "retrieval": "hit" if c.retrieval_hit(found_docs, t["expected_documents"]) else "miss",
                          "numbers": f"{hits_n}/{len(needed)}" if needed else "none needed",
                          "figures_as_percent": f"{strict_n}/{len(needed)}" if needed else "none needed",
                          "passages": [c.citation(d) for d in found_docs]})
    runs["tests"] = {"name": "tests", "k": TEST_K, "rows": tests,
                     "note": "retrieval: an expected report is among the passages (retrieval_hit of notebook 03). "
                             "numbers: needed values found by numbers_present of notebook 03, which accepts any number "
                             "within its tolerance. figures_as_percent: needed values found written as a percentage "
                             "(common.figure_present)."}

    checks, server_date = check_figures(chunks)
    try:
        run_date = parsedate_to_datetime(server_date).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  # noqa: UP017 (Python 3.10)
        date_source = "HTTP Date header of api.worldbank.org"
    except (TypeError, ValueError):  # no Date header was received
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  # noqa: UP017 (Python 3.10)
        date_source = "local clock, unverified"
    out = {
        "description": "Retrieval runs of the public RAG example over World Bank country economic updates for Kenya and Uganda.",
        "run_date": run_date, "run_date_source": date_source, "git_commit": git_commit(),
        "embedding_model": c.EMBEDDING_MODEL, "score": "cosine similarity (InMemoryVectorStore)",
        "store": {k: summary[k] for k in ("documents", "pdf_pages", "pages_with_chunks", "chunks", "characters",
                                          "tokens_embedding_model", "embedding_max_tokens", "chunks_over_max_tokens")},
        "citation_form": "[report id · p. PDF page]; the report id names the series (keu: Kenya Economic Update, "
                         "ueu: Uganda Economic Update) and the edition month",
        "runs": runs, "checks": checks,
    }
    c.RUNS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    failed = [k for k, v in checks.items() if not v["found_in_text"]]
    print(f"wrote {c.RUNS_JSON.name}: {len(runs)} runs, {len(checks)} checks, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
