"""Run the RAG chain on the chosen cases with a live model and write answers.json.

    WORKSHOP_API_KEY=... WORKSHOP_BASE_URL=https://openrouter.ai/api/v1 \
    WORKSHOP_MODEL=anthropic/claude-haiku-5.5 python record_answers.py
    python record_answers.py --dry-run             # print the prompts; no key needed, nothing written
    ... python record_answers.py --only main_one_search,outside

The model is configured as in the session notebooks (workshop.make_model):
WORKSHOP_API_KEY for an OpenAI-compatible chat API, WORKSHOP_BASE_URL
(optional), WORKSHOP_MODEL (default gpt-4o-mini), temperature 0. The key is
never printed or written.

Each case retrieves its passages from the store exactly as `make_rag` does
(store.similarity_search with the case's k and filter), formats them with
their citations, and sends the system and user messages of notebook 03. The
passages are compared with retrieval_runs.json; a difference is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import common as c
from runs import LATEST_QUESTION, MAIN_QUESTION, OUTSIDE_QUESTION, INSIDE_QUESTION, SINCE, SUB_QUERIES

# name: (concept, question, k, date filter or None, how the passages are chosen)
CASES = {
    "main_one_search": ("rag, two parts", MAIN_QUESTION, 4, None, "one search"),
    "main_merged": ("two parts", MAIN_QUESTION, 4, None, "one search per part, merged"),
    "latest_no_filter": ("latest figure", LATEST_QUESTION, 2, None, "one search"),
    "latest_filtered": ("latest figure", LATEST_QUESTION, 2, SINCE, "one search"),
    "inside": ("outside the collection", INSIDE_QUESTION, 3, None, "one search"),
    "outside": ("outside the collection", OUTSIDE_QUESTION, 3, None, "one search"),
}


def passages_for(store, name: str):
    concept, question, k, since, how = CASES[name]
    where = c.since(since) if since else None
    if how == "one search":
        return [(d, s) for d, s in c.hits(store, question, k=k, where=where)]
    merged, scores = [], {}
    for q in SUB_QUERIES:
        for d, s in c.hits(store, q, k=k, where=where):
            if d not in merged:
                merged.append(d)
                scores[id(d)] = s
    return [(d, scores[id(d)]) for d in merged]


def network_date(base_url: str | None) -> tuple[str, str]:
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=10) as resp:
            date = resp.headers.get("Date")
    except urllib.error.HTTPError as err:  # an error status still carries the Date header
        date = err.headers.get("Date")
    except Exception:
        date = None
    if date:
        host = urllib.parse.urlparse(url).netloc
        return parsedate_to_datetime(date).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), f"HTTP Date header of {host}"
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "local clock, unverified"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the prompts that would be sent; no key needed")
    ap.add_argument("--only", default="", help="comma-separated case names (default: all)")
    args = ap.parse_args()

    names = [n for n in args.only.split(",") if n] or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}; known: {', '.join(CASES)}")
        return 1
    api_key = os.environ.get("WORKSHOP_API_KEY")
    if not api_key and not args.dry_run:
        print("WORKSHOP_API_KEY is not set; nothing to record. Set it (and WORKSHOP_BASE_URL, WORKSHOP_MODEL), "
              "or run with --dry-run to print the prompts.")
        return 1
    model_name = os.environ.get("WORKSHOP_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("WORKSHOP_BASE_URL") or None

    store = c.load_store()
    prompt = c.make_prompt()
    recorded = json.loads(c.RUNS_JSON.read_text())["runs"] if c.RUNS_JSON.exists() else {}

    model = None
    if not args.dry_run:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.0)
    print(f"model: {model_name}  base URL: {base_url or '(default)'}  cases: {', '.join(names)}"
          + ("  [dry run]" if args.dry_run else ""))

    results = []
    for name in names:
        concept, question, k, since, how = CASES[name]
        found = passages_for(store, name)
        docs = [d for d, _ in found]
        if name in recorded:
            want = [p["citation"] + f"/{p['chunk']}" for p in recorded[name]["passages"]]
            got = [c.citation(d) + f"/{d.metadata['chunk']}" for d in docs]
            if want != got:
                print(f"note: passages of {name} differ from retrieval_runs.json: {got} vs {want}")
        messages = prompt.format_messages(context=c.format_context(docs), question=question)
        if args.dry_run:
            print(f"\n=== {name} ({concept}; k={k}; filter: {f'date >= {since}' if since else 'none'}; {how})")
            for m in messages:
                print(f"--- {m.type}\n{m.content}")
            continue
        print(f"asking: {name}", flush=True)
        reply = model.invoke(messages)
        meta = reply.response_metadata or {}
        results.append({
            "name": name, "concept": concept, "question": question, "k": k, "model": model_name,
            "filter": f"date >= {since}" if since else None, "passages_chosen_by": how,
            "passages": [{"citation": c.citation(d), "id": d.metadata["id"], "edition": d.metadata["edition"],
                          "page": d.metadata["page"], "chunk": d.metadata["chunk"],
                          "score": round(float(s), 4),  # for merged passages, the score in its own part's search
                          "text": d.page_content} for d, s in found],
            "messages": [{"role": m.type, "content": m.content} for m in messages],
            "answer": reply.content if isinstance(reply.content, str) else json.dumps(reply.content),
            "finish_reason": meta.get("finish_reason"),
            "model_reported": meta.get("model_name"),
            "usage": meta.get("token_usage"),
        })

    if args.dry_run:
        print(f"\ndry run: {len(names)} prompts printed; nothing sent or written")
        return 0
    date, date_source = network_date(base_url)
    for r in results:
        r["date"] = date
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=c.HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    out = {"model": model_name, "base_url_host": urllib.parse.urlparse(base_url).netloc if base_url else None,
           "temperature": 0.0, "date": date, "date_source": date_source, "git_commit": commit,
           "instructions": c.INSTRUCTIONS, "answers": results}
    if c.ANSWERS_JSON.exists() and len(names) < len(CASES):
        # A partial run replaces only its own cases; each answer keeps its own model and date.
        old = json.loads(c.ANSWERS_JSON.read_text())
        kept = [a for a in old.get("answers", []) if a["name"] not in names]
        out["answers"] = kept + results
    c.ANSWERS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} answers to {c.ANSWERS_JSON}")
    for r in results:
        if r["finish_reason"] not in (None, "stop"):
            print(f"warning: {r['name']} finished with {r['finish_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
