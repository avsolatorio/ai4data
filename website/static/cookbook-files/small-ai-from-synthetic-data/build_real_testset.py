"""Build a test set of real abstracts from the World Bank Documents and Reports API.

Downloads the abstracts and topic tags of World Bank Policy Research
Working Papers published in a date range, maps the catalogue's topic tags
to the classifier's labels with a crosswalk (CSV: source_topic, label),
and writes one JSON line per paper that has an abstract of at least 40
words and at least one mapped label:

    id              the document identifier in the catalogue
    title           the display title
    url             the catalogue page
    date            the document date
    text            the abstract
    labels          the classifier labels the crosswalk gives for its tags
    source_topics   the catalogue tags that produced those labels

Prints the number of papers downloaded and kept, and the labels that the
test set can measure. Uses requests and pandas. The API is described at
https://documents.worldbank.org/en/publication/documents-reports/api.

Usage:
    python build_real_testset.py crosswalk.csv real_test.jsonl
    python build_real_testset.py crosswalk.csv real_test.jsonl --start 2016-01-01 --end 2020-12-31

Exit status: 0 when the file is written, 1 when the API returns nothing.

What this does not check: whether the catalogue's tags are complete or
right for each paper. The tags are a reference made independently of the
classifier; a paper can be about a topic its tags do not name, so a
predicted label missing from the tags is not always an error.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
import requests

API = "https://search.worldbank.org/api/v3/wds"
FIELDS = "id,display_title,abstracts,subtopic,teratopic,docdt,url"


def fetch(start: str, end: str, doc_type: str, page: int = 500) -> list[dict]:
    documents: dict[str, dict] = {}
    offset = 0
    while True:
        params = {
            "format": "json",
            "rows": page,
            "os": offset,
            "fl": FIELDS,
            "strdate": start,
            "enddate": end,
            "docty": doc_type,
        }
        reply = requests.get(API, params=params, timeout=90)
        reply.raise_for_status()
        body = reply.json()
        batch = [
            d
            for d in body.get("documents", {}).values()
            if isinstance(d, dict) and "id" in d
        ]
        for document in batch:
            documents[document["id"]] = document
        offset += page
        if not batch or offset >= int(body.get("total", 0)):
            return list(documents.values())
        time.sleep(1)


def abstract(document: dict) -> str:
    value = document.get("abstracts") or ""
    if isinstance(value, dict):
        value = value.get("cdata!", "")
    return " ".join(str(value).split())


def to_testset(documents: list[dict], crosswalk: pd.DataFrame) -> pd.DataFrame:
    mapping = crosswalk.groupby("source_topic")["label"].apply(list).to_dict()
    rows = []
    for document in documents:
        tags = sorted(
            {
                t.strip()
                for f in ("subtopic", "teratopic")
                for t in (document.get(f) or "").split(",")
                if t.strip()
            }
        )
        used = [t for t in tags if t in mapping]
        labels = sorted({label for t in used for label in mapping[t]})
        text = abstract(document)
        if labels and len(text.split()) >= 40:
            rows.append(
                {
                    "id": document["id"],
                    "title": " ".join(str(document.get("display_title", "")).split()),
                    "url": document.get("url", ""),
                    "date": str(document.get("docdt", ""))[:10],
                    "text": text,
                    "labels": labels,
                    "source_topics": used,
                }
            )
    return (
        pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("crosswalk")
    parser.add_argument("out")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--type", default="Policy Research Working Paper")
    args = parser.parse_args(argv)
    documents = fetch(args.start, args.end, args.type)
    if not documents:
        print("the API returned no documents", file=sys.stderr)
        return 1
    testset = to_testset(documents, pd.read_csv(args.crosswalk))
    testset.to_json(args.out, orient="records", lines=True, force_ascii=False)
    counts = testset["labels"].explode().value_counts()
    print(f"papers downloaded   {len(documents)}")
    print(f"papers kept         {len(testset)}")
    print(f"labels measured     {len(counts)}")
    print(
        "most frequent: " + ", ".join(f"{k} ({v})" for k, v in counts.head(5).items())
    )
    print(
        "least frequent: " + ", ".join(f"{k} ({v})" for k, v in counts.tail(3).items())
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
