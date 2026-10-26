"""Convert the published doc-topics synthetic dataset to the example format.

The dataset avsolatorio/doc-topics-synthetic_data on the Hugging Face Hub
has two parts: "processed" (one row per generated sentence: sentence,
topics, model_name) and "raw" (one row per response of the generating
model, with its full text and token counts). The processed part does not
say which response a sentence came from. This script finds each sentence
in the raw responses and writes one JSON line per sentence:

    text         the generated sentence
    labels       the topics the generating model assigned to it
    request_id   the identifier of the response that contains it
    generator    the model that generated it

Sentences that appear in no response get the request_id "unmatched".
Uses pandas, and huggingface_hub with pyarrow to read the files.

Usage:
    python import_hub_dataset.py examples_hub.jsonl
    python import_hub_dataset.py examples_hub.jsonl --revision e6d8416...

Exit status: 0 when the file is written.

What this does not check: the content of the examples. check_examples.py
does that.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import pandas as pd

REPO = "hf://datasets/avsolatorio/doc-topics-synthetic_data@{revision}/{part}/train-00000-of-00001.parquet"
REVISION = "e6d841658894fc096d8e0df3fb0567c201250bfc"
SENTENCE = re.compile(r'"sentence"\s*:\s*"((?:[^"\\]|\\.)*)"')


def sentence_index(raw: pd.DataFrame) -> dict[str, str]:
    index: dict[str, str] = {}
    # Some responses carry no identifier; their row number stands in for it.
    ids = [
        rid if isinstance(rid, str) and rid else f"response-{n:05d}"
        for n, rid in enumerate(raw["id"])
    ]
    for content, response_id in zip(raw["content"], ids):
        for match in SENTENCE.finditer(content or ""):
            try:
                text = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                continue
            index.setdefault(text.strip(), response_id)
    return index


def to_examples(processed: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    index = sentence_index(raw)
    text = processed["sentence"].fillna("").str.strip()
    return pd.DataFrame(
        {
            "text": text,
            "labels": processed["topics"].map(lambda t: [] if t is None else list(t)),
            "request_id": text.map(lambda s: index.get(s, "unmatched")),
            "generator": processed["model_name"],
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args(argv)
    processed = pd.read_parquet(REPO.format(revision=args.revision, part="processed"))
    raw = pd.read_parquet(REPO.format(revision=args.revision, part="raw"))
    examples = to_examples(processed, raw)
    examples.to_json(args.out, orient="records", lines=True, force_ascii=False)
    matched = (examples["request_id"] != "unmatched").sum()
    print(f"responses   {len(raw)}")
    print(f"sentences   {len(examples)}")
    print(f"matched     {matched}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
