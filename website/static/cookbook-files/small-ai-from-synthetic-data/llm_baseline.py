"""Label documents with a large language model, for comparison with the small model.

Sends each document to a server that offers the OpenAI chat completions
interface and asks for the labels from the label file that the document
is about, at most --top of them, in a JSON structure whose values may
only be label names. Writes one JSON line per document in the format of
predict.py (id, scores), with a score of 1 for each chosen label and 0
for the others, so that evaluate.py can compare both with a threshold of
0.5. Prints the token counts and the time per document. A document
already in the output file is skipped.

The connection is read from the same environment variables as
generate_examples.py: SYNTH_BASE_URL, SYNTH_API_KEY, SYNTH_MODEL, and the
optional SYNTH_EXTRA_BODY.

Needs, beyond the cookbook requirements: openai.

Usage:
    python llm_baseline.py labels.csv real_test.jsonl llm_scores.jsonl --limit 300

Exit status: 0 when every document has an answer, 1 when some failed.

What this does not do: tune the prompt. A prompt with label definitions
and examples would likely do better, and would cost more tokens per
document.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pandas as pd
from openai import OpenAI

PROMPT = """Below is the abstract of a document, followed by a list of topics. Choose the topics the document is about, at most {top}, most relevant first. Use only names from the list.

Abstract:
{text}

Topics:
{topics}"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels")
    parser.add_argument("documents")
    parser.add_argument("out")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    labels = sorted(pd.read_csv(args.labels)["label"])
    client = OpenAI(
        base_url=os.environ["SYNTH_BASE_URL"],
        api_key=os.environ["SYNTH_API_KEY"],
        timeout=300,
    )
    model = os.environ["SYNTH_MODEL"]
    documents = pd.read_json(args.documents, lines=True, dtype={"id": str})
    if args.limit:
        documents = documents.head(args.limit)
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {
            json.loads(line)["id"]
            for line in out.read_text(encoding="utf-8").splitlines()
            if line
        }
    todo = [d for d in documents.to_dict("records") if d["id"] not in done]
    schema = {
        "name": "topics",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string", "enum": labels},
                    "maxItems": args.top,
                }
            },
            "required": ["topics"],
            "additionalProperties": False,
        },
    }
    lock, failed, usage = Lock(), [], {"prompt": 0, "completion": 0, "seconds": 0.0}

    def work(document: dict) -> None:
        started = time.time()
        try:
            reply = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            top=args.top,
                            text=document["text"],
                            topics="\n".join(labels),
                        ),
                    }
                ],
                response_format={"type": "json_schema", "json_schema": schema},
                temperature=0,
                extra_body=json.loads(os.environ.get("SYNTH_EXTRA_BODY", "{}")),
            )
            chosen = set(
                json.loads(reply.choices[0].message.content or "{}").get("topics", [])
            )
        except Exception as error:  # noqa: BLE001 - a failed document is retried on the next run
            failed.append((document["id"], str(error)[:200]))
            return
        record = {
            "id": document["id"],
            "scores": {label: float(label in chosen) for label in labels},
        }
        with lock:
            usage["prompt"] += reply.usage.prompt_tokens if reply.usage else 0
            usage["completion"] += reply.usage.completion_tokens if reply.usage else 0
            usage["seconds"] += time.time() - started
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    answered = max(len(todo) - len(failed), 1)
    print(f"model                {model}")
    print(f"documents            {len(todo)}")
    print(f"failed               {len(failed)}")
    print(f"prompt tokens per document      {usage['prompt'] / answered:.0f}")
    print(f"completion tokens per document  {usage['completion'] / answered:.0f}")
    print(
        f"seconds per document (one request at a time)  {usage['seconds'] / answered:.2f}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
