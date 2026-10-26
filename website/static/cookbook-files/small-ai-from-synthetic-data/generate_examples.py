"""Ask a large language model for labelled training sentences.

Reads the request plan from plan_requests.py and sends one request per
line to any server that offers the OpenAI chat completions interface
(a hosted service, or a model the organization runs with vLLM or
Ollama). Each request names its labels and asks for --per-call
sentences in a JSON structure whose label field may only take those
names, so the model cannot invent a label. A request with no labels is
a background request: it asks for sentences of the same kind of document
that are about none of the topics, such as sentences on methods, data,
and the structure of a paper. Every reply is kept in full
in raw.jsonl (request, model, time, prompt version, token counts,
finish reason, content), and the parsed sentences go to examples.jsonl
(text, labels, request_id, generator, kind), where kind is "topic" or
"background". A reply that did not finish
normally is counted as failed and not written. A request already in
raw.jsonl is skipped, so an interrupted run continues where it stopped.
A lock file in the output folder stops a second run from writing to the
same files at the same time.

The connection is read from the environment:

    SYNTH_BASE_URL   for example https://openrouter.ai/api/v1
    SYNTH_API_KEY    the key for that server
    SYNTH_MODEL      for example openai/gpt-oss-120b
    SYNTH_EXTRA_BODY optional JSON with options of that server, for example
                     {"provider": {"require_parameters": true, "sort": "throughput"}}
                     on OpenRouter, which sends the request only to providers
                     that enforce the JSON structure, fastest first

Needs, beyond the cookbook requirements: openai.

Usage:
    python generate_examples.py requests.jsonl out/ --per-call 20 --workers 8
    python generate_examples.py requests.jsonl out/ --limit 5
    python generate_examples.py requests.jsonl out/ --names avoid

--names decides whether the sentences may use the topic names. With
"allow" (the default), they use the words a report would use; with
"avoid", the prompt asks for other words. In the tests for this guide,
the model trained on "allow" sentences scored higher on real abstracts
(micro F1 0.441 and 0.446 on the calibration abstracts, for two runs, against 0.371 to 0.387
for three models trained on "avoid" sentences).

Exit status: 0 when every request has a reply, 1 when some failed, 2 when
another run holds the lock on the output folder.

What this does not check: the quality of the sentences. check_examples.py
counts duplicates, missing labels, and sentences that copy a label name;
the evaluation on real text measures the rest.
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

PROMPT_VERSION = "v2"
PROMPT = """Write {n} different sentences of the kind found in reports, working papers, policy notes, and news about economic and social development.

Each sentence must be about one or more of these topics, and its "labels" must list every topic from this list that the sentence is about:
{topics}

Rules:
- {names_rule}
- Vary the length from 10 to 45 words, and vary the style: findings, policy statements, project descriptions, statistics, questions a study asks.
- Mention places, institutions, and numbers only as an author would; invent nothing that reads as a quotation of a real document.
- About a third of the sentences should be about two or three of the topics at once."""

NAMES_RULES = {
    "avoid": "Do not use the topic names or their key words in the sentences. Describe the subject in other words, as a report would.",
    "allow": "Use the words a report on the subject would use, including the topic names or their key words where an author would use them.",
}

BACKGROUND_PROMPT = """Write {n} different sentences of the kind found in reports, working papers, and policy notes about economic and social development, which are about no particular sector or policy area.

Write sentences that describe methods, data sources, samples, estimation, robustness checks, the structure of the document, earlier literature in general terms, limitations, and acknowledgements. Vary the length from 10 to 45 words. Do not name a sector, a policy area, or a development topic."""


def schema(labels: list[str], n: int) -> dict:
    if not labels:
        item = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
        return {
            "name": "examples",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "examples": {"type": "array", "items": item, "minItems": n}
                },
                "required": ["examples"],
                "additionalProperties": False,
            },
        }
    item = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "labels": {
                "type": "array",
                "items": {"type": "string", "enum": labels},
                "minItems": 1,
            },
        },
        "required": ["text", "labels"],
        "additionalProperties": False,
    }
    return {
        "name": "examples",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"examples": {"type": "array", "items": item, "minItems": n}},
            "required": ["examples"],
            "additionalProperties": False,
        },
    }


def ask(
    client: OpenAI, model: str, request: dict, n: int, names: str = "avoid"
) -> dict:
    started = time.time()
    reply = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    n=n,
                    topics="\n".join(f"- {t}" for t in request["labels"]),
                    names_rule=NAMES_RULES[names],
                )
                if request["labels"]
                else BACKGROUND_PROMPT.format(n=n),
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": schema(request["labels"], n),
        },
        temperature=1.0,
        extra_body=json.loads(os.environ.get("SYNTH_EXTRA_BODY", "{}")),
    )
    choice = reply.choices[0]
    usage = reply.usage.model_dump() if reply.usage else {}
    return {
        "request_id": request["request_id"],
        "model": reply.model or model,
        "created": int(started),
        "seconds": round(time.time() - started, 2),
        "prompt_version": f"{PROMPT_VERSION}-{names}-names"
        if request["labels"]
        else f"{PROMPT_VERSION}-background",
        "usage": {
            k: usage.get(k)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "finish_reason": choice.finish_reason,
        "content": choice.message.content or "",
    }


def parse(raw: dict, allowed: list[str]) -> list[dict]:
    try:
        items = json.loads(raw["content"]).get("examples", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    rows = []
    for item in items:
        text = " ".join(str(item.get("text", "")).split())
        labels = sorted({x for x in item.get("labels", []) if x in allowed})
        if text and (labels or not allowed):
            rows.append(
                {
                    "text": text,
                    "labels": labels,
                    "request_id": raw["request_id"],
                    "generator": raw["model"],
                    "kind": "topic" if allowed else "background",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requests")
    parser.add_argument("out_dir")
    parser.add_argument("--per-call", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--names", choices=sorted(NAMES_RULES), default="allow")
    args = parser.parse_args(argv)
    client = OpenAI(
        base_url=os.environ["SYNTH_BASE_URL"],
        api_key=os.environ["SYNTH_API_KEY"],
        timeout=300,
    )
    model = os.environ["SYNTH_MODEL"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path, examples_path = out / "raw.jsonl", out / "examples.jsonl"
    lock_path = out / "generate.lock"
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        print(f"{lock_path} exists: another run is writing here", file=sys.stderr)
        return 2
    done = set()
    if raw_path.exists():
        done = {
            json.loads(line)["request_id"]
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line
        }
    plan = [
        r
        for r in pd.read_json(args.requests, lines=True).to_dict("records")
        if r["request_id"] not in done
    ]
    if args.limit:
        plan = plan[: args.limit]
    lock, failed = Lock(), []

    def work(request: dict) -> None:
        try:
            raw = ask(client, model, request, args.per_call, args.names)
        except Exception as error:  # noqa: BLE001 - a failed request is retried on the next run
            failed.append((request["request_id"], str(error)[:200]))
            return
        if raw["finish_reason"] != "stop":
            failed.append(
                (request["request_id"], f"finish reason {raw['finish_reason']}")
            )
            return
        rows = parse(raw, request["labels"])
        with lock:
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
            with examples_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, plan))
    finally:
        lock_path.unlink()
    print(f"requests sent   {len(plan)}")
    print(f"failed          {len(failed)}")
    for request_id, error in failed[:5]:
        print(f"  {request_id}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
