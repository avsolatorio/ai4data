"""Generation of labelled examples by two model roles: a writer and a labeller.

The writer receives one request from a plan (its labels, mode, and note)
and returns a list of texts, each with the labels it belongs to. A JSON
schema restricts the label field to the request's labels, so the writer
cannot invent one. The labeller then reads the texts without the
writer's labels and assigns labels from a wider list of candidates: the
request's labels, their siblings, and other labels drawn at random,
plus "none". An example is kept as verified when the two agree: the same
code for a single-label task, and at least one shared label for a
multi-label task, where the shared labels are kept. The agreement rate
per label is a signal for the feedback loop: labels on which the two
roles often disagree have unclear definitions or confusable neighbours.

Any server with the OpenAI chat completions interface can play either
role. The connection is read from the environment:

    SYNTH_BASE_URL     for example https://openrouter.ai/api/v1, or a local vLLM server
    SYNTH_API_KEY      the key for that server
    SYNTH_MODEL        the writer, for example openai/gpt-oss-120b
    SYNTH_LABEL_MODEL  optional: the labeller (default: the writer's model)
    SYNTH_EXTRA_BODY   optional JSON with options of that server

Every reply is kept in full in raw.jsonl (role, request, model, time,
prompt version, token counts, finish reason, content). The examples go
to examples.jsonl. A reply that did not finish normally counts as
failed. A request already in raw.jsonl is skipped, so an interrupted run
continues where it stopped, and a lock file stops two runs from writing
to the same folder.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from .task import Task

PROMPT_VERSION = "h1"

DEFAULT_PROMPT = """Write {n} different {items}.

{context}Each {item} must belong to {cardinality} of these labels, and its "labels" must list {which}:
{labels}

Rules:
{rules}"""

CONTRAST = """
These labels are easily confused. Write {items} close to the boundary between them, as a real {item} could be. Each {item} must still belong clearly to the label it is given, and the details that decide between the labels must be in the text."""

FOCUS = "\nPurpose of this request: {note}"

BACKGROUND = """Write {n} different {items}.

{context}Each {item} must be about none of the subjects in this list, while reading like the rest of the collection:
{labels}

Rules:
{rules}"""


def connection() -> tuple:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["SYNTH_BASE_URL"],
        api_key=os.environ["SYNTH_API_KEY"],
        timeout=300,
    )
    writer = os.environ["SYNTH_MODEL"]
    return client, writer, os.environ.get("SYNTH_LABEL_MODEL", writer)


def writer_prompt(task: Task, request: dict, n: int) -> str:
    g = task.generation
    detail = g.get("label_detail", True)
    if request["mode"] == "background":
        others = random.Random(request["request_id"]).sample(
            task.leaves, min(30, len(task.leaves))
        )
        return BACKGROUND.format(
            n=n,
            items=g["items"],
            item=g["item"],
            context=g.get("context", ""),
            labels="\n".join(f"- {task.labels[x].prompt_line(False)}" for x in others),
            rules="\n".join(
                f"- {r}" for r in g.get("background_rules", g.get("rules", []))
            ),
        )
    labels = "\n".join(
        f"- {task.labels[x].prompt_line(detail)}" for x in request["labels"]
    )
    rules = list(g.get("rules", []))
    if "names" in g:
        rules.insert(0, g["names_rules"][g["names"]])
    template = g.get("prompt", DEFAULT_PROMPT)
    text = template.format(
        n=n,
        items=g["items"],
        item=g["item"],
        context=g.get("context", ""),
        labels=labels,
        topics=labels,
        cardinality="one or more" if task.multi_label else "exactly one",
        which="every label from this list that it belongs to"
        if task.multi_label
        else "that one label",
        rules="\n".join(f"- {r}" for r in rules),
        names_rule=rules[0] if rules else "",
    )
    if request["mode"] == "contrast":
        text += CONTRAST.format(items=g["items"], item=g["item"])
    if request.get("note"):
        text += FOCUS.format(note=request["note"])
    return text


def item_schema(
    enum: list[str], n: int, with_labels: bool = True, index: bool = False
) -> dict:
    props: dict = (
        {"index": {"type": "integer"}} if index else {"text": {"type": "string"}}
    )
    required = ["index"] if index else ["text"]
    if with_labels:
        props["labels"] = {
            "type": "array",
            "items": {"type": "string", "enum": enum},
            "minItems": 1,
        }
        required.append("labels")
    item = {
        "type": "object",
        "properties": props,
        "required": required,
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


def call(
    client, model: str, prompt: str, schema: dict, temperature: float
) -> tuple[dict, object]:
    started = time.time()
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=temperature,
        extra_body=json.loads(os.environ.get("SYNTH_EXTRA_BODY", "{}")),
    )
    choice = reply.choices[0]
    usage = reply.usage.model_dump() if reply.usage else {}
    return {
        "model": reply.model or model,
        "created": int(started),
        "seconds": round(time.time() - started, 2),
        "usage": {
            k: usage.get(k)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "finish_reason": choice.finish_reason,
        "content": choice.message.content or "",
    }, choice


def candidates(task: Task, request: dict, extra: int = 8) -> list[str]:
    """Labels the labeller chooses from: the request's, their siblings, and random others."""
    rng = random.Random("cand-" + request["request_id"])
    chosen = list(request["labels"])
    for x in request["labels"]:
        chosen += task.siblings(x)[:6]
    pool = [x for x in task.leaves if x not in chosen]
    chosen += rng.sample(pool, min(extra, len(pool)))
    seen, out = set(), []
    for x in chosen:
        if x not in seen:
            seen.add(x)
            out.append(x)
    rng.shuffle(out)
    return out


LABELLER = """You assign labels to {items} for the task below. Read each {item} and choose {cardinality} from the list, or "none" when no label fits.

Task: {statement}

Labels:
{labels}

{texts}"""


def label_texts(
    task: Task, client, model: str, request: dict, texts: list[str]
) -> tuple[dict, list[list[str]] | None]:
    cands = candidates(task, request)
    names = {x: x for x in cands}
    g = task.generation
    prompt = LABELLER.format(
        items=g["items"],
        item=g["item"],
        statement=task.statement,
        cardinality="every label that applies"
        if task.multi_label
        else "the one label that applies",
        labels="\n".join(
            f"- {task.labels[x].prompt_line(g.get('label_detail', True))}"
            for x in cands
        ),
        texts="\n".join(f"[{i}] {t}" for i, t in enumerate(texts)),
    )
    raw, choice = call(
        client,
        model,
        prompt,
        item_schema([*names, "none"], len(texts), index=True),
        0.0,
    )
    if choice.finish_reason != "stop":
        return raw, None
    try:
        items = json.loads(raw["content"]).get("examples", [])
    except json.JSONDecodeError:
        return raw, None
    out: list[list[str]] = [[] for _ in texts]
    for item in items:
        i = item.get("index")
        if isinstance(i, int) and 0 <= i < len(texts):
            out[i] = sorted({x for x in item.get("labels", []) if x in names})
    return raw, out


def agree(task: Task, written: list[str], labelled: list[str]) -> list[str]:
    if task.multi_label:
        return sorted(set(written) & set(labelled))
    return written if written == labelled else []


def generate(
    task: Task,
    plan: list[dict],
    out_dir: Path,
    per_call: int,
    workers: int = 8,
    verify: bool = True,
    limit: int | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, examples_path, lock_path = (
        out_dir / "raw.jsonl",
        out_dir / "examples.jsonl",
        out_dir / "generate.lock",
    )
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        raise RuntimeError(f"{lock_path} exists: another run is writing here") from None
    done = set()
    if raw_path.exists():
        done = {
            r["request_id"]
            for r in map(json.loads, raw_path.read_text(encoding="utf-8").splitlines())
            if r.get("role") == "writer"
        }
    todo = [r for r in plan if r["request_id"] not in done][: limit or None]
    client, writer, labeller = connection()
    lock, failed = Lock(), []
    temperature = float(task.generation.get("temperature", 1.0))

    def work(request: dict) -> None:
        try:
            labels = request["labels"]
            prompt = writer_prompt(task, request, per_call)
            enum = labels if labels else []
            raw, choice = call(
                client,
                writer,
                prompt,
                item_schema(enum, per_call, with_labels=bool(labels)),
                temperature,
            )
            raw.update(
                role="writer",
                request_id=request["request_id"],
                prompt_version=PROMPT_VERSION,
                mode=request["mode"],
            )
            if choice.finish_reason != "stop":
                failed.append(
                    (request["request_id"], f"finish reason {choice.finish_reason}")
                )
                return
            items = json.loads(raw["content"]).get("examples", [])
            rows = []
            for item in items:
                text = " ".join(str(item.get("text", "")).split())
                got = sorted({x for x in item.get("labels", []) if x in labels})
                if text and (got or not labels):
                    rows.append(
                        {
                            "text": text,
                            "labels": got,
                            "written_labels": got,
                            "request_id": request["request_id"],
                            "generator": raw["model"],
                            "kind": "background" if not labels else "topic",
                            "mode": request["mode"],
                        }
                    )
            check_raw = None
            if verify and rows:
                check_raw, verdicts = label_texts(
                    task, client, labeller, request, [r["text"] for r in rows]
                )
                check_raw.update(
                    role="labeller",
                    request_id=request["request_id"],
                    prompt_version=PROMPT_VERSION,
                )
                for row, verdict in zip(rows, verdicts or [None] * len(rows)):
                    row["labeller_labels"] = verdict
                    if verdict is None:
                        row["verified"] = None
                    elif not row["labels"]:
                        row["verified"] = verdict in ([], ["none"]) or all(
                            v == "none" for v in verdict
                        )
                    else:
                        kept = agree(task, row["labels"], verdict)
                        row["verified"] = bool(kept)
                        if kept:
                            row["labels"] = kept
        except Exception as error:  # noqa: BLE001 - a failed request is retried on the next run
            failed.append((request["request_id"], str(error)[:200]))
            return
        with lock:
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
                if check_raw:
                    handle.write(json.dumps(check_raw, ensure_ascii=False) + "\n")
            with examples_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))
    finally:
        lock_path.unlink()
    return {"requests": len(todo), "failed": len(failed), "errors": failed[:5]}


def usage(raw_path: Path) -> dict:
    """Token counts of a generation folder, by role."""
    out: dict = {}
    if not raw_path.exists():
        return out
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        role = out.setdefault(
            r.get("role", "writer"),
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        )
        role["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            role[k] += (r.get("usage") or {}).get(k) or 0
    return out


if __name__ == "__main__":
    sys.exit("use: python -m ai4data.small_ai generate ...")
