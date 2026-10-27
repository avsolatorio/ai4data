"""A large model labels the real texts directly, for comparison with the small model.

The model receives the task statement, the full list of leaf labels (code and title), and a batch of
texts, and returns labels for each text through a JSON schema whose label field may only take the
task's codes. Single-label tasks get one label per text; multi-label tasks up to `top` labels per text.
The output has the format of model.score(): one record per text with a score of 1 for each label the
model chose, so evaluate.py scores it in the same way. Token counts are kept for the cost.

The connection is read from SYNTH_BASE_URL, SYNTH_API_KEY, SYNTH_MODEL (or BASELINE_MODEL), and
SYNTH_EXTRA_BODY, as for generation. BASELINE_TEMPERATURE sets the temperature (default 0); "none"
leaves it out, for models that accept only their own sampling settings. Real texts are sent to that server, so the task must allow it
(privacy.real_text_to_external.baseline), or the server must be one the organization runs.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .task import Task

PROMPT = """{statement}

Labels (code and title):
{labels}

For each numbered text below, give {cardinality}. Use only codes from the list.

{texts}"""


def label_batch(
    task: Task, client, model: str, batch: pd.DataFrame, top: int
) -> tuple[list[dict], dict]:
    from .generate import call

    labels = "\n".join(f"- {task.labels[x].prompt_line(False)}" for x in task.leaves)
    prompt = PROMPT.format(
        statement=task.statement,
        labels=labels,
        cardinality=f"up to {top} codes that apply, most relevant first"
        if task.multi_label
        else "the one code that applies",
        texts="\n".join(f"[{i}] {str(t)[:3000]}" for i, t in enumerate(batch["text"])),
    )
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["index", "labels"],
        "properties": {
            "index": {"type": "integer"},
            "labels": {
                "type": "array",
                "items": {"type": "string", "enum": list(task.leaves)},
                "minItems": 1,
                "maxItems": top if task.multi_label else 1,
            },
        },
    }
    schema = {
        "name": "labels",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {"type": "array", "items": item, "minItems": len(batch)}
            },
        },
    }
    setting = os.environ.get("BASELINE_TEMPERATURE", "0")
    temperature = None if setting.lower() == "none" else float(setting)
    raw, choice = call(client, model, prompt, schema, temperature)
    chosen: dict[int, list[str]] = {}
    if choice.finish_reason == "stop":
        try:
            for it in json.loads(raw["content"]).get("items", []):
                chosen[it.get("index")] = [
                    x for x in it.get("labels", []) if x in task.labels
                ]
        except json.JSONDecodeError:
            pass
    out = [
        {"id": str(i), "scores": {x: 1.0 for x in chosen.get(k, [])}}
        for k, i in enumerate(batch["id"])
    ]
    return out, raw


def run(
    task: Task,
    texts: pd.DataFrame,
    batch_size: int = 20,
    workers: int = 8,
    top: int = 5,
) -> tuple[list[dict], dict]:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["SYNTH_BASE_URL"],
        api_key=os.environ["SYNTH_API_KEY"],
        timeout=600,
    )
    model = os.environ.get("BASELINE_MODEL", os.environ["SYNTH_MODEL"])
    batches = [texts.iloc[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(lambda b: label_batch(task, client, model, b, top), batches)
        )
    records = [r for out, _ in results for r in out]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for _, raw in results:
        for k in usage:
            usage[k] += raw["usage"].get(k) or 0
    unlabelled = sum(1 for r in records if not r["scores"])
    return records, {
        "model": model,
        "texts": len(records),
        "unlabelled": unlabelled,
        "batches": len(batches),
        "seconds": round(time.time() - started, 1),
        **usage,
    }
