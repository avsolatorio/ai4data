"""Splitting, training, and scoring with a small encoder.

split():   keeps every reply of the generating model whole, so that
           sentences of one reply never sit on both sides of a split.
train():   fine-tunes a pretrained encoder (by default
           microsoft/deberta-v3-small) with one output per leaf label.
           A multi-label task uses one sigmoid per label and a threshold
           chosen on the dev split; a single-label task uses a softmax.
score():   scores texts. A document is read one sentence at a time when
           the task says so, and the sentence scores are combined per
           label (mean or max).

Tested with torch 2.9.1, transformers 4.57.3, and datasets 4.8.5. With
transformers 5.20 and torch 2.14, DeBERTa-v3 did not train (the loss
became NaN within ten steps), so the versions are pinned in the extra.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .task import Task

BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split(
    examples: pd.DataFrame, dev: float = 0.1, test: float = 0.1, seed: int = 42
) -> dict[str, pd.DataFrame]:
    from sklearn.model_selection import GroupShuffleSplit

    outer = GroupShuffleSplit(n_splits=1, test_size=dev + test, random_state=seed)
    train_idx, rest_idx = next(outer.split(examples, groups=examples["request_id"]))
    rest = examples.iloc[rest_idx]
    inner = GroupShuffleSplit(
        n_splits=1, test_size=test / (dev + test), random_state=seed
    )
    dev_idx, test_idx = next(inner.split(rest, groups=rest["request_id"]))
    return {
        "train": examples.iloc[train_idx],
        "dev": rest.iloc[dev_idx],
        "test": rest.iloc[test_idx],
    }


def encode(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    target = np.zeros((len(frame), len(labels)), dtype=np.float32)
    for row, names in enumerate(frame["labels"]):
        for name in names:
            target[row, index[name]] = 1.0
    return target


def best_threshold(probs: np.ndarray, target: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = [
        f1_score(target, probs >= t, average="micro", zero_division=0) for t in grid
    ]
    return float(grid[int(np.argmax(scores))])


def train(
    task: Task,
    splits: dict[str, pd.DataFrame],
    out: Path,
    seed: int = 42,
    epochs: float | None = None,
    max_steps: int = -1,
    bf16: bool = False,
) -> dict:
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    m = task.model
    base = m.get("base", "microsoft/deberta-v3-small")
    labels = list(task.leaves)
    tokenizer = AutoTokenizer.from_pretrained(base)
    max_length = int(m.get("max_length", 128))
    multi = task.multi_label

    def dataset(frame: pd.DataFrame) -> Dataset:
        if multi:
            target = encode(frame, labels).tolist()
        else:
            index = {x: i for i, x in enumerate(labels)}
            target = [index[ls[0]] for ls in frame["labels"]]
        data = Dataset.from_dict({"text": frame["text"].tolist(), "labels": target})
        return data.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, max_length=max_length
            ),
            batched=True,
        )

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={x: i for i, x in enumerate(labels)},
        problem_type="multi_label_classification"
        if multi
        else "single_label_classification",
    )
    started = time.time()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "checkpoints"),
            num_train_epochs=epochs or float(m.get("epochs", 4)),
            max_steps=max_steps,
            per_device_train_batch_size=int(m.get("batch", 32)),
            per_device_eval_batch_size=128,
            learning_rate=float(m.get("lr", 5e-5)),
            weight_decay=0.01,
            warmup_ratio=0.06,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=bf16,
            logging_steps=200,
            disable_tqdm=True,
            seed=seed,
            report_to=[],
        ),
        train_dataset=dataset(splits["train"]),
        eval_dataset=dataset(splits["dev"]),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(int(m["early_stopping"]))]
        if m.get("early_stopping")
        else None,
    )
    trainer.train()
    seconds = round(time.time() - started, 1)
    logits = {
        name: trainer.predict(dataset(splits[name])).predictions
        for name in ("dev", "test")
    }
    scores, threshold = {}, None
    if multi:
        probs = {k: 1 / (1 + np.exp(-v)) for k, v in logits.items()}
        threshold = best_threshold(probs["dev"], encode(splits["dev"], labels))
        for name in ("dev", "test"):
            target, predicted = encode(splits[name], labels), probs[name] >= threshold
            scores[name] = {
                "micro_f1": round(
                    f1_score(target, predicted, average="micro", zero_division=0), 3
                ),
                "macro_f1": round(
                    f1_score(target, predicted, average="macro", zero_division=0), 3
                ),
            }
    else:
        for name in ("dev", "test"):
            gold = [ls[0] for ls in splits[name]["labels"]]
            guess = [labels[i] for i in logits[name].argmax(axis=1)]
            scores[name] = {
                "accuracy": round(accuracy_score(gold, guess), 3),
                "macro_f1": round(
                    f1_score(gold, guess, average="macro", zero_division=0), 3
                ),
            }
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    config = {
        "task": task.id,
        "base": base,
        "labels": labels,
        "multi_label": multi,
        "threshold": threshold,
        "max_length": max_length,
        "seed": seed,
        "train_seconds": seconds,
        "epochs_run": round(float(trainer.state.epoch or 0), 2),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_examples": len(splits["train"]),
        "scores": scores,
    }
    (out / "classifier.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    parts = [p.strip() for p in BOUNDARY.split(" ".join(str(text).split()))]
    sentences = [p for p in parts if len(p.split()) >= min_words]
    return sentences or [str(text).strip()]


def load(model_dir: Path, device: str | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads((model_dir / "classifier.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = (
        AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        .to(device)
        .eval()
    )
    return config, tokenizer, model, device


def probabilities(
    texts: list[str],
    config: dict,
    tokenizer,
    model,
    device: str,
    max_length: int,
    batch: int = 64,
) -> np.ndarray:
    import torch

    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            enc = tokenizer(
                texts[start : start + batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.float()
            p = (
                torch.sigmoid(logits)
                if config["multi_label"]
                else torch.softmax(logits, dim=-1)
            )
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(config["labels"])))


def score(
    task: Task, model_dir: Path, documents: pd.DataFrame, device: str | None = None
) -> tuple[list[dict], dict]:
    """Label scores for each document: {"id", "scores"}; and timing."""
    config, tokenizer, model, device = load(model_dir, device)
    unit = task.documents.get("unit", "whole")
    how = task.documents.get("aggregate", "mean")
    started, read, out = time.perf_counter(), 0, []
    if unit == "sentence":
        for doc_id, text in zip(documents["id"], documents["text"]):
            sentences = split_sentences(text)
            p = probabilities(
                sentences, config, tokenizer, model, device, config["max_length"]
            )
            agg = p.mean(axis=0) if how == "mean" else p.max(axis=0)
            read += len(sentences)
            out.append(
                {
                    "id": str(doc_id),
                    "scores": dict(zip(config["labels"], np.round(agg, 5).tolist())),
                }
            )
    else:
        p = probabilities(
            [str(t) for t in documents["text"]],
            config,
            tokenizer,
            model,
            device,
            int(task.documents.get("max_length", config["max_length"])),
        )
        read = len(p)
        for doc_id, row in zip(documents["id"], p):
            out.append(
                {
                    "id": str(doc_id),
                    "scores": dict(zip(config["labels"], np.round(row, 5).tolist())),
                }
            )
    elapsed = time.perf_counter() - started
    return out, {
        "device": device,
        "documents": len(documents),
        "texts_read": read,
        "seconds_per_document": round(elapsed / max(len(documents), 1), 4),
    }
