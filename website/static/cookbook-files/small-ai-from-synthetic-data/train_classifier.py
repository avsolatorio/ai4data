"""Train a small multi-label text classifier on the split examples.

Fine-tunes a small pretrained encoder (by default microsoft/deberta-v3-small,
141 million parameters) with one output per label. Each output gives the
probability that the text is about that label, so a text can have several
labels. After training, the script chooses one decision threshold for all
labels on the development file: the threshold, between 0.05 and 0.95,
with the highest micro-averaged F1 score. It saves the model, the
tokenizer, the label list, and the threshold in the output folder and
prints the scores on the development and test files.

Needs, beyond the cookbook requirements: torch, transformers, datasets,
accelerate, sentencepiece. Tested with torch 2.9.1, transformers 4.57.3,
and datasets 4.8.5. With transformers 5.20 and torch 2.14, DeBERTa-v3
did not train (the loss became NaN within ten steps), so pin the
versions. A GPU shortens training from hours to minutes; the trained
model runs on a CPU.

Usage:
    python train_classifier.py splits/ labels.csv model/ --epochs 4

Exit status: 0 when the model is saved.

What this does not check: how the model does on real text. The test file
is synthetic; chapter 5 measures the model on real documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def encode(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    target = np.zeros((len(frame), len(labels)), dtype=np.float32)
    for row, names in enumerate(frame["labels"]):
        for name in names:
            target[row, index[name]] = 1.0
    return target


def best_threshold(probs: np.ndarray, target: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = [
        f1_score(target, probs >= t, average="micro", zero_division=0) for t in grid
    ]
    return float(grid[int(np.argmax(scores))])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("splits")
    parser.add_argument("labels")
    parser.add_argument("out")
    parser.add_argument("--base", default="microsoft/deberta-v3-small")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bf16", action="store_true", help="half-precision training on a recent GPU"
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="stop early, for a quick check"
    )
    args = parser.parse_args(argv)

    labels = sorted(pd.read_csv(args.labels)["label"])
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    parts = {
        name: pd.read_json(Path(args.splits) / f"{name}.jsonl", lines=True)
        for name in ("train", "dev", "test")
    }

    def dataset(frame: pd.DataFrame) -> Dataset:
        data = Dataset.from_dict(
            {"text": frame["text"].tolist(), "labels": encode(frame, labels).tolist()}
        )
        return data.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=128),
            batched=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        problem_type="multi_label_classification",
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(args.out) / "checkpoints"),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=128,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.06,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            bf16=args.bf16,
            logging_steps=200,
            disable_tqdm=True,
            seed=args.seed,
            report_to=[],
        ),
        train_dataset=dataset(parts["train"]),
        eval_dataset=dataset(parts["dev"]),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    scores = {}
    probs = {}
    for name in ("dev", "test"):
        logits = trainer.predict(dataset(parts[name])).predictions
        probs[name] = 1 / (1 + np.exp(-logits))
    threshold = best_threshold(probs["dev"], encode(parts["dev"], labels))
    for name in ("dev", "test"):
        target = encode(parts[name], labels)
        predicted = probs[name] >= threshold
        scores[name] = {
            "micro_f1": round(
                f1_score(target, predicted, average="micro", zero_division=0), 3
            ),
            "macro_f1": round(
                f1_score(target, predicted, average="macro", zero_division=0), 3
            ),
        }

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    config = {
        "base": args.base,
        "labels": labels,
        "threshold": threshold,
        "scores": scores,
    }
    (Path(args.out) / "classifier.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"threshold  {threshold}")
    for name, values in scores.items():
        print(
            f"{name:<5}  micro F1 {values['micro_f1']:.3f}  macro F1 {values['macro_f1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
