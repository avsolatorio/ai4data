"""Score documents with the trained classifier.

Reads documents (JSON lines with id and text) and the model folder that
train_classifier.py wrote. With --unit sentence, each text is split into
sentences, every sentence is scored, and the sentence probabilities are
combined per label with --how (mean or max); with --unit whole, the text
is scored in one pass, truncated to the model's input length. Writes one
JSON line per document with its id, the model folder, and the score of
every label. With --threshold, each line also lists the chosen labels:
those at or above the threshold, at most --top of them, highest first.
Prints the device, the number of texts the model read, and the time per
document.

Needs, beyond the cookbook requirements: torch, transformers,
sentencepiece.

Usage:
    python predict.py model/ real_test.jsonl scores.jsonl --unit sentence --how mean
    python predict.py model/ real_test.jsonl scores.jsonl --device cpu --limit 200
    python predict.py model/ new_papers.jsonl topics.jsonl --threshold 0.02 --top 5

Exit status: 0 when the scores are written.

What this does not check: whether the scores are right. evaluate.py
compares them with reference labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from topic_scores import aggregate, choose, split_sentences
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def score_texts(
    texts: list[str], model, tokenizer, device: str, max_length: int, batch: int = 64
) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            encoded = tokenizer(
                texts[start : start + batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            out.append(torch.sigmoid(model(**encoded).logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config.num_labels))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("documents")
    parser.add_argument("out")
    parser.add_argument("--unit", choices=["sentence", "whole"], default="sentence")
    parser.add_argument("--how", choices=["mean", "max"], default="mean")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    labels = json.loads(
        (Path(args.model) / "classifier.json").read_text(encoding="utf-8")
    )["labels"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model)
        .to(args.device)
        .eval()
    )
    documents = pd.read_json(args.documents, lines=True, dtype={"id": str})
    if args.limit:
        documents = documents.head(args.limit)

    started = time.perf_counter()
    read = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for doc_id, text in zip(documents["id"], documents["text"]):
            if args.unit == "sentence":
                sentences = split_sentences(text)
                scores = aggregate(
                    score_texts(sentences, model, tokenizer, args.device, 128), args.how
                )
                read += len(sentences)
            else:
                scores = score_texts([text], model, tokenizer, args.device, 512)[0]
                read += 1
            record = {
                "id": str(doc_id),
                "model": Path(args.model).resolve().name,
                "scores": {
                    label: round(float(s), 5) for label, s in zip(labels, scores)
                },
            }
            if args.threshold is not None:
                record["labels"] = choose(record["scores"], args.threshold, args.top)
            handle.write(json.dumps(record) + "\n")
    elapsed = time.perf_counter() - started
    print(f"device          {args.device}")
    print(f"documents       {len(documents)}")
    print(f"texts read      {read}")
    print(f"seconds per document {elapsed / max(len(documents), 1):.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
