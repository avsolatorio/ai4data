"""Split examples into training, development, and test files by request.

Sentences generated in one response to the large model share wording and
topics. If some of them are used for training and others for testing,
the test measures how well the model remembers that response. This
script keeps every response whole: all sentences with the same
request_id go to the same file. Writes train.jsonl, dev.jsonl, and
test.jsonl in the output folder and prints their sizes.

Uses pandas and scikit-learn (GroupShuffleSplit).

Usage:
    python split_examples.py clean.jsonl splits/ --dev 0.1 --test 0.1 --seed 42

Exit status: 0 when the files are written.

What this does not check: that the test file resembles the text the
model will meet in use. Synthetic test sentences come from the same
generator as the training sentences, so the score on them is an upper
bound; chapter 5 measures the model on real text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split(
    examples: pd.DataFrame, dev: float, test: float, seed: int
) -> dict[str, pd.DataFrame]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("examples")
    parser.add_argument("out_dir")
    parser.add_argument("--dev", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    examples = pd.read_json(args.examples, lines=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in split(examples, args.dev, args.test, args.seed).items():
        part.to_json(
            out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False
        )
        print(
            f"{name:<6}{len(part):>7} examples  {part['request_id'].nunique():>5} requests"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
