"""Open multilingual pretraining corpus (Wikipedia) -> tokenizer -> uint16 token shards.

  python -m relevance_explainer.tiny.corpus fetch      --root data/corpus
  python -m relevance_explainer.tiny.corpus tokenizer  --root data/corpus --task-data data/v1
  python -m relevance_explainer.tiny.corpus tokenize   --root data/corpus
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

# language -> (Wikipedia config, character budget). Budgets give roughly 1B tokens in total.
LANGS = {
    "English": ("en", 1_200_000_000), "French": ("fr", 450_000_000), "Spanish": ("es", 450_000_000),
    "Portuguese": ("pt", 250_000_000), "Arabic": ("ar", 300_000_000), "Chinese": ("zh", 300_000_000),
    "Russian": ("ru", 350_000_000), "Indonesian": ("id", 200_000_000), "Hindi": ("hi", 150_000_000),
    "Turkish": ("tr", 200_000_000), "Swahili": ("sw", 80_000_000),
}
PAD, EOS = 0, 1


def _fetch_one(args) -> tuple[str, int, int]:
    from datasets import load_dataset

    lang, (code, budget), root = args
    path = Path(root) / "raw" / f"{lang}.jsonl"
    if path.exists():
        return lang, 0, path.stat().st_size
    tmp = path.with_suffix(".tmp")
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{code}", split="train", streaming=True)
    chars = docs = 0
    with tmp.open("w") as f:
        for ex in ds:
            t = ex["text"].strip()
            if len(t) < 300:
                continue
            t = t[:20_000]
            f.write(json.dumps({"t": t}, ensure_ascii=False) + "\n")
            chars, docs = chars + len(t), docs + 1
            if chars >= budget:
                break
    tmp.rename(path)
    return lang, docs, chars


def cmd_fetch(a) -> None:
    (Path(a.root) / "raw").mkdir(parents=True, exist_ok=True)
    with mp.Pool(len(LANGS)) as pool:
        for lang, docs, chars in pool.imap_unordered(_fetch_one, [(l, v, a.root) for l, v in LANGS.items()]):
            print(f"{lang}: {docs} docs, {chars / 1e6:.0f}M chars", flush=True)


# FineWeb-2 (web text, ODC-By) configs and character budgets for the second corpus shard per language.
# English is left out: it already has the most data, and the aim is to lift the smaller languages.
FW2 = {
    "Swahili": ("swh_Latn", 300_000_000), "Hindi": ("hin_Deva", 450_000_000), "Indonesian": ("ind_Latn", 400_000_000),
    "Turkish": ("tur_Latn", 400_000_000), "Russian": ("rus_Cyrl", 450_000_000), "Arabic": ("arb_Arab", 400_000_000),
    "Chinese": ("cmn_Hani", 300_000_000), "Portuguese": ("por_Latn", 250_000_000), "Spanish": ("spa_Latn", 250_000_000),
    "French": ("fra_Latn", 250_000_000),
}


def _fetch_fw2_one(args) -> tuple[str, int, int]:
    from datasets import load_dataset

    lang, (code, budget), root = args
    path = Path(root) / "raw" / f"{lang}_fw2.jsonl"
    if path.exists():
        return lang, 0, path.stat().st_size
    tmp = path.with_suffix(".tmp")
    ds = load_dataset("HuggingFaceFW/fineweb-2", name=code, split="train", streaming=True)
    chars = docs = 0
    with tmp.open("w") as f:
        for ex in ds:
            t = ex["text"].strip()
            if len(t) < 300:
                continue
            t = t[:20_000]
            f.write(json.dumps({"t": t}, ensure_ascii=False) + "\n")
            chars, docs = chars + len(t), docs + 1
            if chars >= budget:
                break
    tmp.rename(path)
    return lang, docs, chars


def cmd_fetch2(a) -> None:
    (Path(a.root) / "raw").mkdir(parents=True, exist_ok=True)
    with mp.Pool(len(FW2)) as pool:
        for lang, docs, chars in pool.imap_unordered(_fetch_fw2_one, [(l, v, a.root) for l, v in FW2.items()]):
            print(f"{lang} (fineweb-2): {docs} docs, {chars / 1e6:.0f}M chars", flush=True)


def _texts(path: Path, max_chars: int):
    n = 0
    with path.open() as f:
        for line in f:
            t = json.loads(line)["t"]
            yield t
            n += len(t)
            if n >= max_chars:
                return


def cmd_tokenizer(a) -> None:
    from tokenizers import ByteLevelBPETokenizer

    root = Path(a.root)
    task = []
    for split in ("train", "val"):
        for line in (Path(a.task_data) / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            task += [r["input"], r["output"]]

    def corpus():
        for lang in LANGS:
            yield from _texts(root / "raw" / f"{lang}.jsonl", a.chars_per_lang)
        for _ in range(a.task_repeat):  # make domain terms and the output format cheap to encode
            yield from task

    tk = ByteLevelBPETokenizer()
    tk.train_from_iterator(corpus(), vocab_size=a.vocab, special_tokens=["<pad>", "<eos>"], show_progress=False)
    tk.save(str(root / "tokenizer.json"))
    print("vocab", tk.get_vocab_size())


def _tok_one(args) -> tuple[str, int]:
    """Tokenize raw/<stem>.jsonl into tok/<stem>.bin (stem is a language name, optionally with a source suffix)."""
    from tokenizers import Tokenizer

    stem, root = args
    tk = Tokenizer.from_file(str(Path(root) / "tokenizer.json"))
    tk.no_padding(), tk.no_truncation()
    out = Path(root) / "tok" / f"{stem}.bin"
    n = 0
    with out.open("wb") as f:
        batch: list[str] = []
        for t in _texts(Path(root) / "raw" / f"{stem}.jsonl", 1 << 62):
            batch.append(t)
            if len(batch) == 1000:
                n += _flush(tk, batch, f)
                batch = []
        if batch:
            n += _flush(tk, batch, f)
    return lang, n


def _flush(tk, batch, f) -> int:
    ids = []
    for enc in tk.encode_batch(batch):
        ids.extend(enc.ids + [EOS])
    np.asarray(ids, dtype=np.uint16).tofile(f)
    return len(ids)


def cmd_tokenize(a) -> None:
    """Tokenize every raw/*.jsonl that has no tok/*.bin yet (reruns only touch new shards)."""
    root = Path(a.root)
    (root / "tok").mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in sorted((root / "raw").glob("*.jsonl")) if not (root / "tok" / f"{p.stem}.bin").exists()]
    total = 0
    with mp.Pool(min(max(len(stems), 1), a.procs)) as pool:
        for stem, n in pool.imap_unordered(_tok_one, [(s, a.root) for s in stems]):
            total += n
            print(f"{stem}: {n / 1e6:.0f}M tokens", flush=True)
    print(f"total new {total / 1e6:.0f}M tokens")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch"); p.add_argument("--root", required=True); p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("fetch2"); p.add_argument("--root", required=True); p.set_defaults(fn=cmd_fetch2)
    p = sub.add_parser("tokenizer"); p.add_argument("--root", required=True); p.add_argument("--task-data", required=True)
    p.add_argument("--vocab", type=int, default=12000); p.add_argument("--chars-per-lang", type=int, default=25_000_000)
    p.add_argument("--task-repeat", type=int, default=10); p.set_defaults(fn=cmd_tokenizer)
    p = sub.add_parser("tokenize"); p.add_argument("--root", required=True); p.add_argument("--procs", type=int, default=11)
    p.set_defaults(fn=cmd_tokenize)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
