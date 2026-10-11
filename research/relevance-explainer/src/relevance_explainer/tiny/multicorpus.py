"""Corpus for a multilingual base model: every FineWeb-2 language with enough text, plus English.

  plan       per-language character budgets from FineWeb-2's sizes: share proportional to size**alpha (small languages
             get more than their share of the web), capped by what each language has, the cap's excess given to the rest
  fetch      stream each language's budget into raw/<config>.jsonl (resumable; finished files are skipped)
  tokenizers train byte-level BPE tokenizers of several sizes on a balanced sample and report bytes per token by language
             on held-out documents, against an existing tokenizer
  tokenize   reuse tiny.corpus tokenize with the chosen tokenizer

  python -m relevance_explainer.tiny.multicorpus plan --sizes fw2_size.json --root data/corpus_multi --chars 18e9
  python -m relevance_explainer.tiny.multicorpus fetch --root data/corpus_multi --procs 16
  python -m relevance_explainer.tiny.multicorpus tokenizers --root data/corpus_multi --vocab 8000 12000 16000 24000 \\
      --compare data/corpus/tokenizer.json --task-data data/v2_g_itemfirst_nosim
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from collections import defaultdict
from pathlib import Path

# The explainer's 11 languages and their FineWeb-2 configs; English is not in FineWeb-2 and comes from Wikipedia.
OURS = {"French": "fra_Latn", "Spanish": "spa_Latn", "Portuguese": "por_Latn", "Arabic": "arb_Arab",
        "Chinese": "cmn_Hani", "Russian": "rus_Cyrl", "Indonesian": "ind_Latn", "Hindi": "hin_Deva",
        "Turkish": "tur_Latn", "Swahili": "swh_Latn"}
ENGLISH = "eng_Latn"
MIN_DOC_CHARS, MAX_DOC_CHARS = 200, 20_000


def cmd_plan(a) -> None:
    d = json.loads(Path(a.sizes).read_text())
    size = defaultdict(int)
    for c in d["size"]["splits"]:
        # "_removed" holds what FineWeb-2's filters dropped; "und_" folders hold text of undetermined language
        if c["split"] == "train" and not c["config"].endswith("_removed") and not c["config"].startswith("und_"):
            size[c["config"]] += c["num_bytes_parquet_files"]
    size = {k: v for k, v in size.items() if v >= a.min_mb * 1e6}
    size[ENGLISH] = a.english_mb * 1e6  # English Wikipedia stands in; its budget follows the same rule
    # available characters: about one character per compressed parquet byte (UTF-8 text compresses 2-4x, and
    # non-Latin scripts take 2-3 bytes per character), with at most `max_epochs` passes over a small language
    avail = {k: v * a.max_epochs for k, v in size.items()}
    w = {k: v ** a.alpha for k, v in size.items()}
    budget, free, total = {}, set(size), a.chars
    while free:  # water-filling: give capped languages their cap and share the rest among the others
        s = sum(w[k] for k in free)
        capped = {k for k in free if total * w[k] / s >= avail[k]}
        if not capped:
            for k in free:
                budget[k] = total * w[k] / s
            break
        for k in capped:
            budget[k] = avail[k]
            total -= avail[k]
        free -= capped
    plan = {k: {"chars": int(budget[k]), "parquet_mb": round(size[k] / 1e6, 1),
                "fetch_chars": int(min(budget[k], size[k]))} for k in sorted(budget, key=lambda k: -budget[k])}
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps(plan, indent=1))
    tot = sum(p["chars"] for p in plan.values())
    print(f"{len(plan)} languages, {tot / 1e9:.1f}B characters planned "
          f"({sum(p['fetch_chars'] for p in plan.values()) / 1e9:.1f}B to fetch; the rest is repetition of small languages)")
    for k in list(plan)[:8] + ["..."] + list(plan)[-5:]:
        if k == "...":
            print("  ...")
            continue
        print(f"  {k:10s} {plan[k]['chars'] / 1e6:8.0f}M chars  (parquet {plan[k]['parquet_mb']:.0f} MB)")
    for name, cfg in OURS.items():
        print(f"  ours {name:11s} {cfg}: {plan.get(cfg, {}).get('chars', 0) / 1e6:.0f}M chars")


def _fetch_one(args) -> tuple[str, int, int]:
    from datasets import load_dataset

    cfg, budget, root, *rest = args
    retry_raw = bool(rest and rest[0])
    path = Path(root) / "raw" / f"{cfg}.jsonl"
    if path.exists():
        return cfg, -1, 0
    tmp = path.with_suffix(".tmp")
    chars = docs = 0
    try:
        if cfg == ENGLISH:
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        elif retry_raw:  # some configs' files have an extra column the declared features lack: read the files directly
            ds = load_dataset("parquet", data_files=f"hf://datasets/HuggingFaceFW/fineweb-2/data/{cfg}/train/*.parquet",
                              split="train", streaming=True).select_columns(["text"])
        else:
            ds = load_dataset("HuggingFaceFW/fineweb-2", name=cfg, split="train", streaming=True)
        with tmp.open("w") as f:
            for ex in ds:
                t = ex["text"].strip()
                if len(t) < MIN_DOC_CHARS:
                    continue
                t = t[:MAX_DOC_CHARS]
                f.write(json.dumps({"t": t}, ensure_ascii=False) + "\n")
                chars, docs = chars + len(t), docs + 1
                if chars >= budget:
                    break
    except Exception as e:  # one language failing (network, missing config) must not stop the others; a rerun retries it
        if type(e).__name__ == "CastError" and not retry_raw:
            return _fetch_one((cfg, budget, root, True))
        return cfg, -2, f"{type(e).__name__}: {str(e)[:200]}"
    tmp.rename(path)
    return cfg, docs, chars


def cmd_fetch(a) -> None:
    root = Path(a.root)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())
    jobs = [(k, v["fetch_chars"], a.root) for k, v in plan.items() if not (root / "raw" / f"{k}.jsonl").exists()]
    print(f"{len(jobs)} languages to fetch", flush=True)
    with mp.Pool(a.procs) as pool:
        failed = 0
        for i, (cfg, docs, chars) in enumerate(pool.imap_unordered(_fetch_one, jobs, chunksize=1)):
            if docs == -2:
                failed += 1
                print(f"[{i + 1}/{len(jobs)}] {cfg}: FAILED {chars}", flush=True)
            else:
                print(f"[{i + 1}/{len(jobs)}] {cfg}: {docs} docs, {chars / 1e6:.0f}M chars", flush=True)
    if failed:  # the driver reruns fetch until every language is in or the failures persist
        print(f"{failed} languages failed", flush=True)
    else:
        print("FETCH-DONE", flush=True)


def _sample(path: Path, chars: int, skip_docs: int = 0):
    """The first `chars` characters of a raw file after skipping `skip_docs` documents."""
    n = 0
    with path.open() as f:
        for j, line in enumerate(f):
            if j < skip_docs:
                continue
            t = json.loads(line)["t"]
            yield t
            n += len(t)
            if n >= chars:
                return


def cmd_tokenizers(a) -> None:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer

    root = Path(a.root)
    plan = json.loads((root / "plan.json").read_text())
    langs = [k for k in plan if (root / "raw" / f"{k}.jsonl").exists()]
    # tokenizer sample: the same alpha-flattened shares as the corpus, then capped, so no language dominates merges
    tot = sum(plan[k]["chars"] for k in langs)
    share = {k: min(a.sample_chars * plan[k]["chars"] / tot, a.max_lang_chars) for k in langs}
    held = {}  # held-out documents per language for measuring: the documents after the first 2,000
    for k in langs:
        held[k] = list(_sample(root / "raw" / f"{k}.jsonl", a.eval_chars, skip_docs=2000))
    task = []
    if a.task_data:
        for line in (Path(a.task_data) / "train.jsonl").open():
            r = json.loads(line)
            task += [r["input"], r["output"]]
            if len(task) >= a.task_rows:
                break

    def corpus():
        order = langs[:]
        random.Random(0).shuffle(order)
        for k in order:
            yield from _sample(root / "raw" / f"{k}.jsonl", share[k])
        for _ in range(a.task_repeat):  # domain terms and the output format stay cheap to encode
            yield from task

    out = root / "tokenizers"
    out.mkdir(exist_ok=True)
    toks = {}
    for v in a.vocab:
        p = out / f"bpe{v}.json"
        if not p.exists():
            tk = ByteLevelBPETokenizer()
            tk.train_from_iterator(corpus(), vocab_size=v, special_tokens=["<pad>", "<eos>"], show_progress=False)
            tk.save(str(p))
            print(f"trained {p}", flush=True)
        toks[f"bpe{v}"] = Tokenizer.from_file(str(p))
    if a.compare:
        toks["current"] = Tokenizer.from_file(a.compare)
    res = {}
    for k in langs:
        nbytes = sum(len(t.encode()) for t in held[k])
        res[k] = {name: nbytes / max(1, sum(len(e.ids) for e in tk.encode_batch(held[k]))) for name, tk in toks.items()}
    (out / "bytes_per_token.json").write_text(json.dumps(res, indent=1))
    names = list(toks)
    print("bytes per token on held-out text (higher is better)")
    print(f"{'language':12s} " + " ".join(f"{n:>9s}" for n in names))
    ours = [OURS[x] for x in OURS] + [ENGLISH]
    for k in ours + ["..."] + sorted(set(langs) - set(ours), key=lambda k: -plan[k]["chars"]):
        if k == "...":
            print("  (other languages)")
            continue
        if k in res:
            print(f"{k:12s} " + " ".join(f"{res[k][n]:9.2f}" for n in names))
    for label, group in (("our 11", [k for k in ours if k in res]), ("other", [k for k in res if k not in ours])):
        if group:
            print(f"{'mean ' + label:12s} " + " ".join(f"{sum(res[k][n] for k in group) / len(group):9.2f}" for n in names))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--sizes", required=True); p.add_argument("--root", required=True)
    p.add_argument("--chars", type=float, default=18e9); p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--min-mb", type=float, default=10); p.add_argument("--max-epochs", type=float, default=4)
    p.add_argument("--english-mb", type=float, default=20000); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("fetch")
    p.add_argument("--root", required=True); p.add_argument("--procs", type=int, default=16); p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("tokenizers")
    p.add_argument("--root", required=True); p.add_argument("--vocab", type=int, nargs="+", default=[8000, 12000, 16000, 24000])
    p.add_argument("--sample-chars", type=float, default=3e9); p.add_argument("--max-lang-chars", type=float, default=60e6)
    p.add_argument("--eval-chars", type=int, default=300_000); p.add_argument("--compare", default=None)
    p.add_argument("--task-data", default=None); p.add_argument("--task-rows", type=int, default=400_000)
    p.add_argument("--task-repeat", type=int, default=3); p.set_defaults(fn=cmd_tokenizers)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
