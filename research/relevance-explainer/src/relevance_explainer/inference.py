"""Inference API for the claim-carrying relevance explainer: the entry point for services and batch jobs.

A model bundle is one directory:

    explainer-int8.pt     model weights (INT8, per-row scales) and tokenizer (tiny/export.py)
    calibration.json      verdict temperature and conformal thresholds (scripts/calibrate_c.py)
    faith_scorer.json     faithfulness scorer (scripts/faith_scorer.py)
    bundle.json           name, version, input format, SHA-256 of each file, and the run and commit it was built from

Use:

    from relevance_explainer.inference import Explainer
    ex = Explainer.from_bundle("dist/relevance-explainer-c-dpo")
    out = ex.explain(record, search="maternal mortality Kenya 2015", language="English")
    out["verdict"], out["explanation"], out["faithfulness"], out["all_verified"]

`record` is a catalog record as built by synth/sources.py (keys such as kind, title, abstract, coverage, years); see
schemas.py for the item types. `explain_text` takes an already rendered input. The command line:

    relevance-explainer bundle --run runs/final-C-dpo --out dist/relevance-explainer-c-dpo
    relevance-explainer explain --bundle dist/relevance-explainer-c-dpo --input requests.jsonl > answers.jsonl

Each input line is {"record": {...}, "search": "...", "language": "...", "facets": {...}} or {"text": "<rendered>"}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

FILES = ("explainer-int8.pt", "calibration.json", "faith_scorer.json")
# Render version of served inputs. Version 5 parses searches and checks records as version 4 (the training data) with
# the level-5 fixes of synth/understand.py: no places from short everyday words or short alternate city names, and
# plural subnational words. Version 4 inputs are unchanged (notebook 2026-10-16).
RENDER_VERSION = 5
INPUT_FORMAT = f"itemfirst_nosim, render version {RENDER_VERSION}"
_SIM = re.compile(r"\nRetrieval similarity: [-0-9.]+")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render(record: dict, search: str, language: str, facets: dict | None = None, understander=None,
           render_version: int = RENDER_VERSION) -> str:
    """The model input for a record and a search, as the app renders it (item-first, query understanding and checks at
    `render_version`, no retrieval similarity line). render_version=4 gives the inputs of the evaluation sets."""
    from .synth.realistic import item_first, render_v2
    from .synth.understand import Understander
    u = understander or Understander()
    p = {"query": search.strip(), "language": language, "record": record, "facet_values": facets or {},
         "render_version": render_version}
    parsed = u.parse(p["query"], p["facet_values"], level=render_version)
    return _SIM.sub("", item_first(render_v2(p, u, parsed, 0.0)))


class Explainer:
    def __init__(self, bundle: str | Path, threads: int | None = None, verify_files: bool = True):
        import torch
        from .answer_c import Assets
        from .tiny.export import load_int8
        from .tiny.scriptguard import token_masks
        self.dir = Path(bundle)
        self.meta = json.loads((self.dir / "bundle.json").read_text())
        if verify_files:
            for name, digest in self.meta["sha256"].items():
                if sha256(self.dir / name) != digest:
                    raise ValueError(f"{name} does not match the checksum in bundle.json")
        if threads:
            torch.set_num_threads(threads)
        model, tk = load_int8(self.dir / "explainer-int8.pt", "cpu")
        self.assets = Assets.from_run(model, tk, self.dir)
        self.masks = token_masks(tk)
        self._u = None

    @classmethod
    def from_bundle(cls, path: str | Path, **kw) -> "Explainer":
        return cls(path, **kw)

    @property
    def understander(self):
        if self._u is None:
            from .synth.understand import Understander
            self._u = Understander()
        return self._u

    def explain(self, record: dict, search: str, language: str, facets: dict | None = None, **kw) -> dict:
        text = render(record, search, language, facets, self.understander)
        return {**self.explain_text(text, **kw), "input": text}

    def explain_text(self, text: str, retries: int = 3, guard: bool = True) -> dict:
        from .answer_c import answer
        from .tiny.scriptguard import banned_for
        t0 = time.time()
        o = answer(self.assets, text, retries=retries, banned=banned_for(self.masks, text) if guard else None)
        ck = o["check"]
        return {"verdict": o["verdict"], "probabilities": o["probs"], "verdict_set": o["verdict_set"],
                "set_level": o["set_level"], "expected_grade": o["expected_grade"],
                "tldr": ck["sections"].get("tldr", ""), "explanation": ck["sections"].get("exp", ""),
                "claims": ck["claims"], "references": ck["refs"], "all_verified": ck["all_verified"],
                "unverified": ck["unverified"], "flags": ck["flags"], "errors": ck["errors"], "faithfulness": o["faith"],
                "retried": o["retried"], "trust": o["trust"], "verdict_view": o["verdict_view"],
                "unsupported_language": o["unsupported_language"],
                "raw": o["text"], "model": self.meta.get("name"),
                "model_version": self.meta.get("version"), "seconds": round(time.time() - t0, 3)}


def build_bundle(run: Path, out: Path, name: str, version: str) -> dict:
    """Copy a run's serving files into a bundle directory and write bundle.json."""
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (run / f).exists():
            raise FileNotFoundError(run / f)
        shutil.copy2(run / f, out / f)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    meta = {"name": name, "version": version, "input_format": INPUT_FORMAT, "built_from_run": str(run),
            "code_commit": commit, "sha256": {f: sha256(out / f) for f in FILES}}
    (out / "bundle.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="relevance-explainer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle", help="assemble a model bundle from a trained run")
    b.add_argument("--run", required=True); b.add_argument("--out", required=True)
    b.add_argument("--name", default="relevance-explainer-c-dpo"); b.add_argument("--version", default="0.2.0")
    e = sub.add_parser("explain", help="explain JSONL requests with a bundle")
    e.add_argument("--bundle", required=True); e.add_argument("--input", default="-")
    e.add_argument("--threads", type=int, default=4); e.add_argument("--retries", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        print(json.dumps(build_bundle(Path(a.run), Path(a.out), a.name, a.version), indent=1))
        return
    ex = Explainer(a.bundle, threads=a.threads)
    src = sys.stdin if a.input == "-" else open(a.input)
    for line in src:
        if not line.strip():
            continue
        r = json.loads(line)
        o = (ex.explain_text(r["text"], retries=a.retries) if "text" in r else
             ex.explain(r["record"], r["search"], r["language"], r.get("facets"), retries=a.retries))
        print(json.dumps(o, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
