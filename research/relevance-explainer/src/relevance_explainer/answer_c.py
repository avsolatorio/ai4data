"""One answer from a claim-carrying (format C) model with its checks, as the app and an API serve it (CPU or GPU).

Steps, all on the student model:
  1. verdict probabilities at the first output step, temperature-scaled (<run>/calibration.json, scripts/calibrate_c.py),
     and the split-conformal verdict set at the calibrated level;
  2. greedy answer without the repeat ban (references repeat claim values by design, claims.NO_REPEAT_TAGGED);
  3. verification of claims and references against the input (claims.verify); when the answer is not fully verified,
     one round of `retries` samples after the same verdict line, keeping the best by the verifier;
  4. the expected grade E[g] = sum_v P(v) E[g | v] from the 11 grade tokens after each verdict line;
  5. the faithfulness score of the kept answer (faith.FaithScorer, <run>/faith_scorer.json) and its band.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .claims import _REF, NO_REPEAT_TAGGED, QUERY_FIELDS, claim_source, from_model, parse_record, record_fields, sanitize_input, verify
from .faith import FaithScorer, model_features, verifier_features

SEP = "\n\n### Judgment\n"
EOS = 1
VERDICTS = ["relevant", "partly_relevant", "irrelevant"]
FAITH_BANDS = ((0.8, "high"), (0.5, "check"), (0.0, "low"))  # judge rated 4-5: 94% at 0.8 or more, 46% below 0.5


@dataclass
class Assets:
    model: torch.nn.Module
    tk: object
    calibration: dict | None = None
    scorer: FaithScorer | None = None

    @classmethod
    def from_run(cls, model, tk, run: str | Path) -> "Assets":
        run = Path(run)
        cal = json.loads((run / "calibration.json").read_text()) if (run / "calibration.json").exists() else None
        sc = FaithScorer(run / "faith_scorer.json") if (run / "faith_scorer.json").exists() else None
        return cls(model, tk, cal, sc)


def segments(section_raw: str, refs: list[dict]) -> list[dict]:
    """A section in attribute form split into plain text and references, each reference with its check."""
    out, at, k = [], 0, 0
    for m in _REF.finditer(section_raw):
        if m.start() > at:
            out.append({"text": section_raw[at:m.start()]})
        r = refs[k] if k < len(refs) else {"ok": False, "reason": "not checked"}
        out.append({"text": m.group(2), "claim": int(m.group(1)), "ok": r["ok"], "reason": r.get("reason", "")})
        at, k = m.end(), k + 1
    if at < len(section_raw):
        out.append({"text": section_raw[at:]})
    return out


def _score(ck) -> tuple:
    return (ck.all_verified, ck.verified_share if ck.verified_share is not None else 1.0, -len(ck.unverified))


def _band(p: float) -> str:
    return next(name for lo, name in FAITH_BANDS if p >= lo)


PLAIN = {"relevant": "matches the search", "partly_relevant": "partly matches the search", "irrelevant": "does not match the search"}
_MET = ("covered", "matches", "mentioned in the metadata", "named in the metadata")
_UNMET = ("not covered", "not mentioned", "differs")


def record_view(text: str) -> dict:
    """The record part of a rendered input, for display: its item type and its fields in order."""
    pairs = parse_record(text)
    return {"item_type": next((v for k, v in pairs if k == "Item type"), ""),
            "fields": [[k, v] for k, v in pairs if k not in QUERY_FIELDS and k not in ("Checks", "Item type")]}


def parse_checks(text: str) -> list[dict]:
    """The rule-based checks of the search's constraints against the record (the input's Checks line), each with a
    state: met, unmet, or info (the status neither confirms nor rules out the constraint)."""
    line = next((v for k, v in parse_record(text) if k == "Checks"), "none")
    out = []
    for part in [] if line.strip() == "none" else line.split(" | "):
        label, _, status = part.partition(": ")
        st = status.strip()
        state = ("unmet" if st.startswith(_UNMET) else "met" if st.startswith(_MET) and "not stated" not in st else "info")
        out.append({"constraint": label.strip(), "status": st, "state": state})
    return out


def trust(o: dict) -> dict:
    """One display state for the explanation from its checks: "checked" (show as is), "check" (show, ask the reader to
    check the record, with the reasons) or "withheld" (show the record; the explanation only on request). Uncertainty
    about the verdict itself is shown by verdict_view."""
    ck, faith = o["check"], o.get("faith")
    reasons = []
    if o.get("unsupported_language"):  # language.py: answers outside the eleven languages were garbled
        return {"state": "withheld", "reasons": [o["unsupported_language"]]}
    if o["verdict"] not in VERDICTS or ck["errors"]:
        return {"state": "withheld", "reasons": ["The model did not produce a complete answer."]}
    if not ck["all_verified"]:
        reasons.append("Some facts in the explanation could not be found in the record.")
    if faith and faith["band"] == "low":
        reasons.append("The explanation is likely not supported by the record.")
    elif faith and faith["band"] == "check":
        reasons.append("The explanation may not fully follow from the record.")
    withheld = faith is not None and faith["band"] == "low" and not ck["all_verified"]
    return {"state": "withheld" if withheld else "check" if reasons else "checked", "reasons": reasons}


def verdict_view(o: dict) -> dict:
    """The verdict as a reader sees it. A conformal set with more than one verdict is shown as one hedged label."""
    vset = o.get("verdict_set") or [o["verdict"]]
    if o["verdict"] not in VERDICTS or o.get("unsupported_language"):  # verdicts in other languages were right 2 times in 4
        return {"key": "invalid", "label": "No answer", "uncertain": True, "between": []}
    labels = {"relevant": "Matches the search", "partly_relevant": "Partly matches", "irrelevant": "Does not match"}
    if len(vset) <= 1:
        return {"key": o["verdict"], "label": labels[o["verdict"]], "uncertain": False, "between": vset}
    if set(vset) == {"relevant", "partly_relevant"}:
        label, key = "May match", "relevant"
    elif set(vset) == {"partly_relevant", "irrelevant"}:
        label, key = "May partly match", "partly_relevant"
    else:
        label, key = "Unclear", "partly_relevant"
    return {"key": key, "label": label, "uncertain": True, "between": [v for v in VERDICTS if v in vset]}


def _checked(ck, fields: dict | None = None) -> dict:
    return {"all_verified": ck.all_verified, "grade": ck.grade,
            "claims": [{"id": c.id, "field": c.field, "value": c.value, "ok": c.ok, "reason": c.reason,
                        "source": claim_source(c, fields) if fields is not None and c.ok else None}
                       for c in ck.claims.values()],
            "refs": [{"claim": r.claim, "text": r.text, "section": r.section, "ok": r.ok, "reason": r.reason} for r in ck.refs],
            "sections": ck.sections, "unverified": ck.unverified, "flags": ck.flags, "errors": ck.errors}


@torch.no_grad()
def _last_logits(model, seqs: list[list[int]]) -> torch.Tensor:
    """Final-position logits of each sequence, one forward pass each (no padding, so positions are exact)."""
    dev = next(model.parameters()).device
    return torch.stack([model(torch.tensor([s], device=dev))[0, -1].float().cpu() for s in seqs])


def answer(a: Assets, text: str, *, retries: int = 3, retry_temperature: float = 0.7, max_new: int = 420,
           level: str = "0.90", banned: torch.Tensor | None = None, seed: int = 0, stream=None) -> dict:
    """`stream(token_ids)` is called with the growing greedy answer; `banned` is a [vocab] mask (tiny.scriptguard)."""
    from .viewer import stream as kv_stream
    model, tk = a.model, a.tk
    t0 = time.time()
    clean = sanitize_input(text)
    prompt = tk.encode(clean + SEP).ids
    first = [tk.encode(v).ids[0] for v in VERDICTS]
    # 1-2. greedy answer; the first step gives the verdict distribution
    toks, logit3 = [], None
    for k, (tok, logits) in enumerate(kv_stream(model, prompt, max_new=max_new, banned=banned, no_repeat=NO_REPEAT_TAGGED)):
        if k == 0:
            logit3 = logits.float()[first].numpy()
        if tok == EOS:
            break
        toks.append(tok)
        if stream is not None:
            stream(toks)
    t_greedy = time.time() - t0
    T = a.calibration["temperature"] if a.calibration else 1.0
    z = logit3 / T
    p = np.exp(z - z.max())
    p = p / p.sum()
    probs = dict(zip(VERDICTS, p.tolist()))
    vset = None
    if a.calibration and level in a.calibration.get("conformal", {}):
        q = a.calibration["conformal"][level]
        vset = [v for v, pv in probs.items() if 1 - pv <= q]
    out = tk.decode(toks, skip_special_tokens=False).strip()
    verdict = out.split("\n", 1)[0].strip()
    ck = verify(from_model(out), text)
    attempts = [{"text": out, "all_verified": ck.all_verified, "kept": True}]
    # 3. one retry round after the same verdict line when the greedy answer is not fully verified
    retried = 0
    if retries and not ck.all_verified and verdict in VERDICTS:
        vline = tk.encode(clean + SEP + verdict + "\n").ids
        torch.manual_seed(seed)
        ban = banned[None].expand(retries, -1) if banned is not None else None
        outs = model.generate_batch([vline] * retries, EOS, max_new, no_repeat=NO_REPEAT_TAGGED,
                                    temperature=retry_temperature, banned=ban)
        retried = retries
        for o in outs:
            c = verdict + "\n" + tk.decode(o, skip_special_tokens=False).strip()
            cc = verify(from_model(c), text)
            attempts.append({"text": c, "all_verified": cc.all_verified, "kept": False})
            if _score(cc) > _score(ck):
                out, ck = c, cc
        for x in attempts:
            x["kept"] = x["text"] == out
    # 4. expected grade
    g_ids = [tk.token_to_id(f"<|g:{i}|>") for i in range(11)]
    eg = None
    if all(g is not None for g in g_ids):
        lg = _last_logits(model, [prompt + tk.encode(v + "\n").ids for v in VERDICTS])[:, g_ids]
        eg_v = (torch.softmax(lg, -1) * torch.arange(11.0)).sum(-1).numpy()
        eg = float((p * eg_v).sum())
    # 5. faithfulness score of the kept answer
    faith = None
    if a.scorer is not None and verdict in VERDICTS:
        body = out.split("\n", 1)[1] if "\n" in out else ""
        gen, hid = model_features(model, [tk.encode(text + SEP + verdict + "\n").ids], [tk.encode(body).ids + [EOS]], batch=1)
        if not np.isnan(gen).any():
            s = float(a.scorer.proba(np.array([verifier_features(out, text)]), gen, hid)[0])
            faith = {"score": round(s, 3), "band": _band(s)}
    o = {"verdict": verdict if verdict in VERDICTS else "invalid", "probs": {k: round(v, 4) for k, v in probs.items()},
         "verdict_set": vset, "set_level": level if vset is not None else None,
         "expected_grade": None if eg is None else round(eg, 2), "check": _checked(ck, record_fields(text)), "faith": faith,
         "text": out, "attempts": attempts, "retried": retried,
         "stats": {"prompt_tokens": len(prompt), "output_tokens": len(toks), "greedy_s": round(t_greedy, 2),
                   "total_s": round(time.time() - t0, 2)}}
    from .language import unsupported
    o["unsupported_language"] = unsupported(text)
    o["trust"] = trust(o)
    o["verdict_view"] = verdict_view(o)
    return o
