"""Interactive viewer for the relevance explainer, as a JupyterLab widget app.

Pick one of 100 curated test inputs (or edit it), choose a model and decoding settings, and watch the INT8 model
generate its verdict and narrative on CPU. The view shows the parsed input and the signals the model receives,
the verdict probabilities, timing, the teacher's reference, the three audit labels (stored, teacher thinking mode,
Gemma 4) and the offline judge scores of the best model on the same input.

  python -m relevance_explainer.viewer build      # writes app/examples.json from the evaluated test sample
  # in a notebook with the relevance-explainer kernel:
  from relevance_explainer.viewer import launch; launch()
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "app" / "examples.json"
SEP = "\n\n### Judgment\n"
EOS = 1
VERDICTS = ["relevant", "partly_relevant", "irrelevant"]
MODELS = {
    "Realistic + DPO (item-first)": "runs/v2b-cont-itemfirst-dpo/explainer-int8.pt",
    "Round 7 + DPO (best)": "runs/f6-dpo/explainer-int8.pt",
    "Round 7 + DPO + copy head": "runs/f6-copy-dpo/explainer-int8.pt",
    "Round 7, fine-tuned only": "runs/f6-sft/explainer-int8.pt",
    "Round 6 + DPO": "runs/f5-dpo/explainer-int8.pt",
}
RTL = {"Arabic"}
# Input format per model: "itemfirst" models read the item-first realistic format with query understanding and checks.
MODEL_FORMAT = {"Realistic + DPO (item-first)": "itemfirst"}


# ----------------------------------------------------------------------------------------------- examples

def build(n: int = 100, seed: int = 3) -> None:
    """Select n examples from the 800 judged test inputs that have all three labels, spread over languages,
    sources and verdicts, with some contested labels."""
    data = ROOT / "data" / "f2"
    test = [json.loads(l) for l in (data / "test.jsonl").read_text().splitlines()]
    gemma = {x["i"]: x["second"] for x in map(json.loads, (data / "audit_test_gemma-4-31b_800.jsonl").read_text().splitlines())}
    think = {x["i"]: x["thinking"] for x in map(json.loads, (data / "think_test_800.jsonl").read_text().splitlines())}
    run = ROOT / "runs" / "f6-dpo"
    gens = [json.loads(l) for l in (run / "f2testsg_generations_int8.jsonl").read_text().splitlines()]
    k_to_i = random.Random(0).sample(range(len(gens)), 800)  # the judge's sample order
    judged: dict[int, dict] = {}
    for suffix, name in (("", "teacher"), ("_gemma4", "gemma4")):
        for it in json.loads((run / f"judge_f2testsg_int8{suffix}.json").read_text())["items"]:
            if it["who"] == "student" and it["score"]:
                judged.setdefault(k_to_i[it["k"]], {})[name] = it["score"]
    cands = [i for i in gemma if gemma[i] and think.get(i)]
    random.Random(seed).shuffle(cands)
    contested = lambda i: len({test[i]["verdict"], think[i], gemma[i]}) > 1  # noqa: E731
    want_v = {"relevant": 25, "partly_relevant": 35, "irrelevant": 40}
    pick, by_lang, by_src, by_v, n_cont = [], Counter(), Counter(), Counter(), 0
    for rule in ("doc", "lang", "fill"):  # documents are rarer, so they get a first pass
        for i in cands:
            if len(pick) >= n or i in pick:
                continue
            r = test[i]
            if rule == "doc" and (r["source"] != "document" or by_src["document"] >= 12):
                continue
            if by_v[r["verdict"]] >= want_v[r["verdict"]]:
                continue
            if rule == "lang" and (by_lang[r["language"]] >= 7 or by_src[r["source"]] >= 30):
                continue
            if contested(i) and n_cont >= 18:
                continue
            pick.append(i)
            by_lang[r["language"]] += 1
            by_src[r["source"]] += 1
            by_v[r["verdict"]] += 1
            n_cont += contested(i)
    out = []
    for i in pick:
        r = test[i]
        out.append({"id": i, "language": r["language"], "source": r["source"], "input": r["input"],
                    "reference": r["output"], "labels": {"stored": r["verdict"], "thinking": think[i], "gemma4": gemma[i]},
                    "offline": {"text": gens[i]["generated"], "judged": judged.get(i, {})}})
    title = lambda t: (m.group(1).strip() if (m := re.search(r"^Result: (.+)$", t, re.M)) else "")  # noqa: E731
    seen = Counter(title(json.loads(l)["input"]) for l in (ROOT / "data" / "f6" / "train.jsonl").read_text().splitlines())
    for x in out:  # how often the same catalog item occurs in the best model's training data, with other searches
        x["train_item_count"] = seen[title(x["input"])]
    out.sort(key=lambda x: (x["language"] != "English", x["language"], x["source"]))
    EXAMPLES.parent.mkdir(exist_ok=True)
    EXAMPLES.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {len(out)} examples to {EXAMPLES}")
    print("languages", dict(by_lang), "\nsources", dict(by_src), "\nverdicts", dict(by_v), "\ncontested", n_cont)


# ----------------------------------------------------------------------------------------------- model

_CACHE: dict[str, tuple] = {}


def load(name: str):
    if name not in _CACHE:
        from .tiny.export import load_int8
        from .tiny.scriptguard import token_masks
        import warnings
        t = time.time()
        path = ROOT / MODELS[name]
        with warnings.catch_warnings():  # torch.load's weights_only notice; the files are this project's own exports
            warnings.simplefilter("ignore", FutureWarning)
            model, tk = load_int8(path, "cpu")
        _CACHE[name] = (model, tk, token_masks(tk), time.time() - t, path.stat().st_size)
    return _CACHE[name]


@torch.no_grad()
def stream(model, prompt: list[int], max_new: int = 260, banned: torch.Tensor | None = None, no_repeat: int = 4,
           temperature: float = 0.0, top_k: int = 40, seed: int = 0):
    """Yield (token, step_logits) one token at a time with a KV cache; same rules as TinyLM.generate_batch."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([prompt])
    caches = [{} for _ in model.order]
    x = model.emb(ids)
    for j, c in zip(model.order, caches):
        x = model.blocks[j](x, model.cos, model.sin, None, c)
    h = model.norm(x)
    if model.c.copy:
        keys, src = F.linear(h, model.copy_k.weight), ids
        logits = model._mix(h[:, -1:], keys, src, torch.ones(1, 1, ids.size(1), dtype=torch.bool))[0, 0]
    else:
        logits = F.linear(h[0, -1], model.emb.weight)
    out, seen, pos = [], {}, len(prompt)
    for _ in range(max_new):
        logits = logits.float()
        if banned is not None:
            logits = logits.masked_fill(banned, float("-inf"))
        if no_repeat and len(out) >= no_repeat - 1:
            block = seen.get(tuple(out[len(out) - no_repeat + 1 :]), ())
            if block:
                logits[list(block)] = float("-inf")
        if temperature > 0:
            top = logits.topk(top_k)
            nxt = top.indices[torch.multinomial(torch.softmax(top.values / temperature, -1), 1, generator=g)].item()
        else:
            nxt = int(logits.argmax())
        yield nxt, logits
        if nxt == EOS:
            return
        if no_repeat and len(out) >= no_repeat - 1:
            seen.setdefault(tuple(out[len(out) - no_repeat + 1 :]), set()).add(nxt)
        out.append(nxt)
        x = model.emb(torch.tensor([[nxt]]))
        at = torch.tensor([pos])
        cos, sin = model.cos[at][:, None], model.sin[at][:, None]  # (1, 1, head_dim/2): this token's position
        mask = torch.ones(1, 1, 1, pos + 1, dtype=torch.bool)
        for j, c in zip(model.order, caches):
            x = model.blocks[j](x, cos, sin, mask, c)
        h = model.norm(x)
        if model.c.copy:
            keys = torch.cat([keys, F.linear(h, model.copy_k.weight)], 1)
            src = torch.cat([src, torch.tensor([[nxt]])], 1)
            logits = model._mix(h, keys, src, torch.ones(1, 1, src.size(1), dtype=torch.bool))[0, 0]
        else:
            logits = F.linear(h[0, -1], model.emb.weight)
        pos += 1


def verdict_probs(tk, logits: torch.Tensor) -> dict[str, float]:
    """Probabilities of the three verdicts from the first output step, renormalized over their first tokens."""
    first = [tk.encode(v).ids[0] for v in VERDICTS]
    p = torch.softmax(logits.float(), -1)[first]
    p = p / p.sum().clamp_min(1e-12)
    return dict(zip(VERDICTS, p.tolist()))


# ----------------------------------------------------------------------------------------------- parsing

def parse_input(text: str) -> dict:
    f, meta = {}, []
    for line in text.split("\n"):
        key, sep, val = line.partition(": ")
        if not sep:
            continue
        if key in ("Respond in", "Item type", "Query", "Filters", "Retrieval similarity", "Result", "Filter check",
                   "Reranker score"):
            f[key] = val.strip()
        else:
            meta.append((key, val.strip()))
    f["meta"] = meta
    return f


def filter_chips(s: str) -> list[tuple[str, str]]:
    chips = []
    for part in [p.strip() for p in s.split(";") if p.strip()]:
        low = part.lower()
        kind = ("bad" if ("not covered" in low or "differs" in low or "outside" in low)
                else "warn" if ("cannot" in low or "not verifiable" in low or "unknown" in low or "partly" in low)
                else "ok" if ("covered" in low or "match" in low)
                else "neutral")
        chips.append((part, kind))
    return chips


# ----------------------------------------------------------------------------------------------- view

CSS = """
<style>
.rx-app{--rx-bg:var(--jp-layout-color0,#fff);--rx-card:var(--jp-layout-color1,#fff);--rx-soft:var(--jp-layout-color2,#f3f4f6);
 --rx-line:var(--jp-border-color2,#e5e7eb);--rx-fg:var(--jp-ui-font-color0,#111827);--rx-mute:var(--jp-ui-font-color2,#6b7280);
 --rx-acc:#4f46e5;--rx-ok:#059669;--rx-okbg:rgba(5,150,105,.12);--rx-warn:#b45309;--rx-warnbg:rgba(217,119,6,.14);
 --rx-bad:#dc2626;--rx-badbg:rgba(220,38,38,.12);font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:var(--rx-fg)}
.rx-app *{box-sizing:border-box}
.rx-head{display:flex;align-items:center;gap:14px;padding:14px 18px;border-radius:14px;
 background:linear-gradient(120deg,#312e81,#4f46e5 55%,#0ea5e9);color:#fff;margin-bottom:12px}
.rx-head h1{font-size:19px;margin:0;font-weight:650;letter-spacing:.2px}
.rx-head p{margin:2px 0 0;font-size:12.5px;opacity:.88}
.rx-logo{width:38px;height:38px;border-radius:10px;background:rgba(255,255,255,.18);display:grid;place-items:center;font-size:20px}
.rx-pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:600;background:var(--rx-soft);color:var(--rx-mute);margin:2px 4px 2px 0}
.rx-card{background:var(--rx-card);border:1px solid var(--rx-line);border-radius:12px;padding:14px 16px;margin-bottom:12px}
.rx-card h3{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--rx-mute);margin:0 0 10px;font-weight:650}
.rx-query{font-size:17px;font-weight:600;line-height:1.35;margin:2px 0 8px}
.rx-title{font-size:15px;font-weight:600;margin:6px 0 4px}
.rx-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.rx-chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:8px;font-size:12.5px;margin:3px 6px 3px 0;border:1px solid var(--rx-line)}
.rx-chip.ok{background:var(--rx-okbg);color:var(--rx-ok);border-color:transparent}
.rx-chip.bad{background:var(--rx-badbg);color:var(--rx-bad);border-color:transparent}
.rx-chip.warn{background:var(--rx-warnbg);color:var(--rx-warn);border-color:transparent}
.rx-meter{height:8px;background:var(--rx-soft);border-radius:99px;overflow:hidden;width:180px}
.rx-meter>div{height:100%;border-radius:99px;background:linear-gradient(90deg,#f59e0b,#10b981)}
.rx-meta{display:grid;grid-template-columns:130px 1fr;gap:4px 12px;font-size:12.5px;margin-top:8px}
.rx-meta dt{color:var(--rx-mute)}.rx-meta dd{margin:0;max-height:5.4em;overflow:auto}
.rx-verdict{display:inline-flex;align-items:center;gap:8px;font-size:15px;font-weight:700;padding:6px 14px;border-radius:10px}
.rx-v-relevant{background:var(--rx-okbg);color:var(--rx-ok)}
.rx-v-partly_relevant{background:var(--rx-warnbg);color:var(--rx-warn)}
.rx-v-irrelevant{background:var(--rx-badbg);color:var(--rx-bad)}
.rx-v-invalid,.rx-v-pending{background:var(--rx-soft);color:var(--rx-mute)}
.rx-narr{font-size:15px;line-height:1.6;margin:12px 0 4px;min-height:3.2em}
.rx-caret{display:inline-block;width:8px;height:1.05em;background:var(--rx-acc);vertical-align:-2px;margin-left:2px;animation:rxb 1s steps(2) infinite}
@keyframes rxb{50%{opacity:0}}
.rx-bars{display:grid;grid-template-columns:118px 1fr 48px;gap:6px 10px;align-items:center;font-size:12.5px;margin-top:6px}
.rx-bar{height:9px;background:var(--rx-soft);border-radius:99px;overflow:hidden}
.rx-bar>div{height:100%;border-radius:99px}
.rx-stats{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--rx-mute);margin-top:12px;border-top:1px dashed var(--rx-line);padding-top:10px}
.rx-stats b{color:var(--rx-fg);font-size:14px;display:block}
.rx-labels{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:6px 0 10px}
.rx-lab{border:1px solid var(--rx-line);border-radius:10px;padding:8px 10px;font-size:12px}
.rx-lab span{display:block;color:var(--rx-mute);margin-bottom:4px}
.rx-note{font-size:12.5px;padding:8px 12px;border-radius:9px;margin-top:8px}
.rx-note.warn{background:var(--rx-warnbg);color:var(--rx-warn)}.rx-note.ok{background:var(--rx-okbg);color:var(--rx-ok)}
.rx-ref{font-size:13.5px;line-height:1.55;color:var(--rx-fg);background:var(--rx-soft);border-radius:9px;padding:10px 12px}
.rx-mute{color:var(--rx-mute);font-size:12px}
.rx-sec{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--rx-mute);font-weight:700;margin:2px 0 4px;
 padding-bottom:6px;border-bottom:2px solid var(--rx-acc);display:inline-block}
.rx-empty{color:var(--rx-mute);font-size:13.5px;padding:18px 4px}
.rx-side .widget-select select{font-size:12.5px;border-radius:10px;border:1px solid var(--jp-border-color2,#e5e7eb);padding:4px}
.rx-gen button,.rx-gen{border-radius:10px!important;font-weight:650!important}
.rx-ta textarea{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;border-radius:10px}
</style>
"""

COLORS = {"relevant": "#10b981", "partly_relevant": "#f59e0b", "irrelevant": "#ef4444"}
NICE = {"relevant": "Relevant", "partly_relevant": "Partly relevant", "irrelevant": "Irrelevant", "invalid": "No verdict",
        "pending": "Thinking"}
ITEM = {"statistical": ("📈", "Data360 indicator"), "survey": ("🗂️", "Microdata Library study"),
        "humanitarian": ("🆘", "HDX dataset"), "research": ("📄", "Policy Research Working Paper")}


def item_pill(item_type: str) -> str:
    icon, name = next((v for k, v in ITEM.items() if item_type.lower().startswith(k)), ("🔎", item_type))
    return f"{icon} {name}"


def _e(s: str) -> str:
    return html.escape(s or "")


def input_html(ex: dict, text: str) -> str:
    f = parse_input(text)
    lang = f.get("Respond in", "")
    d = ' dir="rtl"' if lang in RTL else ""
    sim = f.get("Retrieval similarity")
    try:
        simf = max(0.0, min(1.0, float(sim)))
    except (TypeError, ValueError):
        simf = None
    filters = f.get("Filters", "none")
    fchips = "".join(f'<span class="rx-chip">{_e(x.strip())}</span>' for x in filters.split(";") if x.strip()) \
        if filters and filters != "none" else '<span class="rx-mute">no filters</span>'
    checks = "".join(f'<span class="rx-chip {k}">{"✓" if k == "ok" else "✕" if k == "bad" else "?" if k == "warn" else "•"} {_e(c)}</span>'
                     for c, k in filter_chips(f.get("Filter check", ""))) or '<span class="rx-mute">no checkable filters</span>'
    meta = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in f["meta"])
    item = f.get("Item type", "")
    rr = f.get("Reranker score")
    return f"""
<div class="rx-card"><h3>Search</h3>
 <div class="rx-row"><span class="rx-pill">🌐 {_e(lang)}</span><span class="rx-pill" title="{_e(item)}">{_e(item_pill(item))}</span>
 <span class="rx-pill">test example #{ex.get('id', '–')}</span></div>
 <div class="rx-query"{d}>{_e(f.get('Query', '')) if d else '“' + _e(f.get('Query', '')) + '”'}</div>
 <div class="rx-row">{fchips}</div>
</div>
<div class="rx-card"><h3>Signals given to the model</h3>
 <div class="rx-row"><span class="rx-mute" style="width:150px">Retrieval similarity</span>
  {f'<div class="rx-meter"><div style="width:{simf * 100:.0f}%"></div></div><b>{simf:.2f}</b>' if simf is not None else '<span class="rx-mute">none</span>'}
  {f'<span class="rx-mute" style="margin-left:18px">Reranker</span><b>{_e(rr)}</b>' if rr else ''}</div>
 <div style="margin-top:8px"><span class="rx-mute">Deterministic filter check</span><div class="rx-row" style="margin-top:4px">{checks}</div></div>
</div>
<div class="rx-card"><h3>Result shown to the user</h3>
 <div class="rx-title">{_e(f.get('Result', ''))}</div>
 <dl class="rx-meta">{meta}</dl>
</div>"""


def output_html(state: dict) -> str:
    v = state.get("verdict", "pending")
    lang = state.get("language", "")
    d = ' dir="rtl"' if lang in RTL else ""
    caret = '<span class="rx-caret"></span>' if state.get("running") else ""
    narr = _e(state.get("narrative", "")) + caret
    if not state.get("started"):
        return '<div class="rx-card"><h3>Model output</h3><div class="rx-empty">Press <b>Generate</b> to run the 20 MB model on this input.</div></div>'
    bars = ""
    for k in VERDICTS:
        p = state.get("probs", {}).get(k)
        if p is None:
            continue
        bars += (f'<span>{NICE[k]}</span><div class="rx-bar"><div style="width:{p * 100:.1f}%;background:{COLORS[k]}"></div></div>'
                 f'<span style="text-align:right">{p * 100:.1f}%</span>')
    st = state.get("stats", {})
    stats = "".join(f"<div><b>{_e(str(val))}</b>{_e(lab)}</div>" for lab, val in st.items())
    return f"""
<div class="rx-card"><h3>Model output</h3>
 <span class="rx-verdict rx-v-{v}">{'●' if v != 'pending' else '◌'} {NICE.get(v, v)}</span>
 <div class="rx-narr"{d}>{narr}</div>
 {f'<h3 style="margin-top:14px">Verdict confidence (first output step)</h3><div class="rx-bars">{bars}</div>' if bars else ''}
 <div class="rx-stats">{stats}</div>
</div>"""


def compare_html(ex: dict, edited: bool, model_name: str, live: str | None) -> str:
    if not ex.get("labels"):
        return '<div class="rx-card"><h3>Reference</h3><div class="rx-empty">Custom input: no reference or labels.</div></div>'
    lab = ex["labels"]
    cells = "".join(f'<div class="rx-lab"><span>{name}</span><b class="rx-verdict rx-v-{lab[k]}" style="font-size:12px;padding:3px 8px">{NICE[lab[k]]}</b></div>'
                    for k, name in (("stored", "Teacher label (training target)"), ("thinking", "Teacher, thinking mode"), ("gemma4", "Gemma 4 (independent)")))
    agree = len(set(lab.values())) == 1
    note = ('<div class="rx-note ok">All three labelers agree on this example.</div>' if agree else
            '<div class="rx-note warn">Contested label: the labelers disagree, so more than one verdict is defensible here.</div>')
    ref_v, _, ref_n = ex["reference"].partition("\n")
    lang = ex.get("language", "")
    d = ' dir="rtl"' if lang in RTL else ""
    off = ex.get("offline", {})
    j = off.get("judged", {})
    judged = ""
    if j and not edited:
        bits = " · ".join(f"{who}: faithful {s['faithful']}/5, coherent {s['coherent']}/5" for who, s in (("Teacher judge", j.get("teacher")), ("Gemma 4", j.get("gemma4"))) if s)
        same = (live is not None and model_name == "Round 7 + DPO (best)" and not edited and live.strip() == off.get("text", "").strip())
        judged = (f'<h3 style="margin-top:14px">Offline judges, best model on this input</h3><div class="rx-mute">{_e(bits)}'
                  f'{" · identical to the live output above" if same else ""}</div>')
    return f"""
<div class="rx-card"><h3>Labels from the data audit</h3>
 <div class="rx-labels">{cells}</div>{note}
 {'<div class="rx-note warn">Input edited: labels and reference refer to the original input.</div>' if edited else ''}
 <h3 style="margin-top:14px">Teacher reference (Qwen3.8-27B)</h3>
 <span class="rx-verdict rx-v-{ref_v.strip()}" style="font-size:12px;padding:3px 8px">{NICE.get(ref_v.strip(), ref_v)}</span>
 <div class="rx-ref"{d} style="margin-top:8px">{_e(ref_n)}</div>
 {judged}
</div>"""


ABOUT = """
<div class="rx-card"><h3>How it works</h3><div style="font-size:13px;line-height:1.6">
A 20.6M-parameter decoder (7 layers, width 448, 12k byte-level BPE vocabulary) stored as one 21.6 MB INT8 file runs here
on CPU. It reads the search, the filters, the retriever's similarity score, the deterministic filter check and the item's
metadata, and writes a verdict line followed by two or three sentences in the query's language. Decoding is greedy with a
4-gram repeat block and a script constraint that blocks characters of scripts absent from the requested language and the
input. Training data: 201k synthetic examples written by Qwen3.8-27B over World Bank Data360 indicators, Microdata Library
studies, Policy Research Working Papers and HDX datasets, followed by preference tuning (DPO).
<br><br><b>Reading the scores.</b> Verdict accuracy on the standard test set is about 86%; the labels themselves are contested
on about one example in five (see the audit labels). Faithfulness of narratives is about 4.6 of 5 by an independent judge.
</div></div>"""


def launch():
    """Build and display the app in a notebook."""
    import ipywidgets as W
    from IPython.display import display

    examples = json.loads(EXAMPLES.read_text())
    torch.set_num_threads(4)

    def label(ex):
        flag = "" if len(set(ex["labels"].values())) == 1 else "  ⚑"
        q = parse_input(ex["input"]).get("Query", "")
        return f"{ex['language'][:3]} · {ex['source'][:4]} · {NICE[ex['labels']['stored']][:6]} · {q[:52]}{flag}"

    langs = ["All languages"] + sorted({e["language"] for e in examples})
    srcs = ["All sources"] + sorted({e["source"] for e in examples})
    verds = ["All labels", "Relevant", "Partly relevant", "Irrelevant", "Contested labels"]
    f_lang = W.Dropdown(options=langs, layout=W.Layout(width="100%"))
    f_src = W.Dropdown(options=srcs, layout=W.Layout(width="100%"))
    f_ver = W.Dropdown(options=verds, layout=W.Layout(width="100%"))
    f_txt = W.Text(placeholder="Search queries and titles…", layout=W.Layout(width="100%"))
    pick = W.Select(options=[], rows=22, layout=W.Layout(width="100%"))
    count = W.HTML()
    model_dd = W.Dropdown(options=list(MODELS), value="Round 7 + DPO (best)", description="Model", layout=W.Layout(width="100%"))
    mode = W.ToggleButtons(options=["Greedy", "Sample"], value="Greedy", style={"button_width": "88px"})
    temp = W.FloatSlider(value=0.7, min=0.1, max=1.5, step=0.05, description="Temperature", readout_format=".2f", layout=W.Layout(width="100%"))
    guard = W.Checkbox(value=True, description="Script constraint", indent=False)
    gen = W.Button(description="Generate", icon="bolt", button_style="primary", layout=W.Layout(width="100%", height="40px"))
    gen.add_class("rx-gen")
    rand = W.Button(description="Random example", icon="random", layout=W.Layout(width="100%"))
    edit = W.Textarea(layout=W.Layout(width="100%", height="230px"))
    edit.add_class("rx-ta")
    edit_box = W.Accordion(children=[edit], titles=("Edit the raw input the model sees",))
    edit_box.selected_index = None
    inp_view, out_view, cmp_view = W.HTML(), W.HTML(), W.HTML()
    current = {"ex": None, "live": None}

    def filtered():
        out = []
        for i, e in enumerate(examples):
            labs = e["labels"]
            if f_lang.value != "All languages" and e["language"] != f_lang.value:
                continue
            if f_src.value != "All sources" and e["source"] != f_src.value:
                continue
            if f_ver.value == "Contested labels" and len(set(labs.values())) == 1:
                continue
            if f_ver.value in ("Relevant", "Partly relevant", "Irrelevant") and NICE[labs["stored"]] != f_ver.value:
                continue
            if f_txt.value and f_txt.value.lower() not in e["input"].lower():
                continue
            out.append((label(e), i))
        return out

    def refresh(*_):
        opts = filtered()
        pick.options = opts
        count.value = f'<span class="rx-mute">{len(opts)} of {len(examples)} examples · ⚑ contested label</span>'
        if opts and (pick.value is None or pick.value not in [v for _, v in opts]):
            pick.value = opts[0][1]

    def show(*_):
        if pick.value is None:
            return
        ex = examples[pick.value]
        current["ex"], current["live"] = ex, None
        edit.value = ex["input"]
        inp_view.value = input_html(ex, ex["input"])
        out_view.value = output_html({"started": False})
        cmp_view.value = compare_html(ex, False, model_dd.value, None)

    def on_edit(change):
        if current["ex"] is not None:
            inp_view.value = input_html(current["ex"], edit.value)

    def run(*_):
        ex = current["ex"]
        if ex is None:
            return
        text = edit.value
        edited = text.strip() != ex["input"].strip()
        lang = parse_input(text).get("Respond in", "")
        gen.disabled, gen.description = True, "Generating…"
        try:
            t0 = time.time()
            model, tk, masks, load_s, size = load(model_dd.value)
            from .tiny.scriptguard import banned_for
            prompt = tk.encode(text + SEP).ids
            ban = banned_for(masks, text) if guard.value else None
            state = {"started": True, "running": True, "verdict": "pending", "narrative": "", "language": lang, "stats": {}}
            out_view.value = output_html(state)
            toks, t1, first_t = [], time.time(), None
            for k, (tok, logits) in enumerate(stream(model, prompt, banned=ban, temperature=temp.value if mode.value == "Sample" else 0.0,
                                                     seed=int(time.time()))):
                if k == 0:
                    first_t = time.time() - t1
                    state["probs"] = verdict_probs(tk, logits)
                if tok == EOS:
                    break
                toks.append(tok)
                text_out = tk.decode(toks)
                head, nl, rest = text_out.partition("\n")
                if nl:
                    state["verdict"] = head.strip() if head.strip() in VERDICTS else "invalid"
                    state["narrative"] = rest.strip()
                if k % 3 == 0:
                    out_view.value = output_html(state)
            total = time.time() - t1
            text_out = tk.decode(toks).strip()
            head, _, rest = text_out.partition("\n")
            state.update(running=False, verdict=head.strip() if head.strip() in VERDICTS else "invalid", narrative=rest.strip())
            state["stats"] = {"prompt tokens": len(prompt), "output tokens": len(toks),
                              "time to first token": f"{first_t * 1000:.0f} ms", "total": f"{total:.2f} s",
                              "per token": f"{(total - first_t) / max(1, len(toks) - 1) * 1000:.1f} ms",
                              "model file": f"{size / 1e6:.1f} MB"}
            if load_s and time.time() - t0 > total + 0.05:
                state["stats"]["load"] = f"{load_s:.2f} s"
            out_view.value = output_html(state)
            current["live"] = text_out
            cmp_view.value = compare_html(ex, edited, model_dd.value, text_out)
        finally:
            gen.disabled, gen.description = False, "Generate"

    def randomize(*_):
        opts = pick.options
        if opts:
            pick.value = random.choice(opts)[1]

    for w in (f_lang, f_src, f_ver, f_txt):
        w.observe(refresh, names="value")
    pick.observe(show, names="value")
    edit.observe(on_edit, names="value")
    gen.on_click(run)
    rand.on_click(randomize)

    header = W.HTML(CSS + """<div class="rx-app"><div class="rx-head"><div class="rx-logo">✦</div><div>
      <h1>Relevance Explainer · interactive viewer</h1>
      <p>A 21.6 MB model that explains why a catalog result fits a search · 11 languages · runs on CPU in this notebook</p></div></div></div>""")
    side = W.VBox([W.HTML('<div class="rx-app"><div class="rx-sec">Examples</div>'
                          '<div class="rx-mute" style="margin-bottom:6px">100 test inputs from themes the model never saw in training</div></div>'),
                   f_txt, W.HBox([f_lang, f_src]), f_ver, count, pick, rand,
                   W.HTML('<div class="rx-app"><div class="rx-sec" style="margin-top:14px">Settings</div></div>'),
                   model_dd, mode, temp, guard, gen], layout=W.Layout(width="34%", padding="0 12px 0 0"))
    side.add_class("rx-side")
    for w in (inp_view, out_view, cmp_view):
        w.add_class("rx-app")
    main = W.VBox([W.HBox([W.VBox([inp_view], layout=W.Layout(width="50%", padding="0 6px 0 0")),
                           W.VBox([out_view, cmp_view], layout=W.Layout(width="50%", padding="0 0 0 6px"))]),
                   edit_box, W.HTML('<div class="rx-app">' + ABOUT + "</div>")], layout=W.Layout(width="66%"))
    app = W.VBox([header, W.HBox([side, main])])
    app.parts = {"pick": pick, "gen": gen, "edit": edit, "model": model_dd, "mode": mode, "filters": (f_lang, f_src, f_ver, f_txt),
                 "views": (inp_view, out_view, cmp_view)}  # handles for tests
    refresh()
    show()
    display(app)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.n)


if __name__ == "__main__":
    main()
