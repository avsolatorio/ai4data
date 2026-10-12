"""Web app for the relevance explainer: a single-page viewer with live CPU generation.

Serves the page, the curated examples and a streaming generation endpoint. All browser requests use relative URLs,
so the app works at the root or under a path prefix behind a reverse proxy. If the proxy forwards the prefix
(nginx `proxy_pass http://host:3410;` without a trailing slash), set BASE_PATH to it and it is stripped here.

  BASE_PATH=/relevance-explainer uvicorn relevance_explainer.webapp.server:app --host 0.0.0.0 --port 3410

Endpoints (relative to the base path):
  GET  /                 the page
  GET  /api/examples     curated test inputs with labels, teacher reference and offline judge scores
  GET  /api/models       available models
  POST /api/generate     {"input", "model", "temperature", "guard"} -> NDJSON stream of events
  GET  /healthz          liveness
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import torch
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from .. import viewer as V
from ..tiny.scriptguard import banned_for

STATIC = Path(__file__).parent / "static"
BASE_PATH = os.environ.get("BASE_PATH", "").rstrip("/")
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "12000"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "4"))
THREADS = int(os.environ.get("TORCH_THREADS", "4"))
torch.set_num_threads(THREADS)
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
FIELDS = ("id", "language", "source", "input", "reference", "labels", "offline", "train_item_count", "labels_missing", "theme",
          "query", "facets", "style", "relation")
_REALISTIC: dict[str, dict] = {}  # id -> example with its catalog record (kept on the server)


def _load_examples() -> list[dict]:
    """The held-out test examples ("test") and, when built, the fresh set with unseen items ("fresh")."""
    out = []
    for path, name in ((V.EXAMPLES.parent / "examples_realistic.json", "realistic"), (V.EXAMPLES, "test"),
                       (V.EXAMPLES.parent / "examples_fresh.json", "fresh")):
        if path.exists():
            for e in json.loads(path.read_text()):
                if name == "realistic":
                    _REALISTIC[str(e["id"])] = e
                out.append({**{k: e[k] for k in FIELDS if k in e}, "set": name})
    return out


_U = None
_ST = None


def render_for(example_id: str, search: str | None) -> dict:
    """Inputs for a realistic example and a (possibly edited) search: item-first and first-round formats."""
    global _U, _ST
    from ..synth.realistic import item_first, render_v1, render_v2
    from ..synth.understand import Understander
    from ..synth.signals import MODEL as ST_MODEL, item_text
    e = _REALISTIC[example_id]
    q = (search or e["query"]).strip()
    if q == e["query"]:
        sim = e["similarity"]
    else:
        if _ST is None:
            from sentence_transformers import SentenceTransformer
            _ST = SentenceTransformer(ST_MODEL, device="cpu")
        a, b = _ST.encode([q, item_text(e["record"])], normalize_embeddings=True)
        sim = float(a @ b)
    if _U is None:
        _U = Understander()
    # facets belong to the original search; an edited search is treated as typed with no facets set
    p = {"query": q, "language": e["language"], "record": e["record"], "facet_values": (e.get("facets") or {}) if q == e["query"] else {}, "render_version": 4}
    parsed = _U.parse(q, p["facet_values"], level=p["render_version"])
    v2 = render_v2(p, _U, parsed, sim)
    return {"itemfirst": item_first(v2), "v1": render_v1(p, p["facet_values"], sim), "similarity": round(sim, 2)}


_PUBLIC_EXAMPLES = _load_examples()


class StripPrefix:
    """ASGI middleware: remove BASE_PATH from the request path when the proxy forwards it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if BASE_PATH and scope["type"] == "http" and scope["path"].startswith(BASE_PATH):
            scope = dict(scope)
            scope["path"] = scope["path"][len(BASE_PATH):] or "/"
            scope["root_path"] = BASE_PATH
        await self.app(scope, receive, send)


async def index(request: Request) -> Response:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


async def examples(request: Request) -> Response:
    return JSONResponse(_PUBLIC_EXAMPLES, headers={"Cache-Control": "public, max-age=300"})


async def models(request: Request) -> Response:
    out = []
    for name, path in V.MODELS.items():
        p = V.ROOT / path
        if p.exists():
            out.append({"name": name, "size_mb": round(p.stat().st_size / 1e6, 1), "default": name == "Round 7 + DPO (best)",
                        "format": V.MODEL_FORMAT.get(name, "v1")})
    return JSONResponse(out)


async def render(request: Request) -> Response:
    try:
        body = await request.json()
        eid, search = str(body.get("example_id", "")), body.get("search")
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if eid not in _REALISTIC or (search is not None and (not str(search).strip() or len(str(search)) > 500)):
        return JSONResponse({"error": "unknown example or invalid search"}, status_code=400)
    return JSONResponse(render_for(eid, search))


async def health(request: Request) -> Response:
    return PlainTextResponse("ok")


def _events(text: str, model_name: str, temperature: float, guard: bool):
    """Generate one answer and yield NDJSON lines: start, probs, token (cumulative text), done."""
    if not _slots.acquire(timeout=30):
        yield json.dumps({"type": "error", "message": "The server is busy; try again in a moment."}) + "\n"
        return
    try:
        t0 = time.time()
        model, tk, masks, load_s, size = V.load(model_name)
        prompt = tk.encode(text + V.SEP).ids
        if len(prompt) > model.c.max_len - 280:  # leave room for the answer within the model's context
            yield json.dumps({"type": "error", "message": f"Input is {len(prompt)} tokens; the limit is {model.c.max_len - 280}. Shorten the metadata."}) + "\n"
            return
        ban = banned_for(masks, text) if guard else None
        yield json.dumps({"type": "start", "prompt_tokens": len(prompt), "model_mb": round(size / 1e6, 1)}) + "\n"
        toks, t1, first, probs, trace = [], time.time(), None, None, []
        for k, (tok, logits) in enumerate(V.stream(model, prompt, banned=ban, temperature=temperature, seed=int(t0 * 1000) % 2**31,
                                                   trace=trace)):
            if k == 0:
                first = time.time() - t1
                probs = V.verdict_probs(tk, logits)
                yield json.dumps({"type": "probs", "probs": probs}) + "\n"
            if tok == V.EOS:
                break
            toks.append(tok)
            if k % 2 == 0:
                yield json.dumps({"type": "token", "text": tk.decode(toks)}) + "\n"
        total = time.time() - t1
        out = tk.decode(toks).strip()
        out_ids, out_trace, out_prefix = toks, trace, 0  # the generated tokens and step stats behind `out`
        from ..tiny.numguard import repair_ranges, unsupported
        resampled = 0
        if guard:  # an explanation with a number absent from the input is resampled (tiny/numguard.py)
            for r in range(6):
                if not unsupported(out, text):
                    break
                cand, ctr = [], []
                for tok, _ in V.stream(model, prompt, banned=ban, temperature=0.7, seed=1000 + r, trace=ctr):
                    if tok == V.EOS:
                        break
                    cand.append(tok)
                c = tk.decode(cand).strip()
                resampled += 1
                if c.split("\n")[0].strip() == out.split("\n")[0].strip() and not unsupported(c, text):
                    out, out_ids, out_trace = c, cand, ctr
        head, _, rest = out.partition("\n")
        verdict = head.strip() if head.strip() in V.VERDICTS else "invalid"
        # answer confidence (tiny/answer_score.py) from the step statistics recorded while decoding
        score, first_score, retried, attempts = None, None, 0, []
        scorer = V.score_head(model_name)
        if scorer is not None and probs is not None:
            from ..tiny.answer_score import features
            nl = tk.encode("\n").ids[-1]
            vp = probs.get(verdict, max(probs.values()))
            score = first_score = scorer(features(out_trace, out_ids, nl, vp, bool(unsupported(out, text)), out_prefix))
            if score < V.RETRY_BELOW and verdict != "invalid":
                # low confidence: write at most RETRY_SAMPLES more explanations after the same verdict line, within
                # RETRY_MAX_S seconds, and keep the most confident; rewrites use the page's temperature in sample mode.
                # Every attempt is reported so the page can show the ones that were not kept.
                yield json.dumps({"type": "retry", "score": round(score, 3)}) + "\n"
                attempts.append({"text": out, "score": score})
                vline = tk.encode(text + V.SEP + verdict + "\n").ids
                forced = vline[len(prompt):]
                rtemp, rseed, r0 = temperature or V.RETRY_TEMP, (2000 if temperature == 0 else int(t0 * 1000) % 2**31), time.time()
                if vline[: len(prompt)] == prompt and forced and forced[-1] == nl:
                    for r in range(V.RETRY_SAMPLES):
                        if r and time.time() - r0 > V.RETRY_MAX_S:
                            break
                        cand, ctr = [], []
                        for tok, _ in V.stream(model, vline, banned=ban, temperature=rtemp, seed=rseed + r, trace=ctr):
                            if tok == V.EOS:
                                break
                            cand.append(tok)
                        retried += 1
                        ids = forced + cand
                        c = tk.decode(ids).strip()
                        bad = bool(unsupported(c, text))
                        if guard and bad:  # never shown as the answer
                            attempts.append({"text": c, "score": None, "note": "states a number absent from the record"})
                            continue
                        sc = scorer(features(ctr, ids, nl, vp, bad, len(forced)))
                        attempts.append({"text": c, "score": sc})
                        if sc > score:
                            out, score = c, sc
                    head, _, rest = out.partition("\n")
        if guard:
            out = repair_ranges(out, text)
            head, _, rest = out.partition("\n")
        best = max(range(len(attempts)), key=lambda j: -1 if attempts[j]["score"] is None else attempts[j]["score"], default=None)
        attempts = [{"narrative": (repair_ranges(a["text"], text) if guard else a["text"]).partition("\n")[2].strip(),
                     "score": None if a["score"] is None else round(a["score"], 3), "kept": j == best,
                     **({"note": a["note"]} if "note" in a else {})} for j, a in enumerate(attempts)]
        total = time.time() - t1
        yield json.dumps({"type": "done", "text": out, "verdict": verdict, "narrative": rest.strip(), "attempts": attempts,
                          "score": None if score is None else round(score, 3), "stats": {
            "prompt_tokens": len(prompt), "output_tokens": len(toks), "first_token_ms": round(first * 1000),
            "total_s": round(total, 2), "per_token_ms": round((total - first) / max(1, len(toks) - 1) * 1000, 1),
            "model_mb": round(size / 1e6, 1), "resampled": resampled, "retried": retried,
            "first_score": None if first_score is None else round(first_score, 3)}}) + "\n"
    except Exception as exc:  # report to the page instead of breaking the stream
        yield json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}) + "\n"
    finally:
        _slots.release()


async def generate(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    text = str(body.get("input", ""))
    model_name = str(body.get("model", "Round 7 + DPO (best)"))
    eid = str(body.get("example_id", ""))
    if eid in _REALISTIC and not body.get("raw"):  # realistic example: the server renders the input for the model's format
        search = body.get("search")
        if search is not None and (not str(search).strip() or len(str(search)) > 500):
            return JSONResponse({"error": "invalid search"}, status_code=400)
        r = render_for(eid, search)
        fmt = V.MODEL_FORMAT.get(model_name, "v1")
        text = r["v1"] if fmt == "v1" else r["itemfirst"]
        if fmt == "itemfirst_nosim":
            text = re.sub(r"\nRetrieval similarity: [-0-9.]+", "", text)
    elif V.MODEL_FORMAT.get(model_name, "v1").startswith("itemfirst"):
        if not body.get("raw"):
            return JSONResponse({"error": "this model reads the realistic format; pick an example from the realistic set"}, status_code=400)
        if V.MODEL_FORMAT[model_name] == "itemfirst_nosim":  # a raw realistic input (live catalog page): drop the score line
            text = re.sub(r"\nRetrieval similarity: [-0-9.]+", "", text)
    if not text.strip() or len(text) > MAX_INPUT_CHARS:
        return JSONResponse({"error": f"input must be 1 to {MAX_INPUT_CHARS} characters"}, status_code=400)
    if model_name not in V.MODELS or not (V.ROOT / V.MODELS[model_name]).exists():
        return JSONResponse({"error": "unknown model"}, status_code=400)
    try:
        temperature = min(max(float(body.get("temperature", 0.0)), 0.0), 1.5)
    except (TypeError, ValueError):
        temperature = 0.0
    guard = bool(body.get("guard", True))
    return StreamingResponse(_events(text, model_name, temperature, guard), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def live_page(request: Request) -> Response:
    return FileResponse(STATIC / "live.html", headers={"Cache-Control": "no-cache"})


async def live_render(request: Request) -> Response:
    """{"url", "search", "language"} -> the rendered input for a live catalog record (see webapp/live.py)."""
    from starlette.concurrency import run_in_threadpool
    from . import live
    try:
        body = await request.json()
        url, search, lang = str(body.get("url", "")).strip(), str(body.get("search", "")).strip(), body.get("language")
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not url or len(url) > 500 or not search or len(search) > 300:
        return JSONResponse({"error": "give a catalog link and a search of up to 300 characters"}, status_code=400)
    try:
        return JSONResponse(await run_in_threadpool(live.render, url, search, lang))
    except live.LiveError as e:
        return JSONResponse({"error": str(e)}, status_code=422)


app = Starlette(routes=[
    Route("/", index), Route("/index.html", index), Route("/api/examples", examples), Route("/api/models", models),
    Route("/api/generate", generate, methods=["POST"]), Route("/api/render", render, methods=["POST"]), Route("/healthz", health),
    Route("/live", live_page), Route("/api/live", live_render, methods=["POST"]),
], middleware=[Middleware(StripPrefix)])

# Load the default model at startup so the first request is fast.
V.load("Round 7 + DPO (best)")
