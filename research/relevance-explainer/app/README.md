# Relevance Explainer web app

A single-page viewer for the 21.6 MB model. Pick one of 100 test inputs (or edit one), choose a model and decoding
settings, and the model generates its verdict and narrative on CPU, streamed token by token. The page also shows the
signals the model receives, its verdict probabilities, timing, the teacher's reference, the three audit labels and
the offline judge scores, and can run every model on the same input side by side.

- Server: `src/relevance_explainer/webapp/server.py` (Starlette on uvicorn); page: `webapp/static/index.html`.
- Examples, two sets selectable on the page:
  - Unseen items, `app/examples_fresh.json`: 100 searches and catalog items that occur in no data round, built
    after training from new themes with `scripts/build_fresh.sh data/fresh 100 --anchor-unseen 140`.
  - Held-out searches, `app/examples.json`: 100 test-split searches from held-out themes; their catalog items do
    occur in training with other searches (the page shows how often). Built with `python -m relevance_explainer.viewer build`.
- Models: the INT8 exports listed in `relevance_explainer.viewer.MODELS` (files under `runs/`, not tracked by git).

## Run

```
scripts/serve_app.sh                      # port 3410 on all interfaces, CPU only
PORT=3410 BASE_PATH=/relevance-explainer MAX_CONCURRENT=4 TORCH_THREADS=4 scripts/serve_app.sh
```

`BASE_PATH` is stripped from incoming paths when present, so the app works whether or not the proxy forwards the
prefix. The page uses relative URLs. `MAX_CONCURRENT` limits simultaneous generations (others wait up to 30 s);
each generation takes about 0.5 to 1 s with 4 threads. Inputs are limited to 12,000 characters and to the model's
context length.

## Behind nginx

```nginx
# Relevance Explainer web app (Starlette/uvicorn on the host, port 3410).
location = /relevance-explainer {
    return 301 /relevance-explainer/;
}

location /relevance-explainer/ {
    proxy_pass http://host.docker.internal:3410/;

    proxy_http_version 1.1;

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Generation is streamed as NDJSON; do not buffer it.
    proxy_buffering off;
    proxy_cache off;

    client_max_body_size 1m;

    proxy_connect_timeout 30s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;
}
```

The trailing slash in `proxy_pass http://host.docker.internal:3410/;` removes the `/relevance-explainer` prefix
before the request reaches the app. Without it the prefix is forwarded, which also works because the server strips
`BASE_PATH`. The app also sends `X-Accel-Buffering: no` on the streaming endpoint.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the page |
| GET | `/api/examples` | the 100 examples with labels, reference and offline judge scores |
| GET | `/api/models` | available models and file sizes |
| POST | `/api/generate` | `{"input", "model", "temperature", "guard"}`; returns NDJSON events `start`, `probs`, `token`, `done` or `error` |
| GET | `/healthz` | liveness |
| GET | `/live` | page that explains a live catalog record (link plus typed search) |
| POST | `/api/live` | `{"url", "search", "language"}`; fetches a Data360 indicator or a study from an allowlisted NADA catalog (Microdata Library, catalog.ihsn.org, NADA demo, Reproducible Research Repository), converts it with `relevance_explainer.schemas` and returns the item-first input, the detected language and the record summary. Generation then goes through `/api/generate` with `"raw": true` |

## Notes

- Live output can differ from the offline evaluation in a word or two: the evaluation decoded on GPU in bfloat16 and
  the app decodes on CPU in float32, so near-tied tokens can flip. In a check of 32 examples, 26 were identical and
  all 32 had the same verdict. The page says when the live output is identical to the judged one.
- The process is started with `setsid nohup` and does not restart after a reboot.
