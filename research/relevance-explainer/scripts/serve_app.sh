#!/usr/bin/env bash
# Serve the relevance-explainer web app on CPU (default port 3410, all interfaces so a containerized proxy can reach it).
# Usage: [PORT=3410] [BASE_PATH=/relevance-explainer] [WORKERS=1] scripts/serve_app.sh
# BASE_PATH only matters if the proxy forwards the path prefix; with `proxy_pass http://host:3410/;` it is stripped there.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=""
export BASE_PATH=${BASE_PATH:-}
export TORCH_THREADS=${TORCH_THREADS:-4}
export MAX_CONCURRENT=${MAX_CONCURRENT:-4}
exec "$RELEXP_PY" -m uvicorn relevance_explainer.webapp.server:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-3410}" --workers "${WORKERS:-1}" \
  --proxy-headers --forwarded-allow-ips "*" --timeout-keep-alive 30
