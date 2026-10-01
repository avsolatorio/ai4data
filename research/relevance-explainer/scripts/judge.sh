#!/usr/bin/env bash
# Judge generated narratives with the vLLM-served teacher.
# Usage: scripts/judge.sh runs/t-sim int8 300
set -euo pipefail
cd "$(dirname "$0")/.."

export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1}
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
export PYTHONPATH=src

python -m relevance_explainer.tiny.judge --run "${1:?run dir}" --prec "${2:-int8}" --n "${3:-300}" --tag "${4:-test}"
