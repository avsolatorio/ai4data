#!/usr/bin/env bash
# Pairwise usefulness preference between two runs with Gemma 4. Usage: scripts/pairwise.sh <run-a> <run-b>
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export LLM_BASE_URL=${GEMMA_URL:-http://localhost:8002/v1}
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
exec "$HOME/venvs/relexp/bin/python" -m relevance_explainer.tiny.pairwise "$1" "$2" --model gemma-4-31b --n 800 "${@:3}"
