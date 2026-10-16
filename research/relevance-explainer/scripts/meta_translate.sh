#!/usr/bin/env bash
# Metadata-translation set from realistic chunks. Usage: scripts/meta_translate.sh <out> <n> <seed> <chunk_dir>...
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
OUT=${1:?out}; N=${2:?n}; SEED=${3:?seed}; shift 3
export PYTHONPATH=src LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
"$RELEXP_PY" -m relevance_explainer.synth.meta_translate --src "$@" --out "$OUT" --n "$N" --seed "$SEED" 2>&1 | grep -v Warning
