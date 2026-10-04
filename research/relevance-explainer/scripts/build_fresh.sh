#!/usr/bin/env bash
# Build a fresh evaluation set (searches and items absent from every data round) with the vLLM teacher and Gemma 4.
# Usage: scripts/build_fresh.sh [out=data/fresh] [n=100] [--anchor-unseen 120 ...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export LANG_WEIGHTS='{"English":12,"French":9,"Spanish":9,"Portuguese":8,"Arabic":9,"Chinese":9,"Russian":9,"Indonesian":9,"Hindi":9,"Turkish":8,"Swahili":8}'
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1,http://localhost:8001/v1}
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b TEACHER_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
exec "$HOME/venvs/relexp/bin/python" -m relevance_explainer.synth.fresh --out "${1:-data/fresh}" --n "${2:-100}" "${@:3}"
