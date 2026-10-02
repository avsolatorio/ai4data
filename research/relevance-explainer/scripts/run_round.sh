#!/usr/bin/env bash
# A further data round with the flat language mix against the vLLM teacher on port 8000. Resumable.
# Usage: scripts/run_round.sh data/v4 300 [theme_combos] [anchored_combos]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output directory}
SEED=${2:?seed (use a new one per round so the theme and source draws differ)}

export LANG_WEIGHTS='{"English":20,"French":9,"Spanish":9,"Portuguese":6,"Arabic":10,"Chinese":10,"Russian":10,"Indonesian":8,"Hindi":8,"Turkish":6,"Swahili":4}'
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1}
export QUERY_MODEL=qwen3.8-27b
export NARRATE_MODEL=qwen3.8-27b
export LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
export WORKERS=${WORKERS:-128}
export RETRIEVE_WORKERS=${RETRIEVE_WORKERS:-12}
export SEED
export STAGES=${STAGES:-"queries anchored retrieve narrate"}

exec scripts/run_pipeline.sh "$OUT" "${3:-1500}" "${4:-1500}"
