#!/usr/bin/env bash
# Third data round: 1,500 theme and 1,500 anchored (theme, source) combinations with a flatter language mix,
# against the vLLM teacher on port 8000. Resumable; rerun to continue.
set -euo pipefail
cd "$(dirname "$0")/.."

export LANG_WEIGHTS='{"English":20,"French":9,"Spanish":9,"Portuguese":6,"Arabic":10,"Chinese":10,"Russian":10,"Indonesian":8,"Hindi":8,"Turkish":6,"Swahili":4}'
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1}
export QUERY_MODEL=qwen3.8-27b
export NARRATE_MODEL=qwen3.8-27b
export LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
export WORKERS=${WORKERS:-128}
export RETRIEVE_WORKERS=${RETRIEVE_WORKERS:-12}
export SEED=200
export STAGES=${STAGES:-"queries anchored retrieve narrate"}

exec scripts/run_pipeline.sh data/v3 1500 1500
