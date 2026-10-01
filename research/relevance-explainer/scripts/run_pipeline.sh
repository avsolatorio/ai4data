#!/usr/bin/env bash
# Run the synthetic-data pipeline against a local Ollama server. Retrieval and narration are resumable;
# `queries` and `anchored` add new (theme, source) combinations each time they run, so select stages explicitly.
# Usage: STAGES="queries anchored retrieve narrate assemble" scripts/run_pipeline.sh data/v1 [theme_combos] [anchored_combos]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:?output directory}
THEME=${2:-100}
ANCHORED=${3:-160}
STAGES=${STAGES:-"queries anchored retrieve narrate assemble"}
WORKERS=${WORKERS:-16}
SEED=${SEED:-10}

export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:11435/v1}
export QUERY_MODEL=${QUERY_MODEL:-qwen3.8:27b-q4_K_M}
export NARRATE_MODEL=${NARRATE_MODEL:-qwen3.8:27b-q4_K_M}
export LLM_REASONING=${LLM_REASONING:-none}
export PYTHONPATH=src

G="python -m relevance_explainer.synth.generate"
for s in $STAGES; do
  case $s in
    queries)  $G queries  --out "$OUT" --combos "$THEME" --per-combo 4 --workers "$WORKERS" --seed "$SEED" ;;
    anchored) $G anchored --out "$OUT" --combos "$ANCHORED" --items 2 --workers "$WORKERS" --seed "$((SEED + 1))" ;;
    retrieve) $G retrieve --out "$OUT" --workers "${RETRIEVE_WORKERS:-12}" --seed "$((SEED + 2))" ;;
    narrate)  $G narrate  --out "$OUT" --workers "$WORKERS" ;;
    assemble) $G assemble --out "$OUT" ;;
    *) echo "unknown stage $s" >&2; exit 1 ;;
  esac
done
