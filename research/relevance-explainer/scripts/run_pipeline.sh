#!/usr/bin/env bash
# Run the full synthetic-data pipeline against a local Ollama server. Every stage is resumable.
# Usage: scripts/run_pipeline.sh data/v1 [theme_combos] [anchored_combos]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:?output directory}
THEME=${2:-100}
ANCHORED=${3:-160}

export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:11435/v1}
export QUERY_MODEL=${QUERY_MODEL:-qwen3.8:27b-q4_K_M}
export NARRATE_MODEL=${NARRATE_MODEL:-qwen3.8:27b-q4_K_M}
export LLM_REASONING=${LLM_REASONING:-none}
export PYTHONPATH=src

python -m relevance_explainer.synth.generate queries  --out "$OUT" --combos "$THEME" --per-combo 4 --workers 16 --seed 10
python -m relevance_explainer.synth.generate anchored --out "$OUT" --combos "$ANCHORED" --items 2 --workers 16 --seed 11
python -m relevance_explainer.synth.generate retrieve --out "$OUT" --workers 12 --seed 12
python -m relevance_explainer.synth.generate narrate  --out "$OUT" --workers 16
python -m relevance_explainer.synth.generate assemble --out "$OUT"
