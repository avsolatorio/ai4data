#!/usr/bin/env bash
# Realistic-search pilot: spec-driven searches, anchored searches, retrieval, rendering, narration.
# Usage: scripts/realistic_pilot.sh [out=data/real_pilot] [n_theme=400] [n_anchor=400] [seed=11]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-data/real_pilot}; NT=${2:-400}; NA=${3:-400}; SEED=${4:-11}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1,http://localhost:8001/v1}
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
PY="$HOME/venvs/relexp/bin/python -m relevance_explainer.synth.realistic"
$PY queries --out "$OUT" --n "$NT" --seed "$SEED"
$PY anchored --out "$OUT" --n "$NA" --seed "$SEED"
$PY retrieve --out "$OUT" --seed "$SEED"
$PY render --out "$OUT"
$PY narrate --out "$OUT"
echo REALISTIC-PILOT-DONE
