#!/usr/bin/env bash
# Generate realistic-search data in resumable chunks: each chunk runs the full pipeline with its own seed.
# Usage: scripts/realistic_scale.sh [chunks=10] [theme_per_chunk=2000] [anchor_per_chunk=2500] [first=1]
# Chunks go to data/real/cN; a chunk with examples.jsonl is skipped. Themes: data/themes.json (split by theme later).
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
N=${1:-10}; NT=${2:-2000}; NA=${3:-2500}; FIRST=${4:-1}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1,http://localhost:8001/v1}
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
PY="$RELEXP_PY -m relevance_explainer.synth.realistic"
for k in $(seq "$FIRST" "$N"); do
  OUT=data/real/c$k; SEED=$((100 + k))
  [ -f "$OUT/examples.jsonl" ] && { echo "chunk $k done"; continue; }
  echo "chunk $k start $(date -u +%H:%M)"
  [ -f "$OUT/queries.jsonl" ] || $PY queries --out "$OUT" --n "$NT" --seed "$SEED" --themes data/themes.json || continue
  [ -f "$OUT/anchored.jsonl" ] || $PY anchored --out "$OUT" --n "$NA" --seed "$SEED" --themes data/themes.json --retrieve-workers 20 || continue
  [ -f "$OUT/pairs.jsonl" ] || $PY retrieve --out "$OUT" --seed "$SEED" --retrieve-workers 20 || continue
  [ -f "$OUT/rendered.jsonl" ] || $PY render --out "$OUT" || continue
  $PY narrate --out "$OUT" --workers 160 || continue
  echo "chunk $k done $(date -u +%H:%M): $(wc -l < "$OUT/examples.jsonl") examples"
done
echo REALISTIC-SCALE-DONE
