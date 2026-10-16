#!/usr/bin/env bash
# Realistic chunks for the metadata-schemas types from the local record pools (tables, map layers, scripts).
# Same stages as scripts/realistic_long.sh; stop with data/real_types/STOP. Usage: scripts/types_long.sh [first=1] [last=8]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
FIRST=${1:-1}; LAST=${2:-8}
until grep -q POOLS-DONE /tmp/claude-1002/-home-jupyter-wb536061-WBG-ai4data/f9459d82-c4fb-49c4-a88f-41cc1ffd548f/scratchpad/pools.log 2>/dev/null || [ -f data/pools/READY ]; do sleep 60; done
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1,http://localhost:8001/v1}
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
tables=$(ls data/pools/table_*.jsonl | sed 's/.*/"&"/' | paste -sd, -)
export RECORD_POOLS="{\"table\": [$tables], \"geospatial\": [\"data/pools/geospatial_hdx.jsonl\"], \"script\": [\"data/pools/script_rrr.jsonl\"]}"
export SOURCE_WEIGHTS='{"table": 55, "geospatial": 25, "script": 20}'
PY="$RELEXP_PY -m relevance_explainer.synth.realistic"
for k in $(seq "$FIRST" "$LAST"); do
  [ -f data/real_types/STOP ] && { echo "stop file found"; break; }
  OUT=data/real_types/c$k; SEED=$((5000 + k))
  [ -f "$OUT/examples.jsonl" ] && continue
  echo "types chunk $k start $(date -u +%H:%M)"
  [ -f "$OUT/queries.jsonl" ] || $PY queries --out "$OUT" --n 1000 --seed "$SEED" --themes data/themes.json || continue
  [ -f "$OUT/anchored.jsonl" ] || $PY anchored --out "$OUT" --n 1500 --seed "$SEED" --themes data/themes.json --retrieve-workers 20 || continue
  [ -f "$OUT/pairs.jsonl" ] || $PY retrieve --out "$OUT" --seed "$SEED" --retrieve-workers 20 || continue
  [ -f "$OUT/rendered.jsonl" ] || $PY render --out "$OUT" || continue
  $PY narrate --out "$OUT" --workers 120 || continue
  echo "types chunk $k done $(date -u +%H:%M): $(wc -l < "$OUT/examples.jsonl") examples"
done
echo TYPES-LONG-DONE
