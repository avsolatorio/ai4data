#!/usr/bin/env bash
# Long realistic-data run after scripts/realistic_scale.sh: chunks FIRST..LAST (coverage-steered through
# data/real/coverage.json), and after every second chunk a translation set of about 20% of those two chunks.
# Stop cleanly by creating data/real/STOP. Usage: scripts/realistic_long.sh [first=11] [last=200]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
FIRST=${1:-11}; LAST=${2:-200}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1,http://localhost:8001/v1}
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
PY="$RELEXP_PY -m relevance_explainer.synth.realistic"
for k in $(seq "$FIRST" "$LAST"); do
  [ -f data/real/STOP ] && { echo "stop file found"; break; }
  OUT=data/real/c$k; SEED=$((100 + k))
  if [ ! -f "$OUT/examples.jsonl" ]; then
    echo "chunk $k start $(date -u +%H:%M)"
    [ -f "$OUT/queries.jsonl" ] || $PY queries --out "$OUT" --n 2000 --seed "$SEED" --themes data/themes.json || continue
    [ -f "$OUT/anchored.jsonl" ] || $PY anchored --out "$OUT" --n 2500 --seed "$SEED" --themes data/themes.json --retrieve-workers 20 || continue
    [ -f "$OUT/pairs.jsonl" ] || $PY retrieve --out "$OUT" --seed "$SEED" --retrieve-workers 20 || continue
    [ -f "$OUT/rendered.jsonl" ] || $PY render --out "$OUT" || continue
    $PY narrate --out "$OUT" --workers 160 || continue
    echo "chunk $k done $(date -u +%H:%M): $(wc -l < "$OUT/examples.jsonl") examples"
  fi
  if [ $((k % 2)) -eq 0 ] && [ ! -f "data/real/t$k/examples.jsonl" ]; then
    n=$(( ($(wc -l < data/real/c$((k - 1))/examples.jsonl) + $(wc -l < "$OUT/examples.jsonl")) / 5 ))
    "$RELEXP_PY" -m relevance_explainer.synth.translate --src "data/real/c$((k - 1))" "$OUT" --out "data/real/t$k" --n "$n" --seed "$k" --workers 24 \
      && echo "translation t$k: $(wc -l < "data/real/t$k/examples.jsonl") examples"
  fi
done
echo REALISTIC-LONG-DONE
