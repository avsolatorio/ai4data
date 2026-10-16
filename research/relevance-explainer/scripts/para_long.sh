#!/usr/bin/env bash
# Paraphrase-focused anchored chunks: fully-answering searches in everyday words, and hard cases with the same wording
# and a wrong place, time or group, or a related concept. Stop with data/real_para/STOP. Usage: scripts/para_long.sh [first=1] [last=10]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
FIRST=${1:-1}; LAST=${2:-10}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1,http://localhost:8001/v1}
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
export RELATION_WEIGHTS='{"full": 50, "wrong_place": 12, "wrong_time": 10, "wrong_group": 8, "related": 14, "other_topic": 6}'
export PARAPHRASE_SHARE=0.8 PARAPHRASE_RELATIONS=full,wrong_place,wrong_time,wrong_group,related
PY="$RELEXP_PY -m relevance_explainer.synth.realistic"
for k in $(seq "$FIRST" "$LAST"); do
  [ -f data/real_para/STOP ] && { echo "stop file found"; break; }
  OUT=data/real_para/c$k; SEED=$((7000 + k))
  [ -f "$OUT/examples.jsonl" ] && continue
  echo "para chunk $k start $(date -u +%H:%M)"
  mkdir -p "$OUT"; [ -f "$OUT/queries.jsonl" ] || : > "$OUT/queries.jsonl"
  [ -f "$OUT/anchored.jsonl" ] || $PY anchored --out "$OUT" --n 3000 --seed "$SEED" --themes data/themes.json --retrieve-workers 20 || continue
  [ -f "$OUT/pairs.jsonl" ] || $PY retrieve --out "$OUT" --seed "$SEED" --retrieve-workers 20 || continue
  [ -f "$OUT/rendered.jsonl" ] || $PY render --out "$OUT" || continue
  $PY narrate --out "$OUT" --workers 120 || continue
  echo "para chunk $k done $(date -u +%H:%M): $(wc -l < "$OUT/examples.jsonl") examples"
done
echo PARA-LONG-DONE
