#!/usr/bin/env bash
# Build the metadata-schemas test set and evaluate a run on it. Usage: scripts/schemas_eval.sh [run] [src] [out]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/v2b-cont-itemfirst-dpo}
SRC=${2:-data/nada_demo.jsonl}
OUT=${3:-data/schemas_eval}
export PYTHONPATH=src LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
PY="$HOME/venvs/relexp/bin/python"
[ -f "$OUT/searches.jsonl" ] || $PY -m relevance_explainer.synth.schemas_eval searches --src "$SRC" --out "$OUT"
[ -f "$OUT/labeled.jsonl" ] || $PY -m relevance_explainer.synth.schemas_eval label --out "$OUT"
$PY -m relevance_explainer.synth.schemas_eval build --out "$OUT"
CUDA_VISIBLE_DEVICES=1 $PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data "$OUT/itemfirst" --precisions int8 --tag schemas --script-guard
$PY -m relevance_explainer.synth.score_real --data "$OUT/itemfirst" --run "$RUN" --tag schemas
$PY -m relevance_explainer.tiny.judge --run "$RUN" --prec int8 --tag schemas --n 800 --workers 48
LLM_BASE_URL=http://localhost:8002/v1 $PY -m relevance_explainer.tiny.judge --run "$RUN" --prec int8 --tag schemas --n 800 --workers 16 --model gemma-4-31b --out-suffix _gemma4
echo SCHEMAS-EVAL-DONE "$RUN"
