#!/usr/bin/env bash
# Score a no-score run on the realistic, paraphrase, nine-type and challenge test sets.
# Usage: RUN_DIR=<run> [SUF=_v3] [TAG_SUF=_v3] scripts/eval_sets.sh
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
SUF=${SUF:-}; TS=${TAG_SUF:-$SUF}
for pair in "data/eval_real/itemfirst_nosim realtest" "data/eval_para/itemfirst_nosim para" "data/schemas_eval2/itemfirst_nosim schemas" "data/eval_challenge/itemfirst_nosim challenge"; do
  set -- $pair
  D="$1$SUF"; T="$2$TS"
  for attempt in 1 2 3; do  # the GPU is shared with vLLM and training; retry transient memory errors
    [ -f "$RUN_DIR/${T}_generations_int8.jsonl" ] && break
    $PY -m relevance_explainer.tiny.evaluate --run "$RUN_DIR" --data "$D" --precisions int8 --tag "$T" --script-guard 2>&1 | grep -E "Error|int8 " | tail -2
    [ -f "$RUN_DIR/${T}_generations_int8.jsonl" ] || sleep 60
  done
  echo "== $T"; $PY -m relevance_explainer.synth.score_real --data "$D" --run "$RUN_DIR" --tag "$T" | grep -E "teacher|by_mode"
done
