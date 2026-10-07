#!/usr/bin/env bash
# Score a no-score run on the realistic, paraphrase, nine-type and challenge test sets. Usage: scripts/eval_sets.sh <run>
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$HOME/venvs/relexp/bin/python"
for pair in "data/eval_real/itemfirst_nosim realtest" "data/eval_para/itemfirst_nosim para" "data/schemas_eval2/itemfirst_nosim schemas" "data/eval_challenge/itemfirst_nosim challenge"; do
  set -- $pair
  [ -f "$RUN_DIR/${2}_generations_int8.jsonl" ] || $PY -m relevance_explainer.tiny.evaluate --run "$RUN_DIR" --data "$1" --precisions int8 --tag "$2" --script-guard 2>&1 | grep -E "Error|int8 "
  echo "== $2"; $PY -m relevance_explainer.synth.score_real --data "$1" --run "$RUN_DIR" --tag "$2" | grep -E "teacher|by_mode|by_source"
done
