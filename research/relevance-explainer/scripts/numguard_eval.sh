#!/usr/bin/env bash
# Evaluate a run with the number guard on the version-3 test sets (tags <set>_v3_rs) and judge the realistic test.
# Usage: scripts/numguard_eval.sh [run=runs/v2e-nosim-dpo]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/v2e-nosim-dpo}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$HOME/venvs/relexp/bin/python"
for pair in "data/eval_real/itemfirst_nosim_v3 realtest_v3_rs" "data/eval_para/itemfirst_nosim_v3 para_v3_rs" "data/schemas_eval2/itemfirst_nosim_v3 schemas_v3_rs" "data/eval_challenge/itemfirst_nosim_v3 challenge_v3_rs"; do
  set -- $pair
  for attempt in 1 2 3; do
    [ -f "$RUN/${2}_generations_int8.jsonl" ] && break
    $PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data "$1" --precisions int8 --tag "$2" --script-guard --num-guard 2>&1 | grep -E "Error|int8 " | tail -2
    [ -f "$RUN/${2}_generations_int8.jsonl" ] || sleep 60
  done
done
scripts/judge_tag.sh "$RUN" realtest_v3_rs 2>&1 | grep -E "^student|DONE"
echo NUMGUARD-EVAL-DONE
