#!/usr/bin/env bash
# Correctness head experiment (tiny/selfcheck.py) on the deployed model. Every step skips when its output exists.
#   train set: 6,000 round G validation inputs x (greedy + 3 sampled explanations)
#   test set:  the realistic test (1,495 inputs) x (greedy + 3 sampled explanations)
# Usage: scripts/selfcheck.sh [run=runs/v2g-from-dapt-8ep-dpo] [n_train=6000] [out_dir=<run>/selfcheck]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1
PY="$HOME/venvs/relexp/bin/python"
RUN=${1:-runs/v2g-from-dapt-8ep-dpo}; N=${2:-6000}; O=${3:-$RUN/selfcheck}
mkdir -p "$O"
SC="$PY -m relevance_explainer.tiny.selfcheck"
echo "selfcheck $RUN start $(date -u +%H:%M)"
step() {  # name data split n
  local NAME=$1 D=$2 SPLIT=$3 NN=$4
  $SC sample --run "$RUN" --data "$D" --split "$SPLIT" --n "$NN" --k 3 --out "$O/${NAME}_samples.jsonl" || return 1
  [ "$(cat "$O/${NAME}_scored.jsonl" 2>/dev/null | wc -l)" -ge "$(wc -l < "$O/${NAME}_samples.jsonl")" ] || \
    $SC score --data "$D" --split "$SPLIT" --samples "$O/${NAME}_samples.jsonl" --out "$O/${NAME}_scored.jsonl" || return 1
  $SC features --run "$RUN" --data "$D" --split "$SPLIT" --samples "$O/${NAME}_samples.jsonl" \
    --scored "$O/${NAME}_scored.jsonl" --out "$O/${NAME}_features.pt" || return 1
  echo "done $NAME $(date -u +%H:%M)"
}
step test data/eval_real/itemfirst_nosim_v4 test "${NT:-100000}" || exit 1
step train data/v2_g_itemfirst_nosim val "$N" || exit 1
$SC train --train-features "$O/train_features.pt" --test-features "$O/test_features.pt" --out "$O/results.json" \
  --save-head "$O/head.pt"
echo "SELFCHECK-DONE $(date -u +%H:%M)"
