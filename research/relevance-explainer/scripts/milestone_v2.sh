#!/usr/bin/env bash
# Train and compare the three input formats once enough realistic data exists.
# Usage: scripts/milestone_v2.sh <name> <min_chunks> [old_share=0.6]
# Waits for <min_chunks> finished chunks and the labelled realistic test set, assembles data/v2_<name>[_min|_itemfirst],
# trains runs/v2<name>-{full,min,itemfirst} with the round recipe, and scores each on data/eval_real/<format>.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
NAME=${1:?name}; MINC=${2:?min chunks}; SHARE=${3:-0.6}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
until [ "$(ls data/real/c*/examples.jsonl 2>/dev/null | wc -l)" -ge "$MINC" ] && [ -f data/real_test/examples_labeled.jsonl ]; do sleep 300; done
echo "milestone $NAME: $(ls data/real/c*/examples.jsonl | wc -l) chunks $(date -u +%H:%M)"
$PY -m relevance_explainer.synth.assemble_v2 --out "data/v2_$NAME" --old-share "$SHARE" || exit 1
[ -f data/eval_real/full/test.jsonl ] || $PY -m relevance_explainer.synth.eval_sets || exit 1
$PY -m relevance_explainer.synth.realistic_eval --data data/real_test --run runs/f6-dpo > "data/real_test/baseline_f6-dpo.txt" 2>&1
for fmt in full min itemfirst; do
  data="data/v2_$NAME"; [ "$fmt" = full ] || data="data/v2_${NAME}_$fmt"
  run="runs/v2$NAME-$fmt"
  if [ ! -f "$run/model_fp32.pt" ]; then
    $PY -m relevance_explainer.tiny.train --data "$data" --out "$run" --tokenizer data/corpus/tokenizer.json \
      --init runs/pre-wide-full2/model.pt --lr 1e-3 --epochs 8 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
      > "$run.train.log" 2>&1 || { echo "train $fmt failed"; continue; }
  fi
  $PY -m relevance_explainer.tiny.evaluate --run "$run" --data "data/eval_real/$fmt" --precisions int8 --tag realtest --script-guard > "$run.eval.log" 2>&1
  $PY -m relevance_explainer.synth.score_real --data "data/eval_real/$fmt" --run "$run" --tag realtest
done
echo "MILESTONE-$NAME-DONE"
