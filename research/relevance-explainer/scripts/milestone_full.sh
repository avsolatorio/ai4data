#!/usr/bin/env bash
# Models at a data milestone: each listed format trained and scored on the realistic test set; on-policy DPO for the
# first format. Usage: scripts/milestone_full.sh <name> <min_chunks> [old_share=0.5] [formats="itemfirst full"]
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?name}; MINC=${2:?min chunks}; SHARE=${3:-0.5}; FORMATS=${4:-"itemfirst full"}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1
PY="$HOME/venvs/relexp/bin/python"
until [ "$(ls data/real/c*/examples.jsonl 2>/dev/null | wc -l)" -ge "$MINC" ]; do sleep 300; done
echo "milestone $NAME: $(ls data/real/c*/examples.jsonl | wc -l) chunks, $(ls data/real/t*/examples.jsonl 2>/dev/null | wc -l) translation sets $(date -u +%H:%M)"
[ -f "data/v2_$NAME/train.jsonl" ] || $PY -m relevance_explainer.synth.assemble_v2 --out "data/v2_$NAME" --old-share "$SHARE" || exit 1
first=""
for fmt in $FORMATS; do
  D="data/v2_$NAME"; [ "$fmt" = full ] || D="data/v2_${NAME}_$fmt"
  RUN="runs/v2$NAME-$fmt"
  if [ ! -f "$RUN/model_fp32.pt" ]; then
    $PY -m relevance_explainer.tiny.train --data "$D" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
      --init runs/pre-wide-full2/model.pt --lr 1e-3 --epochs 8 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
      > "$RUN.train.log" 2>&1 || { echo "train $fmt failed"; continue; }
  fi
  $PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data "data/eval_real/$fmt" --precisions int8 --tag realtest --script-guard > "$RUN.eval.log" 2>&1
  $PY -m relevance_explainer.synth.score_real --data "data/eval_real/$fmt" --run "$RUN" --tag realtest
  [ -z "$first" ] && first="$fmt"
done
# on-policy DPO for the first format
fmt=$first; D="data/v2_$NAME"; [ "$fmt" = full ] || D="data/v2_${NAME}_$fmt"; RUN="runs/v2$NAME-$fmt"
$PY -m relevance_explainer.distill.onpolicy sample --run "$RUN" --data "$D" --n 8000 --k 4 --out onpolicy_samples.jsonl || exit 1
$PY -m relevance_explainer.distill.onpolicy score --data "$D" --samples onpolicy_samples.jsonl --out onpolicy_scored.jsonl --workers 160 || exit 1
$PY -m relevance_explainer.distill.onpolicy pairs --data "$D" --scored onpolicy_scored.jsonl --out dpo_pairs.jsonl || exit 1
$PY -m relevance_explainer.tiny.dpo --run "$RUN" --data "$D" --pairs dpo_pairs.jsonl --out "$RUN-dpo" --beta 0.3 --lr 2e-5 --epochs 2 --bs 16 --sft 0.5 || exit 1
$PY -m relevance_explainer.tiny.evaluate --run "$RUN-dpo" --data "data/eval_real/$fmt" --precisions int8 --tag realtest --script-guard > "$RUN-dpo.eval.log" 2>&1
$PY -m relevance_explainer.synth.score_real --data "data/eval_real/$fmt" --run "$RUN-dpo" --tag realtest
$PY -m relevance_explainer.tiny.export --run "$RUN-dpo" --out "$RUN-dpo/explainer-int8.pt"
echo "MILESTONE-FULL-$NAME-DONE"
