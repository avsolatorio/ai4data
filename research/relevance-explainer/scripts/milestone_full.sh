#!/usr/bin/env bash
# Full-format model at a data milestone, then on-policy DPO, each scored on the realistic test set.
# Usage: scripts/milestone_full.sh <name> <min_chunks> [old_share=0.5]
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?name}; MINC=${2:?min chunks}; SHARE=${3:-0.5}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1
PY="$HOME/venvs/relexp/bin/python"
until [ "$(ls data/real/c*/examples.jsonl 2>/dev/null | wc -l)" -ge "$MINC" ]; do sleep 300; done
echo "milestone $NAME: $(ls data/real/c*/examples.jsonl | wc -l) chunks, $(ls data/real/t*/examples.jsonl 2>/dev/null | wc -l) translation sets $(date -u +%H:%M)"
D="data/v2_$NAME"
[ -f "$D/train.jsonl" ] || $PY -m relevance_explainer.synth.assemble_v2 --out "$D" --old-share "$SHARE" || exit 1
RUN="runs/v2$NAME-full"
if [ ! -f "$RUN/model_fp32.pt" ]; then
  $PY -m relevance_explainer.tiny.train --data "$D" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
    --init runs/pre-wide-full2/model.pt --lr 1e-3 --epochs 8 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
    > "$RUN.train.log" 2>&1 || exit 1
fi
$PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data data/eval_real/full --precisions int8 --tag realtest --script-guard > "$RUN.eval.log" 2>&1
$PY -m relevance_explainer.synth.score_real --data data/eval_real/full --run "$RUN" --tag realtest
# on-policy DPO with the round settings
$PY -m relevance_explainer.distill.onpolicy sample --run "$RUN" --data "$D" --n 8000 --k 4 --out onpolicy_samples.jsonl || exit 1
$PY -m relevance_explainer.distill.onpolicy score --data "$D" --samples onpolicy_samples.jsonl --out onpolicy_scored.jsonl --workers 160 || exit 1
$PY -m relevance_explainer.distill.onpolicy pairs --data "$D" --scored onpolicy_scored.jsonl --out dpo_pairs.jsonl || exit 1
$PY -m relevance_explainer.tiny.dpo --run "$RUN" --data "$D" --pairs dpo_pairs.jsonl --out "runs/v2$NAME-dpo" --beta 0.3 --lr 2e-5 --epochs 2 --bs 16 --sft 0.5 || exit 1
$PY -m relevance_explainer.tiny.evaluate --run "runs/v2$NAME-dpo" --data data/eval_real/full --precisions int8 --tag realtest --script-guard > "runs/v2$NAME-dpo.eval.log" 2>&1
$PY -m relevance_explainer.synth.score_real --data data/eval_real/full --run "runs/v2$NAME-dpo" --tag realtest
$PY -m relevance_explainer.tiny.export --run "runs/v2$NAME-dpo" --out "runs/v2$NAME-dpo/explainer-int8.pt"
echo "MILESTONE-FULL-$NAME-DONE"
