#!/usr/bin/env bash
# No-score arm: milestone C's recipe (continue runs/f6-dpo on data/v2_c_itemfirst, lr 3e-4, 3 epochs) with the
# retrieval-similarity line removed from every training and test input. Supervised stage only.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
D=data/v2_c_itemfirst_nosim
RUN=runs/v2c-cont-itemfirst-nosim
[ -f "$D/train.jsonl" ] || $PY scripts/strip_similarity.py data/v2_c_itemfirst "$D" || exit 1
[ -f "$RUN/model_fp32.pt" ] || $PY -m relevance_explainer.tiny.train --data "$D" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
  --init runs/f6-dpo/model_fp32.pt --lr 3e-4 --epochs 3 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
  > "$RUN.train.log" 2>&1 || exit 1
# realistic test set without the line (data/sim_variants/none)
$PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data data/sim_variants/none --precisions int8 --tag realtest --script-guard > "$RUN.eval.log" 2>&1
$PY -m relevance_explainer.synth.score_real --data data/sim_variants/none --run "$RUN" --tag realtest
scripts/judge_tag.sh "$RUN" realtest 2>&1 | grep -E "^student|DONE"
echo "ARM-NOSIM-DONE"
