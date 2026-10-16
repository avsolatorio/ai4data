#!/usr/bin/env bash
# After scripts/dapt.sh: can one fine-tuning stage plus DPO replace the eight-stage chain?
# 1. Of the three bases fine-tuned for 2 epochs, take the one with the best validation verdict accuracy.
# 2. Fine-tune that base for 4 epochs (the chain saw about 8.6M row-passes; 2 epochs of round G's data is about 2M).
# 3. DPO on whichever of the 2- and 4-epoch models has the better validation accuracy, as in the chain's last step.
# 4. Evaluate on all test sets and judge, for comparison with runs/v2g-nosim-dpo (the chain).
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
until grep -q "DAPT-DONE" $RELEXP_LOGS/dapt.log; do sleep 300; done
echo "followup start $(date -u +%H:%M)"
BEST=$($PY scripts/pick_best.py runs/v2g-from-base runs/v2g-from-cont runs/v2g-from-dapt) || exit 1
ARM=${BEST#runs/v2g-from-}
case $ARM in base) INIT=runs/pre-wide-full2/model.pt ;; cont) INIT=runs/pre-cont/model.pt ;; dapt) INIT=runs/pre-dapt/model.pt ;; *) exit 1 ;; esac
echo "best base by validation: $ARM"
DN=data/v2_g_itemfirst_nosim
R4="runs/v2g-from-$ARM-4ep"
[ -f "$R4/model_fp32.pt" ] || $PY -m relevance_explainer.tiny.train --data "$DN" --out "$R4" --tokenizer data/corpus/tokenizer.json \
  --init "$INIT" --lr 1e-3 --epochs 4 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc --test-limit 3000 \
  > "$R4.train.log" 2>&1 || exit 1
echo "trained 4 epochs $(date -u +%H:%M)"
SFT=$($PY scripts/pick_best.py "$BEST" "$R4") || exit 1
echo "DPO from $SFT"
[ -f "$SFT-dpo/model_fp32.pt" ] || scripts/dpo_run.sh "$SFT" "$DN" itemfirst_nosim_v4 > "$SFT.dpo.log" 2>&1
grep -E "pairs|^student" "$SFT.dpo.log"
for RUN in "$R4" "$SFT-dpo"; do
  RUN_DIR="$RUN" SUF=_v4 scripts/eval_sets.sh
  [ -f "$RUN/tables_generations_int8.jsonl" ] || $PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data data/eval_tables/itemfirst_nosim \
    --precisions int8 --tag tables --script-guard 2>&1 | grep -E "Error|int8 " | tail -2
  echo "== tables $RUN"; $PY -m relevance_explainer.synth.score_real --data data/eval_tables/itemfirst_nosim --run "$RUN" --tag tables | grep -E "teacher"
done
echo "FOLLOWUP-DONE $(date -u +%H:%M)"
