#!/usr/bin/env bash
# Single-stage recipe: one fine-tune of a pretrained base on round G's data, then DPO, then all evaluations and judges.
# The best epoch is chosen by validation verdict accuracy, so a long run also covers shorter ones.
# Usage: scripts/single_stage.sh <epochs> [base=runs/pre-dapt/model.pt] [name=v2g-from-dapt-<epochs>ep]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
EP=${1:?epochs}; BASE=${2:-runs/pre-dapt/model.pt}; RUN=runs/${3:-v2g-from-dapt-${EP}ep}
DN=data/v2_g_itemfirst_nosim
echo "single stage $RUN start $(date -u +%H:%M)"
[ -f "$RUN/model_fp32.pt" ] || $PY -m relevance_explainer.tiny.train --data "$DN" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
  --init "$BASE" --lr 1e-3 --epochs "$EP" --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc --test-limit 3000 \
  > "$RUN.train.log" 2>&1 || exit 1
grep "'epoch'" "$RUN.train.log" | cut -c1-160
echo "trained $(date -u +%H:%M)"
[ -f "$RUN-dpo/model_fp32.pt" ] || scripts/dpo_run.sh "$RUN" "$DN" itemfirst_nosim_v4 > "$RUN.dpo.log" 2>&1
grep -E "pairs|^student" "$RUN.dpo.log"
for R in "$RUN" "$RUN-dpo"; do
  RUN_DIR="$R" SUF=_v4 scripts/eval_sets.sh
  [ -f "$R/tables_generations_int8.jsonl" ] || $PY -m relevance_explainer.tiny.evaluate --run "$R" --data data/eval_tables/itemfirst_nosim \
    --precisions int8 --tag tables --script-guard 2>&1 | grep -E "Error|int8 " | tail -2
  echo "== tables $R"; $PY -m relevance_explainer.synth.score_real --data data/eval_tables/itemfirst_nosim --run "$R" --tag tables | grep -E "teacher"
done
echo "SINGLE-STAGE-DONE $RUN $(date -u +%H:%M)"
