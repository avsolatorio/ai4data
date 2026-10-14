#!/usr/bin/env bash
# Train and evaluate one format-test run with the settings of scripts/fmt_pipeline.sh (used to rerun a run that failed).
#   scripts/fmt_train_one.sh <data dir> <run dir> <GPU> [extra train flags]
#   scripts/fmt_train_one.sh data/fmt_s/B runs/fmt-s-B 0 --special-tokens
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="$HOME/venvs/relexp/bin/python"
DATA=$1; RUN=$2; GPU=$3; shift 3
if [ ! -f "$RUN/model_fp32.pt" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -m relevance_explainer.tiny.train --data "$DATA" --out "$RUN" \
    --tokenizer data/corpus_multi_final/tokenizer.json --init runs/base-2k/model.pt --lr 1e-3 --epochs 4 --bs 32 \
    --dropout 0.1 --verdict-weight 5 --select val_verdict_acc --max-len 2048 --max-new 420 --test-limit 3000 "$@" \
    > "$RUN.train.log" 2>&1 || { echo "training failed: $RUN.train.log"; exit 1; }
fi
RUN_DIR=$RUN SUF=_v4 scripts/eval_sets.sh  # GPU 1
[ -f "$RUN/tables_generations_int8.jsonl" ] || CUDA_VISIBLE_DEVICES=$GPU $PY -m relevance_explainer.tiny.evaluate --run "$RUN" \
  --data data/eval_tables/itemfirst_nosim --precisions int8 --tag tables --script-guard 2>&1 | grep -E "Error|int8 " | tail -2
echo "DONE $RUN $(date -u +%H:%M)"
