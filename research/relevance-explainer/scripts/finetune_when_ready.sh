#!/usr/bin/env bash
# Wait for a pretraining run to reach its final step, then fine-tune on the task data with the standard recipe.
# Usage: GPU=1 scripts/finetune_when_ready.sh runs/pre-wide-v2corpus 24414 runs/f1-v2corpus data/f1
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
PRE=${1:?pretraining run dir}
FINAL=${2:?final step}
OUT=${3:?output run dir}
DATA=${4:-data/f1}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONPATH=src

until [ -f "$PRE/log.json" ] && grep -q "\"step\": $FINAL" "$PRE/log.json"; do sleep 60; done
sleep 30  # let the final checkpoint finish writing

exec "$RELEXP_PY" -m relevance_explainer.tiny.train --data "$DATA" --out "$OUT" \
  --tokenizer data/corpus/tokenizer.json --init "$PRE/model.pt" \
  --lr 1e-3 --epochs 8 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc
