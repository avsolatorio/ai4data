#!/usr/bin/env bash
# After a data round finishes narrating, assemble the combined set, fine-tune from the base, run the on-policy
# preference pipeline, and evaluate. Everything after narration is unattended.
# Usage: GPU=1 scripts/round_when_ready.sh data/v4 data/f3 "data/v3 data/v2 data/v1" runs/pre-wide-full2/model.pt runs/f3
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
ROUND=${1:?round directory being narrated}
OUT=${2:?combined dataset directory}
ALSO=${3:?earlier round directories}
INIT=${4:?pretrained model.pt}
RUN=${5:?run prefix}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONPATH=src
export TEACHER_URLS=${TEACHER_URLS:-http://localhost:8000/v1}
PY="$RELEXP_PY"

until ! pgrep -u "$USER" -f "run_pipeline.sh $ROUND" > /dev/null; do sleep 60; done

mkdir -p "$OUT"
cp "$ROUND/examples.jsonl" "$OUT/examples.jsonl"
$PY -m relevance_explainer.synth.generate assemble --out "$OUT" --also $ALSO --max-ratio 100 --similarity
$PY -m relevance_explainer.tiny.train --data "$OUT" --out "$RUN-sft" --tokenizer data/corpus/tokenizer.json --init "$INIT" \
  --lr 1e-3 --epochs 8 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc
$PY -m relevance_explainer.distill.onpolicy sample --run "$RUN-sft" --data "$OUT" --n 8000 --k 4
python -m relevance_explainer.distill.onpolicy score --data "$OUT" --workers 192
python -m relevance_explainer.distill.onpolicy pairs --data "$OUT"
$PY -m relevance_explainer.tiny.dpo --run "$RUN-sft" --data "$OUT" --out "$RUN-dpo" --beta 0.3 --lr 2e-5 --epochs 2 --bs 16 --sft 0.5
$PY -m relevance_explainer.tiny.evaluate --run "$RUN-dpo" --data "$OUT" --precisions int8 int4 --tag ev
$PY -m relevance_explainer.tiny.export --run "$RUN-dpo" --out "$RUN-dpo/explainer-int8.pt"
echo ROUND-PIPELINE-DONE
