#!/usr/bin/env bash
# Continue training an existing model on new-format data, then score on the realistic test set and judge narratives.
# Usage: scripts/continue_train.sh <init_run> <data_dir> <out_run> <format> [lr=3e-4] [epochs=3]
set -uo pipefail
cd "$(dirname "$0")/.."
INIT=${1:?init run}; D=${2:?data}; RUN=${3:?out run}; FMT=${4:?format}; LR=${5:-3e-4}; EP=${6:-3}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$HOME/venvs/relexp/bin/python"
$PY -m relevance_explainer.tiny.train --data "$D" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
  --init "$INIT/model_fp32.pt" --lr "$LR" --epochs "$EP" --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
  > "$RUN.train.log" 2>&1 || exit 1
$PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data "data/eval_real/$FMT" --precisions int8 --tag realtest --script-guard > "$RUN.eval.log" 2>&1
$PY -m relevance_explainer.synth.score_real --data "data/eval_real/$FMT" --run "$RUN" --tag realtest
scripts/judge_tag.sh "$RUN" realtest 2>&1 | grep -E "^student|DONE"
echo "CONTINUE-DONE $RUN"
