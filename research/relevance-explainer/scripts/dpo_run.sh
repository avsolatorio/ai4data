#!/usr/bin/env bash
# On-policy DPO for a trained run on its data, then realistic-test scoring and judging.
# Usage: scripts/dpo_run.sh <run> <data_dir> <format>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?run}; D=${2:?data}; FMT=${3:?format}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1
PY="$HOME/venvs/relexp/bin/python"
TAG=$(basename "$RUN")
$PY -m relevance_explainer.distill.onpolicy sample --run "$RUN" --data "$D" --n 8000 --k 4 --out "onpolicy_${TAG}_samples.jsonl" || exit 1
$PY -m relevance_explainer.distill.onpolicy score --data "$D" --samples "onpolicy_${TAG}_samples.jsonl" --out "onpolicy_${TAG}_scored.jsonl" --workers 160 || exit 1
$PY -m relevance_explainer.distill.onpolicy pairs --data "$D" --scored "onpolicy_${TAG}_scored.jsonl" --out "dpo_pairs_${TAG}.jsonl" || exit 1
$PY -m relevance_explainer.tiny.dpo --run "$RUN" --data "$D" --pairs "dpo_pairs_${TAG}.jsonl" --out "$RUN-dpo" --beta 0.3 --lr 2e-5 --epochs 2 --bs 16 --sft 0.5 || exit 1
$PY -m relevance_explainer.tiny.evaluate --run "$RUN-dpo" --data "data/eval_real/$FMT" --precisions int8 --tag realtest --script-guard > "$RUN-dpo.eval.log" 2>&1
$PY -m relevance_explainer.synth.score_real --data "data/eval_real/$FMT" --run "$RUN-dpo" --tag realtest
scripts/judge_tag.sh "$RUN-dpo" realtest 2>&1 | grep -E "^student|DONE"
$PY -m relevance_explainer.tiny.export --run "$RUN-dpo" --out "$RUN-dpo/explainer-int8.pt"
echo "DPO-RUN-DONE $RUN"
