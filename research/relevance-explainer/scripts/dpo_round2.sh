#!/usr/bin/env bash
# A second on-policy DPO round from an already preference-tuned model, with the same number of teacher judge calls
# as a GRPO run (N inputs x K samples). Usage: scripts/dpo_round2.sh <run> <data> <out> [N] [K] [seed]
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
run=$1; data=$2; out=$3; n=${4:-8000}; k=${5:-4}; seed=${6:-1}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export TEACHER_URLS=${TEACHER_URLS:-http://localhost:8000/v1,http://localhost:8001/v1}
export PYTHONPATH=src
py="$RELEXP_PY"
tag=r2s$seed
$py -m relevance_explainer.distill.onpolicy sample --run "$run" --data "$data" --n "$n" --k "$k" --seed "$seed" --out "onpolicy_${tag}_samples.jsonl"
$py -m relevance_explainer.distill.onpolicy score --data "$data" --samples "onpolicy_${tag}_samples.jsonl" --out "onpolicy_${tag}_scored.jsonl" --workers 128
$py -m relevance_explainer.distill.onpolicy pairs --data "$data" --scored "onpolicy_${tag}_scored.jsonl" --out "dpo_pairs_${tag}.jsonl"
$py -m relevance_explainer.tiny.dpo --run "$run" --data "$data" --pairs "dpo_pairs_${tag}.jsonl" --out "$out" --beta 0.3 --lr 2e-5 --epochs 2 --sft 0.5
echo DPO-ROUND2-DONE
