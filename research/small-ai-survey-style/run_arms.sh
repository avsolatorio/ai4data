#!/bin/bash
# usage: run_arms.sh GPU "arm1 arm2" "seed1 seed2"
# PY is the Python of the harness environment (default: python). Logs go to $SMALL_AI_RUNS/isco-survey/logs.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
L=${SMALL_AI_RUNS:-../../runs}/isco-survey/logs/arms_gpu$1.log
mkdir -p "$(dirname "$L")"
export CUDA_VISIBLE_DEVICES=$1
for arm in $2; do for seed in $3; do
  echo "== $arm $seed $(date -u +%H:%M:%S)" >> "$L"
  $PY arms.py $arm $seed >> "$L" 2>&1 || echo "FAILED $arm $seed" >> "$L"
done; done
echo "done $(date -u +%H:%M:%S)" >> "$L"
grep -E '^(== |FAILED|done|\{"arm")' "$L" | cut -c1-700 | tail -20
