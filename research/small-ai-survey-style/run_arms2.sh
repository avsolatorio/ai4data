#!/bin/bash
# usage: run_arms2.sh GPU TAG "arm:seed arm:seed ..."
# PY is the Python of the harness environment (default: python). Logs go to $SMALL_AI_RUNS/isco-survey/logs.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
L=${SMALL_AI_RUNS:-../../runs}/isco-survey/logs/arms_$2.log
mkdir -p "$(dirname "$L")"
export CUDA_VISIBLE_DEVICES=$1
for job in $3; do
  arm=${job%%:*}; seed=${job##*:}
  echo "== $arm $seed $(date -u +%H:%M:%S)" >> "$L"
  $PY arms.py $arm $seed >> "$L" 2>&1 || echo "FAILED $arm $seed" >> "$L"
done
echo "done $(date -u +%H:%M:%S)" >> "$L"
grep -E '^(== |FAILED|done|\{"arm")' "$L" | cut -c1-900 | tail -20
