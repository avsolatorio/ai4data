#!/bin/bash
# usage: run_learn.sh GPU "seed1 seed2"
# PY is the Python of the harness environment (default: python).
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
D=${SMALL_AI_RUNS:-../../runs}/isco-survey
mkdir -p "$D/learn"
export CUDA_VISIBLE_DEVICES=$1
for seed in $2; do
  echo "== $seed $(date -u +%H:%M:%S)" >> "$D/learn/seed$seed.log"
  $PY learn_record.py $seed >> "$D/learn/seed$seed.log" 2>&1 || echo "FAILED $seed" >> "$D/learn/seed$seed.log"
done
echo "done $(date -u +%H:%M:%S)" >> "$D/learn/gpu$1.done"
