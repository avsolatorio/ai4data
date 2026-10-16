#!/usr/bin/env bash
# Evaluate a task model at several copy-bias strengths. Usage: scripts/sweep_copy_bias.sh runs/f1-pre2 data/f1 "1.0 2.0 4.0"
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONPATH=src
for b in ${3:-1.0 2.0 4.0}; do
  echo "copy-bias $b"
  "$RELEXP_PY" -m relevance_explainer.tiny.evaluate --run "$1" --data "$2" --precisions int8 --copy-bias "$b" --tag "cb$b" 2>&1 | grep -E "^int8"
done
