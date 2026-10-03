#!/usr/bin/env bash
# GRPO on the narratives with the teacher as the judge. Usage: scripts/grpo.sh <run> <data> <out> [extra args]
# Environment: GPU (default 1), TEACHER_URLS (default both local servers).
set -euo pipefail
cd "$(dirname "$0")/.."
run=$1; data=$2; out=$3; shift 3
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export TEACHER_URLS=${TEACHER_URLS:-http://localhost:8000/v1,http://localhost:8001/v1}
export PYTHONPATH=src
exec "$HOME/venvs/relexp/bin/python" -m relevance_explainer.tiny.grpo --run "$run" --data "$data" --out "$out" "$@"
