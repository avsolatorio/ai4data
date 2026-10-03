#!/usr/bin/env bash
# Run a project module with the training environment. Usage: GPU=1 scripts/py.sh relevance_explainer.tiny.mbr --run ...
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONPATH=src
exec "$HOME/venvs/relexp/bin/python" -m "$@"
