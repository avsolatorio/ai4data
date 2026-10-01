#!/usr/bin/env bash
# Detached-friendly pretraining launcher. Usage: GPU=1 scripts/pretrain.sh runs/pre-40m 1.6e9 --layers 8 --d-model 576 --heads 9 --d-ff 1536
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output directory}
TOKENS=${2:?token budget}
shift 2
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONPATH=src
exec "$HOME/venvs/relexp/bin/python" -m relevance_explainer.tiny.pretrain --root data/corpus --out "$OUT" --tokens "$TOKENS" \
  --init-emb Qwen/Qwen3-0.6B --micro 64 --lr "${LR:-2e-3}" --eval-every 2000 "$@"
