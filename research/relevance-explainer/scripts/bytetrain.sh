#!/usr/bin/env bash
# Detached-friendly launcher for the ByteFlow task fine-tune. Usage: GPU=0 scripts/bytetrain.sh runs/byteflow-task runs/byteflow-a/model.pt 3
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output run dir}
INIT=${2:?pretrained model.pt}
EPOCHS=${3:-3}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH=src
exec "$HOME/venvs/relexp/bin/python" -m relevance_explainer.tiny.bytetrain --data "${DATA:-data/f2}" --out "$OUT" --init "$INIT" \
  --epochs "$EPOCHS" --lr 5e-4 --bs 16 --micro 8
