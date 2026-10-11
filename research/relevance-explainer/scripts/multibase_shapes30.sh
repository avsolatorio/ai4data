#!/usr/bin/env bash
# Multilingual base model, 30 MB budget: the same comparison as scripts/multibase_shapes.sh (1B tokens each, held-out
# bits per byte) for shapes that keep 7 or 8 layers at width 448 with a larger vocabulary. It reuses the 20 MB
# comparison's tokenized corpora and waits for each to be ready. Every step skips when its output exists.
# Usage: scripts/multibase_shapes30.sh [tokens=1e9]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU:-1}
PY="$HOME/venvs/relexp/bin/python"
TOKENS=${1:-1e9}
echo "shapes30 start $(date -u +%H:%M) on GPU $CUDA_VISIBLE_DEVICES"
CANDIDATES=(
  "v24k-d448-l7 24000 448 7 7"
  "v32k-d448-l7 32000 448 7 7"
  "v24k-d448-l8 24000 448 8 7"
)
for c in "${CANDIDATES[@]}"; do
  set -- $c
  NAME=$1; V=$2; D=$3; L=$4; H=$5
  R=data/corpus_multi_v$((V / 1000))k
  until [ -f "$R/tok/.done" ]; do sleep 60; done  # tokenized by scripts/multibase_shapes.sh, possibly still running
  OUT=runs/mb-$NAME
  [ -f "$OUT/done" ] || { $PY -m relevance_explainer.tiny.pretrain --root "$R" --all-shards --holdout 200000 --eval-seqs 8 \
      --out "$OUT" --tokens "$TOKENS" --d-model "$D" --layers "$L" --heads "$H" --d-ff 1024 --lr 2e-3 --micro 64 \
      --eval-every 2000 --init-emb Qwen/Qwen3-0.6B > "$OUT.log" 2>&1 && touch "$OUT/done"; } || { echo "pretrain $NAME failed"; continue; }
  echo "$NAME: $(grep params "$OUT.log" | head -1); $(grep 'mean held-out bits per byte' "$OUT.log" | tail -1)"
done
$PY scripts/multibase_compare.py runs/mb-v12k-d448-l7 runs/mb-v16k-d448-l6 runs/mb-v24k-d384-l6 runs/mb-v32k-d384-l5 \
  runs/mb-v24k-d448-l7 runs/mb-v32k-d448-l7 runs/mb-v24k-d448-l8
echo "SHAPES30-DONE $(date -u +%H:%M)"
