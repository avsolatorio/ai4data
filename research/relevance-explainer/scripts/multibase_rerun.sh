#!/usr/bin/env bash
# Rerun 20 MB shapes from scripts/multibase_shapes.sh that failed (out of memory next to the teacher server on GPU 1),
# with 32 sequences per forward pass. Gradient accumulation keeps the effective batch at 128, so results are comparable.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU:-1}
PY="$RELEXP_PY"
until grep -q SHAPES-DONE $RELEXP_LOGS/shapes.log; do sleep 60; done
echo "rerun start $(date -u +%H:%M)"
for c in "v24k-d384-l6 24000 384 6 6" "v32k-d384-l5 32000 384 5 6"; do
  set -- $c
  NAME=$1; V=$2; D=$3; L=$4; H=$5
  R=data/corpus_multi_v$((V / 1000))k
  OUT=runs/mb-$NAME
  [ -f "$OUT/done" ] && continue
  until [ -f "$R/tok/.done" ]; do sleep 60; done
  mv "$OUT.log" "$OUT.oom.log" 2>/dev/null
  $PY -m relevance_explainer.tiny.pretrain --root "$R" --all-shards --holdout 200000 --eval-seqs 8 \
    --out "$OUT" --tokens 1e9 --d-model "$D" --layers "$L" --heads "$H" --d-ff 1024 --lr 2e-3 --micro 32 \
    --eval-every 2000 --init-emb Qwen/Qwen3-0.6B > "$OUT.log" 2>&1 && touch "$OUT/done" || { echo "pretrain $NAME failed again"; continue; }
  echo "$NAME: $(grep params "$OUT.log" | head -1); $(grep 'mean held-out bits per byte' "$OUT.log" | tail -1)"
done
echo "RERUN-DONE $(date -u +%H:%M)"
