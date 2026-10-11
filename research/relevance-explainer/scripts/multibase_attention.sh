#!/usr/bin/env bash
# Multilingual base, attention variants on the 32k, width 448 shape (same 1B-token comparison as
# scripts/multibase_shapes30.sh, baseline runs/mb-v32k-d448-l7):
#   gqa: 8 layers with one key/value head shared by the 7 query heads (grouped-query attention), about the same size
#   gate: 7 layers with a per-head sigmoid gate on the attention output (Qiu et al. 2025, Qwen3-Next)
#   gqa-gate: both
#   seed1: the baseline again with another seed, to measure run-to-run noise (the held-out text does not change)
# Usage: GPU=0 scripts/multibase_attention.sh A   (runs gate, seed1)
#        GPU=1 scripts/multibase_attention.sh B   (runs gqa, gqa-gate; 32 sequences per pass beside the teacher server)
#        GPU=0 scripts/multibase_attention.sh C   (runs 8 layers with the gate)
# Every run skips when its output exists.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU:-0}
PY="$HOME/venvs/relexp/bin/python"
R=data/corpus_multi_v32k
PART=$1
if [ "$PART" = A ]; then
  RUNS=("v32k-d448-l7-gate 7 0 head 0 64" "v32k-d448-l7-seed1 7 0 '' 1 64")
else
  RUNS=("v32k-d448-l8-gqa1 8 1 '' 0 32" "v32k-d448-l8-gqa1-gate 8 1 head 0 32")
fi
if [ "$PART" = C ]; then
  RUNS=("v32k-d448-l8-gate 8 0 head 0 64")
fi
echo "attention $PART start $(date -u +%H:%M) on GPU $CUDA_VISIBLE_DEVICES"
for c in "${RUNS[@]}"; do
  eval set -- $c
  NAME=$1; L=$2; KV=$3; GATE=$4; SEED=$5; MICRO=$6
  OUT=runs/mb-$NAME
  [ -f "$OUT/done" ] || { $PY -m relevance_explainer.tiny.pretrain --root "$R" --all-shards --holdout 200000 --eval-seqs 8 \
      --out "$OUT" --tokens 1e9 --d-model 448 --layers "$L" --heads 7 --kv-heads "$KV" --attn-gate "$GATE" --seed "$SEED" \
      --d-ff 1024 --lr 2e-3 --micro "$MICRO" --eval-every 2000 --init-emb Qwen/Qwen3-0.6B > "$OUT.log" 2>&1 \
      && touch "$OUT/done"; } || { echo "pretrain $NAME failed"; continue; }
  echo "$NAME: $(grep params "$OUT.log" | head -1); $(grep 'mean held-out bits per byte' "$OUT.log" | tail -1)"
done
echo "ATTENTION-$PART-DONE $(date -u +%H:%M)"
