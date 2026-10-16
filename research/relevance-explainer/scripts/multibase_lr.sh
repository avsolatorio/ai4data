#!/usr/bin/env bash
# Multilingual base: learning rate for the gated 32k, 8-layer shape. Qiu et al. 2025 report that the attention-output
# gate tolerates larger learning rates; this runs the scripts/multibase_attention.sh setup (1B tokens) at each rate given.
# runs/mb-v32k-d448-l8-gate is the 2e-3 point. Every run skips when its output exists.
# Usage: GPU=1 MICRO=32 scripts/multibase_lr.sh 4e-3 [more rates]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU:-1}
PY="$RELEXP_PY"
R=data/corpus_multi_v32k
echo "lr start $(date -u +%H:%M) on GPU $CUDA_VISIBLE_DEVICES"
for LR in "$@"; do
  OUT=runs/mb-v32k-d448-l8-gate-lr$LR
  [ -f "$OUT/done" ] || { $PY -m relevance_explainer.tiny.pretrain --root "$R" --all-shards --holdout 200000 --eval-seqs 8 \
      --out "$OUT" --tokens 1e9 --d-model 448 --layers 8 --heads 7 --attn-gate head --d-ff 1024 --lr "$LR" \
      --micro "${MICRO:-64}" --eval-every 2000 --init-emb Qwen/Qwen3-0.6B > "$OUT.log" 2>&1 && touch "$OUT/done"; } \
    || { echo "pretrain lr $LR failed"; continue; }
  echo "lr $LR: $(grep params "$OUT.log" | head -1); $(grep 'mean held-out bits per byte' "$OUT.log" | tail -1)"
done
echo "LR-DONE $(date -u +%H:%M)"
