#!/usr/bin/env bash
# Multilingual base, final run: 32k vocabulary, width 448, 8 layers, 7 heads, per-head attention gate (31.8M
# parameters), 80B tokens on both GPUs, warmup-stable-decay at 3e-3 (decay over the last 20%). The stable phase writes
# a resumable state every 38,147 steps (5B tokens), so decay branches can start from 20B (step 152,588) and 40B
# (step 305,176). Waits for scripts/multibase_final_data.sh; after an interruption it resumes from the latest state.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src
PY="$RELEXP_PY"
R=data/corpus_multi_final
OUT=runs/base-final
until [ -f "$R/tok/.done" ]; do sleep 120; done
[ -f "$OUT/done" ] && { echo "BASE-FINAL-DONE (already)"; exit 0; }
mkdir -p "$OUT"
LATEST=$(ls "$OUT"/state-*.pt 2>/dev/null | tail -1)
echo "base final start $(date -u +%H:%M)${LATEST:+, resuming from $LATEST}"
CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc-per-node 2 --master-port 29541 \
  -m relevance_explainer.tiny.pretrain --root "$R" --all-shards --holdout 200000 --eval-seqs 8 --out "$OUT" \
  --tokens 80e9 --d-model 448 --layers 8 --heads 7 --attn-gate head --d-ff 1024 --lr 3e-3 --schedule wsd \
  --decay-frac 0.2 --micro 64 --eval-every 19073 --save-every 38147 --init-emb Qwen/Qwen3-0.6B \
  ${LATEST:+--resume-state "$LATEST"} >> "$OUT.log" 2>&1 && touch "$OUT/done" || { echo "base final stopped $(date -u +%H:%M)"; exit 1; }
echo "BASE-FINAL-DONE $(date -u +%H:%M)"
