#!/usr/bin/env bash
# Multilingual base with 2,048-token context: resume the final run's 60B state (step 457,764, stable phase) and train
# the rest of the run on 2,048-token sequences with 64 sequences per step, so a step holds the same 131,072 tokens and
# the step schedule is unchanged: 4B more tokens at lr 3e-3 to step 488,280, then the decay to 3e-4 at 80B. The rotary
# base stays 10,000 (adequate at 2,048). After an interruption it resumes from the latest state of this run.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src
PY="$RELEXP_PY"
R=data/corpus_multi_final
OUT=runs/base-2k
[ -f "$OUT/done" ] && { echo "BASE-2K-DONE (already)"; exit 0; }
mkdir -p "$OUT"
LATEST=$(ls "$OUT"/state-*.pt 2>/dev/null | tail -1)
LATEST=${LATEST:-runs/base-final/state-457764.pt}
echo "base 2k start $(date -u +%H:%M), resuming from $LATEST"
CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc-per-node 2 --master-port 29542 \
  -m relevance_explainer.tiny.pretrain --root "$R" --all-shards --holdout 200000 --eval-seqs 8 --out "$OUT" \
  --tokens 80e9 --seq 2048 --max-len 2048 --bs 64 --micro 32 --d-model 448 --layers 8 --heads 7 --attn-gate head \
  --d-ff 1024 --lr 3e-3 --schedule wsd --decay-start 488280 --eval-every 19073 --save-every 38147 \
  --resume-state "$LATEST" >> "$OUT.log" 2>&1 && touch "$OUT/done" || { echo "base 2k stopped $(date -u +%H:%M)"; exit 1; }
echo "BASE-2K-DONE $(date -u +%H:%M)"
