#!/usr/bin/env bash
# Seed spread for the 300-step comparison of scripts/check_ddp.sh: one GPU and two GPUs, seeds 1 and 2.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="$HOME/venvs/relexp/bin/python"
TR="$PY -m torch.distributed.run --nproc-per-node 2 --master-port 29531 -m relevance_explainer.tiny.pretrain"
T=${TMPDIR:-/tmp}/ddp_check
COMMON=(--root data/corpus_multi_v32k --all-shards --holdout 200000 --eval-seqs 4 --seq 128 --bs 16 --micro 4
        --d-model 448 --layers 2 --heads 7 --attn-gate head --lr 2e-3 --schedule wsd --decay-frac 0.5 --tokens 614400
        --eval-every 300)
for S in 1 2; do
  CUDA_VISIBLE_DEVICES=0 $PY -m relevance_explainer.tiny.pretrain "${COMMON[@]}" --seed $S --out "$T/one-s$S" > "$T.one-s$S.log" 2>&1
  CUDA_VISIBLE_DEVICES=0,1 $TR "${COMMON[@]}" --seed $S --out "$T/two-s$S" > "$T.two-s$S.log" 2>&1
  for R in one two; do
    echo "$R GPU seed $S: $(grep 'mean held-out' "$T.$R-s$S.log" | tail -1)"
  done
done
