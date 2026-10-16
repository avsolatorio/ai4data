#!/usr/bin/env bash
# Checks of two-GPU data-parallel pretraining (torchrun) in tiny/pretrain.py:
#   1. a two-GPU run straight through and the same run resumed from its step-20 state end identically
#   2. 300 steps on one GPU and on two GPUs (same global batch, different random windows) reach similar held-out scores
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src
PY="$RELEXP_PY"
TR="$PY -m torch.distributed.run --nproc-per-node 2 --master-port 29531 -m relevance_explainer.tiny.pretrain"
T=${TMPDIR:-/tmp}/ddp_check
rm -rf "$T"
COMMON=(--root data/corpus_multi_v32k --all-shards --holdout 200000 --eval-seqs 1 --seq 128 --bs 16 --micro 4
        --d-model 448 --layers 2 --heads 7 --attn-gate head --lr 2e-3 --schedule wsd --decay-frac 0.5)
CUDA_VISIBLE_DEVICES=0,1 $TR "${COMMON[@]}" --tokens 81920 --eval-every 20 --out "$T/full" --save-every 20 > "$T.full.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1 $TR "${COMMON[@]}" --tokens 81920 --eval-every 20 --out "$T/resumed" \
  --resume-state "$T/full/state-000020.pt" > "$T.resumed.log" 2>&1
CUDA_VISIBLE_DEVICES=0 $PY -m relevance_explainer.tiny.pretrain "${COMMON[@]}" --tokens 614400 --eval-every 300 \
  --out "$T/one" > "$T.one.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1 $TR "${COMMON[@]}" --tokens 614400 --eval-every 300 --out "$T/two" > "$T.two.log" 2>&1
$PY - "$T" <<'EOF'
import json, sys
t = sys.argv[1]
last = lambda r: json.load(open(f"{t}/{r}/log.json"))["log"][-1]
mean = lambda e: sum(v for k, v in e["heldout_ppl"].items() if k.endswith("_bpb")) / sum(k.endswith("_bpb") for k in e["heldout_ppl"])
f, r = last("full"), last("resumed")
print(f"resume on two GPUs, step {f['step']} vs {r['step']}: train loss {f['train_loss']:.5f} vs {r['train_loss']:.5f}; "
      f"held-out bits per byte {mean(f):.5f} vs {mean(r):.5f}")
o, w = last("one"), last("two")
print(f"300 steps: one GPU {mean(o):.4f}, two GPUs {mean(w):.4f} held-out bits per byte")
EOF
grep -h "tok/s" "$T.one.log" "$T.two.log" | tail -2
