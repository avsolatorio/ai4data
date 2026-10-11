#!/usr/bin/env bash
# Check of the warmup-stable-decay schedule and resumable states in tiny/pretrain.py: a 40-step run straight through
# and the same run resumed from its step-20 state must end with the same held-out score and learning rates.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU:-0}
PY="$HOME/venvs/relexp/bin/python"
T=${TMPDIR:-/tmp}/wsd_check
rm -rf "$T"
COMMON=(--root data/corpus_multi_v32k --all-shards --holdout 200000 --eval-seqs 1 --tokens 40960 --seq 128 --bs 8
        --micro 8 --d-model 448 --layers 2 --heads 7 --attn-gate head --lr 2e-3 --schedule wsd --decay-frac 0.5
        --eval-every 20)
$PY -m relevance_explainer.tiny.pretrain "${COMMON[@]}" --out "$T/full" --save-every 20 > "$T.full.log" 2>&1
$PY -m relevance_explainer.tiny.pretrain "${COMMON[@]}" --out "$T/resumed" --resume-state "$T/full/state-000020.pt" > "$T.resumed.log" 2>&1
$PY - "$T" <<'EOF'
import json, sys
t = sys.argv[1]
full, res = (json.load(open(f"{t}/{r}/log.json"))["log"] for r in ("full", "resumed"))
f, r = full[-1], res[-1]
mean = lambda e: sum(v for k, v in e["heldout_ppl"].items() if k.endswith("_bpb")) / sum(k.endswith("_bpb") for k in e["heldout_ppl"])
print(f"step {f['step']} vs {r['step']}: train loss {f['train_loss']:.5f} vs {r['train_loss']:.5f}; "
      f"held-out bits per byte {mean(f):.5f} vs {mean(r):.5f}")
EOF
grep -h "resumed" "$T.resumed.log"
