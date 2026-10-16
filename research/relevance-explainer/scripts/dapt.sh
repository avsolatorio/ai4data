#!/usr/bin/env bash
# Domain pretraining experiment, after round G. Three bases, each fine-tuned the same way on round G's data:
#   base  runs/pre-wide-full2 as it is (the base of the current fine-tuning chain)
#   cont  base + 600M more tokens of the general corpus only (control for "more pretraining")
#   dapt  base + 600M tokens with 30% of sequences from the domain corpus (synth/domain_corpus.py)
# Every step skips when its output exists, so a restart resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
until grep -q "ROUND-G-DONE" $RELEXP_LOGS/round_g.log; do sleep 300; done
echo "dapt start $(date -u +%H:%M)"
# 1. domain corpus (after round G, so its labelled sets and the Eurostat test half exist)
DC=data/corpus_domain
[ -f "$DC/raw/English.jsonl" ] || $PY -m relevance_explainer.synth.domain_corpus --out "$DC" 2>&1 | grep -v Warning || exit 1
[ -f "$DC/tokenizer.json" ] || cp data/corpus/tokenizer.json "$DC/"
[ -f "$DC/tok/English.bin" ] || $PY -m relevance_explainer.tiny.corpus tokenize --root "$DC" || exit 1
# 2. continued pretraining, domain mix and general-only control
PRE="--root data/corpus --tokens 6e8 --resume runs/pre-wide-full2/model.pt --lr 3e-4 --micro 64 --eval-every 1000"
[ -f runs/pre-dapt/model.pt ] || $PY -m relevance_explainer.tiny.pretrain $PRE --out runs/pre-dapt --domain-root "$DC" --domain-share 0.3 > runs/pre-dapt.log 2>&1 || exit 1
[ -f runs/pre-cont/model.pt ] || $PY -m relevance_explainer.tiny.pretrain $PRE --out runs/pre-cont > runs/pre-cont.log 2>&1 || exit 1
grep "heldout ppl" runs/pre-dapt.log | tail -1
grep "heldout ppl" runs/pre-cont.log | tail -1
echo "pretrained $(date -u +%H:%M)"
# 3. fine-tune each base on round G's data (2 epochs) and evaluate
DN=data/v2_g_itemfirst_nosim
evals() {
  RUN_DIR="$1" SUF=_v4 scripts/eval_sets.sh
  [ -f "$1/tables_generations_int8.jsonl" ] || $PY -m relevance_explainer.tiny.evaluate --run "$1" --data data/eval_tables/itemfirst_nosim \
    --precisions int8 --tag tables --script-guard 2>&1 | grep -E "Error|int8 " | tail -2
  echo "== tables $1"; $PY -m relevance_explainer.synth.score_real --data data/eval_tables/itemfirst_nosim --run "$1" --tag tables | grep -E "teacher|unanimous"
}
for pair in "dapt runs/pre-dapt/model.pt" "base runs/pre-wide-full2/model.pt" "cont runs/pre-cont/model.pt"; do
  set -- $pair
  RUN=runs/v2g-from-$1
  [ -f "$RUN/model_fp32.pt" ] || $PY -m relevance_explainer.tiny.train --data "$DN" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
    --init "$2" --lr 1e-3 --epochs 2 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc --test-limit 3000 \
    > "$RUN.train.log" 2>&1 || { echo "train $1 failed"; continue; }
  echo "trained $1 $(date -u +%H:%M)"
  evals "$RUN"
done
echo "DAPT-DONE $(date -u +%H:%M)"
