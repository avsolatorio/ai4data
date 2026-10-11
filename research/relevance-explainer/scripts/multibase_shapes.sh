#!/usr/bin/env bash
# Multilingual base model: which vocabulary and shape fit best in about 20.6M parameters (the 20 MB INT8 budget)?
# Each candidate "name vocab d_model layers heads" is tokenized with its vocabulary (shards per vocabulary share the
# raw text by symlink), pretrained for TOKENS on the 243-language corpus with the plan's language weights, and compared
# on held-out bits per byte, which does not depend on the vocabulary. Every step skips when its output exists.
# Usage: scripts/multibase_shapes.sh [tokens=1e9]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU:-1}
PY="$HOME/venvs/relexp/bin/python"
TOKENS=${1:-1e9}
SRC=data/corpus_multi
echo "shapes start $(date -u +%H:%M)"
CANDIDATES=(
  "v12k-d448-l7 12000 448 7 7"
  "v16k-d448-l6 16000 448 6 7"
  "v24k-d384-l6 24000 384 6 6"
  "v32k-d384-l5 32000 384 5 6"
)
for c in "${CANDIDATES[@]}"; do
  set -- $c
  NAME=$1; V=$2; D=$3; L=$4; H=$5
  R=data/corpus_multi_v$((V / 1000))k
  mkdir -p "$R"
  [ -e "$R/raw" ] || ln -s "$(readlink -f $SRC/raw)" "$R/raw"
  until [ -f "$SRC/tokenizers/bpe$V.json" ]; do sleep 60; done  # trained by scripts/multibase.sh, possibly still running
  [ -f "$R/tokenizer.json" ] || cp "$SRC/tokenizers/bpe$V.json" "$R/tokenizer.json"
  [ -f "$R/plan.json" ] || cp "$SRC/plan.json" "$R/plan.json"
  [ -f "$R/tok/.done" ] || { $PY -m relevance_explainer.tiny.corpus tokenize --root "$R" --procs 96 | tail -1 && touch "$R/tok/.done"; } || exit 1
  OUT=runs/mb-$NAME
  [ -f "$OUT/done" ] || { $PY -m relevance_explainer.tiny.pretrain --root "$R" --all-shards --holdout 200000 --eval-seqs 8 \
      --out "$OUT" --tokens "$TOKENS" --d-model "$D" --layers "$L" --heads "$H" --d-ff 1024 --lr 2e-3 --micro 64 \
      --eval-every 2000 --init-emb Qwen/Qwen3-0.6B > "$OUT.log" 2>&1 && touch "$OUT/done"; } || { echo "pretrain $NAME failed"; continue; }
  echo "$NAME: $(grep params "$OUT.log" | head -1); $(grep 'mean held-out bits per byte' "$OUT.log" | tail -1)"
done
$PY scripts/multibase_compare.py runs/mb-v12k-d448-l7 runs/mb-v16k-d448-l6 runs/mb-v24k-d384-l6 runs/mb-v32k-d384-l5
echo "SHAPES-DONE $(date -u +%H:%M)"
