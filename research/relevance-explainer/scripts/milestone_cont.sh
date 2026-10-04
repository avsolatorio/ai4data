#!/usr/bin/env bash
# Milestone by continued training: at <min_chunks> chunks, assemble data, continue training <init> on the item-first
# variant, score and judge on the realistic test set, then on-policy DPO.
# Usage: scripts/milestone_cont.sh <name> <min_chunks> [init=runs/f6-dpo] [old_share=0.5]
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?name}; MINC=${2:?min chunks}; INIT=${3:-runs/f6-dpo}; SHARE=${4:-0.5}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$HOME/venvs/relexp/bin/python"
until [ "$(ls data/real/c*/examples.jsonl 2>/dev/null | wc -l)" -ge "$MINC" ]; do sleep 300; done
echo "milestone $NAME: $(ls data/real/c*/examples.jsonl | wc -l) chunks, $(ls data/real/t*/examples.jsonl 2>/dev/null | wc -l) translation sets $(date -u +%H:%M)"
[ -f "data/v2_${NAME}_itemfirst/train.jsonl" ] || $PY -m relevance_explainer.synth.assemble_v2 --out "data/v2_$NAME" --old-share "$SHARE" || exit 1
RUN="runs/v2$NAME-cont-itemfirst"
[ -f "$RUN/model_fp32.pt" ] || scripts/continue_train.sh "$INIT" "data/v2_${NAME}_itemfirst" "$RUN" itemfirst 3e-4 3
scripts/dpo_run.sh "$RUN" "data/v2_${NAME}_itemfirst" itemfirst
echo "MILESTONE-CONT-$NAME-DONE"
