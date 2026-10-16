#!/usr/bin/env bash
# Fit the live faithfulness scorer for a format-C run on its own samples: sample (GPU 0, teachers off), verifier and
# teacher judge (scripts/teachers.sh up/down around it), fit (scripts/faith_scorer.py, GPU 1), then evaluate on the
# run's judged realistic-test answers against the teacher and Gemma 4. A seed other than the DPO round's (0) draws
# inputs that run did not train on. Steps skip when their outputs exist.
#   scripts/faith_refit.sh runs/final-C-dpo [n inputs=8000] [seed=1]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?run}; N=${2:-8000}; SEED=${3:-1}; D=data/final/C
export PYTHONPATH=src TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1
PY="$HOME/venvs/relexp/bin/python"
TAG=$(basename "$RUN")_s$SEED
S="faith_${TAG}_samples.jsonl"; SC="faith_${TAG}_scored.jsonl"
[ -f "$D/$S" ] || CUDA_VISIBLE_DEVICES=0 $PY -m relevance_explainer.distill.onpolicy_c sample --run "$RUN" --data "$D" \
  --n "$N" --k 8 --seed "$SEED" --out "$S" || exit 1
if [ ! -f "$D/$SC" ]; then
  scripts/teachers.sh up || exit 1
  $PY -m relevance_explainer.distill.onpolicy_c score --data "$D" --samples "$S" --out "$SC" || { scripts/teachers.sh down; exit 1; }
  scripts/teachers.sh down
fi
CUDA_VISIBLE_DEVICES=1 $PY scripts/faith_scorer.py "$RUN" "$D" "$SC" --batch 32 2>&1 | grep -v -i -E "warn|torch.load|^  [0-9]"
for SUF in "" _gemma4; do
  echo "== realistic test, judge${SUF:- teacher}"
  CUDA_VISIBLE_DEVICES=1 $PY scripts/faith_scorer_eval.py "$RUN" --suffix "$SUF" 2>&1 | grep -v -i -E "warn|torch.load"
done
echo "FAITH-REFIT-DONE $RUN $(date -u +%H:%M)"
