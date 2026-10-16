#!/usr/bin/env bash
# Answer-confidence calibration beyond the realistic test: greedy outputs of the deployed model on the challenge suite,
# the held-out Eurostat tables and the paraphrase set, judged by the teacher, scored by the saved 9-parameter head.
# Every step skips when its output exists.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1
PY="$RELEXP_PY"
RUN=${1:-runs/v2g-from-dapt-8ep-dpo}; O=$RUN/selfcheck
SC="$PY -m relevance_explainer.tiny.selfcheck"
echo "selfcheck ood start $(date -u +%H:%M)"
for pair in "challenge data/eval_challenge/itemfirst_nosim_v4" "tables data/eval_tables/itemfirst_nosim" "para data/eval_para/itemfirst_nosim_v4"; do
  set -- $pair
  NAME=$1; D=$2
  $SC sample --run "$RUN" --data "$D" --split test --n 100000 --k 0 --out "$O/${NAME}_samples.jsonl" || exit 1
  [ "$(cat "$O/${NAME}_scored.jsonl" 2>/dev/null | wc -l)" -ge "$(wc -l < "$O/${NAME}_samples.jsonl")" ] || \
    $SC score --data "$D" --split test --samples "$O/${NAME}_samples.jsonl" --out "$O/${NAME}_scored.jsonl" || exit 1
  $SC features --run "$RUN" --data "$D" --split test --samples "$O/${NAME}_samples.jsonl" \
    --scored "$O/${NAME}_scored.jsonl" --out "$O/${NAME}_features.pt" || exit 1
  echo "done $NAME $(date -u +%H:%M)"
done
$PY scripts/calibration_report.py "$O" test challenge tables para
echo "SELFCHECK-OOD-DONE $(date -u +%H:%M)"
