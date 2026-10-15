#!/usr/bin/env bash
# Re-grade 600 realistic-test rows with the teacher (scripts/teacher_grade_repeat.py) once the teachers are free of
# the remaining-rows run and the arms' judge (scripts/final_pipeline.sh step 2).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
LOGD=/data/wb536061/ai4data-relevance-explainer
until [ -f data/fmt/teacher_rest.done ] && grep -q JUDGE-DECODE-DONE "$LOGD/judge_decode_arms.log" 2>/dev/null; do sleep 120; done
[ -f data/fmt_s/teacher_grade_repeat.jsonl ] || "$HOME/venvs/relexp/bin/python" scripts/teacher_grade_repeat.py --n 600
echo "GRADE-REPEAT-DONE $(date -u +%H:%M)"
