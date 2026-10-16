#!/usr/bin/env bash
# Label a sample of realistic test pairs twice, with the version-3 setup (NARRATE_V2, render 3) and with the verdict
# rules (NARRATE_V3, render 4), to measure how often the rules change ordinary labels. Usage: scripts/policy_check.sh [dir]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
OUT=${1:-data/policy_check_real}
CH="$RELEXP_PY -m relevance_explainer.synth.challenge"
[ -f "$OUT/labeled.jsonl" ] || $CH label --out "$OUT" --prompt v2 --render-version 3 --labeled labeled.jsonl --workers 24
[ -f "$OUT/labeled_v4.jsonl" ] || $CH label --out "$OUT" --prompt v3 --render-version 4 --labeled labeled_v4.jsonl --workers 24
echo POLICY-CHECK-DONE
