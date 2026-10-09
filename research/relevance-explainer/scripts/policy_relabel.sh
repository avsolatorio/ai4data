#!/usr/bin/env bash
# Relabel the challenge suite at render version 4 with the verdict rules (NARRATE_V3), to compare the teacher-Gemma 4
# agreement with the version-3 labels. Usage: scripts/policy_relabel.sh [dir=data/challenge]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
OUT=${1:-data/challenge}
[ -f "$OUT/labeled_v4.jsonl" ] || "$HOME/venvs/relexp/bin/python" -m relevance_explainer.synth.challenge label --out "$OUT" \
  --prompt v3 --render-version 4 --labeled labeled_v4.jsonl --workers 48
echo POLICY-RELABEL-DONE
