#!/usr/bin/env bash
# Large models asked directly on the survey-style test set, then the time of the small model. Resumable.
# Needs OPENROUTER_API_KEY in the environment and Qwen3.8-27B served at localhost:8100.
# PY is the Python of the harness environment (default: python).
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
until curl -s --max-time 5 http://localhost:8100/v1/models | grep -q qwen3.8-27b; do sleep 20; done
ROWS=("qwen3.8-27b|http://localhost:8100/v1|local|qwen3.8-27b|{\"chat_template_kwargs\": {\"enable_thinking\": false}}|0|32"
      "gemini-3.8-flash|https://openrouter.ai/api/v1|$OPENROUTER_API_KEY|google/gemini-3.8-flash|{\"provider\": {\"require_parameters\": true}}|0|8"
      "claude-sonnet-5.5|https://openrouter.ai/api/v1|$OPENROUTER_API_KEY|anthropic/claude-sonnet-5.5|{\"provider\": {\"require_parameters\": true}}|none|8")
for r in "${ROWS[@]}"; do
  IFS='|' read -r name url key model extra temp workers <<< "$r"
  SYNTH_BASE_URL=$url SYNTH_API_KEY=$key SYNTH_MODEL=$model SYNTH_EXTRA_BODY="$extra" BASELINE_TEMPERATURE=$temp WORKERS=$workers \
    $PY direct.py $name 2>&1 | sed 's/sk-[A-Za-z0-9_-]*/<hidden>/g'
done
CUDA_VISIBLE_DEVICES="" $PY speed.py cpu 4
CUDA_VISIBLE_DEVICES=1 $PY speed.py cuda
echo DONE
