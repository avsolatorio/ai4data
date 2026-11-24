#!/usr/bin/env bash
# Large models asked directly on the report set of one task.  baselines.sh isco|wb_topics
# Qwen3.8-27B answers from the local server; Gemini 3.8 Flash and Claude Sonnet 5.5 answer through OpenRouter
# with the key in OPENROUTER_API_KEY. A model with a .done file is not asked again. The key is never printed.
HERE="$(cd "$(dirname "$0")" && pwd)"
KEY="${OPENROUTER_API_KEY:-}"
source "$HERE/env.sh"
NAME="${1:?usage: baselines.sh isco|wb_topics}"
TASK="$REPO/src/ai4data/small_ai/tasks/$NAME" REF="$TASK/data/report.jsonl" O="$RUNS/baseline_local/$NAME"; mkdir -p "$O"
ROWS=("qwen3.8-27b|http://localhost:8100/v1|local|qwen3.8-27b|{\"chat_template_kwargs\": {\"enable_thinking\": false}}|0|32"
      "gemini-3.8-flash|https://openrouter.ai/api/v1|$KEY|google/gemini-3.8-flash|{\"provider\": {\"require_parameters\": true}}|0|8"
      "claude-sonnet-5.5|https://openrouter.ai/api/v1|$KEY|anthropic/claude-sonnet-5.5|{\"provider\": {\"require_parameters\": true}}|none|8")
for r in "${ROWS[@]}"; do
  IFS='|' read -r name url key model extra temp workers <<< "$r"
  [ -f "$O/$name.done" ] || { SYNTH_BASE_URL=$url SYNTH_API_KEY=$key SYNTH_MODEL=$model SYNTH_EXTRA_BODY="$extra" BASELINE_TEMPERATURE=$temp \
    H baseline "$TASK" "$REF" "$O/$name.jsonl" --workers "$workers" > "$O/$name.log" 2>&1 && touch "$O/$name.done"; }
  H evaluate "$TASK" "$REF" "$O/$name.jsonl" > "$O/${name}_eval.txt" 2>&1
  echo "== $name $(date -u +%T) $(grep -m1 -E '^accuracy|micro' "$O/${name}_eval.txt")"
done
