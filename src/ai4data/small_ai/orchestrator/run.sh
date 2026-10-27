#!/usr/bin/env bash
# Run the orchestrator on one run folder of the feedback loop.
#
#   HARNESS_PYTHON   Python with ai4data.small_ai importable (set PYTHONPATH if it is not installed)
#   OPENROUTER_API_KEY or another provider key that pi reads
#   ORCH_PROVIDER    pi provider name (default: openrouter)
#   ORCH_MODEL       model id for the orchestrator (default: openai/gpt-oss-120b)
#   ORCH_ATTEMPTS    how many times to start or resume the session (default: 6)
#   ORCH_BASE_URL    optional: an OpenAI-compatible server for the orchestrator, for example a local
#                    vLLM server; pi then uses a provider "local" with the model ORCH_MODEL, described in
#                    RUN_FOLDER/pi-agent/models.json (ORCH_CONTEXT sets its context window, default 32768)
#
#   src/ai4data/small_ai/orchestrator/run.sh RUN_FOLDER
#
# pi starts with its built-in tools switched off and only the harness tools allowed, without MCP
# servers, skills, or context files. The event stream is appended to RUN_FOLDER/orchestrator.jsonl and
# the session is kept in RUN_FOLDER/pi-session/. A model can end its turn without calling a tool, for
# example by writing the call into its reasoning; when the session ends while the run is neither
# stopped nor out of rounds, the script resumes the same session and asks the agent to continue.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(cd "${1:?usage: run.sh RUN_FOLDER}" && pwd)"
export HARNESS_RUN="$RUN" PI_TELEMETRY=0
PY="${HARNESS_PYTHON:-python}"
TOOLS=harness_status,gap_report,find_labels,label_examples,label_errors,run_round,stop_run
FIRST="Run the feedback loop for this run folder. Start with harness_status and gap_report, then plan and run each round until you stop the run."
AGAIN="Continue the feedback loop: call harness_status, then plan and run the next round, or call stop_run if the run should end. Call the tools; do not write tool calls as text."
cd "$RUN"
if [ -n "${ORCH_BASE_URL:-}" ]; then
  export PI_CODING_AGENT_DIR="$RUN/pi-agent"
  mkdir -p "$PI_CODING_AGENT_DIR"
  ORCH_MODEL="${ORCH_MODEL:?set ORCH_MODEL}" ORCH_BASE_URL="$ORCH_BASE_URL" "$PY" - "$PI_CODING_AGENT_DIR/models.json" <<PYEOF
import json, os, sys
model = {"id": os.environ["ORCH_MODEL"], "reasoning": True,
         "contextWindow": int(os.environ.get("ORCH_CONTEXT", 32768)), "maxTokens": 8192}
provider = {"baseUrl": os.environ["ORCH_BASE_URL"], "api": "openai-completions", "apiKey": "local",
            "models": [model]}
json.dump({"providers": {"local": provider}}, open(sys.argv[1], "w"), indent=2)
PYEOF
  ORCH_PROVIDER=local
fi

remaining() {
  "$PY" -m ai4data.small_ai status "$RUN" | "$PY" -c \
    'import json,sys; s=json.load(sys.stdin); print(0 if s["stopped"] else s["planned_rounds_remaining"])'
}

for attempt in $(seq 1 "${ORCH_ATTEMPTS:-6}"); do
  [ "$(remaining)" -gt 0 ] || break
  if [ "$attempt" -eq 1 ] && [ ! -d "$RUN/pi-session" ]; then resume=(); prompt="$FIRST"; else resume=(--continue); prompt="$AGAIN"; fi
  echo "{\"type\":\"harness_attempt\",\"attempt\":$attempt}" >> "$RUN/orchestrator.jsonl"
  "$HERE/node_modules/.bin/pi" "${resume[@]}" \
    --provider "${ORCH_PROVIDER:-openrouter}" --model "${ORCH_MODEL:-openai/gpt-oss-120b}" \
    --extension "$HERE/extension.ts" --no-builtin-tools --tools "$TOOLS" \
    --no-mcp --no-skills --no-context-files --no-extensions \
    --append-system-prompt "$HERE/instructions.md" \
    --session-dir "$RUN/pi-session" --mode json "$prompt" >> "$RUN/orchestrator.jsonl" || true
done
