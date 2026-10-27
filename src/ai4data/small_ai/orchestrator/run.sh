#!/usr/bin/env bash
# Run the orchestrator on one run folder of the feedback loop.
#
#   HARNESS_PYTHON   Python with ai4data.small_ai importable (set PYTHONPATH if it is not installed)
#   OPENROUTER_API_KEY or another provider key that pi reads
#   ORCH_PROVIDER    pi provider name (default: openrouter)
#   ORCH_MODEL       model id for the orchestrator (default: openai/gpt-oss-120b)
#
#   src/ai4data/small_ai/orchestrator/run.sh RUN_FOLDER
#
# pi starts with its built-in tools switched off and only the harness tools allowed, without MCP
# servers, skills, or context files. The full event stream goes to RUN_FOLDER/orchestrator.jsonl and the
# session to RUN_FOLDER/pi-session/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(cd "${1:?usage: run.sh RUN_FOLDER}" && pwd)"
export HARNESS_RUN="$RUN" PI_TELEMETRY=0
TOOLS=harness_status,gap_report,find_labels,label_examples,label_errors,run_round,stop_run
cd "$RUN"
"$HERE/node_modules/.bin/pi" \
  --provider "${ORCH_PROVIDER:-openrouter}" --model "${ORCH_MODEL:-openai/gpt-oss-120b}" \
  --extension "$HERE/extension.ts" --no-builtin-tools --tools "$TOOLS" \
  --no-mcp --no-skills --no-context-files --no-extensions \
  --append-system-prompt "$HERE/instructions.md" \
  --session-dir "$RUN/pi-session" --mode json \
  "Run the feedback loop for this run folder. Start with harness_status and gap_report, then plan and run each round until you stop the run." \
  > "$RUN/orchestrator.jsonl"
