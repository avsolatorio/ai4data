# Shared settings of the feedback-loop runs. Source this file; it prints nothing and holds no key.
#   PY               Python with the harness dependencies (default: python)
#   SMALL_AI_RUNS    folder of the run folders and logs (default: runs/ in the repository)
#   The writer, labeller and orchestrator are local vLLM servers started with serve_local.sh:
#   Qwen3.8-27B on port 8100 (writer and orchestrator) and Gemma 4 31B on port 8102 (labeller).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="$(cd "$HERE/../.." && pwd)"
export PY="${PY:-python}"
export HARNESS_PYTHON="${HARNESS_PYTHON:-$PY}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" PI_TELEMETRY=0
export RUNS="${SMALL_AI_RUNS:-$REPO/runs}"
NOTHINK='{"chat_template_kwargs": {"enable_thinking": false}}'
export SYNTH_BASE_URL=http://localhost:8100/v1 SYNTH_API_KEY=local SYNTH_MODEL=qwen3.8-27b SYNTH_EXTRA_BODY="$NOTHINK"
export SYNTH_LABEL_BASE_URL=http://localhost:8102/v1 SYNTH_LABEL_MODEL=gemma-4-31b SYNTH_LABEL_EXTRA_BODY="$NOTHINK"
export SYNTH_WORKERS="${SYNTH_WORKERS:-128}"
export ORCH_BASE_URL=http://localhost:8100/v1 ORCH_MODEL=qwen3.8-27b ORCH_CONTEXT=131072
H() { "$PY" -m ai4data.small_ai "$@"; }
