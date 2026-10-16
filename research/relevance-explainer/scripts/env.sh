# Shared settings for the shell scripts; sourced from the project folder (research/relevance-explainer) after `cd`.
#
#   RELEXP_PY     Python interpreter of the project environment (default: python3 on the PATH)
#   RELEXP_VLLM   vllm executable of the serving environment (default: vllm on the PATH)
#   RELEXP_LOGS   directory for pipeline logs, the keepalive's pause files and large side files (default: logs)
#
# Values for one machine go in scripts/local.env, which is not committed (see scripts/local.env.example). A value
# already set in the environment takes precedence over local.env.
_relexp_py=${RELEXP_PY-}; _relexp_vllm=${RELEXP_VLLM-}; _relexp_logs=${RELEXP_LOGS-}
[ -f scripts/local.env ] && . scripts/local.env
RELEXP_PY=${_relexp_py:-${RELEXP_PY:-$(command -v python3)}}
RELEXP_VLLM=${_relexp_vllm:-${RELEXP_VLLM:-vllm}}
RELEXP_LOGS=${_relexp_logs:-${RELEXP_LOGS:-logs}}
unset _relexp_py _relexp_vllm _relexp_logs
mkdir -p "$RELEXP_LOGS"
export RELEXP_PY RELEXP_VLLM RELEXP_LOGS
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
