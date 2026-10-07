#!/usr/bin/env bash
# Keep the model servers, the web app and a long job running: every minute, start whatever is down.
# Meant to run in a tmux session that outlives the shell that started it:
#   tmux new -d -s relexp 'scripts/keepalive.sh scripts/round_e.sh ROUND-E-DONE'
# Arguments: an optional job script and the marker it prints when finished (the job is restarted until the marker appears;
# the round scripts skip completed steps).
cd "$(dirname "$0")/.."
JOB=${1:-}; DONE=${2:-}
LOGD=/data/wb536061/ai4data-relevance-explainer
up() { curl -s -o /dev/null --max-time 5 -w "%{http_code}" "localhost:$1/$2" | grep -q 200; }
running() { pgrep -u "$(id -u)" -f "$1" > /dev/null; }
while true; do
  running "vllm serve.*--port 8000" || up 8000 v1/models || { echo "$(date -u +%T) start teacher GPU 0"; GPU=0 MEM=0.43 setsid nohup scripts/serve_vllm.sh >> "$LOGD/vllm0.log" 2>&1 < /dev/null & sleep 240; }
  running "vllm serve.*--port 8001" || up 8001 v1/models || { echo "$(date -u +%T) start teacher GPU 1"; GPU=1 MEM=0.55 PORT=8001 setsid nohup scripts/serve_vllm.sh >> "$LOGD/vllm1.log" 2>&1 < /dev/null & sleep 60; }
  if up 8000 v1/models && ! running "vllm serve.*--port 8002" && ! up 8002 v1/models; then
    echo "$(date -u +%T) start Gemma 4"
    REPO=RedHatAI/gemma-4-31B-it-FP8-dynamic NAME=gemma-4-31b PORT=8002 GPU=0 MEM=0.5 MAXLEN=3072 SEQS=64 \
      setsid nohup scripts/serve_vllm.sh >> "$LOGD/gemma.log" 2>&1 < /dev/null &
    sleep 240
  fi
  running "uvicorn relevance_explainer.webapp.server" || up 3410 healthz || { echo "$(date -u +%T) start app"; setsid nohup scripts/serve_app.sh >> "$LOGD/app_3410.log" 2>&1 < /dev/null & }
  if [ -n "$JOB" ] && ! grep -q "$DONE" "$LOGD/$(basename "$JOB" .sh).log" 2>/dev/null && ! running "$(basename "$JOB")"; then
    if up 8000 v1/models && up 8001 v1/models && up 8002 v1/models; then
      echo "$(date -u +%T) start $JOB"; setsid nohup "$JOB" >> "$LOGD/$(basename "$JOB" .sh).log" 2>&1 < /dev/null &
    fi
  fi
  sleep 60
done
