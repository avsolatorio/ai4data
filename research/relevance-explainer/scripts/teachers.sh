#!/usr/bin/env bash
# Start the teacher servers only for the steps that need them, through the keepalive's pause files.
#   scripts/teachers.sh up      remove pause_gpu0 and pause_gpu1, wait until ports 8000 and 8001 answer (keepalive starts them)
#   scripts/teachers.sh down    create both pause files, stop this user's vLLM servers on ports 8000 and 8001
#   scripts/teachers.sh status
# pause_gemma is left as it is. Up waits at most 30 minutes and exits 1 if a server has not answered by then.
set -uo pipefail
LOGD=/data/wb536061/ai4data-relevance-explainer
up() { curl -s -o /dev/null --max-time 5 -w "%{http_code}" "localhost:$1/v1/models" | grep -q 200; }
case "${1:?up, down or status}" in
  up)
    rm -f "$LOGD/pause_gpu0" "$LOGD/pause_gpu1"
    for _ in $(seq 1 180); do
      up 8000 && up 8001 && { echo "$(date -u +%T) teachers up"; exit 0; }
      sleep 10
    done
    echo "$(date -u +%T) teachers not up after 30 minutes" >&2; exit 1 ;;
  down)
    touch "$LOGD/pause_gpu0" "$LOGD/pause_gpu1"
    for port in 8000 8001; do
      pkill -u "$(id -u)" -f "vllm serve.*--port $port( |$)"
    done
    for _ in $(seq 1 30); do
      pgrep -u "$(id -u)" -f "vllm serve.*--port 800[01]( |$)" > /dev/null || { echo "$(date -u +%T) teachers down"; exit 0; }
      sleep 5
    done
    echo "$(date -u +%T) teachers still running after 150 s" >&2; exit 1 ;;
  status)
    for port in 8000 8001; do up "$port" && echo "$port up" || echo "$port down"; done
    ls "$LOGD" | grep pause ;;
esac
