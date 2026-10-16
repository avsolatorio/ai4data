#!/usr/bin/env bash
# Start the three model servers used for data generation and judging, detached, one after another:
# teacher on GPU 0 (port 8000), teacher on GPU 1 (port 8001), then Gemma 4 on GPU 0 (port 8002) once GPU 0's teacher is up.
# Usage: scripts/serve_all_llms.sh <log_dir>
cd "$(dirname "$0")/.."
. scripts/env.sh
LOG=${1:-$RELEXP_LOGS}
up() { curl -s -o /dev/null --max-time 5 -w "%{http_code}" "localhost:$1/v1/models" | grep -q 200; }
wait_up() { for _ in $(seq 1 120); do up "$1" && return 0; sleep 10; done; echo "port $1 did not come up"; return 1; }
up 8000 || GPU=0 MEM=0.43 setsid nohup scripts/serve_vllm.sh > "$LOG/vllm0.log" 2>&1 < /dev/null &
up 8001 || GPU=1 MEM=0.55 PORT=8001 setsid nohup scripts/serve_vllm.sh > "$LOG/vllm1.log" 2>&1 < /dev/null &
wait_up 8000 && { up 8002 || REPO=RedHatAI/gemma-4-31B-it-FP8-dynamic NAME=gemma-4-31b PORT=8002 GPU=0 MEM=0.5 MAXLEN=3072 SEQS=64 \
  setsid nohup scripts/serve_vllm.sh > "$LOG/gemma.log" 2>&1 < /dev/null & }
wait_up 8001; wait_up 8002
for p in 8000 8001 8002; do echo "port $p: $(up $p && echo up || echo down)"; done
