#!/usr/bin/env bash
# Serve one local model for the harness with vLLM.  serve_local.sh qwen|gemma
#   VLLM             the vllm command (default: $HOME/venvs/vllm/bin/vllm)
#   HF_HUB_CACHE     the Hugging Face cache with the model snapshots (default: $HOME/.cache/huggingface/hub)
#   SMALL_AI_RUNS    the log is written to $SMALL_AI_RUNS/logs/serve_<name>.log
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VLLM="${VLLM:-$HOME/venvs/vllm/bin/vllm}"
HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
LOGS="${SMALL_AI_RUNS:-$HERE/../../runs}/logs"; mkdir -p "$LOGS"
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$(dirname "$VLLM"):$PATH"
case "${1:?usage: serve_local.sh qwen|gemma}" in
  qwen)  export CUDA_VISIBLE_DEVICES=0
         M=$(ls -d "$HUB"/models--Qwen--Qwen3.8-27B-FP8/snapshots/* | head -1)
         ARGS=(--served-model-name qwen3.8-27b --port 8100 --max-model-len 131072 --gpu-memory-utilization 0.78
               --max-num-seqs 256 --enable-prefix-caching --language-model-only
               --structured-outputs-config.backend xgrammar --structured-outputs-config.disable_any_whitespace true
               --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3) ;;
  gemma) export CUDA_VISIBLE_DEVICES=1
         M=$(ls -d "$HUB"/models--RedHatAI--gemma-4-31B-it-FP8-dynamic/snapshots/* | head -1)
         ARGS=(--served-model-name gemma-4-31b --port 8102 --max-model-len 16384 --gpu-memory-utilization 0.90
               --max-num-seqs 256 --enable-prefix-caching --language-model-only --kv-cache-dtype fp8 --attention-backend TRITON_ATTN
               --structured-outputs-config.backend xgrammar --structured-outputs-config.disable_any_whitespace true) ;;
  *) echo "usage: serve_local.sh qwen|gemma" >&2; exit 2 ;;
esac
exec "$VLLM" serve "$M" "${ARGS[@]}" >> "$LOGS/serve_$1.log" 2>&1
