#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data2/model_zoo/Qwen2.5-Coder-7B-Instruct}
if [ ! -e "$MODEL_PATH" ]; then
  ALT_LOCAL=${MODEL_PATH_ALT_LOCAL:-/data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct}
  if [ -e "$ALT_LOCAL" ]; then
    MODEL_PATH="$ALT_LOCAL"
  else
    MODEL_PATH=${MODEL_PATH_FALLBACK:-Qwen/Qwen2.5-Coder-7B-Instruct}
  fi
fi
PORT=${PORT:-8001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen-coder-7b}
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-0}
GDN_PREFILL_BACKEND=${GDN_PREFILL_BACKEND:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}
export CUDA_VISIBLE_DEVICES
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

if command -v vllm >/dev/null 2>&1; then
  VLLM_CMD=(vllm)
elif command -v uv >/dev/null 2>&1; then
  VLLM_CMD=(uv run vllm)
else
  echo "Neither vllm nor uv is available on PATH." >&2
  exit 127
fi

CMD=("${VLLM_CMD[@]}" serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --enable-prefix-caching)

if [ "$LANGUAGE_MODEL_ONLY" = "1" ] || [ "$LANGUAGE_MODEL_ONLY" = "true" ]; then
  CMD+=(--language-model-only)
fi
if [ -n "$GDN_PREFILL_BACKEND" ]; then
  CMD+=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
fi
if [ "$ENFORCE_EAGER" = "1" ] || [ "$ENFORCE_EAGER" = "true" ]; then
  CMD+=(--enforce-eager)
fi
if [ -n "$EXTRA_VLLM_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $EXTRA_VLLM_ARGS )
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Launching vLLM with prefix caching:"
printf ' %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
