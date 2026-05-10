#!/usr/bin/env bash
set -euo pipefail

SERVER=${SERVER:-http://localhost:8000/v1}
METRICS_URL=${METRICS_URL:-http://localhost:8000/metrics}
MODEL=${MODEL:-qwen-coder-7b}
TOKENIZER_PATH=${TOKENIZER_PATH:-/data2/model_zoo/Qwen2.5-Coder-7B-Instruct}
OUT_DIR=${OUT_DIR:-data/profiles}
RUN_ID=${RUN_ID:-pr2_v2}
REPEATS=${REPEATS:-3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
INCLUDE_PREFIX_BASELINE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --metrics-url) METRICS_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --include-prefix-baseline) INCLUDE_PREFIX_BASELINE=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

if [ ! -e "$TOKENIZER_PATH" ]; then
  ALT_LOCAL=${TOKENIZER_PATH_ALT_LOCAL:-/data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct}
  if [ -e "$ALT_LOCAL" ]; then
    TOKENIZER_PATH="$ALT_LOCAL"
  else
    TOKENIZER_PATH=${TOKENIZER_PATH_FALLBACK:-Qwen/Qwen2.5-Coder-7B-Instruct}
  fi
fi

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

"${PYTHON[@]}" -m agentweaver.profiling.profile_vllm \
  --server "$SERVER" \
  --metrics-url "$METRICS_URL" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --mode length \
  --out-dir "$OUT_DIR" \
  --run-id "$RUN_ID" \
  --repeats "$REPEATS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --stream

"${PYTHON[@]}" -m agentweaver.profiling.profile_vllm \
  --server "$SERVER" \
  --metrics-url "$METRICS_URL" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --mode concurrency \
  --out-dir "$OUT_DIR" \
  --run-id "$RUN_ID" \
  --repeats "$REPEATS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --stream

if [ "$INCLUDE_PREFIX_BASELINE" = "1" ]; then
  "${PYTHON[@]}" -m agentweaver.profiling.profile_vllm \
    --server "$SERVER" \
    --metrics-url "$METRICS_URL" \
    --model "$MODEL" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --mode prefix \
    --out-dir "$OUT_DIR" \
    --run-id "${RUN_ID}_noprefix" \
    --output-file "h100_profile_prefix_noprefix_${RUN_ID}.csv" \
    --repeats "$REPEATS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --stream
fi
