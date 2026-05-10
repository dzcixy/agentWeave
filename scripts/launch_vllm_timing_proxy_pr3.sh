#!/usr/bin/env bash
set -euo pipefail

BACKEND_SERVER=${BACKEND_SERVER:-http://localhost:8001/v1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8010}
RUN_ID=${RUN_ID:-pr3_v2_proxy}
TOKENIZER_PATH=${TOKENIZER_PATH:-/data2/model_zoo/Qwen2.5-Coder-7B-Instruct}
LOG_PATH=${LOG_PATH:-data/logs/vllm_proxy_timing_${RUN_ID}.jsonl}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-server|--backend) BACKEND_SERVER="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; LOG_PATH="data/logs/vllm_proxy_timing_${2}.jsonl"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --log-path) LOG_PATH="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_PATH")"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

exec uv run python -m agentweaver.profiling.vllm_timing_proxy \
  --host "$HOST" \
  --port "$PORT" \
  --backend "$BACKEND_SERVER" \
  --log-path "$LOG_PATH" \
  --tokenizer-path "$TOKENIZER_PATH"
