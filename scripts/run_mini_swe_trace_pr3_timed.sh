#!/usr/bin/env bash
set -euo pipefail

SERVER=http://localhost:8010/v1
BACKEND_SERVER=http://localhost:8001/v1
MODEL=qwen-coder-7b
TOKENIZER_PATH=/data2/model_zoo/Qwen2.5-Coder-7B-Instruct
INSTANCE_LIST=""
NUM_INSTANCES=5
ROLLOUTS=1
MAX_STEPS=10
RUN_ID=mini_swe_lite5_timed
OUT_DIR=data/traces/mini_swe_lite5_timed
TRAJ_DIR=""
RUN_REAL=0
RESULTS_DIR=data/results
PROCESSED_DIR=""
RAW_ROOT=""
TIMING_SIDECAR=""
RUN_ID_EXPLICIT=0
PROXY_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --backend-server|--backend) BACKEND_SERVER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --instance-list) INSTANCE_LIST="$2"; shift 2 ;;
    --num-instances|--instances) NUM_INSTANCES="$2"; shift 2 ;;
    --rollouts) ROLLOUTS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; OUT_DIR="data/traces/$2"; RUN_ID_EXPLICIT=1; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --traj-dir) TRAJ_DIR="$2"; shift 2 ;;
    --raw-root) RAW_ROOT="$2"; shift 2 ;;
    --timing-sidecar) TIMING_SIDECAR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --processed-dir) PROCESSED_DIR="$2"; shift 2 ;;
    --run-real) RUN_REAL=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ "$RUN_ID_EXPLICIT" = "0" ] && [ "$OUT_DIR" != "data/traces/mini_swe_lite5_timed" ]; then
  RUN_ID=$(basename "$OUT_DIR")
fi
PROCESSED_DIR=${PROCESSED_DIR:-data/processed/$RUN_ID}
RAW_ROOT=${RAW_ROOT:-data/raw_trajs/$RUN_ID}
TIMING_SIDECAR=${TIMING_SIDECAR:-data/logs/vllm_proxy_timing_${RUN_ID}.jsonl}

if [ "$ROLLOUTS" -ne 1 ]; then
  echo "Timed Lite-5 mode requires --rollouts 1; use run_mini_swe_multibranch_pr3_timed.sh for branch fanout" >&2
  exit 2
fi

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

cleanup() {
  if [ -n "$PROXY_PID" ]; then
    kill "$PROXY_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

proxy_host_port() {
  "${PYTHON[@]}" - "$SERVER" <<'PY'
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1])
print(u.hostname or "127.0.0.1")
print(u.port or (443 if u.scheme == "https" else 80))
PY
}

ensure_proxy() {
  if curl -fsS "$SERVER/models" >/dev/null 2>&1; then
    echo "Using existing OpenAI-compatible server at $SERVER"
    return
  fi
  if ! curl -fsS "$BACKEND_SERVER/models" >/dev/null 2>&1; then
    echo "backend vLLM server is not reachable at $BACKEND_SERVER" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$TIMING_SIDECAR")"
  : > "$TIMING_SIDECAR"
  mapfile -t hp < <(proxy_host_port)
  echo "Starting vLLM timing proxy at $SERVER -> $BACKEND_SERVER"
  bash scripts/launch_vllm_timing_proxy_pr3.sh \
    --backend-server "$BACKEND_SERVER" \
    --host "${hp[0]}" \
    --port "${hp[1]}" \
    --run-id "$RUN_ID" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --log-path "$TIMING_SIDECAR" &
  PROXY_PID=$!
  for _ in $(seq 1 60); do
    if curl -fsS "$SERVER/models" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "timing proxy did not become reachable at $SERVER" >&2
  exit 1
}

convert_and_replay() {
  local src_dir=$1
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR" "$RESULTS_DIR"
  local found=0
  while IFS= read -r traj; do
    found=1
    local rel parent instance base rollout_num branch_id out_name
    rel=${traj#"$src_dir"/}
    parent=$(dirname "$rel")
    instance=$(printf '%s\n' "$parent" | cut -d/ -f1)
    base=$(basename "$traj")
    rollout_num=$(printf '%s\n' "$base" | sed -n 's/.*rollout_\([0-9][0-9]*\).*/\1/p')
    rollout_num=${rollout_num:-0}
    branch_id="b${rollout_num}"
    if [ "$parent" != "." ] && [ -n "$instance" ]; then
      out_name="${instance}_${branch_id}.jsonl"
      "${PYTHON[@]}" -m agentweaver.tracing.mini_swe_trace_adapter \
        --traj "$traj" \
        --model "$TOKENIZER_PATH" \
        --instance-id "$instance" \
        --branch-id "$branch_id" \
        --parent-branch-id root \
        --rollout-id "rollout_${rollout_num}" \
        --timing-sidecar "$TIMING_SIDECAR" \
        --out "$OUT_DIR/$out_name"
    else
      local name
      name=${base%.json}
      name=${name%.traj}
      "${PYTHON[@]}" -m agentweaver.tracing.mini_swe_trace_adapter \
        --traj "$traj" \
        --model "$TOKENIZER_PATH" \
        --branch-id b0 \
        --parent-branch-id root \
        --rollout-id rollout_0 \
        --timing-sidecar "$TIMING_SIDECAR" \
        --out "$OUT_DIR/${name}.jsonl"
    fi
  done < <(find "$src_dir" -type f \( -name '*.traj' -o -name '*.traj.json' -o -name '*.json' \) ! -path '*/_mini_extra_*/*' | sort)
  if [ "$found" = "0" ]; then
    echo "no mini-SWE-agent trajectory files found under $src_dir" >&2
    exit 1
  fi

  "${PYTHON[@]}" -m agentweaver.tracing.trace_validate \
    --trace-dir "$OUT_DIR" \
    --out "$RESULTS_DIR/${RUN_ID}_trace_validation.csv"
  "${PYTHON[@]}" -m agentweaver.tracing.pr3_pipeline summarize \
    --trace-dir "$OUT_DIR" \
    --run-id "$RUN_ID" \
    --mode lite5 \
    --results-dir "$RESULTS_DIR"
  "${PYTHON[@]}" -m agentweaver.tracing.pr3_pipeline replay \
    --trace-dir "$OUT_DIR" \
    --processed-dir "$PROCESSED_DIR" \
    --run-id "$RUN_ID" \
    --results-dir "$RESULTS_DIR"
  "${PYTHON[@]}" -m agentweaver.workloads.extract_swebench_predictions \
    --traj-root "$src_dir" \
    --out "$RESULTS_DIR/${RUN_ID}_predictions.jsonl" \
    --model-name-or-path "$MODEL"
  "${PYTHON[@]}" -m agentweaver.tracing.pr3_v2_report \
    --results-dir "$RESULTS_DIR" \
    --out "$RESULTS_DIR/pr3_v2_report.md" || true
  echo "converted, validated, summarized, and replayed timed trajectories under $OUT_DIR"
}

if [ -n "$TRAJ_DIR" ]; then
  convert_and_replay "$TRAJ_DIR"
  exit 0
fi

if [ "$RUN_REAL" != "1" ]; then
  cat >&2 <<EOF
No --traj-dir was provided.

Adapter-only mode:
  bash scripts/run_mini_swe_trace_pr3_timed.sh --traj-dir <real_traj_dir> --run-id $RUN_ID --server $SERVER --timing-sidecar $TIMING_SIDECAR

Runner mode:
  bash scripts/run_mini_swe_trace_pr3_timed.sh --run-real --instance-list data/results/mini_swe_lite5_instances.txt --num-instances $NUM_INSTANCES --max-steps $MAX_STEPS --run-id $RUN_ID --server $SERVER --backend-server $BACKEND_SERVER

This script does not generate fake traces.
EOF
  exit 1
fi

if [ -z "$INSTANCE_LIST" ]; then
  echo "--run-real requires --instance-list" >&2
  exit 2
fi

ensure_proxy

"${PYTHON[@]}" -m agentweaver.tracing.miniswe_runner \
  --instance-list "$INSTANCE_LIST" \
  --num-instances "$NUM_INSTANCES" \
  --rollouts 1 \
  --max-steps "$MAX_STEPS" \
  --run-id "$RUN_ID" \
  --raw-root "$RAW_ROOT" \
  --server "$SERVER" \
  --model "$MODEL" \
  --results-dir "$RESULTS_DIR" \
  --timed

convert_and_replay "$RAW_ROOT"
