#!/usr/bin/env bash
set -euo pipefail

SERVER=http://localhost:8001/v1
MODEL=qwen-coder-7b
TOKENIZER_PATH=/data2/model_zoo/Qwen2.5-Coder-7B-Instruct
INSTANCE_LIST=""
NUM_INSTANCES=10
ROLLOUTS=4
MAX_STEPS=10
RUN_ID=mini_swe_lite10_r4
OUT_DIR=data/traces/mini_swe_lite10_r4
TRAJ_ROOT=""
RAW_ROOT=""
RUN_REAL=0
RESULTS_DIR=data/results
PROCESSED_DIR=""
RUN_ID_EXPLICIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --instance-list) INSTANCE_LIST="$2"; shift 2 ;;
    --num-instances|--instances) NUM_INSTANCES="$2"; shift 2 ;;
    --rollouts) ROLLOUTS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; OUT_DIR="data/traces/$2"; RUN_ID_EXPLICIT=1; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --traj-root|--traj-dir) TRAJ_ROOT="$2"; shift 2 ;;
    --raw-root) RAW_ROOT="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --processed-dir) PROCESSED_DIR="$2"; shift 2 ;;
    --run-real) RUN_REAL=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ "$RUN_ID_EXPLICIT" = "0" ] && [ "$OUT_DIR" != "data/traces/mini_swe_lite10_r4" ]; then
  RUN_ID=$(basename "$OUT_DIR")
fi
PROCESSED_DIR=${PROCESSED_DIR:-data/processed/$RUN_ID}
RAW_ROOT=${RAW_ROOT:-data/raw_trajs/$RUN_ID}

if [ "$ROLLOUTS" -lt 2 ]; then
  echo "multi-branch collection requires --rollouts >= 2" >&2
  exit 1
fi

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

convert_and_replay() {
  local src_root=$1
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR" "$RESULTS_DIR"
  local found=0
  declare -A seen_rollouts=()
  while IFS= read -r traj; do
    found=1
    local rel instance base rollout_num branch_id rollout_id out_name
    rel=${traj#"$src_root"/}
    instance=$(dirname "$rel" | cut -d/ -f1)
    base=$(basename "$traj")
    rollout_num=$(printf '%s\n' "$base" | sed -n 's/.*rollout_\([0-9][0-9]*\).*/\1/p')
    rollout_num=${rollout_num:-0}
    branch_id="b${rollout_num}"
    rollout_id="rollout_${rollout_num}"
    seen_rollouts["$instance:$rollout_id"]=1
    out_name="${instance}_${rollout_id}.jsonl"
    "${PYTHON[@]}" -m agentweaver.tracing.mini_swe_trace_adapter \
      --traj "$traj" \
      --model "$TOKENIZER_PATH" \
      --instance-id "$instance" \
      --branch-id "$branch_id" \
      --parent-branch-id root \
      --rollout-id "$rollout_id" \
      --out "$OUT_DIR/$out_name"
  done < <(find "$src_root" -type f \( -name '*.traj' -o -name '*.traj.json' -o -name '*.json' \) ! -path '*/_mini_extra_*/*' | sort)
  if [ "$found" = "0" ]; then
    echo "no mini-SWE-agent trajectory files found under $src_root" >&2
    exit 1
  fi

  local missing_file="$RESULTS_DIR/${RUN_ID}_missing_rollouts.txt"
  : > "$missing_file"
  while IFS= read -r instance_dir; do
    local instance
    instance=$(basename "$instance_dir")
    for ((i=0; i<ROLLOUTS; i++)); do
      local key="$instance:rollout_${i}"
      if [ -z "${seen_rollouts[$key]+x}" ]; then
        echo "$instance missing rollout_${i}" >> "$missing_file"
      fi
    done
  done < <(find "$src_root" -mindepth 1 -maxdepth 1 -type d | sort)

  "${PYTHON[@]}" -m agentweaver.tracing.trace_validate \
    --trace-dir "$OUT_DIR" \
    --out "$RESULTS_DIR/${RUN_ID}_trace_validation.csv"
  "${PYTHON[@]}" -m agentweaver.tracing.pr3_pipeline summarize \
    --trace-dir "$OUT_DIR" \
    --run-id "$RUN_ID" \
    --mode lite10_r4 \
    --results-dir "$RESULTS_DIR"
  "${PYTHON[@]}" -m agentweaver.tracing.pr3_pipeline replay \
    --trace-dir "$OUT_DIR" \
    --processed-dir "$PROCESSED_DIR" \
    --run-id "$RUN_ID" \
    --results-dir "$RESULTS_DIR"
  "${PYTHON[@]}" -m agentweaver.tracing.pr3_pipeline report \
    --results-dir "$RESULTS_DIR" \
    --out "$RESULTS_DIR/pr3_report.md"
  echo "converted, validated, summarized, and replayed multi-branch trajectories under $OUT_DIR"
  if [ -s "$missing_file" ]; then
    echo "missing rollout report: $missing_file" >&2
  fi
}

if [ -n "$TRAJ_ROOT" ]; then
  convert_and_replay "$TRAJ_ROOT"
  exit 0
fi

if [ "$RUN_REAL" != "1" ]; then
  cat >&2 <<EOF
No --traj-root was provided.

Adapter-only mode:
  bash scripts/run_mini_swe_multibranch_pr3.sh --traj-root <traj_root> --run-id $RUN_ID --server $SERVER --tokenizer-path $TOKENIZER_PATH --rollouts $ROLLOUTS

Runner mode:
  bash scripts/run_mini_swe_multibranch_pr3.sh --run-real --instance-list data/results/mini_swe_lite10_instances.txt --num-instances $NUM_INSTANCES --rollouts $ROLLOUTS --max-steps $MAX_STEPS --run-id $RUN_ID --server $SERVER --tokenizer-path $TOKENIZER_PATH

This script does not generate fake traces.
EOF
  exit 1
fi

if [ -z "$INSTANCE_LIST" ]; then
  echo "--run-real requires --instance-list" >&2
  exit 2
fi

if ! curl -fsS "$SERVER/models" >/dev/null 2>&1; then
  echo "local OpenAI-compatible vLLM server is not reachable at $SERVER" >&2
  echo "start vLLM first, then rerun this script" >&2
  exit 1
fi

"${PYTHON[@]}" -m agentweaver.tracing.miniswe_runner \
  --instance-list "$INSTANCE_LIST" \
  --num-instances "$NUM_INSTANCES" \
  --rollouts "$ROLLOUTS" \
  --max-steps "$MAX_STEPS" \
  --run-id "$RUN_ID" \
  --raw-root "$RAW_ROOT" \
  --server "$SERVER" \
  --model "$MODEL" \
  --results-dir "$RESULTS_DIR"

convert_and_replay "$RAW_ROOT"
