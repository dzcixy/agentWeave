#!/usr/bin/env bash
set -euo pipefail

SERVER=http://localhost:8010/v1
BACKEND_SERVER=http://localhost:8001/v1
MODEL=qwen-coder-7b
TOKENIZER_PATH=/data2/model_zoo/Qwen2.5-Coder-7B-Instruct
INSTANCE_LIST=data/results/mini_swe_lite5_instances.txt
NUM_INSTANCES=5
MAX_STEPS=15
RUN_ID=mini_swe_lite5_patchcap
RESULTS_DIR=data/results
RAW_ROOT=data/raw_trajs/mini_swe_lite5_patchcap
TRACE_DIR=data/traces/mini_swe_lite5_patchcap
MAX_WORKERS=1
RUN_REAL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --backend-server|--backend) BACKEND_SERVER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --instance-list) INSTANCE_LIST="$2"; shift 2 ;;
    --num-instances|--instances) NUM_INSTANCES="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; RAW_ROOT="data/raw_trajs/$2"; TRACE_DIR="data/traces/$2"; shift 2 ;;
    --raw-root) RAW_ROOT="$2"; shift 2 ;;
    --out-dir|--trace-dir) TRACE_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --run-real) RUN_REAL=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

mkdir -p "$RESULTS_DIR"

if [ "$RUN_REAL" != "1" ]; then
  echo "patch capture requires --run-real trajectories; this script does not generate fake traces" >&2
  exit 2
fi

AGENTWEAVER_CAPTURE_PATCH=1 bash scripts/run_mini_swe_trace_pr3_timed.sh \
  --server "$SERVER" \
  --backend-server "$BACKEND_SERVER" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --instance-list "$INSTANCE_LIST" \
  --num-instances "$NUM_INSTANCES" \
  --max-steps "$MAX_STEPS" \
  --run-id "$RUN_ID" \
  --raw-root "$RAW_ROOT" \
  --out-dir "$TRACE_DIR" \
  --results-dir "$RESULTS_DIR" \
  --patch-capture \
  --run-real

"${PYTHON[@]}" -m agentweaver.workloads.patch_capture_debug \
  --traj-root "$RAW_ROOT" \
  --out "$RESULTS_DIR/patch_capture_debug_pr3_v4.csv"

"${PYTHON[@]}" -m agentweaver.workloads.extract_swebench_predictions \
  --traj-root "$RAW_ROOT" \
  --out "$RESULTS_DIR/${RUN_ID}_predictions.jsonl" \
  --model-name-or-path agentweaver-mini-swe-qwen-coder-7b \
  --report "$RESULTS_DIR/${RUN_ID}_patch_extraction_report.csv"

bash scripts/run_swebench_eval_pr3.sh \
  --predictions "$RESULTS_DIR/${RUN_ID}_predictions.jsonl" \
  --run-id "${RUN_ID}_agentweaver" \
  --max-workers "$MAX_WORKERS" \
  --summary "$RESULTS_DIR/${RUN_ID}_official_eval_summary.csv"

if awk -F, 'NR==2 && $2=="true" && $4+0 >= 1 {found=1} END {exit found ? 0 : 1}' "$RESULTS_DIR/${RUN_ID}_official_eval_summary.csv"; then
  report_json=$(find "data/logs/swebench_eval/${RUN_ID}_agentweaver" -maxdepth 1 -name "*.${RUN_ID}_agentweaver.json" -type f | head -1 || true)
  if [ -n "$report_json" ]; then
    "${PYTHON[@]}" -m agentweaver.workloads.merge_swebench_verifier \
      --trace-dir "$TRACE_DIR" \
      --official-results "$report_json" \
      --out-dir "data/traces/${RUN_ID}_verified" \
      --summary-out "$RESULTS_DIR/${RUN_ID}_verified_merge_summary.csv"
    "${PYTHON[@]}" -m agentweaver.tracing.pr3_pipeline summarize \
      --trace-dir "data/traces/${RUN_ID}_verified" \
      --run-id "${RUN_ID}_verified" \
      --mode lite5 \
      --results-dir "$RESULTS_DIR"
    "${PYTHON[@]}" -m agentweaver.tracing.pr3_pipeline replay \
      --trace-dir "data/traces/${RUN_ID}_verified" \
      --processed-dir "data/processed/${RUN_ID}_verified" \
      --run-id "${RUN_ID}_verified" \
      --results-dir "$RESULTS_DIR"
  else
    echo "official summary says completed, but final harness report JSON was not found; leaving verifier results unknown" >&2
  fi
fi

"${PYTHON[@]}" -m agentweaver.analysis.bes_positioning_pr3_v4 \
  --out "$RESULTS_DIR/bes_positioning_pr3_v4.md"

"${PYTHON[@]}" -m agentweaver.tracing.pr3_v4_report \
  --results-dir "$RESULTS_DIR" \
  --out "$RESULTS_DIR/pr3_v4_report.md"

cat "$RESULTS_DIR/pr3_v4_report.md"
