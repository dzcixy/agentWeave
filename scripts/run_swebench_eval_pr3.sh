#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS=data/results/mini_swe_lite5_predictions.jsonl
RUN_ID=mini_swe_lite5_agentweaver
MAX_WORKERS=2
DATASET=princeton-nlp/SWE-bench_Lite
RESULTS_DIR=data/results
LOG_DIR=data/logs/swebench_eval
SUMMARY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions) PREDICTIONS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --dataset-name|--dataset) DATASET="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --summary) SUMMARY="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ -z "$SUMMARY" ]; then
  pred_base=$(basename "$PREDICTIONS")
  if [[ "$pred_base" == *_predictions.jsonl ]]; then
    SUMMARY="$RESULTS_DIR/${pred_base%_predictions.jsonl}_official_eval_summary.csv"
  else
    SUMMARY="$RESULTS_DIR/mini_swe_lite5_official_eval_summary.csv"
  fi
fi
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
mkdir -p "$RESULTS_DIR" "$RUN_LOG_DIR"

prediction_count() {
  if [ -s "$PREDICTIONS" ]; then
    wc -l < "$PREDICTIONS" | tr -d ' '
  else
    echo 0
  fi
}

write_summary() {
  local status=$1
  local official=$2
  local evaluated=$3
  local passed=$4
  local failed=$5
  local message=$6
  {
    echo "run_id,official_verifier_used,status,official_verifier_num_evaluated,official_verifier_num_pass,official_verifier_num_fail,message"
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$RUN_ID" "$official" "$status" "$evaluated" "$passed" "$failed" "$(printf '%s' "$message" | tr ',' ';')"
  } > "$SUMMARY"
}

if [ ! -s "$PREDICTIONS" ]; then
  write_summary "SKIPPED" "false" "0" "0" "0" "predictions file missing or empty: $PREDICTIONS"
  cat "$SUMMARY"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1 || ! docker ps >/dev/null 2>&1; then
  write_summary "SKIPPED" "false" "0" "0" "0" "Docker is unavailable or current user cannot run docker"
  cat "$SUMMARY"
  exit 0
fi

if ! python - <<'PY' >/dev/null 2>&1
import swebench
PY
then
  write_summary "SKIPPED" "false" "0" "0" "0" "swebench package is not installed"
  cat "$SUMMARY"
  exit 0
fi

LOG_FILE="$RUN_LOG_DIR/run_evaluation.log"
set +e
python -m swebench.harness.run_evaluation \
  --dataset_name "$DATASET" \
  --predictions_path "$PREDICTIONS" \
  --max_workers "$MAX_WORKERS" \
  --run_id "$RUN_ID" > "$LOG_FILE" 2>&1
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
  n=$(prediction_count)
  write_summary "COMPLETED" "true" "$n" "" "" "official SWE-bench harness completed; see $LOG_FILE"
else
  write_summary "FAILED" "false" "0" "0" "0" "official SWE-bench harness failed with rc=$rc; see $LOG_FILE"
fi

cat "$SUMMARY"
