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
PREDICTIONS_ABS=$(realpath "$PREDICTIONS" 2>/dev/null || printf '%s' "$PREDICTIONS")
if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

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

if ! "${PYTHON[@]}" - <<'PY' >/dev/null 2>&1
import swebench
PY
then
  write_summary "SKIPPED" "false" "0" "0" "0" "swebench package is not installed"
  cat "$SUMMARY"
  exit 0
fi

LOG_FILE="$RUN_LOG_DIR/run_evaluation.log"
set +e
(
  cd "$RUN_LOG_DIR" || exit 1
  "${PYTHON[@]}" -m swebench.harness.run_evaluation \
  --dataset_name "$DATASET" \
  --predictions_path "$PREDICTIONS_ABS" \
  --max_workers "$MAX_WORKERS" \
  --run_id "$RUN_ID" \
  --report_dir "$RUN_LOG_DIR"
) > "$LOG_FILE" 2>&1
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
  read -r completed passed failed report_file < <(python - "$RUN_LOG_DIR" "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
run_id = sys.argv[2]
candidates = sorted(root.glob(f"*.{run_id}.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    print("0 0 0 ''")
    raise SystemExit
report = json.loads(candidates[0].read_text())
print(
    int(report.get("completed_instances", 0)),
    int(report.get("resolved_instances", 0)),
    int(report.get("unresolved_instances", 0)),
    str(candidates[0]),
)
PY
  )
  if [ "${completed:-0}" -ge 1 ]; then
    write_summary "COMPLETED" "true" "$completed" "${passed:-0}" "${failed:-0}" "official SWE-bench harness completed; report=${report_file}; see $LOG_FILE"
  else
    n=$(prediction_count)
    write_summary "COMPLETED_NO_EVALUATIONS" "false" "0" "0" "0" "harness exited 0 but no completed instance report was found; predictions=$n; see $LOG_FILE"
  fi
else
  write_summary "FAILED" "false" "0" "0" "0" "official SWE-bench harness failed with rc=$rc; see $LOG_FILE"
fi

cat "$SUMMARY"
