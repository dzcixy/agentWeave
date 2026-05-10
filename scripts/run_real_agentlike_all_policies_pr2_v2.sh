#!/usr/bin/env bash
set -euo pipefail

SERVER=${SERVER:-http://localhost:8001/v1}
MODEL=${MODEL:-qwen-coder-7b}
TOKENIZER_PATH=${TOKENIZER_PATH:-/data2/model_zoo/Qwen2.5-Coder-7B-Instruct}
INSTANCES=5
BRANCH_FANOUT=4
RUN_ID=real_agentlike_pr2_v2
TRACE_DIR=data/traces/real_agentlike_h100
PROCESSED_DIR=data/processed/real_agentlike_h100
RESULTS_DIR=data/results
TMP_DIR="$RESULTS_DIR/.tmp_real_agentlike_pr2_v2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --instances) INSTANCES="$2"; shift 2 ;;
    --branch-fanout) BRANCH_FANOUT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --trace-dir) TRACE_DIR="$2"; shift 2 ;;
    --processed-dir) PROCESSED_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; TMP_DIR="$2/.tmp_real_agentlike_pr2_v2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RESULTS_DIR" "$TMP_DIR"
export TMP_DIR RESULTS_DIR
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

rm -rf "$TRACE_DIR" "$PROCESSED_DIR" "$TMP_DIR"
mkdir -p "$TRACE_DIR" "$PROCESSED_DIR" "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

"${PYTHON[@]}" -m agentweaver.workloads.real_agentlike_trace \
  --server "$SERVER" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --run-id "$RUN_ID" \
  --instances "$INSTANCES" \
  --branch-fanout "$BRANCH_FANOUT" \
  --stream \
  --out "$TRACE_DIR" \
  --summary-out "$RESULTS_DIR/real_agentlike_trace_summary_pr2_v2.csv" \
  --latency-plot-out data/plots/real_agentlike_latency_breakdown_pr2_v2.pdf \
  --context-plot-out data/plots/real_agentlike_context_reuse_pr2_v2.pdf

"${PYTHON[@]}" -m agentweaver.analysis.context_segment_graph \
  --trace-dir "$TRACE_DIR" \
  --out "$PROCESSED_DIR" \
  --config configs/default.yaml

if [ -f data/profiles/h100_latency_model_pr2_v2.json ]; then
  cp data/profiles/h100_latency_model_pr2_v2.json "$PROCESSED_DIR/h100_latency_model.json"
else
  echo "warning: data/profiles/h100_latency_model_pr2_v2.json missing; replay will use placeholder latency model" >&2
fi

POLICIES=(naive_wafer static_branch_pinning wafer_fcfs acd_only acd_bes acd_nisp full_agentweaver)
for POLICY in "${POLICIES[@]}"; do
  "${PYTHON[@]}" -m agentweaver.simulator.replay \
    --processed "$PROCESSED_DIR" \
    --wafer-config configs/wafer_6x6.yaml \
    --policy "$POLICY" \
    --run-id "$RUN_ID" \
    --out "$TMP_DIR/replay_${POLICY}.csv"
done

"${PYTHON[@]}" - <<'PY'
import csv
import glob
import os
from pathlib import Path

tmp = Path(os.environ.get("TMP_DIR", "data/results/.tmp_real_agentlike_pr2_v2"))
out = Path(os.environ.get("RESULTS_DIR", "data/results")) / "real_agentlike_replay_all_policies_pr2_v2.csv"
rows = []
fields = []
for path in sorted(glob.glob(str(tmp / "replay_*.csv"))):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            for key in row:
                if key not in fields:
                    fields.append(key)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(out)
PY

"${PYTHON[@]}" -m agentweaver.profiling.pr2_v2 real-comparison \
  --all-policies-csv "$RESULTS_DIR/real_agentlike_replay_all_policies_pr2_v2.csv" \
  --out "$RESULTS_DIR/real_agentlike_policy_comparison_pr2_v2.csv"

"${PYTHON[@]}" -m agentweaver.profiling.pr2_v2 report \
  --out "$RESULTS_DIR/pr2_v2_report.md" \
  --real-all-policies-csv "$RESULTS_DIR/real_agentlike_replay_all_policies_pr2_v2.csv" \
  --model "$MODEL" \
  --model-path "$TOKENIZER_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --prefix-server "$SERVER"

cat "$RESULTS_DIR/pr2_v2_report.md"
