#!/usr/bin/env bash
set -euo pipefail

NO_PREFIX_SERVER=""
NO_PREFIX_METRICS_URL=""
PREFIX_SERVER=""
PREFIX_METRICS_URL=""
MODEL=""
TOKENIZER_PATH=""
RUN_ID=pr2_v2
OUT_DIR=data/profiles
RESULTS_DIR=data/results
PLOTS_DIR=data/plots
REPEATS=${REPEATS:-3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
SINGLE_SERVER_DEBUG=0

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/run_h100_profile_pr2_v2.sh \
  --noprefix-server http://localhost:8000/v1 \
  --noprefix-metrics-url http://localhost:8000/metrics \
  --prefix-server http://localhost:8001/v1 \
  --prefix-metrics-url http://localhost:8001/metrics \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct \
  --run-id pr2_v2

Use --single-server-debug only when intentionally profiling both modes against
one server; the report will mark PROFILE_ISOLATION = FAIL_SINGLE_SERVER_DEBUG.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --noprefix-server) NO_PREFIX_SERVER="$2"; shift 2 ;;
    --noprefix-metrics-url) NO_PREFIX_METRICS_URL="$2"; shift 2 ;;
    --prefix-server) PREFIX_SERVER="$2"; shift 2 ;;
    --prefix-metrics-url) PREFIX_METRICS_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --plots-dir) PLOTS_DIR="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --single-server-debug) SINGLE_SERVER_DEBUG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg $1" >&2; usage; exit 2 ;;
  esac
done

if [ "$SINGLE_SERVER_DEBUG" = "1" ]; then
  if [ -z "$PREFIX_SERVER" ]; then PREFIX_SERVER="$NO_PREFIX_SERVER"; fi
  if [ -z "$PREFIX_METRICS_URL" ]; then PREFIX_METRICS_URL="$NO_PREFIX_METRICS_URL"; fi
fi

if [ -z "$NO_PREFIX_SERVER" ] || [ -z "$NO_PREFIX_METRICS_URL" ] || [ -z "$PREFIX_SERVER" ] || [ -z "$PREFIX_METRICS_URL" ] || [ -z "$MODEL" ] || [ -z "$TOKENIZER_PATH" ]; then
  usage
  exit 2
fi

mkdir -p "$OUT_DIR" "$RESULTS_DIR" "$PLOTS_DIR"
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
MODEL_TEXT="${MODEL,,} ${TOKENIZER_PATH,,}"
MODEL_FAMILY=unknown
if [[ "$MODEL_TEXT" == *coder* ]]; then
  MODEL_FAMILY=coder
elif [[ "$MODEL_TEXT" == *qwen2.5* ]]; then
  MODEL_FAMILY=general
fi

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

write_fail_report() {
  local reason="$1"
  cat > "$RESULTS_DIR/pr2_v2_report.md" <<EOF
# PR2-v2 Report

PR1_GATE = PASS
OLD_PR2_V1_STATUS = INVALID_FOR_PAPER
OLD_PR2_V1_INVALID_REASON = PR2-v1 used high-error latency fit, non-streaming TTFT gaps, target output tokens, fuzzy metrics, possible prefix-cache contamination, and only full_agentweaver real replay.
H100_PROFILE = FAIL
LATENCY_MODEL_QUALITY = FAIL
PROFILE_ISOLATION = FAIL
TTFT_COVERAGE = 0
OUTPUT_LENGTH_CONTROL = FAIL
PREFIX_CACHE_METRICS_RELIABLE = false
PREFIX_METRICS_SANITY = FAIL
PREFIX_REUSE_LATENCY_BENEFIT = FAIL
REAL_AGENTLIKE_ALL_POLICIES = FAIL
MODEL_PATH = $TOKENIZER_PATH
MODEL_FAMILY = $MODEL_FAMILY
TOKENIZER_PATH = $TOKENIZER_PATH
NO_PREFIX_SERVER = $NO_PREFIX_SERVER
PREFIX_SERVER = $PREFIX_SERVER
LENGTH_SWEEP_SUCCESS_CASES = 0
CONCURRENCY_SWEEP_SUCCESS_CASES = 0
PREFIX_SWEEP_SUCCESS_CASES = 0
MAX_SUCCESSFUL_CONCURRENCY = 0
MEDIAN_LATENCY_MODEL_ERROR = 
P95_LATENCY_MODEL_ERROR = 
LATENCY_MODEL_MODE = 
REMAINING_BLOCKERS_FOR_SWE_AGENT = $reason
EOF
}

trap 'write_fail_report "profile pipeline failed at line $LINENO"; cat "$RESULTS_DIR/pr2_v2_report.md"; exit 1' ERR

bash scripts/run_h100_profile_noprefix.sh \
  --server "$NO_PREFIX_SERVER" \
  --metrics-url "$NO_PREFIX_METRICS_URL" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --out-dir "$OUT_DIR" \
  --run-id "$RUN_ID" \
  --repeats "$REPEATS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --include-prefix-baseline

bash scripts/run_h100_profile_prefix.sh \
  --server "$PREFIX_SERVER" \
  --metrics-url "$PREFIX_METRICS_URL" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --out-dir "$OUT_DIR" \
  --run-id "$RUN_ID" \
  --repeats "$REPEATS" \
  --max-model-len "$MAX_MODEL_LEN"

"${PYTHON[@]}" -m agentweaver.profiling.pr2_v2 combine-profiles \
  --out "$OUT_DIR/h100_profile_raw_${RUN_ID}.csv" \
  "$OUT_DIR/h100_profile_length_${RUN_ID}.csv" \
  "$OUT_DIR/h100_profile_concurrency_${RUN_ID}.csv" \
  "$OUT_DIR/h100_profile_prefix_${RUN_ID}.csv"

"${PYTHON[@]}" -m agentweaver.profiling.collect_vllm_metrics \
  --audit-raw \
  "$OUT_DIR/vllm_metrics_raw_${RUN_ID}_length.txt" \
  "$OUT_DIR/vllm_metrics_raw_${RUN_ID}_concurrency.txt" \
  "$OUT_DIR/vllm_metrics_raw_${RUN_ID}_prefix.txt" \
  "$OUT_DIR/vllm_metrics_raw_${RUN_ID}_noprefix_prefix.txt" \
  --audit-out "$RESULTS_DIR/vllm_metrics_source_audit_${RUN_ID}.md"

"${PYTHON[@]}" -m agentweaver.profiling.pr2_v2 prefix-effect \
  --noprefix-csv "$OUT_DIR/h100_profile_prefix_noprefix_${RUN_ID}.csv" \
  --prefix-csv "$OUT_DIR/h100_profile_prefix_${RUN_ID}.csv" \
  --out "$RESULTS_DIR/prefix_reuse_effect_${RUN_ID}.csv"

"${PYTHON[@]}" -m agentweaver.profiling.fit_latency_model \
  --raw "$OUT_DIR/h100_profile_raw_${RUN_ID}.csv" \
  --out "$OUT_DIR/h100_latency_model_${RUN_ID}.json" \
  --plot-dir "$PLOTS_DIR" \
  --plot-suffix "$RUN_ID" \
  --report-csv "$RESULTS_DIR/h100_latency_fit_report_${RUN_ID}.csv" \
  --report-md "$RESULTS_DIR/h100_latency_fit_report_${RUN_ID}.md"

REPORT_ARGS=()
if [ "$SINGLE_SERVER_DEBUG" = "1" ]; then
  REPORT_ARGS+=(--single-server-debug)
fi

"${PYTHON[@]}" -m agentweaver.profiling.pr2_v2 report \
  --out "$RESULTS_DIR/pr2_v2_report.md" \
  --length-csv "$OUT_DIR/h100_profile_length_${RUN_ID}.csv" \
  --concurrency-csv "$OUT_DIR/h100_profile_concurrency_${RUN_ID}.csv" \
  --prefix-csv "$OUT_DIR/h100_profile_prefix_${RUN_ID}.csv" \
  --prefix-effect-csv "$RESULTS_DIR/prefix_reuse_effect_${RUN_ID}.csv" \
  --fit-report-md "$RESULTS_DIR/h100_latency_fit_report_${RUN_ID}.md" \
  --model-json "$OUT_DIR/h100_latency_model_${RUN_ID}.json" \
  --model "$MODEL" \
  --model-path "$TOKENIZER_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --noprefix-server "$NO_PREFIX_SERVER" \
  --prefix-server "$PREFIX_SERVER" \
  "${REPORT_ARGS[@]}"

cat "$RESULTS_DIR/pr2_v2_report.md"
