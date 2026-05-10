#!/usr/bin/env bash
set -euo pipefail

SERVER=http://localhost:8000/v1
METRICS_URL=http://localhost:8000/metrics
MODEL=qwen-coder-7b
TOKENIZER_PATH=${TOKENIZER_PATH:-/data2/model_zoo/Qwen2.5-Coder-7B-Instruct}
OUT_DIR=data/profiles
RUN_ID=h100_profile
RESULTS_DIR=data/results
PLOTS_DIR=data/plots

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --metrics-url) METRICS_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --config) shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR" "$RESULTS_DIR" "$PLOTS_DIR"
REPORT="$RESULTS_DIR/pr2_h100_profile_report.md"
export OUT_DIR RESULTS_DIR PLOTS_DIR
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

if [ ! -e "$TOKENIZER_PATH" ]; then
  FALLBACK="Qwen/Qwen2.5-Coder-7B-Instruct"
  echo "Tokenizer path $TOKENIZER_PATH not found; falling back to $FALLBACK"
  TOKENIZER_PATH="$FALLBACK"
fi

write_fail_report() {
  local reason="$1"
  local status="${2:-FAIL_SERVER_UNAVAILABLE}"
  cat > "$REPORT" <<EOF
# PR2 H100 Profile Report

PR1_GATE = PASS
H100_PROFILE = $status
LATENCY_MODEL_QUALITY = FAIL
REAL_AGENTLIKE_TRACE = FAIL
VLLM_SERVER_URL = $SERVER
MODEL_PATH = $TOKENIZER_PATH
MODEL_NAME = $MODEL
TOKENIZER_PATH = $TOKENIZER_PATH
VLLM_METRICS_URL = $METRICS_URL
VLLM_METRICS_AVAILABLE = false
PREFIX_CACHE_METRICS_OBSERVED = false
LENGTH_SWEEP_SUCCESS_CASES = 0
LENGTH_SWEEP_FAILED_CASES = 0
CONCURRENCY_SWEEP_SUCCESS_CASES = 0
CONCURRENCY_SWEEP_FAILED_CASES = 0
PREFIX_SWEEP_SUCCESS_CASES = 0
PREFIX_SWEEP_FAILED_CASES = 0
MAX_SUCCESSFUL_CONCURRENCY = 0
MEDIAN_LATENCY_MODEL_ERROR = 
P95_LATENCY_MODEL_ERROR = 
PROFILE_OUTPUT_PATHS = 
REAL_AGENTLIKE_TRACE_PATH = 
REAL_AGENTLIKE_REPLAY_PATH = 
REMAINING_BLOCKERS_FOR_SWE_AGENT = $reason
EOF
}

"${PYTHON[@]}" - <<PY || { write_fail_report "vLLM server health check failed"; cat "$REPORT"; exit 1; }
import requests
server="$SERVER".rstrip("/")
urls=[server.replace("/v1","") + "/health", server + "/models"]
session=requests.Session()
session.trust_env=False
for url in urls:
    try:
        r=session.get(url, timeout=5)
        if r.status_code < 500:
            raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY

trap 'write_fail_report "profile pipeline failed at line $LINENO" "FAIL"; cat "$REPORT"; exit 1' ERR

"${PYTHON[@]}" -m agentweaver.profiling.profile_vllm \
  --server "$SERVER" \
  --metrics-url "$METRICS_URL" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --mode sanity \
  --out-dir "$OUT_DIR"

"${PYTHON[@]}" -m agentweaver.profiling.profile_vllm \
  --server "$SERVER" \
  --metrics-url "$METRICS_URL" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --mode length \
  --out-dir "$OUT_DIR"

"${PYTHON[@]}" -m agentweaver.profiling.profile_vllm \
  --server "$SERVER" \
  --metrics-url "$METRICS_URL" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --mode concurrency \
  --out-dir "$OUT_DIR"

"${PYTHON[@]}" -m agentweaver.profiling.profile_vllm \
  --server "$SERVER" \
  --metrics-url "$METRICS_URL" \
  --model "$MODEL" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --mode prefix \
  --out-dir "$OUT_DIR"

"${PYTHON[@]}" - <<'PY'
import csv
import os
from pathlib import Path
out_dir = Path(os.environ["OUT_DIR"])
files=[
  out_dir / "h100_profile_length_raw.csv",
  out_dir / "h100_profile_concurrency_raw.csv",
  out_dir / "h100_profile_prefix_raw.csv",
]
rows=[]
fields=[]
for p in files:
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            for k in r:
                if k not in fields:
                    fields.append(k)
with (out_dir / "h100_profile_raw.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader(); w.writerows(rows)
PY

"${PYTHON[@]}" -m agentweaver.profiling.fit_latency_model \
  --raw "$OUT_DIR/h100_profile_raw.csv" \
  --out "$OUT_DIR/h100_latency_model.json" \
  --plot-dir "$PLOTS_DIR" \
  --report-csv "$RESULTS_DIR/h100_latency_fit_report.csv" \
  --report-md "$RESULTS_DIR/h100_latency_fit_report.md"

"${PYTHON[@]}" - <<PY
import csv, json
from pathlib import Path
from agentweaver.profiling.prompt_factory import make_prompt_exact_tokens, make_shared_prefix_prompts
raw=list(csv.DictReader(open("$OUT_DIR/h100_profile_raw.csv")))
def count(mode, ok):
    return sum(1 for r in raw if r.get("mode")==mode and str(r.get("status","")).startswith("ok")) if ok else sum(1 for r in raw if r.get("mode")==mode and not str(r.get("status","")).startswith("ok"))
length_ok=count("length", True)
length_fail=count("length", False)
conc_ok=count("concurrency", True)
conc_fail=count("concurrency", False)
prefix_ok=count("prefix", True)
prefix_fail=count("prefix", False)
max_conc=0
for r in raw:
    if r.get("mode")=="concurrency" and str(r.get("status","")).startswith("ok"):
        max_conc=max(max_conc, int(float(r.get("concurrency") or 0)))
metrics_available=any((r.get("metrics_error","")=="") for r in raw)
prefix_observed=any(float(r.get("prefix_cache_hits_delta") or 0)>0 or float(r.get("cached_prompt_tokens_delta") or 0)>0 for r in raw if r.get("mode")=="prefix")
fit_md=Path("$RESULTS_DIR/h100_latency_fit_report.md").read_text()
model=json.load(open("$OUT_DIR/h100_latency_model.json"))
quality=model.get("quality","WARNING")
prompt_checks=[]
for target in [512,4096,8192]:
    _, actual = make_prompt_exact_tokens(target, seed=target, tokenizer_path="$TOKENIZER_PATH")
    prompt_checks.append((target, actual, abs(actual-target)/target))
shared = make_shared_prefix_prompts(1024, 256, 2, seed=123, tokenizer_path="$TOKENIZER_PATH")
shared_ok = shared[0]["shared_prefix_token_ids"] == shared[1]["shared_prefix_token_ids"]
median=""
p95=""
for line in fit_md.splitlines():
    if "median_absolute_percentage_error" in line:
        median=line.split("=")[-1].strip()
    if "p95_absolute_percentage_error" in line:
        p95=line.split("=")[-1].strip()
status="PASS" if quality=="PASS" and length_ok and conc_ok and prefix_ok else "WARNING"
Path("$REPORT").write_text(f"""# PR2 H100 Profile Report

PR1_GATE = PASS
H100_PROFILE = {status}
LATENCY_MODEL_QUALITY = {quality}
REAL_AGENTLIKE_TRACE = NOT_RUN
VLLM_SERVER_URL = $SERVER
MODEL_PATH = $TOKENIZER_PATH
MODEL_NAME = $MODEL
TOKENIZER_PATH = $TOKENIZER_PATH
VLLM_METRICS_URL = $METRICS_URL
VLLM_METRICS_AVAILABLE = {str(metrics_available).lower()}
PREFIX_CACHE_METRICS_OBSERVED = {str(prefix_observed).lower()}
PROMPT_FACTORY_512 = target:{prompt_checks[0][0]},actual:{prompt_checks[0][1]},relative_error:{prompt_checks[0][2]:.6f}
PROMPT_FACTORY_4096 = target:{prompt_checks[1][0]},actual:{prompt_checks[1][1]},relative_error:{prompt_checks[1][2]:.6f}
PROMPT_FACTORY_8192 = target:{prompt_checks[2][0]},actual:{prompt_checks[2][1]},relative_error:{prompt_checks[2][2]:.6f}
SHARED_PREFIX_EXACT_MATCH = {str(shared_ok).lower()}
LENGTH_SWEEP_SUCCESS_CASES = {length_ok}
LENGTH_SWEEP_FAILED_CASES = {length_fail}
CONCURRENCY_SWEEP_SUCCESS_CASES = {conc_ok}
CONCURRENCY_SWEEP_FAILED_CASES = {conc_fail}
PREFIX_SWEEP_SUCCESS_CASES = {prefix_ok}
PREFIX_SWEEP_FAILED_CASES = {prefix_fail}
MAX_SUCCESSFUL_CONCURRENCY = {max_conc}
MEDIAN_LATENCY_MODEL_ERROR = {median}
P95_LATENCY_MODEL_ERROR = {p95}
PROFILE_OUTPUT_PATHS = $OUT_DIR/h100_profile_raw.csv,$OUT_DIR/h100_latency_model.json
REAL_AGENTLIKE_TRACE_PATH = 
REAL_AGENTLIKE_REPLAY_PATH = 
REMAINING_BLOCKERS_FOR_SWE_AGENT = collect real SWE-agent trajectories and official harness correctness
""", encoding="utf-8")
print(Path("$REPORT").read_text())
PY
