# PR2-v2 Report

PR1_GATE = PASS
OLD_PR2_V1_STATUS = INVALID_FOR_PAPER
OLD_PR2_V1_INVALID_REASON = PR2-v1 used high-error latency fit, non-streaming TTFT gaps, target output tokens, fuzzy metrics, possible prefix-cache contamination, and only full_agentweaver real replay.
H100_PROFILE = FAIL
LATENCY_MODEL_QUALITY = FAIL
PROFILE_ISOLATION = FAIL
TTFT_COVERAGE = 0.000000
OUTPUT_LENGTH_CONTROL = FAIL
PREFIX_CACHE_METRICS_RELIABLE = false
PREFIX_METRICS_SANITY = WARNING
PREFIX_REUSE_LATENCY_BENEFIT = FAIL
REAL_AGENTLIKE_ALL_POLICIES = FAIL
MODEL_PATH = /data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct
MODEL_FAMILY = coder
TOKENIZER_PATH = /data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct
NO_PREFIX_SERVER = http://localhost:8000/v1
PREFIX_SERVER = http://localhost:8001/v1
LENGTH_SWEEP_SUCCESS_CASES = 0
CONCURRENCY_SWEEP_SUCCESS_CASES = 0
PREFIX_SWEEP_SUCCESS_CASES = 0
MAX_SUCCESSFUL_CONCURRENCY = 0
MEDIAN_LATENCY_MODEL_ERROR = 
P95_LATENCY_MODEL_ERROR = 
LATENCY_MODEL_MODE = 
REMAINING_BLOCKERS_FOR_SWE_AGENT = complete isolated no-prefix and prefix H100 profile; latency model does not satisfy median<15% and p95<25% with TTFT coverage>=95%; profile isolation failed or single-server debug was used; real agent-like replay does not include all policies

## Notes
PROFILE_ISOLATION_NOTE = no no-prefix length/concurrency rows available
OUTPUT_LENGTH_CONTROL_NOTE = no successful length rows
PREFIX_METRICS_SANITY_NOTE = strict parser did not find reliable prefix cache counters
PREFIX_REUSE_LATENCY_BENEFIT_NOTE = no comparable no-prefix vs prefix cases
REAL_AGENTLIKE_ALL_POLICIES_NOTE = missing real agent-like policy CSV
REAL_AGENTLIKE_FULL_AGENTWEAVER_BENEFIT_NOTE = missing naive_wafer or full_agentweaver aggregate
PROFILE_ROWS_TOTAL = 0
