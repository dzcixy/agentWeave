# PR2-v2 Report

PR1_GATE = PASS
OLD_PR2_V1_STATUS = INVALID_FOR_PAPER
OLD_PR2_V1_INVALID_REASON = PR2-v1 used high-error latency fit, non-streaming TTFT gaps, target output tokens, fuzzy metrics, possible prefix-cache contamination, and only full_agentweaver real replay.
H100_PROFILE = PASS
LATENCY_MODEL_QUALITY = PASS
PROFILE_ISOLATION = PASS
TTFT_COVERAGE = 1.000000
OUTPUT_LENGTH_CONTROL = PASS
PREFIX_CACHE_METRICS_RELIABLE = false
PREFIX_METRICS_SANITY = WARNING
PREFIX_REUSE_LATENCY_BENEFIT = PASS
REAL_AGENTLIKE_ALL_POLICIES = PASS
MODEL_PATH = /data2/model_zoo/Qwen2.5-Coder-7B-Instruct
MODEL_FAMILY = coder
TOKENIZER_PATH = /data2/model_zoo/Qwen2.5-Coder-7B-Instruct
NO_PREFIX_SERVER = http://localhost:8000/v1
PREFIX_SERVER = http://localhost:8001/v1
LENGTH_SWEEP_SUCCESS_CASES = 96
CONCURRENCY_SWEEP_SUCCESS_CASES = 90
PREFIX_SWEEP_SUCCESS_CASES = 135
MAX_SUCCESSFUL_CONCURRENCY = 16
MEDIAN_LATENCY_MODEL_ERROR = 0.011287
P95_LATENCY_MODEL_ERROR = 0.041557
LATENCY_MODEL_MODE = parametric
REMAINING_BLOCKERS_FOR_SWE_AGENT = SWE-agent/SWE-bench intentionally not started in PR2-v2

## Notes
PROFILE_ISOLATION_NOTE = no reliable no-prefix cache deltas observed
OUTPUT_LENGTH_CONTROL_NOTE = target_32_actual_ge_0.8_target_rate=1.000; target_128_actual_ge_0.8_target_rate=1.000; target_512_actual_ge_0.8_target_rate=1.000; target_1024_actual_ge_0.8_target_rate=1.000
PREFIX_METRICS_SANITY_NOTE = shared=0 hits=48.0 cached=48.0 ratio=0.084211 per_request=24.000; shared=0 hits=48.0 cached=48.0 ratio=0.084211 per_request=24.000; shared=0 hits=48.0 cached=48.0 ratio=0.084211 per_request=24.000; shared=0 hits=128.0 cached=128.0 ratio=0.112281 per_request=32.000; shared=0 hits=128.0 cached=128.0 ratio=0.112281 per_request=32.000; shared=0 hits=128.0 cached=128.0 ratio=0.112281 per_request=32.000; shared=0 hits=256.0 cached=256.0 ratio=0.112281 per_request=32.000; shared=0 hits=256.0 cached=256.0 ratio=0.112281 per_request=32.000; shared=0 hits=256.0 cached=256.0 ratio=0.112281 per_request=32.000; shared=0 hits=48.0 cached=48.0 ratio=0.044362 per_request=24.000; treating vLLM prefix cache counters as unreliable for evidence because zero client-shared-prefix prompts still report small block/template-level cache deltas; prefix reuse benefit is evaluated by no-prefix vs prefix latency comparison
PREFIX_REUSE_LATENCY_BENEFIT_NOTE = 36/36 shared-prefix cases faster on prefix server
REAL_AGENTLIKE_ALL_POLICIES_NOTE = 5 instances plus aggregate include all policies
REAL_AGENTLIKE_FULL_AGENTWEAVER_BENEFIT_NOTE = full_agentweaver_jct_reduction_vs_naive = 0.176378
PROFILE_ROWS_TOTAL = 321
