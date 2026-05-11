# PR4-v12 Policy Metric Consistency

METRIC_CONSISTENCY = PASS
METRIC_CONSISTENCY_PASS = true
SCHEDULE_ROWS = 35
POLICY_COMPARISON_ROWS = 12
CACHE_METRICS_NONZERO = true
SCHEDULE_MATCH_ERROR_OK = true
ASTRA_BYTES_MATCH_SCHEDULE = true
FAILED_CHECKS = []

## Check Details
acd_nisp_cache_hit_tokens = PASS rel_error=0.000000000
acd_nisp_resume_prefill_tokens = PASS rel_error=0.000000000
acd_nisp_local_context_bytes = PASS rel_error=0.000000000
acd_nisp_remote_context_bytes = PASS rel_error=0.000000000
acd_nisp_remote_kv_bytes = PASS rel_error=0.000000000
TAPS-C-v3_cache_hit_tokens = PASS rel_error=0.000000000
TAPS-C-v3_resume_prefill_tokens = PASS rel_error=0.000000000
TAPS-C-v3_local_context_bytes = PASS rel_error=0.000000000
TAPS-C-v3_remote_context_bytes = PASS rel_error=0.000000000
TAPS-C-v3_remote_kv_bytes = PASS rel_error=0.000000000
TAPS-C-v3 + STP-AE-top3_cache_hit_tokens = PASS rel_error=0.000000000
TAPS-C-v3 + STP-AE-top3_resume_prefill_tokens = PASS rel_error=0.000000000
TAPS-C-v3 + STP-AE-top3_local_context_bytes = PASS rel_error=0.000000000
TAPS-C-v3 + STP-AE-top3_remote_context_bytes = PASS rel_error=0.000000000
TAPS-C-v3 + STP-AE-top3_remote_kv_bytes = PASS rel_error=0.000000000
full AgentWeaver_cache_hit_tokens = PASS rel_error=0.000000000
full AgentWeaver_resume_prefill_tokens = PASS rel_error=0.000000000
full AgentWeaver_local_context_bytes = PASS rel_error=0.000000000
full AgentWeaver_remote_context_bytes = PASS rel_error=0.000000000
full AgentWeaver_remote_kv_bytes = PASS rel_error=0.000000000
