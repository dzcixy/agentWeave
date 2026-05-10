# PR3-v2 Report

PR3_V2_GATE = WARNING
TIMING_INSTRUMENTATION = PATCHED_MINISWE
LLM_TIMING_MATCH_RATE_LITE5 = 1.0000
LLM_TIMING_MATCH_RATE_LITE10_R4 = 1.0000
TOOL_TIMING_AVAILABLE = true
MEASURED_AGENT_JCT_AVAILABLE = true
MINI_SWE_LITE5_TIMED = PASS
MINI_SWE_LITE10_R4_TIMED = PASS
OFFICIAL_VERIFIER_USED = false
OFFICIAL_VERIFIER_NUM_EVALUATED = 0
UNKNOWN_VERIFIER_RESULTS = 45
SHARED_CONTEXT_REUSE_OBSERVED = true
BRANCH_SKEW_OBSERVED = true
TOOL_STALL_RESUME_OBSERVED = true
ALL_POLICY_REPLAY_TIMED_LITE5 = PASS
ALL_POLICY_REPLAY_TIMED_LITE10_R4 = PASS
BES_REAL_TRACE_EFFECT = WARNING_ACD_BES_EQUALS_ACD_ONLY
READY_FOR_PR4_SCALEUP = true

## Notes
- No timestamps, tool latency, or verifier pass/fail are fabricated.
- Official verifier status: predictions file missing or empty: data/results/mini_swe_lite5_timed_predictions.jsonl.
- Solved rate is not reported unless official SWE-bench harness completes.
- BES remains identical to ACD-only on this 6x6/4-branch timed trace if BES_REAL_TRACE_EFFECT is WARNING; use PR4 scale/constrained-resource runs to isolate BES scheduling pressure.
