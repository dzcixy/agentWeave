# PR4 Algorithm Report

PR4_ALGO_GATE = PASS
CDF_GAIN = NOT_OBSERVED
CDF_ADDED_REUSABLE_TOKENS = 0
CDF_REUSABLE_RATIO_BEFORE = 0.988254
CDF_REUSABLE_RATIO_AFTER = 0.988254
CDF_MODEL_SIDE_SPEEDUP = 0.000000
TAPS_GAIN = OBSERVED
TAPS_THROUGHPUT_GAIN_AT_8_SESSIONS = 0.010632
TAPS_P95_JCT_GAIN_AT_8_SESSIONS = 0.019453
TAPS_REGION_UTIL_GAIN_AT_8_SESSIONS = -0.022530
PABB_GAIN = OBSERVED
PABB_TIME_TO_PATCH_GAIN = 0.555556
PABB_TOKENS_TO_PATCH_GAIN = 0.555556
OFFICIAL_VERIFIER_USED = true
BES_DEPRECATED = true
READY_FOR_PR4_SCALE = true

## Notes
- CDF results are canonicalized replay potential; raw traces are unchanged.
- CDF_GAIN may be NOT_OBSERVED when the collected mini-SWE traces already expose almost all repeated shared context as exact-prefix reusable.
- TAPS is a multi-session replay using measured mini-SWE tool latencies and H100-calibrated model latency.
- TAPS can improve throughput/p95 while region_utilization drops if context reuse removes enough model-side work in a tool-dominated trace.
- PABB reports patch/progress cost metrics; no solved rate is reported from unknown verifier results.
- Old BES is deprecated for real mini-SWE main-result attribution.
