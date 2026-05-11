# PR4 Algorithm v3 Report

PR4_ALGO_V3_GATE = PASS
CDF_GAIN = WEAK
CDF_ADDED_REUSABLE_TOKENS = 3942
CDF_USED_AS_MAIN_RESULT = false
TAPS_V3_GAIN = WEAK
TAPS_V3_BEST_CONFIG = {"lambda_n": 0, "lambda_s": 0, "memory_budget_bytes": 68719476736.0, "noc_budget_bytes": 137438953472.0, "prefetch_window": 5.0, "tau": 1, "w_a": 0.01, "w_c": 0, "w_l": 0, "w_r": 0, "w_s": 0}
TAPS_V3_P95_GAIN_AT_16 = 0.000068
TAPS_V3_P95_GAIN_AT_32 = -0.002185
TAPS_V3_P95_GAIN_AT_64 = 0.000502
TAPS_V3_READY_WAIT_GAIN_AT_16 = 0.000316
TAPS_V3_THROUGHPUT_GAIN_AT_16 = 0.000101
TAPS_V3_VALIDATION_OVERFIT = false
PABB_ONLINE_IMPLEMENTED = true
PABB_NO_FUTURE_LEAKAGE_TESTS = PASS
PABB_ONLINE_GAIN = MODERATE
PABB_ORACLE_GAP = 0.270798
PABB_USED_AS_MAIN_RESULT = true
OLD_BES_DEPRECATED = true
REAL_MINISWE_MAIN_MECHANISMS = ACD/NISP/TAPS
CDF_STATUS = optional
PABB_STATUS = main
READY_FOR_PR4_SCALE = true

## Evidence Boundaries
- TAPS-v3 numbers are predictive, non-oracle replay results unless explicitly named taps_oracle_upper_bound.
- PABB-v3 online uses only event-prefix-visible signals; oracle rows are upper bounds.
- CDF is strict/block-prefix replay accounting and remains secondary for the current mini-SWE traces.
- Metrics are replay/model-side or measured-tool-side as labeled in their CSVs; no solved-rate claim is made without official verifier coverage.
