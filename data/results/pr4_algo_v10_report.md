# PR4 Algorithm v10 Report

PR4_ALGO_V10_GATE = WARNING
ARTIFACT_SANITY = FAIL
VALIDATION_CSV_NONEMPTY = true
REPORT_CONSISTENCY = false
INVALID_ROWS = 0
INVALID_ROWS_BY_POLICY = {}
VALID_CONFIGS_ALL_POLICIES = 320
STARVATION_FIXED = true
TAPS_C_V3_IMPLEMENTED = true
BEST_FIXED_POLICY = taps_unified_adaptive_v6
RANDOM_SPLIT_P95_GAIN_OVER_BEST_FIXED = -0.000180
RANDOM_SPLIT_THROUGHPUT_GAIN_OVER_BEST_FIXED = -0.013965
ALL_SPLITS_P95_GAIN_OVER_BEST_FIXED = 0.007308
ALL_SPLITS_REGRET_TO_ORACLE = 0.011887
WORST_CASE_REGRET = 0.126531
INVALID_SELECTION_RATE = 0.000000
FAILURE_CONFIGS = 26
TAPS_C_V3_GAIN = MODERATE
STP_IMPLEMENTED = true
SAFE_TOOL_COVERAGE = 0.431579
NEXT_TOOL_TOP1_ACC = 0.410526
STP_P95_GAIN = 0.000000
STP_MEAN_JCT_GAIN = 0.033292
STP_WASTED_WORK_OVERHEAD = 0.094511
STP_SAFETY_VIOLATIONS = 0
STP_GAIN = WEAK
ASTRA_POLICY_AWARE_EXPORT_V3 = PASS
ASTRA_EXPORT_USES_REAL_SCHEDULE = true
ASTRA_REMOTE_REDUCTION = 0.053759
ASTRA_COMPUTE_REDUCTION = 0.173921
ASTRA_SIM_RUN_COMPLETED = false
READY_FOR_PR4_SCALE = false
NO_ORACLE_OR_FUTURE_INFO_USED = true
NO_FAKE_ASTRA_OUTPUT = true

## Notes
- Invalid/incomplete/starved rows are kept in audit artifacts and excluded from compiler training/evaluation grids.
- TAPS-C v3 uses workload/config features and train-split learned parameters only; validation labels and oracle envelopes are not runtime inputs.
- STP launches only commands classified SAFE_READ_ONLY. The oracle STP row is reported separately as an upper bound.
- ASTRA v3 export consumes schedule JSONL and does not infer cached tokens unless explicitly requested; ASTRA_SIM_RUN_COMPLETED remains false because no ASTRA binary was run.
