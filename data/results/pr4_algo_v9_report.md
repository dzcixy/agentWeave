# PR4 Algorithm v9 Report

PR4_ALGO_V9_GATE = PASS
VALID_ROWS = 1487
INVALID_ROWS = 433
INVALID_ROWS_BY_POLICY = {"taps_domain_v4": 121, "taps_unified_adaptive_v6": 156, "taps_unified_v5": 156}
VALID_CONFIGS_ALL_POLICIES = 128
STRATIFIED_GRID_CONFIGS = 320
TAPS_C_V2_IMPLEMENTED = true
BEST_FIXED_POLICY = taps_admission_v4
RANDOM_SPLIT_P95_GAIN_OVER_BEST_FIXED = 0.005351
RANDOM_SPLIT_THROUGHPUT_GAIN_OVER_BEST_FIXED = -0.013830
RANDOM_SPLIT_REGRET_TO_ORACLE = 0.031038
ALL_SPLITS_P95_GAIN_OVER_BEST_FIXED = 0.005014
ALL_SPLITS_REGRET_TO_ORACLE = 0.034035
INVALID_SELECTION_RATE = 0.000000
WORST_CASE_REGRET = 0.368286
FAILURE_CONFIGS = 166
TAPS_C_V2_GAIN = MODERATE
ASTRA_POLICY_AWARE_EXPORT_V2 = PASS
ASTRA_EXPORT_USES_REAL_SCHEDULE = true
ASTRA_REMOTE_REDUCTION = 0.000000
ASTRA_SIM_RUN_COMPLETED = false
SPECULATIVE_TOOL_PREFETCH_ANALYSIS = PASS
SAFE_TOOL_COVERAGE = 0.551502
NEXT_TOOL_TOP1_ACC = 0.392704
POTENTIAL_LATENCY_SAVED = 14.773739
READY_FOR_PR4_SCALE = true

## Notes
- Invalid/incomplete/starved policy rows are audited and excluded from best-fixed, oracle-envelope, and default TAPS-C v2 training.
- Oracle envelope is only an upper bound; it is not a deployable baseline.
- TAPS-C v2 uses workload-only features and train-split pairwise/dominance models. Validation labels are not used at runtime.
- ASTRA policy-aware export consumes the simulator schedule JSONL. It does not infer cached tokens from policy names.
