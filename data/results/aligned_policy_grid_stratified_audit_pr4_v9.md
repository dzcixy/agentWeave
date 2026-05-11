# Aligned Policy Grid Audit PR4-v9

INPUT_GRID = data/results/aligned_policy_grid_stratified_pr4_v9.csv
TOTAL_ROWS = 1920
VALID_ROWS = 1487
INVALID_ROWS = 433
VALID_CONFIGS_ALL_RECORDED_POLICIES = 128
VALID_CONFIGS_ALL_POLICIES = 128
CONFIGS_WITH_AT_LEAST_ONE_VALID_POLICY = 320

## Invalid Rows By Policy
- taps_domain_v4: 121
- taps_unified_adaptive_v6: 156
- taps_unified_v5: 156

## Invalid Reasons

- incomplete_sessions: 204
- starvation: 433

## Policy Validity Rule
Rows are valid only when all sessions complete, starvation is zero, core latency/throughput metrics are positive, ordering constraints hold, and no NaN/inf appears.
Invalid rows are kept in the audit CSV but excluded from best-fixed, oracle-envelope, and default cost-model training.

## Policies Frequently Invalid
taps_domain_v4, taps_unified_adaptive_v6, taps_unified_v5
