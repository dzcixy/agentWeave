# Aligned Policy Grid Audit PR4-v9

INPUT_GRID = data/results/aligned_policy_grid_pr4_v8.csv
TOTAL_ROWS = 720
VALID_ROWS = 601
INVALID_ROWS = 119
VALID_CONFIGS_ALL_RECORDED_POLICIES = 66
VALID_CONFIGS_ALL_POLICIES = 66
CONFIGS_WITH_AT_LEAST_ONE_VALID_POLICY = 120

## Invalid Rows By Policy
- taps_domain_v4: 36
- taps_unified_adaptive_v6: 41
- taps_unified_v5: 42

## Invalid Reasons

- incomplete_sessions: 63
- starvation: 119

## Policy Validity Rule
Rows are valid only when all sessions complete, starvation is zero, core latency/throughput metrics are positive, ordering constraints hold, and no NaN/inf appears.
Invalid rows are kept in the audit CSV but excluded from best-fixed, oracle-envelope, and default cost-model training.

## Policies Frequently Invalid
taps_domain_v4, taps_unified_adaptive_v6, taps_unified_v5
