# Aligned Policy Grid Audit PR4-v10

INPUT_GRID = data/results/aligned_policy_grid_pr4_v10.csv
TOTAL_ROWS = 1920
VALID_ROWS = 1920
INVALID_ROWS = 0
VALID_CONFIGS_ALL_RECORDED_POLICIES = 320
VALID_CONFIGS_ALL_POLICIES = 320
CONFIGS_WITH_AT_LEAST_ONE_VALID_POLICY = 320

## Invalid Rows By Policy
- none

## Invalid Reasons

- none

## Policy Validity Rule
Rows are valid only when all sessions complete, starvation is zero, core latency/throughput metrics are positive, ordering constraints hold, and no NaN/inf appears.
Invalid rows are kept in the audit CSV but excluded from best-fixed, oracle-envelope, and default cost-model training.

## Policies Frequently Invalid
none
