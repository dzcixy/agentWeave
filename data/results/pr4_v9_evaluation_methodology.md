# PR4-v9 Evaluation Methodology

This methodology replaces the PR4-v8 wording that made the per-configuration strongest policy look like a deployable baseline.

VALID_GRID = data/results/aligned_policy_grid_stratified_valid_pr4_v9.csv
TRAIN_CONFIGS = 265
VALIDATION_CONFIGS = 55
FIXED_DEPLOYABLE_BASELINES = reactive_admission,acd_nisp,taps_domain_v4,taps_admission_v4,taps_unified_v5,taps_unified_adaptive_v6
BEST_FIXED_POLICY = taps_domain_v4
BEST_FIXED_SELECTED_ON_TRAIN_ONLY = true
ORACLE_ENVELOPE_NOT_BASELINE = true
INVALID_ROWS_EXCLUDED = true
BEST_FIXED_GAIN_OVER_REACTIVE_P95 = -0.006300
BEST_FIXED_GAIN_OVER_ACD_NISP_P95 = -0.006300
BEST_FIXED_REGRET_TO_ORACLE_P95 = 0.084002
WORST_CASE_REGRET = 0.618298
INVALID_SELECTION_RATE = 0.290909

## Baseline Classes
1. Fixed deployable baselines: one policy is chosen before validation and applied unchanged.
2. Best fixed policy: selected on the train split by normalized objective, then applied to validation configs.
3. Oracle envelope: the best valid policy per validation config after outcomes are known. It is an upper bound only, not a deployable baseline.

## Validity
Invalid/starved/incomplete rows are excluded from best-fixed selection, oracle-envelope computation, and default cost-model training. See data/results/aligned_policy_grid_audit_pr4_v9.md.

## Policy Coverage In Valid Grid
- acd_nisp: 320
- reactive_admission: 320
- taps_admission_v4: 320
- taps_domain_v4: 199
- taps_unified_adaptive_v6: 164
- taps_unified_v5: 164
