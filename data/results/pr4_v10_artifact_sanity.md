# PR4-v10 Artifact Sanity

ARTIFACT_SANITY = FAIL
VALIDATION_CSV_NONEMPTY = true
REPORT_BEST_FIXED_CONSISTENT = false
REPORT_INVALID_RATE_CONSISTENT = false
DEFAULT_PATHS_USE_CURRENT_PR = true
READY_GATE_TRUSTWORTHY = false
VALID_ROWS_MATCH_AUDIT = true
REPORT_BEST_FIXED_POLICY = taps_admission_v4
OBJECTIVE_BEST_FIXED_POLICY = taps_admission_v4
METHODOLOGY_BEST_FIXED_POLICY = taps_domain_v4
REPORT_INVALID_SELECTION_RATE = 0.000000
OBJECTIVE_INVALID_SELECTION_RATE = 0.000000
METHODOLOGY_INVALID_SELECTION_RATE = 0.290909
VALIDATION_INVALID_SELECTION_RATE = 0.000000
AUDIT_COUNTS = {"invalid_by_policy": {"taps_domain_v4": 121, "taps_unified_adaptive_v6": 156, "taps_unified_v5": 156}, "invalid_rows": 433, "valid_rows": 1487}

## Inputs
- audit: data/results/aligned_policy_grid_stratified_audit_pr4_v9.csv
- grid: data/results/aligned_policy_grid_stratified_pr4_v9.csv
- methodology: data/results/pr4_v9_evaluation_methodology.md
- objectives: data/results/taps_compiler_v2_objectives_pr4_v9.csv
- report: data/results/pr4_algo_v9_report.md
- valid_grid: data/results/aligned_policy_grid_stratified_valid_pr4_v9.csv
- validation: data/results/taps_compiler_v2_validation_pr4_v9.csv
