# PR4-v11 Artifact Sanity

ARTIFACT_SANITY = PASS
VALIDATION_CSV_NONEMPTY = true
OBJECTIVES_REPRODUCIBLE_FROM_VALIDATION = true
REPORT_BEST_FIXED_CONSISTENT = true
REPORT_INVALID_RATE_CONSISTENT = true
NO_STALE_PR4_V9_PATHS = true
MATCHED_COMPARISON_EXISTS = true
READY_GATE_TRUSTWORTHY = true
ASTRA_SIM_DECLARED_FALSE = true
ASTRA_EXPORT_USES_REAL_SCHEDULE = true
AGENTWEAVER_V11_COMPARISON_EXISTS = true
AUDIT_INVALID_ROWS = 0
VALIDATION_ROWS = 5360
OBJECTIVE_ROWS = 68
OBJECTIVE_MISMATCHES = 
STALE_PR4_V9_PATHS = 

## Inputs
- astra: data/results/astra_policy_aware_export_v4_report.md
- audit_csv: data/results/aligned_policy_grid_audit_pr4_v10.csv
- audit_md: data/results/aligned_policy_grid_audit_pr4_v10.md
- comparison: data/results/agentweaver_v11_policy_comparison.csv
- grid: data/results/aligned_policy_grid_pr4_v10.csv
- matched: data/results/matched_policy_comparison_pr4_v11.csv
- matched_summary: data/results/matched_policy_comparison_summary_pr4_v11.md
- objectives: data/results/taps_compiler_v3_objectives_pr4_v11.csv
- report: data/results/pr4_algo_v11_report.md
- stp: data/results/stp_v2_simulation_pr4_v11.csv
- validation: data/results/taps_compiler_v3_validation_pr4_v11.csv
