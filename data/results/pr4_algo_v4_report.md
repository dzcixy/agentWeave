# PR4 Algorithm v4 Report

PR4_ALGO_V4_GATE = PASS
CDF_STATUS = optional
CDF_GAIN = WEAK
DOMAIN_SCHEDULER_GAIN = STRONG
DOMAIN_REMOTE_KV_REDUCTION = 0.688283
DOMAIN_P95_JCT_GAIN = 0.044508
ADMISSION_CONTROL_GAIN = STRONG
ADMISSION_THROUGHPUT_GAIN = 2.447189
ADMISSION_P95_JCT_GAIN = 0.268853
ADMISSION_REGION_UTIL_GAIN = 0.316013
MEMORY_BUDGET_RESULTS = PASS
MIN_BUDGET_WITH_STABLE_GAIN = 4
SNAPSHOT_PROGRESS_GAIN = WEAK
PATCH_SNAPSHOT_EVENTS_AVAILABLE = 0
TIME_TO_FILE_MODIFICATION_GAIN = -0.133517
COST_TO_PATCH_GAIN = 0.056262
PABB_ORACLE_GAP = 0.270798
BEST_NON_ORACLE_P95_GAIN = 0.321479
BEST_NON_ORACLE_THROUGHPUT_GAIN = 2.447189
BEST_NON_ORACLE_READY_WAIT_GAIN = 0.420754
BEST_NON_ORACLE_PATCH_COST_GAIN = 0.056262
PABB_S_NO_FUTURE_LEAKAGE_TESTS = PASS
OLD_BES_DEPRECATED = true
READY_FOR_PR4_SCALE = true

## Interpretation
- Non-oracle mechanisms only are used for main status fields. Oracle rows are retained only as upper bounds in CSVs.
- CDF is explicitly optional because current strict/block-prefix gain remains weak.
- TAPS-D targets domain locality and remote KV traffic; TAPS-A targets serving admission under tool stalls; TAPS-M tests explicit memory pressure.
- PABB-S uses only event-visible tool snapshot/file-modification signals. Existing PR3 traces may lack true patch snapshot fields; those cases are reported by snapshot_events_available in the CSV.
- When PATCH_SNAPSHOT_EVENTS_AVAILABLE is 0, PABB-S gains are capped at WEAK because they come from command-visible file-modification proxies rather than real incremental git snapshots.
- No solved-rate claim is made without official verifier coverage.
