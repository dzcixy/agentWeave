# PR4 Algorithm v5 Report

PR4_ALGO_V5_GATE = PASS
TAPS_U_GAIN = STRONG
TAPS_U_BEST_CONFIG = {"admission_domain": 1.0, "admission_llm": 0.5, "admission_mem": 0.5, "admission_stall": 1.0, "admission_tail": 1.0, "admission_tool_penalty": 0.5, "memory_pressure_threshold": 0.85, "ready_depth_factor": 1, "slo_target": null, "w_age": 0.05, "w_batch": 0, "w_domain": 1, "w_mem": 0.5, "w_remote": 0.1, "w_resume": 0.5, "w_short": 1, "w_switch": 0.1, "w_tail": 4.0}
TAPS_U_P95_GAIN_OVER_REACTIVE = 0.117662
TAPS_U_P95_GAIN_OVER_ACD_NISP = 0.117662
TAPS_U_THROUGHPUT_GAIN_OVER_REACTIVE = 0.282915
TAPS_U_READY_WAIT_GAIN = 0.095667
TAPS_U_REGION_UTIL_GAIN = 0.191719
TAPS_U_STARVATION_COUNT = 0
TAPS_U_MEMORY_BUDGET_USED = 32
PATCH_SNAPSHOT_EVENTS_AVAILABLE = 0
PABB_S_GAIN = NOT_OBSERVED
PABB_S_COST_TO_PATCH_GAIN = 0.000000
PABB_S_ORACLE_GAP = 0.270798
CDF_STATUS = optional
CDF_GAIN = WEAK
BEST_NON_ORACLE_P95_GAIN_OVER_STRONG_BASELINE = 0.000682
BEST_NON_ORACLE_THROUGHPUT_GAIN_OVER_STRONG_BASELINE = 0.000000
READY_FOR_PR4_SCALE = true

## Notes
- TAPS-U is compared against `reactive_admission` and `acd_nisp`; `static_admission` is not the only baseline.
- `taps_unified_pr4_v5.csv` is a stratified pressure sweep covering every requested value family; the simulator module still exposes the full Cartesian sweep for longer offline runs.
- TAPS-U scheduling uses predicted tool latency and observed residency/queue state, not actual future tool completion time.
- PABB-S is not promoted when `PATCH_SNAPSHOT_EVENTS_AVAILABLE = 0`; snapshot instrumentation must be rerun on fresh mini-SWE trajectories for a stronger claim.
- CDF remains optional because current strict/block-prefix gains are weak.
- No oracle rows or verifier-unknown outcomes are used as main correctness claims.
