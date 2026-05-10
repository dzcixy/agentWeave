# PR2-v2.1 Report

PR2_V2_1_GATE = PASS
PREFIX_CACHE_METRICS_RELIABLE = false
PREFIX_CACHE_COUNTERS_USED_AS_EVIDENCE = false
PREFIX_REUSE_LATENCY_COMPARISON = PASS
HOLDOUT_LATENCY_MODEL_QUALITY = PASS
SYNTHETIC_REPLAY_H100CALIB = PASS
REAL_AGENTLIKE_ALL_POLICIES = PASS
READY_FOR_PR3 = true
CONTROLLED_REAL_AGENTLIKE_NOTE = real_agentlike_h100 is a controlled pseudo workload, not SWE-bench
random_split_median_ape = 0.010156
random_split_p95_ape = 0.039105
leave_input_median_ape = 0.011889
leave_input_p95_ape = 0.047919
leave_output_median_ape = 0.014950
leave_output_p95_ape = 0.056283
worst_bucket = concurrency_holdout:8:p95=0.978972

## Notes
PREFIX_REUSE_LATENCY_COMPARISON_NOTE = 36/36 shared-prefix cases have lower latency on prefix server
SYNTHETIC_REPLAY_H100CALIB_NOTE = S1/S2/S3_not_worse=3/3; positive_mean_benefit=0.057931; S4_benefit=0.003248; S5_benefit=0.000062; negative_controls_reduced=True; ablation_direction_ok=True
REAL_AGENTLIKE_ALL_POLICIES_NOTE = 5 instances plus aggregate include all policies
README_PR3_SECTION = true
PR3_SCRIPTS_EXIST = true
S1_context_heavy_full_vs_naive_jct_reduction = 0.020271
S2_branch_heavy_full_vs_naive_jct_reduction = 0.120126
S3_tool_stall_heavy_full_vs_naive_jct_reduction = 0.033395
S4_low_reuse_negative_full_vs_naive_jct_reduction = 0.003248
S5_tool_dominated_negative_full_vs_naive_jct_reduction = 0.000062
