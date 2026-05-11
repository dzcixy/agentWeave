# PR4 Algorithm v3 Diagnosis

## CDF
added_reusable_tokens = 3942
model_side_speedup = 0.003845
estimated_prefill_saved = 0.211235
block_prefix_mode = true
diagnosis = CDF is implemented with strict/block-prefix accounting, but the observed mini-SWE gain is weak because these traces already render most stable task/tool/repo context near the prompt prefix and the remaining canonicalizable context is small relative to total tool time.
paper_ready = secondary_only

## TAPS
- sessions=1: throughput_gain=0.000014, mean_jct_gain=0.000014, p95_jct_gain=0.000014, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000003
- sessions=2: throughput_gain=0.000014, mean_jct_gain=0.000077, p95_jct_gain=0.000017, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000020
- sessions=4: throughput_gain=0.000014, mean_jct_gain=0.000062, p95_jct_gain=0.000015, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000026
- sessions=8: throughput_gain=0.001617, mean_jct_gain=0.014539, p95_jct_gain=0.008492, ready_queue_wait_gain=0.166825, region_utilization_change=0.000357
- sessions=16: throughput_gain=0.008368, mean_jct_gain=0.145838, p95_jct_gain=0.095639, ready_queue_wait_gain=0.350937, region_utilization_change=0.002129
diagnosis = TAPS is strongest under high session pressure where tool stalls create ready-queue contention. It is weak at low session counts because the trace is tool-time dominated and there is little hidden work to schedule.
paper_ready = needs_v3_predictive_stress_test

## PABB
legacy_rows = 270
CURRENT_PABB_HAS_FUTURE_LEAKAGE = true
leakage_fields = patch_nonempty, git_diff_bytes, duplicate_patch_hash, no_file_modification, llm_tokens_used, official_verifier_result
diagnosis = The legacy pabb_budget path ranks branches using full-branch summaries before those branches have executed. It is invalid as an online scheduler and must be replaced by event-level replay.
paper_ready = false_until_v3_online_tests_pass
