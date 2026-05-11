# PR4 Algorithm v5 Diagnosis

PR4_V4_GATE = PASS
PR4_V4_READY_FOR_SCALE = true

## Domain Scheduler
MEAN_JCT_GAIN_MEDIAN_VS_ACD_NISP = 0.146334
P95_JCT_GAIN_MEDIAN_VS_ACD_NISP = -0.039691
READY_QUEUE_WAIT_GAIN_MEDIAN_VS_ACD_NISP = 0.282363
REMOTE_KV_REDUCTION_MEDIAN_VS_ACD_NISP = 0.354934
P95_WORSENED_CONFIGS = 39
- arrival=bursty, regions=1, sessions=16, p95_gain=-0.037223
- arrival=bursty, regions=1, sessions=32, p95_gain=-0.258195
- arrival=bursty, regions=1, sessions=64, p95_gain=-0.086951
- arrival=bursty, regions=2, sessions=32, p95_gain=-0.214770
- arrival=bursty, regions=2, sessions=64, p95_gain=-0.220172
- arrival=bursty, regions=4, sessions=32, p95_gain=-0.084696
- arrival=bursty, regions=4, sessions=64, p95_gain=-0.252190
- arrival=bursty, regions=8, sessions=16, p95_gain=-0.009809
- arrival=bursty, regions=8, sessions=32, p95_gain=-0.045181
- arrival=bursty, regions=8, sessions=64, p95_gain=-0.124933
- arrival=bursty, regions=16, sessions=32, p95_gain=-0.027975
- arrival=bursty, regions=16, sessions=64, p95_gain=-0.071595
- arrival=closed_loop, regions=1, sessions=16, p95_gain=-0.045491
- arrival=closed_loop, regions=1, sessions=32, p95_gain=-0.181047
- arrival=closed_loop, regions=1, sessions=64, p95_gain=-0.077450
- arrival=closed_loop, regions=2, sessions=16, p95_gain=-0.001635
- arrival=closed_loop, regions=2, sessions=32, p95_gain=-0.163882
- arrival=closed_loop, regions=2, sessions=64, p95_gain=-0.205068
- arrival=closed_loop, regions=4, sessions=32, p95_gain=-0.055503
- arrival=closed_loop, regions=4, sessions=64, p95_gain=-0.204488
- arrival=closed_loop, regions=8, sessions=16, p95_gain=-0.010165
- arrival=closed_loop, regions=8, sessions=32, p95_gain=-0.054487
- arrival=closed_loop, regions=8, sessions=64, p95_gain=-0.106456
- arrival=closed_loop, regions=16, sessions=32, p95_gain=-0.042159
- arrival=closed_loop, regions=16, sessions=64, p95_gain=-0.060674
- arrival=poisson, regions=1, sessions=8, p95_gain=-0.004616
- arrival=poisson, regions=1, sessions=16, p95_gain=-0.069515
- arrival=poisson, regions=1, sessions=32, p95_gain=-0.347053
- arrival=poisson, regions=1, sessions=64, p95_gain=-0.161469
- arrival=poisson, regions=2, sessions=32, p95_gain=-0.251220

## Admission
TAPS_ADMISSION_P95_GAIN_MEDIAN_VS_STATIC = 0.028948
TAPS_ADMISSION_THROUGHPUT_GAIN_MEDIAN_VS_STATIC = 0.453886
TAPS_ADMISSION_P95_GAIN_MEDIAN_VS_REACTIVE = 0.000000
TAPS_ADMISSION_THROUGHPUT_GAIN_MEDIAN_VS_REACTIVE = -0.022580
REACTIVE_BEATS_TAPS_ADMISSION_CONFIGS = 17
ADMISSION_DIAGNOSIS = taps_admission is strong versus static_admission, but reactive_admission is the stronger baseline and must be retained.

## Memory
- budget=4GB: hit_rate=0.002981, eviction_count=3417, jct_over_64gb=0.002967
- budget=8GB: hit_rate=0.006302, eviction_count=2491, jct_over_64gb=0.002412
- budget=16GB: hit_rate=0.009379, eviction_count=1287, jct_over_64gb=0.001898
- budget=32GB: hit_rate=0.016637, eviction_count=719, jct_over_64gb=0.000688
- budget=64GB: hit_rate=0.020757, eviction_count=299, jct_over_64gb=0.000000
MEMORY_DIAGNOSIS = memory budget evaluation is present, but low hit_rate means TAPS-M is not a main performance source.

## PABB
PATCH_SNAPSHOT_EVENTS_AVAILABLE = 0
PABB_SNAPSHOT_GAIN_MEAN_VS_FCFS = 0.056262
PABB_DIAGNOSIS = snapshot progress remains weak when trace events do not contain real git snapshot fields.
PABB_REQUIRED_FIX = rerun mini-SWE with AGENTWEAVER_CAPTURE_PATCH_SNAPSHOTS=1 or patched tool hooks that record git status/diff-stat after tool events.

## Final Diagnosis
STRONG_COMPONENTS = ACD,NISP
WEAK_COMPONENTS = CDF,TAPS-M,PABB-S,TAPS-D p95 stability
REQUIRED_FIXES = unify admission/domain/memory/tail scheduling; compare against reactive_admission and acd_nisp; rerun patch snapshot instrumentation or mark PABB-S weak.
