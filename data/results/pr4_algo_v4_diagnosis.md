# PR4 Algorithm v4 Diagnosis

## CDF
CDF_STATUS = optional
CDF_ADDED_REUSABLE_TOKENS = 3942
CDF_ESTIMATED_PREFILL_SAVED = 0.211235
CDF_BLOCK_PREFIX_MODE = true
CDF_MAIN_MECHANISM = false
CDF_DIAGNOSIS = CDF should remain secondary: added reusable tokens are small and block-prefix mode is approximate because raw token ids are unavailable.

## TAPS
- sessions=1, arrival=bursty, regions=1: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000012
- sessions=1, arrival=bursty, regions=2: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000006
- sessions=1, arrival=bursty, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000003
- sessions=1, arrival=bursty, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000002
- sessions=1, arrival=bursty, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000001
- sessions=1, arrival=closed_loop, regions=1: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000012
- sessions=1, arrival=closed_loop, regions=2: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000006
- sessions=1, arrival=closed_loop, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000003
- sessions=1, arrival=closed_loop, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000002
- sessions=1, arrival=closed_loop, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000001
- sessions=1, arrival=poisson, regions=1: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000012
- sessions=1, arrival=poisson, regions=2: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000006
- sessions=1, arrival=poisson, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000003
- sessions=1, arrival=poisson, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000002
- sessions=1, arrival=poisson, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_acd=0.000014, p95_jct_gain_vs_naive=0.011989, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000001
- sessions=2, arrival=bursty, regions=1: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000074, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.013180, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000078
- sessions=2, arrival=bursty, regions=2: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000039
- sessions=2, arrival=bursty, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000020
- sessions=2, arrival=bursty, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000010
- sessions=2, arrival=bursty, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000005
- sessions=2, arrival=closed_loop, regions=1: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000074, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.013176, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000078
- sessions=2, arrival=closed_loop, regions=2: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000039
- sessions=2, arrival=closed_loop, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000020
- sessions=2, arrival=closed_loop, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000010
- sessions=2, arrival=closed_loop, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000005
- sessions=2, arrival=poisson, regions=1: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000076, p95_jct_gain_vs_acd=0.000018, p95_jct_gain_vs_naive=0.013456, ready_queue_wait_gain=0.000046, region_utilization_change=-0.000075
- sessions=2, arrival=poisson, regions=2: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000039
- sessions=2, arrival=poisson, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000019
- sessions=2, arrival=poisson, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000010
- sessions=2, arrival=poisson, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000077, p95_jct_gain_vs_acd=0.000017, p95_jct_gain_vs_naive=0.012359, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000005
- sessions=4, arrival=bursty, regions=1: throughput_gain_vs_acd=0.000089, mean_jct_gain_vs_acd=0.000164, p95_jct_gain_vs_acd=0.000094, p95_jct_gain_vs_naive=0.061487, ready_queue_wait_gain=0.000514, region_utilization_change=-0.000075
- sessions=4, arrival=bursty, regions=2: throughput_gain_vs_acd=0.000027, mean_jct_gain_vs_acd=0.000069, p95_jct_gain_vs_acd=0.000027, p95_jct_gain_vs_naive=0.013691, ready_queue_wait_gain=0.000205, region_utilization_change=-0.000050
- sessions=4, arrival=bursty, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012050, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000026
- sessions=4, arrival=bursty, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012050, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000013
- sessions=4, arrival=bursty, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012050, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000007
- sessions=4, arrival=closed_loop, regions=1: throughput_gain_vs_acd=0.000089, mean_jct_gain_vs_acd=0.000162, p95_jct_gain_vs_acd=0.000094, p95_jct_gain_vs_naive=0.056008, ready_queue_wait_gain=0.000491, region_utilization_change=-0.000075
- sessions=4, arrival=closed_loop, regions=2: throughput_gain_vs_acd=0.000027, mean_jct_gain_vs_acd=0.000068, p95_jct_gain_vs_acd=0.000027, p95_jct_gain_vs_naive=0.013678, ready_queue_wait_gain=0.000178, region_utilization_change=-0.000050
- sessions=4, arrival=closed_loop, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012050, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000026
- sessions=4, arrival=closed_loop, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012050, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000013
- sessions=4, arrival=closed_loop, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012050, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000007
- sessions=4, arrival=poisson, regions=1: throughput_gain_vs_acd=0.000082, mean_jct_gain_vs_acd=0.000165, p95_jct_gain_vs_acd=0.000092, p95_jct_gain_vs_naive=0.085864, ready_queue_wait_gain=0.000639, region_utilization_change=-0.000075
- sessions=4, arrival=poisson, regions=2: throughput_gain_vs_acd=0.000022, mean_jct_gain_vs_acd=0.000067, p95_jct_gain_vs_acd=0.000023, p95_jct_gain_vs_naive=0.018785, ready_queue_wait_gain=0.000196, region_utilization_change=-0.000050
- sessions=4, arrival=poisson, regions=4: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012132, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000026
- sessions=4, arrival=poisson, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012050, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000013
- sessions=4, arrival=poisson, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000062, p95_jct_gain_vs_acd=0.000015, p95_jct_gain_vs_naive=0.012050, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000007
- sessions=8, arrival=bursty, regions=1: throughput_gain_vs_acd=0.000036, mean_jct_gain_vs_acd=0.000191, p95_jct_gain_vs_acd=0.000067, p95_jct_gain_vs_naive=0.081809, ready_queue_wait_gain=0.000287, region_utilization_change=-0.000117
- sessions=8, arrival=bursty, regions=2: throughput_gain_vs_acd=0.000036, mean_jct_gain_vs_acd=0.000142, p95_jct_gain_vs_acd=0.000041, p95_jct_gain_vs_naive=0.028682, ready_queue_wait_gain=0.000289, region_utilization_change=-0.000066
- sessions=8, arrival=bursty, regions=4: throughput_gain_vs_acd=0.000016, mean_jct_gain_vs_acd=0.000101, p95_jct_gain_vs_acd=0.000019, p95_jct_gain_vs_naive=0.020430, ready_queue_wait_gain=0.000436, region_utilization_change=-0.000038
- sessions=8, arrival=bursty, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000073, p95_jct_gain_vs_acd=0.000016, p95_jct_gain_vs_naive=0.012145, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000019
- sessions=8, arrival=bursty, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000073, p95_jct_gain_vs_acd=0.000016, p95_jct_gain_vs_naive=0.012145, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000010
- sessions=8, arrival=closed_loop, regions=1: throughput_gain_vs_acd=0.000040, mean_jct_gain_vs_acd=0.000194, p95_jct_gain_vs_acd=0.000070, p95_jct_gain_vs_naive=0.090928, ready_queue_wait_gain=0.000284, region_utilization_change=-0.000114
- sessions=8, arrival=closed_loop, regions=2: throughput_gain_vs_acd=0.000035, mean_jct_gain_vs_acd=0.000157, p95_jct_gain_vs_acd=0.000063, p95_jct_gain_vs_naive=0.032962, ready_queue_wait_gain=0.000324, region_utilization_change=-0.000066
- sessions=8, arrival=closed_loop, regions=4: throughput_gain_vs_acd=0.000016, mean_jct_gain_vs_acd=0.000081, p95_jct_gain_vs_acd=0.000540, p95_jct_gain_vs_naive=0.016893, ready_queue_wait_gain=0.000163, region_utilization_change=-0.000038
- sessions=8, arrival=closed_loop, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000073, p95_jct_gain_vs_acd=0.000016, p95_jct_gain_vs_naive=0.012145, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000019
- sessions=8, arrival=closed_loop, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000073, p95_jct_gain_vs_acd=0.000016, p95_jct_gain_vs_naive=0.012145, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000010
- sessions=8, arrival=poisson, regions=1: throughput_gain_vs_acd=0.000034, mean_jct_gain_vs_acd=0.000193, p95_jct_gain_vs_acd=0.000065, p95_jct_gain_vs_naive=0.085696, ready_queue_wait_gain=0.000296, region_utilization_change=-0.000117
- sessions=8, arrival=poisson, regions=2: throughput_gain_vs_acd=0.000088, mean_jct_gain_vs_acd=0.000207, p95_jct_gain_vs_acd=0.000078, p95_jct_gain_vs_naive=0.034897, ready_queue_wait_gain=0.000568, region_utilization_change=-0.000052
- sessions=8, arrival=poisson, regions=4: throughput_gain_vs_acd=0.000015, mean_jct_gain_vs_acd=0.000162, p95_jct_gain_vs_acd=0.000046, p95_jct_gain_vs_naive=0.017018, ready_queue_wait_gain=0.001563, region_utilization_change=-0.000038
- sessions=8, arrival=poisson, regions=8: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000073, p95_jct_gain_vs_acd=0.000016, p95_jct_gain_vs_naive=0.012145, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000019
- sessions=8, arrival=poisson, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000073, p95_jct_gain_vs_acd=0.000016, p95_jct_gain_vs_naive=0.012145, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000010
- sessions=16, arrival=bursty, regions=1: throughput_gain_vs_acd=0.000039, mean_jct_gain_vs_acd=0.000245, p95_jct_gain_vs_acd=0.000038, p95_jct_gain_vs_naive=0.092153, ready_queue_wait_gain=0.000281, region_utilization_change=-0.000210
- sessions=16, arrival=bursty, regions=2: throughput_gain_vs_acd=0.000045, mean_jct_gain_vs_acd=0.000250, p95_jct_gain_vs_acd=0.000112, p95_jct_gain_vs_naive=0.049166, ready_queue_wait_gain=0.000333, region_utilization_change=-0.000115
- sessions=16, arrival=bursty, regions=4: throughput_gain_vs_acd=0.000101, mean_jct_gain_vs_acd=0.000189, p95_jct_gain_vs_acd=0.000068, p95_jct_gain_vs_naive=0.038663, ready_queue_wait_gain=0.000316, region_utilization_change=-0.000051
- sessions=16, arrival=bursty, regions=8: throughput_gain_vs_acd=0.000018, mean_jct_gain_vs_acd=0.000159, p95_jct_gain_vs_acd=0.000095, p95_jct_gain_vs_naive=0.024425, ready_queue_wait_gain=0.000584, region_utilization_change=-0.000037
- sessions=16, arrival=bursty, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000107, p95_jct_gain_vs_acd=0.000020, p95_jct_gain_vs_naive=0.012401, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000019
- sessions=16, arrival=closed_loop, regions=1: throughput_gain_vs_acd=0.000051, mean_jct_gain_vs_acd=0.000243, p95_jct_gain_vs_acd=0.000127, p95_jct_gain_vs_naive=0.087741, ready_queue_wait_gain=0.000275, region_utilization_change=-0.000183
- sessions=16, arrival=closed_loop, regions=2: throughput_gain_vs_acd=0.000046, mean_jct_gain_vs_acd=0.000237, p95_jct_gain_vs_acd=0.000119, p95_jct_gain_vs_naive=0.065191, ready_queue_wait_gain=0.000306, region_utilization_change=-0.000112
- sessions=16, arrival=closed_loop, regions=4: throughput_gain_vs_acd=0.000049, mean_jct_gain_vs_acd=0.000197, p95_jct_gain_vs_acd=0.000052, p95_jct_gain_vs_naive=0.042235, ready_queue_wait_gain=0.000324, region_utilization_change=-0.000062
- sessions=16, arrival=closed_loop, regions=8: throughput_gain_vs_acd=0.000018, mean_jct_gain_vs_acd=0.000153, p95_jct_gain_vs_acd=0.000028, p95_jct_gain_vs_naive=0.022322, ready_queue_wait_gain=0.000485, region_utilization_change=-0.000037
- sessions=16, arrival=closed_loop, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000107, p95_jct_gain_vs_acd=0.000020, p95_jct_gain_vs_naive=0.012401, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000019
- sessions=16, arrival=poisson, regions=1: throughput_gain_vs_acd=0.000171, mean_jct_gain_vs_acd=-0.001304, p95_jct_gain_vs_acd=0.000143, p95_jct_gain_vs_naive=0.094319, ready_queue_wait_gain=-0.001786, region_utilization_change=-0.000120
- sessions=16, arrival=poisson, regions=2: throughput_gain_vs_acd=0.000024, mean_jct_gain_vs_acd=0.000250, p95_jct_gain_vs_acd=0.000097, p95_jct_gain_vs_naive=0.059329, ready_queue_wait_gain=0.000371, region_utilization_change=-0.000133
- sessions=16, arrival=poisson, regions=4: throughput_gain_vs_acd=0.000039, mean_jct_gain_vs_acd=0.000207, p95_jct_gain_vs_acd=0.000104, p95_jct_gain_vs_naive=0.040149, ready_queue_wait_gain=0.000389, region_utilization_change=-0.000066
- sessions=16, arrival=poisson, regions=8: throughput_gain_vs_acd=0.000016, mean_jct_gain_vs_acd=0.000161, p95_jct_gain_vs_acd=0.000024, p95_jct_gain_vs_naive=0.025493, ready_queue_wait_gain=0.000746, region_utilization_change=-0.000038
- sessions=16, arrival=poisson, regions=16: throughput_gain_vs_acd=0.000014, mean_jct_gain_vs_acd=0.000107, p95_jct_gain_vs_acd=0.000020, p95_jct_gain_vs_naive=0.012401, ready_queue_wait_gain=0.000000, region_utilization_change=-0.000019
- sessions=32, arrival=bursty, regions=1: throughput_gain_vs_acd=0.000707, mean_jct_gain_vs_acd=0.001693, p95_jct_gain_vs_acd=0.001248, p95_jct_gain_vs_naive=0.157618, ready_queue_wait_gain=0.001864, region_utilization_change=-0.000338
- sessions=32, arrival=bursty, regions=2: throughput_gain_vs_acd=0.000851, mean_jct_gain_vs_acd=0.000829, p95_jct_gain_vs_acd=-0.000029, p95_jct_gain_vs_naive=0.078878, ready_queue_wait_gain=0.000958, region_utilization_change=-0.000160
- sessions=32, arrival=bursty, regions=4: throughput_gain_vs_acd=0.000070, mean_jct_gain_vs_acd=-0.000076, p95_jct_gain_vs_acd=-0.002185, p95_jct_gain_vs_naive=0.041024, ready_queue_wait_gain=-0.000369, region_utilization_change=-0.000276
- sessions=32, arrival=bursty, regions=8: throughput_gain_vs_acd=0.000009, mean_jct_gain_vs_acd=-0.000985, p95_jct_gain_vs_acd=-0.001586, p95_jct_gain_vs_naive=0.032110, ready_queue_wait_gain=-0.003350, region_utilization_change=-0.000153
- sessions=32, arrival=bursty, regions=16: throughput_gain_vs_acd=0.000008, mean_jct_gain_vs_acd=0.000579, p95_jct_gain_vs_acd=0.001763, p95_jct_gain_vs_naive=0.019215, ready_queue_wait_gain=0.002448, region_utilization_change=-0.000079
- sessions=32, arrival=closed_loop, regions=1: throughput_gain_vs_acd=0.000710, mean_jct_gain_vs_acd=0.001216, p95_jct_gain_vs_acd=0.001060, p95_jct_gain_vs_naive=0.110334, ready_queue_wait_gain=0.001317, region_utilization_change=-0.000348
- sessions=32, arrival=closed_loop, regions=2: throughput_gain_vs_acd=0.000328, mean_jct_gain_vs_acd=0.001119, p95_jct_gain_vs_acd=0.000295, p95_jct_gain_vs_naive=0.076716, ready_queue_wait_gain=0.001312, region_utilization_change=-0.000382
- sessions=32, arrival=closed_loop, regions=4: throughput_gain_vs_acd=-0.001504, mean_jct_gain_vs_acd=0.000827, p95_jct_gain_vs_acd=-0.001622, p95_jct_gain_vs_naive=0.052416, ready_queue_wait_gain=0.001094, region_utilization_change=-0.000660
- sessions=32, arrival=closed_loop, regions=8: throughput_gain_vs_acd=0.000008, mean_jct_gain_vs_acd=-0.000690, p95_jct_gain_vs_acd=0.000270, p95_jct_gain_vs_naive=0.027884, ready_queue_wait_gain=-0.002405, region_utilization_change=-0.000154
- sessions=32, arrival=closed_loop, regions=16: throughput_gain_vs_acd=0.000007, mean_jct_gain_vs_acd=0.000407, p95_jct_gain_vs_acd=-0.000103, p95_jct_gain_vs_naive=0.018119, ready_queue_wait_gain=0.000642, region_utilization_change=-0.000079
- sessions=32, arrival=poisson, regions=1: throughput_gain_vs_acd=0.000734, mean_jct_gain_vs_acd=0.001358, p95_jct_gain_vs_acd=0.001370, p95_jct_gain_vs_naive=0.147065, ready_queue_wait_gain=0.001517, region_utilization_change=-0.000304
- sessions=32, arrival=poisson, regions=2: throughput_gain_vs_acd=0.000578, mean_jct_gain_vs_acd=0.000771, p95_jct_gain_vs_acd=-0.000978, p95_jct_gain_vs_naive=0.108402, ready_queue_wait_gain=0.000926, region_utilization_change=-0.000265
- sessions=32, arrival=poisson, regions=4: throughput_gain_vs_acd=0.000411, mean_jct_gain_vs_acd=0.001940, p95_jct_gain_vs_acd=-0.000682, p95_jct_gain_vs_naive=0.044174, ready_queue_wait_gain=0.003271, region_utilization_change=-0.000189
- sessions=32, arrival=poisson, regions=8: throughput_gain_vs_acd=0.000022, mean_jct_gain_vs_acd=0.001173, p95_jct_gain_vs_acd=-0.000616, p95_jct_gain_vs_naive=0.033814, ready_queue_wait_gain=0.003026, region_utilization_change=-0.000149
- sessions=32, arrival=poisson, regions=16: throughput_gain_vs_acd=0.000212, mean_jct_gain_vs_acd=0.000748, p95_jct_gain_vs_acd=0.000206, p95_jct_gain_vs_naive=0.017935, ready_queue_wait_gain=0.006904, region_utilization_change=-0.000065
- sessions=64, arrival=bursty, regions=1: throughput_gain_vs_acd=0.000579, mean_jct_gain_vs_acd=0.000749, p95_jct_gain_vs_acd=0.000748, p95_jct_gain_vs_naive=0.172013, ready_queue_wait_gain=0.000781, region_utilization_change=-0.000104
- sessions=64, arrival=bursty, regions=2: throughput_gain_vs_acd=0.001919, mean_jct_gain_vs_acd=0.000645, p95_jct_gain_vs_acd=-0.000518, p95_jct_gain_vs_naive=0.118089, ready_queue_wait_gain=0.000699, region_utilization_change=0.000771
- sessions=64, arrival=bursty, regions=4: throughput_gain_vs_acd=0.001344, mean_jct_gain_vs_acd=0.000214, p95_jct_gain_vs_acd=0.000502, p95_jct_gain_vs_naive=0.099788, ready_queue_wait_gain=0.000212, region_utilization_change=0.000272
- sessions=64, arrival=bursty, regions=8: throughput_gain_vs_acd=0.000623, mean_jct_gain_vs_acd=0.000775, p95_jct_gain_vs_acd=-0.000060, p95_jct_gain_vs_naive=0.042226, ready_queue_wait_gain=0.001128, region_utilization_change=-0.000019
- sessions=64, arrival=bursty, regions=16: throughput_gain_vs_acd=-0.000176, mean_jct_gain_vs_acd=-0.000041, p95_jct_gain_vs_acd=-0.000091, p95_jct_gain_vs_naive=0.031893, ready_queue_wait_gain=-0.000491, region_utilization_change=-0.000111
- sessions=64, arrival=closed_loop, regions=1: throughput_gain_vs_acd=0.000574, mean_jct_gain_vs_acd=0.000712, p95_jct_gain_vs_acd=0.000705, p95_jct_gain_vs_naive=0.164649, ready_queue_wait_gain=0.000740, region_utilization_change=-0.000112
- sessions=64, arrival=closed_loop, regions=2: throughput_gain_vs_acd=-0.000356, mean_jct_gain_vs_acd=0.000632, p95_jct_gain_vs_acd=0.000770, p95_jct_gain_vs_naive=0.102344, ready_queue_wait_gain=0.000681, region_utilization_change=-0.000697
- sessions=64, arrival=closed_loop, regions=4: throughput_gain_vs_acd=0.000323, mean_jct_gain_vs_acd=0.000359, p95_jct_gain_vs_acd=0.000588, p95_jct_gain_vs_naive=0.054906, ready_queue_wait_gain=0.000394, region_utilization_change=-0.000161
- sessions=64, arrival=closed_loop, regions=8: throughput_gain_vs_acd=0.000017, mean_jct_gain_vs_acd=0.000259, p95_jct_gain_vs_acd=-0.000779, p95_jct_gain_vs_naive=0.042939, ready_queue_wait_gain=0.000282, region_utilization_change=-0.000165
- sessions=64, arrival=closed_loop, regions=16: throughput_gain_vs_acd=0.000009, mean_jct_gain_vs_acd=0.000980, p95_jct_gain_vs_acd=-0.000999, p95_jct_gain_vs_naive=0.030243, ready_queue_wait_gain=0.002084, region_utilization_change=-0.000114
- sessions=64, arrival=poisson, regions=1: throughput_gain_vs_acd=0.000536, mean_jct_gain_vs_acd=0.000885, p95_jct_gain_vs_acd=0.000813, p95_jct_gain_vs_naive=0.171287, ready_queue_wait_gain=0.000934, region_utilization_change=-0.000134
- sessions=64, arrival=poisson, regions=2: throughput_gain_vs_acd=0.001447, mean_jct_gain_vs_acd=0.000846, p95_jct_gain_vs_acd=0.001590, p95_jct_gain_vs_naive=0.122475, ready_queue_wait_gain=0.000938, region_utilization_change=0.000453
- sessions=64, arrival=poisson, regions=4: throughput_gain_vs_acd=0.000287, mean_jct_gain_vs_acd=0.001309, p95_jct_gain_vs_acd=-0.000454, p95_jct_gain_vs_naive=0.110535, ready_queue_wait_gain=0.001673, region_utilization_change=-0.000167
- sessions=64, arrival=poisson, regions=8: throughput_gain_vs_acd=0.000143, mean_jct_gain_vs_acd=0.000733, p95_jct_gain_vs_acd=0.000472, p95_jct_gain_vs_naive=0.032435, ready_queue_wait_gain=0.001148, region_utilization_change=-0.000128
- sessions=64, arrival=poisson, regions=16: throughput_gain_vs_acd=-0.000479, mean_jct_gain_vs_acd=0.000463, p95_jct_gain_vs_acd=0.002675, p95_jct_gain_vs_naive=0.028044, ready_queue_wait_gain=0.001214, region_utilization_change=-0.000144
TAPS_BEST_SCENARIO = sessions=64, arrival=poisson, regions=16 p95_gain=0.002675
TAPS_WORST_SCENARIO = sessions=32, arrival=bursty, regions=4 p95_gain=-0.002185
TAPS_DIAGNOSIS = v3 ready-queue priority alone is too weak; v4 should optimize domain locality, admission, and memory pressure.

## PABB
PABB_ONLINE_GAIN_MEDIAN = 0.000000
PABB_ONLINE_GAIN_MEAN = 0.056262
PABB_ORACLE_GAP_MEAN = 0.270798
PABB_PATCH_PRODUCING_INSTANCES = 2
PABB_ONLINE_FAIL_CASES = astropy__astropy-12907, django__django-11815, django__django-11848, django__django-11964, django__django-12125, django__django-12308, django__django-12708, django__django-13028, django__django-13033
PABB_DIAGNOSIS = PABB is online and no-leakage, but progress signals often arrive late; v4 should use measured incremental patch snapshots when available.

## Workload
TOOL_TIME_SHARE_MEDIAN = 0.253409
LLM_TIME_SHARE_MEDIAN = 0.702971
MEASURED_JCT_MEDIAN = 12.855230
BRANCH_JCT_CV_MEDIAN = 0.383733
WORKLOAD_DIAGNOSIS = mini-SWE is often tool-time dominated with high branch skew; pure prompt reuse alone cannot create large end-to-end JCT gains.

## Conclusions
MAIN_EFFECTIVE_MECHANISMS = ACD,NISP,TAPS-A/TAPS-D if validation shows non-oracle gain
WEAK_MECHANISMS = CDF,TAPS-v3-ready-queue-only
NEXT_OPTIMIZATION_TARGETS = domain locality, admission control, SRAM-bounded residency, earlier online patch progress
PR4_V3_REPORTED_TAPS_GAIN = WEAK
PR4_V3_REPORTED_PABB_GAIN = MODERATE
