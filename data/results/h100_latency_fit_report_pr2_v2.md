# H100 Latency Fit Report PR2-v2

- training_samples = 96
- failed_length_samples_excluded = 0
- ttft_coverage = 1.000000
- prefill_proxy = streaming TTFT; no e2e proportional split is used.
- decode_proxy = e2e_latency - TTFT.
- output_token_source = actual completion tokens from server usage or client tokenizer.
- parametric_median_absolute_percentage_error = 0.011287
- parametric_p95_absolute_percentage_error = 0.041557
- parametric_max_error = 0.220896
- interpolation_fallback_used = false
- LATENCY_MODEL_MODE = parametric
- median_absolute_percentage_error = 0.011287
- p95_absolute_percentage_error = 0.041557
- max_error = 0.220896
- LATENCY_MODEL_QUALITY = PASS

## Error By Input Length Bucket

- input<=1024: median APE=0.0101, n=12
- input<=16384: median APE=0.0258, n=12
- input<=2048: median APE=0.0083, n=12
- input<=32768: median APE=0.0241, n=24
- input<=4096: median APE=0.0092, n=12
- input<=512: median APE=0.0116, n=12
- input<=8192: median APE=0.0135, n=12
