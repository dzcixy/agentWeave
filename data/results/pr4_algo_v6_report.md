# PR4 Algorithm v6 Report

PR4_ALGO_V6_GATE = WARNING
ADAPTIVE_TAPS_IMPLEMENTED = true
REGIME_CLASSIFIER_IMPLEMENTED = true
TAPS_ADAPTIVE_P95_GAIN_OVER_REACTIVE = 0.104180
TAPS_ADAPTIVE_P95_GAIN_OVER_STRONGEST_BASELINE = 0.037950
TAPS_ADAPTIVE_THROUGHPUT_GAIN_OVER_STRONGEST_BASELINE = -0.031421
TAPS_ADAPTIVE_READY_WAIT_GAIN_OVER_STRONGEST_BASELINE = 0.011286
TAPS_ADAPTIVE_VALIDATION_P95_GAIN = -0.015280
TAPS_ADAPTIVE_VALIDATION_THROUGHPUT_GAIN = -0.000942
TAPS_ADAPTIVE_STARVATION_COUNT = 0
TAPS_ADAPTIVE_GAIN = NOT_OBSERVED
READY_FOR_PR4_SCALE = false

## Metric Separation
- Simple-baseline gains use reactive_admission / acd_nisp.
- Strongest-baseline gains use the best non-oracle policy available for the same configuration.
- Validation gains use held-out instances only.
- Adaptive scheduling uses online regime state and predicted tool latency only; no future JCT/tool completion is used.
