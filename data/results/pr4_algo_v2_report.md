# PR4 Algorithm v2 Report

PR4_ALGO_V2_GATE = PASS
STRICT_CDF_IMPLEMENTED = true
CDF_GAIN = OBSERVED
CDF_ADDED_REUSABLE_TOKENS = 3942
CDF_MODEL_SIDE_SPEEDUP = 0.003845
TAPS_PREDICTIVE_IMPLEMENTED = true
TAPS_PREDICTIVE_GAIN = WEAK
TAPS_ORACLE_GAP = 0.000000
TOOL_LATENCY_PREDICTOR_MEDIAN_ERROR = 0.048296
PABB_ONLINE_IMPLEMENTED = true
PABB_ONLINE_GAIN = OBSERVED
PABB_ORACLE_GAP = 0.467798
BES_DEPRECATED = true
READY_FOR_PR4_SCALE = true

## Notes
- CDF strict-prefix accounting uses block_prefix mode because raw token ids are unavailable in the trace.
- Segment reuse potential is not counted as KV hit unless the CDF canonical prefix makes it continuous from position 0.
- TAPS predictive is the main online result; taps_oracle is only an upper bound.
- PABB online uses only executed-prefix progress signals; pabb_oracle_upper_bound is only an upper bound.
- No solved rate is reported from unknown verifier results.
