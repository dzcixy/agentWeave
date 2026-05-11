# PR4 Algorithm v7 Report

PR4_ALGO_V7_GATE = WARNING
TAPS_P_IMPLEMENTED = true
SELECTORS_IMPLEMENTED = RuleTable,NearestNeighbor,ConservativeFallback
BEST_SELECTOR = NearestNeighbor
TAPS_P_VALIDATION_P95_GAIN_OVER_REACTIVE = -0.077991
TAPS_P_VALIDATION_P95_GAIN_OVER_STRONGEST = -0.159286
TAPS_P_VALIDATION_THROUGHPUT_GAIN_OVER_STRONGEST = -0.035704
TAPS_P_READY_WAIT_GAIN_OVER_STRONGEST = -0.063821
TAPS_P_ORACLE_REGRET_P95 = 0.159286
TAPS_P_SAFE_FALLBACK_RATE = 1.000000
TAPS_P_FAILURE_CONFIGS = 7
READY_FOR_PR4_SCALE = false

## Fairness Notes
- TAPS-P selectors are trained on train configurations only.
- Validation labels are used only to score held-out configurations.
- The strongest-baseline columns include the best available non-oracle policy for each configuration.
- Table-based selector evaluation is not presented as a fresh replay.

## Failure Configs
aligned:ts128:al32:er8:arrpoisson:mem32, aligned:ts16:al4:er1:arrpoisson:mem8, aligned:ts32:al8:er2:arrclosed_loop:mem16, aligned:ts64:al16:er1:arrbursty:mem32, aligned:ts64:al16:er2:arrbursty:mem32, aligned:ts64:al16:er4:arrbursty:mem16, aligned:ts64:al16:er4:arrclosed_loop:mem32
