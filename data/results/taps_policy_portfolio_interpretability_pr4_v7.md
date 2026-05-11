# TAPS-P Portfolio Interpretability PR4-v7

SELECTED_POLICY_COUNTS = taps_domain_v4:7, taps_unified_v5:5, taps_admission_v4:3, taps_unified_adaptive_v6:3
FAILURE_CONFIGS = 13

## Rule Table
- if ['bursty', 'high', 'high', 'high'] -> taps_unified_adaptive_v6 (support=1)
- if ['bursty', 'high', 'high', 'low'] -> taps_admission_v4 (support=1)
- if ['bursty', 'high', 'high', 'mid'] -> taps_unified_v5 (support=1)
- if ['bursty', 'high', 'low', 'mid'] -> taps_unified_v5 (support=1)
- if ['bursty', 'high', 'mid', 'mid'] -> taps_domain_v4 (support=1)
- if ['bursty', 'low', 'high', 'mid'] -> taps_domain_v4 (support=1)
- if ['bursty', 'mid', 'high', 'mid'] -> taps_unified_v5 (support=1)
- if ['poisson', 'high', 'high', 'mid'] -> taps_domain_v4 (support=1)

## Nearest-Neighbor Examples
- aligned:ts128:al16:er4:arrbursty:mem32 -> taps_unified_v5 (arrival=bursty, session_pressure=8.0, region_pressure=4.0)
- aligned:ts16:al16:er4:arrbursty:mem32 -> taps_domain_v4 (arrival=bursty, session_pressure=1.0, region_pressure=4.0)
- aligned:ts32:al16:er4:arrbursty:mem32 -> taps_domain_v4 (arrival=bursty, session_pressure=2.0, region_pressure=4.0)
- aligned:ts64:al16:er16:arrbursty:mem32 -> taps_unified_v5 (arrival=bursty, session_pressure=4.0, region_pressure=1.0)
- aligned:ts64:al16:er4:arrbursty:mem32 -> taps_unified_adaptive_v6 (arrival=bursty, session_pressure=4.0, region_pressure=4.0)
- aligned:ts64:al16:er4:arrbursty:mem64 -> taps_unified_adaptive_v6 (arrival=bursty, session_pressure=4.0, region_pressure=4.0)
- aligned:ts64:al16:er4:arrbursty:mem8 -> taps_admission_v4 (arrival=bursty, session_pressure=4.0, region_pressure=4.0)
- aligned:ts64:al16:er4:arrpoisson:mem32 -> taps_domain_v4 (arrival=poisson, session_pressure=4.0, region_pressure=4.0)

## Failure Analysis
- RuleTable aligned:ts128:al32:er16:arrbursty:mem64: selected=taps_domain_v4 strongest=taps_unified_v5 p95_gain=-0.6979
- RuleTable aligned:ts128:al32:er8:arrpoisson:mem32: selected=taps_domain_v4 strongest=taps_unified_v5 p95_gain=-0.8872
- RuleTable aligned:ts16:al4:er1:arrpoisson:mem8: selected=taps_domain_v4 strongest=taps_admission_v4 p95_gain=-0.0588
- RuleTable aligned:ts64:al16:er1:arrbursty:mem32: selected=taps_unified_v5 strongest=taps_unified_adaptive_v6 p95_gain=-0.1143
- RuleTable aligned:ts64:al16:er4:arrbursty:mem16: selected=taps_admission_v4 strongest=taps_unified_adaptive_v6 p95_gain=-0.0223
- RuleTable aligned:ts64:al16:er4:arrclosed_loop:mem32: selected=taps_domain_v4 strongest=taps_admission_v4 p95_gain=-0.2414
- NearestNeighbor aligned:ts128:al32:er8:arrpoisson:mem32: selected=taps_domain_v4 strongest=taps_unified_v5 p95_gain=-0.8872
- NearestNeighbor aligned:ts16:al4:er1:arrpoisson:mem8: selected=taps_domain_v4 strongest=taps_admission_v4 p95_gain=-0.0588
- NearestNeighbor aligned:ts32:al8:er2:arrclosed_loop:mem16: selected=taps_admission_v4 strongest=taps_domain_v4 p95_gain=-0.0162
- NearestNeighbor aligned:ts64:al16:er1:arrbursty:mem32: selected=taps_unified_v5 strongest=taps_unified_adaptive_v6 p95_gain=-0.1143
- NearestNeighbor aligned:ts64:al16:er2:arrbursty:mem32: selected=taps_unified_adaptive_v6 strongest=taps_unified_v5 p95_gain=-0.3218
- NearestNeighbor aligned:ts64:al16:er4:arrbursty:mem16: selected=taps_admission_v4 strongest=taps_unified_adaptive_v6 p95_gain=-0.0223

TAPS-P uses train-derived rules or nearest train configurations. Validation labels are used only for offline evaluation.
