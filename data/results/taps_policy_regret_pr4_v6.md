# TAPS Policy Regret PR4-v6

CONFIGS_ANALYZED = 20
DOMAIN_V4_ROWS = 180
ADMISSION_V4_ROWS = 225
REGIMES_IDENTIFIED = ADMISSION_STARVED,DOMAIN_HOT,MEMORY_PRESSURE,TAIL_RISK

## Best Policy By Regime
- ADMISSION_STARVED: taps_admission_v4=2, taps_unified_v5=1
- DOMAIN_HOT: taps_domain_v4=2, taps_unified_v5=1
- MEMORY_PRESSURE: taps_admission_v4=1, taps_unified_v5=1
- TAIL_RISK: taps_unified_v5=8, taps_domain_v4=3, taps_admission_v4=1

## TAPS-U-v5 Regret Sources
taps_domain_v4=5, taps_admission_v4=4

## Interpretation
- Regime labels are derived from current-state pressure features, not future JCT.
- v4 CSVs are used for diagnosis context; aligned v5 replay rows are preferred for per-config regret to avoid mixing simulator axes.
- If one policy dominates all regimes, the report says so rather than forcing artificial regime separation.
