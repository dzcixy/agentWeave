# AgentWeaver v13 Mode Summary

MATCHED_CONFIGS = 320
VALIDATION_ROWS = 1340
ACD_ONLY_MODEL_SIDE_GAIN = 0.000000
ACD_ONLY_REMOTE_REDUCTION = 0.674392
NISP_ONLY_RESUME_PREFILL_REDUCTION = 0.989383
ACD_NISP_MODEL_SIDE_GAIN = 0.000000
ACD_NISP_REMOTE_REDUCTION = 0.674392
TAPS_C_INCREMENTAL_P95_GAIN = 0.064870
STP_AE_INCREMENTAL_P95_GAIN = 0.000000
FULL_AGENTWEAVER_P95_GAIN_OVER_REACTIVE = 0.064870
FULL_AGENTWEAVER_P95_GAIN_OVER_BEST_FIXED = 0.030189

## Attribution Notes
- acd_only and nisp_only are distinct accounting rows over the same matched config set.
- acd_only attributes shared immutable context residency and wafer-local bytes.
- nisp_only attributes private parked state only; shared ACD context is excluded to avoid double-counting.
- If JCT rows are equal for acd_only/nisp_only, the isolated claim is limited to model-side token and traffic accounting.
