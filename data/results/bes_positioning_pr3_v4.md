# BES Positioning for PR3-v4

BES_REAL_TRACE_EFFECT = NOT_OBSERVED
BES_USED_FOR_REAL_MINISWE_MAIN_RESULT = false
BES_RETAINED_FOR_SYNTHETIC_BRANCH_HEAVY = true
ACD_NISP_USED_FOR_REAL_MINISWE_MAIN_RESULT = true

BES remains in the mechanism suite and synthetic branch-heavy evaluation, but the timed mini-SWE real traces did not show an independent BES gain over ACD-only.
The real mini-SWE main result should emphasize ACD/NISP context reuse and model-side savings, while reporting BES as not observed on this workload.
