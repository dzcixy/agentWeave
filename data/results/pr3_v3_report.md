# PR3-v3 Report

PR3_V3_GATE = WARNING
PATCH_EXTRACTION = FAIL
PATCH_EXTRACTION_NUM_PREDICTIONS = 0
PATCH_EXTRACTION_REASON = FAIL_NO_PATCH: no non-empty submission/final_patch/patch/diff/unified-diff block found
OFFICIAL_VERIFIER_USED = false
OFFICIAL_VERIFIER_STATUS = SKIPPED
OFFICIAL_VERIFIER_NUM_EVALUATED = 0
OFFICIAL_VERIFIER_NUM_PASS = 0
OFFICIAL_VERIFIER_NUM_FAIL = 0
OFFICIAL_VERIFIER_MESSAGE = predictions file missing or empty: data/results/mini_swe_lite5_timed_predictions.jsonl
UNKNOWN_VERIFIER_RESULTS_AFTER_MERGE = 45
BES_STRESS_EVALUATION = PASS
BES_REAL_TRACE_EFFECT = NOT_OBSERVED
BES_MAX_GAIN_VS_ACD_ONLY = 0.000000
BES_RESOURCE_CONFIGURATIONS = 11
MODEL_TOOL_BREAKDOWN = PASS
MEASURED_TOOL_TIME_SHARE_LITE10_R4 = 0.636920
MEASURED_LLM_TIME_SHARE_LITE10_R4 = 0.351982
READY_FOR_PR4_PILOT = true

Notes:
- Official SWE-bench verifier results are only merged when the harness evaluates non-empty patches.
- Missing patches remain unknown and are not counted as pass or fail.
- BES stress uses constrained effective compute regions; any reported gain is measured by replay, not hardcoded.
- Measured wall-clock tool/LLM time is reported separately from simulated H100 model-side replay time.
