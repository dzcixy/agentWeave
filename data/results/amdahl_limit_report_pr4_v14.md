# Amdahl Limit Report PR4-v14

MODEL_TIME_SHARE = 0.285056
TOOL_TIME_SHARE = 0.714944
MODEL_SIDE_GAIN = 0.664800
AMDHAL_MAX_E2E_GAIN = 0.189505
ACTUAL_E2E_GAIN = 0.042685

If tool latency dominates the trace, end-to-end JCT gain is bounded by model_time_share * model_side_gain even when ACD/NISP reduce model-side work.
