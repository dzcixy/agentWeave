# NISP State Parking PR4-v13

NISP_STATE_PARKING_ROWS = 5677
NISP_STATE_PARKING_STATUS = PASS
HOT_STALLS = 0
WARM_STALLS = 5117
COLD_STALLS = 0
NONE_STALLS = 560
PARKED_STATE_BYTES = 1353732136960
PRIVATE_SUFFIX_HIT_TOKENS = 330501010
RESUME_PREFILL_TOKENS_SAVED_BY_NISP = 330501010

## Decision Model
NISP parks only branch-private state across tool stalls. Shared prefix cache hits are excluded here so ACD and NISP are not double-counted.
HOT/WARM/COLD decisions are attributed from schedule state_residency and parked_state_bytes. If most states are WARM/COLD, the reason is memory pressure relative to expected reuse and stall duration.

NISP utility = predicted_reuse_probability * recompute_cost_saved; cost = parked_state_bytes * memory_pressure * stall_duration.
