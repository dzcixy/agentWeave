# PR-1 Final Gate Report

## Validation

- `bash scripts/run_all_small.sh --run-id pr1_final`: completed.
- `pytest -q`: must be run by the caller before this script; PR2 phase 0 records the actual result separately.
- Required policies present: True
- Synthetic branch shape LLM0 -> TOOL -> LLM1 -> VERIFIER: True

## Mechanism Checks

### S1_context_heavy

- avg_hops_before = 3.195235
- avg_hops_after = 0.230559
- noc_bytes_before = 123810226176
- noc_bytes_after = 31375884288
- hotspot_before = 3.768183
- hotspot_after = 3.204667
- ACD locality check = True
- hotspot/NoC check = True

### S2_branch_heavy

- naive branch_wasted_tokens = 70626
- full branch_wasted_tokens = 48946
- acd_bes branch_wasted_tokens = 70626
- full blocked_compute_time_avoided = 80.181514
- acd_bes blocked_compute_time_avoided = 80.181514
- branch waste check = True
- blocked compute release check = True

### S3_tool_stall_heavy

- naive resume_prefill_tokens = 20782
- acd_nisp resume_prefill_tokens = 6840
- full resume_prefill_tokens = 6840
- acd_nisp HOT/WARM/COLD = 2/2/0
- full HOT/WARM/COLD = 2/2/0
- resume prefill check = True
- parking diversity check = True

### Negative Controls

- S1 benefit = 0.092012
- S2 benefit = 0.061831
- S4_low_reuse_negative benefit = 0.010542
- S5_tool_dominated_negative benefit = 0.000263
- S4 low-reuse reduced-benefit check = True
- S5 tool-dominated reduced-benefit check = True

## Remaining Limitations

- Default latency model is still placeholder until PR2 H100 profiling succeeds.
- Wafer results remain trace-driven simulation.

PR1_GATE = PASS
