# PR4-v8 Evaluation Methodology

EVAL_METHODOLOGY_FIXED = true
ORACLE_ENVELOPE_NOT_USED_AS_BASELINE = true

## Baseline Classes

### Fixed Deployable Baselines
These policies are single deployable choices and can be selected before serving validation configurations:
- `reactive_admission`
- `acd_nisp`
- `taps_domain_v4`
- `taps_admission_v4`
- `taps_unified_v5`
- `taps_unified_adaptive_v6`

### Best Fixed Policy
The best fixed policy is selected on the train split by average objective. It is deployable because the same single policy is used for every validation configuration.

### Oracle Envelope
The oracle envelope chooses the best true policy result separately for each validation configuration. It is not deployable and is not a baseline. It is used only as an upper bound for regret.

## Primary Metrics
- gain over best fixed policy
- gain over `reactive_admission` and `acd_nisp`
- regret to oracle envelope
- worst-case regret

The previous phrase `strongest baseline` is avoided because per-config strongest is an oracle envelope, not a normal baseline.
