# Final Paper Claims v14

MAIN_CLAIM = MODEL_TRAFFIC_REDUCTION
H100_CALIBRATION_STATUS = NOT_RUN
ASTRA_SIM_RUN_COMPLETED = false
FULL_AGENTWEAVER_P95_GAIN_OVER_REACTIVE = 0.042685

Strong analytic-replay claim: ACD/NISP reduce model-side prefill work and NoC traffic in real mode replay.
Not yet paper-ready: H100 calibration has not run, so latency magnitudes remain analytic.
Weak/optional: STP-AE is demoted because p95 gain remains zero.
Threats to validity: current final run is analytic-only because H100 calibration did not run; ASTRA export is not an ASTRA-sim cycle result.
