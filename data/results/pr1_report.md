# PR-1 Report: Synthetic Correctness and Simulator Core Fix

## Modified files

- `agentweaver/workloads/synthetic_fork_join.py`
- `agentweaver/simulator/replay.py`
- `agentweaver/simulator/acd_mapping.py`
- `agentweaver/simulator/bes_scheduler.py`
- `agentweaver/simulator/nisp.py`
- `scripts/run_all_small.sh`
- `tests/test_synthetic_expected_properties.py`

## Validation

- `pytest -q`: `13 passed in 0.97s`.
- `bash scripts/run_all_small.sh`: completed and generated the required CSV/PDF/PNG artifacts.
- Required result files generated:
  - `data/results/synthetic_trace_summary.csv`
  - `data/results/context_reuse_summary.csv`
  - `data/results/acd_mapping.csv`
  - `data/results/gpu_cache_baselines.csv`
  - `data/results/wafer_replay_summary.csv`
  - `data/results/ablation.csv`
  - `data/results/sensitivity_mesh.csv`
  - `data/results/sensitivity_link_bw.csv`
  - `data/results/sensitivity_tool_latency.csv`

## Mechanism checks

- ACD is exercised by exact-prefix shared context in S1/S2/S3 and reports lower post-mapping hop/traffic metrics than naive shared-bank placement.
- BES is exercised by comparing branch-elastic policies against `static_branch_pinning`, where tool-blocked branches retain regions in the baseline.
- NISP is exercised by the mandatory LLM_1 after each tool, with HOT/WARM/COLD restore changing resume prefill tokens.

## Negative controls

- S4 has low exact-prefix reuse, so AgentWeaver benefit is intentionally small.
- S5 is tool dominated, so wafer-side LLM/context benefits drop relative to total JCT.

## Remaining limitations

- The default latency model remains a placeholder until H100 profiling is collected.
- Real vLLM/SGLang/SWE-agent measurements are not fabricated by this script.

## H100 profiling blockers

- Start vLLM/SGLang with the target local Qwen model under `/data2/model_zoo`.
- Collect `data/profiles/h100_profile_raw.csv` and fit `data/profiles/h100_latency_model.json`.
- Run real multi-rollout SWE traces and convert them through the adapters before full evaluation.
