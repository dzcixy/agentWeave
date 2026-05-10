# AgentWeaver

Artifact repository for the planned HPCA paper:

**AgentWeaver: Context-Domain Mapping and Branch-Elastic Execution for Fork-Join LLM Agents on Wafer-Scale Architectures**

The artifact targets LLM-based software engineering agents such as SWE-agent, mini-SWE-agent, and OpenHands-style coding agents. It provides trace collection/adaptation, context segment graph analysis, H100-calibrated latency modeling, GPU cache baselines, a chiplet-wafer mesh simulator, AgentWeaver policies, plotting scripts, and sanity tests.

## Research Goal

Agent workloads are modeled as fork-join execution DAGs with shared task/repo/tool context, planner/worker/critic/verifier LLM calls, multiple patch trajectories, shell/test/build tools, branch joins, and growing state. The repository evaluates how exact-prefix context reuse, branch elasticity, and non-invasive state parking affect trace-driven replay.

## Challenges and Mechanisms

Challenges:

- **Agent Context-Domain Fragmentation**: branches repeatedly prefill the same issue/repo/tool context, while request-level prefix caches do not map consumers to physical locality domains.
- **Branch-Region Skew**: branch output lengths, tool latencies, and verifier distances diverge, so static branch-to-region placement wastes compute and amplifies stragglers.
- **Tool-Stall Residency Conflict**: branches blocked in pytest/compiler/shell either occupy KV capacity or pay recompute after tool return.

Mechanisms:

- **ACD-Mapping**: builds an exact-prefix Context Segment Graph and maps shared-context consumers plus KV homes into compact chiplet mesh domains.
- **BES**: schedules only `READY_LLM` branches onto compute regions, releases regions on tool stalls, and performs safe sibling cancellation only after measured verifier/test pass.
- **NISP**: chooses HOT/WARM/COLD state parking using measured tool latency CDFs, recompute cost, SRAM pressure, and foreground NoC slack.

No semantic KV reuse or speculative correctness-changing pruning is implemented.

## Hardware Assumptions

- Real profiling and GPU baselines are intended for **2 x H100** using vLLM/SGLang.
- Wafer/chiplet results are **trace-driven simulation** using H100-calibrated prefill/decode latency models.
- No real wafer hardware is required.
- The simulator models context/KV residency, chiplet memory, mesh traffic, and scheduling; it does not claim exact Cerebras or any commercial wafer behavior.
- Tool latency is measured from traces and is not accelerated by the wafer simulator.

Default model path:

- Local first: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- Served model name: `qwen2.5-7b`
- Fallback name: `Qwen/Qwen2.5-Coder-7B-Instruct`

## Installation

```bash
cd /home/duzc/data/agent_trace/AgentWeaver
uv sync
```

Optional profiling/SWE-bench dependencies. The profiling extra includes vLLM/Torch and should be installed with `uv` on the H100 host:

```bash
uv sync --extra profiling --extra swebench --extra dev
```

## Small Synthetic Pipeline

This path requires no GPU and no SWE-bench download.

```bash
pytest -q
bash scripts/run_all_small.sh --run-id pr1_final
```

It runs:

1. Synthetic fork-join trace generation.
2. Context Segment Graph and DAG export.
3. GPU cache baselines.
4. Wafer replay for `naive_wafer`, `static_branch_pinning`, `wafer_fcfs`, `acd_only`, `acd_bes`, `acd_nisp`, and `full_agentweaver`.
5. Plot generation.

Outputs:

- `data/traces/<run_id>/*.jsonl`
- `data/processed/<run_id>/context_segments.csv`
- `data/processed/<run_id>/context_graph.json`
- `data/results/*.csv`
- `data/plots/*.pdf` and `.png`
- `data/results/pr1_final_report.md`

## Launch vLLM Prefix Server

In a separate terminal:

```bash
CUDA_VISIBLE_DEVICES=1 \
TENSOR_PARALLEL_SIZE=1 \
MODEL_PATH=/data2/model_zoo/Qwen2.5-7B-Instruct \
PORT=8000 \
MAX_MODEL_LEN=32768 \
GPU_MEMORY_UTILIZATION=0.90 \
bash scripts/launch_vllm_prefix.sh
```

The launch script uses `/data2/model_zoo/Qwen2.5-7B-Instruct` first and falls back to `Qwen/Qwen2.5-Coder-7B-Instruct` if the local directory is missing. The default PR2 profiling launch uses only GPU 1 with `CUDA_VISIBLE_DEVICES=1` and `TENSOR_PARALLEL_SIZE=1`. If your installed vLLM version rejects `--enable-prefix-caching`, remove that flag manually and record the version limitation in the profile report.

For a no-prefix baseline:

```bash
CUDA_VISIBLE_DEVICES=1 TENSOR_PARALLEL_SIZE=1 PORT=8000 bash scripts/launch_vllm_noprefix.sh
```

## PR2 H100 Profiling

Start vLLM manually, then run:

```bash
bash scripts/run_h100_profile.sh \
  --server http://localhost:8000/v1 \
  --metrics-url http://localhost:8000/metrics \
  --model qwen2.5-7b \
  --tokenizer-path /data2/model_zoo/Qwen2.5-7B-Instruct \
  --out-dir data/profiles \
  --run-id h100_qwen7b_prefix
```

The profile pipeline runs:

1. server health check
2. tokenizer-controlled sanity requests
3. length sweep
4. true async concurrency sweep
5. exact shared-prefix sweep
6. raw CSV merge
7. latency model fitting
8. profile report generation

Outputs:

- `data/profiles/vllm_sanity_raw.csv`
- `data/profiles/h100_profile_raw.csv`
- `data/profiles/h100_profile_length_raw.csv`
- `data/profiles/h100_profile_concurrency_raw.csv`
- `data/profiles/h100_profile_prefix_raw.csv`
- `data/profiles/h100_latency_model.json`
- `data/results/pr2_h100_profile_report.md`
- `data/results/h100_latency_fit_report.md`
- `data/plots/profile_fit_prefill.pdf`
- `data/plots/profile_fit_decode.pdf`
- `data/plots/profile_fit_prefix_reuse.pdf`
- `data/plots/profile_fit_concurrency.pdf`

If the server is unavailable, the script writes `H100_PROFILE = FAIL_SERVER_UNAVAILABLE` and does not create fake H100 measurements.

## vLLM Metrics

```bash
python -m agentweaver.profiling.collect_vllm_metrics \
  --metrics-url http://localhost:8000/metrics \
  --out data/profiles/vllm_metrics_profile.csv \
  --missing-out data/profiles/vllm_metrics_missing.json \
  --duration 10
```

Metric names vary by vLLM version. The collector uses fuzzy matching, writes raw rows to CSV, and records missing metric groups in JSON instead of treating missing metrics as zero.

## Real Agent-Like vLLM Trace

Before connecting a full SWE-agent harness, PR2 provides a controlled real local vLLM trace:

```bash
python -m agentweaver.workloads.real_agentlike_trace \
  --server http://localhost:8000/v1 \
  --model qwen2.5-7b \
  --tokenizer-path /data2/model_zoo/Qwen2.5-7B-Instruct \
  --instances 5 \
  --branch-fanout 4 \
  --out data/traces/real_agentlike_h100
```

This trace uses real vLLM latency for LLM0/LLM1 and real local subprocess timing for tools. The verifier is pseudo pass/fail for controlled replay only; it is not SWE-bench correctness.

Build the graph and replay:

```bash
python -m agentweaver.analysis.context_segment_graph \
  --trace-dir data/traces/real_agentlike_h100 \
  --out data/processed/real_agentlike_h100

python -m agentweaver.simulator.replay \
  --processed data/processed/real_agentlike_h100 \
  --wafer-config configs/wafer_6x6.yaml \
  --policy full_agentweaver \
  --out data/results/real_agentlike_full_agentweaver.csv
```

Outputs:

- `data/results/real_agentlike_trace_summary.csv`
- `data/results/real_agentlike_replay_summary.csv`
- `data/plots/real_agentlike_latency_breakdown.pdf`
- `data/plots/real_agentlike_context_reuse.pdf`

## SGLang Baseline

```bash
bash scripts/launch_sglang.sh
```

Then point profiling or agent execution at `http://localhost:30000/v1` if the installed SGLang version exposes the OpenAI-compatible route.

## SWE-Agent and Mini-SWE-Agent Traces

Post-process existing trajectory files:

```bash
python -m agentweaver.tracing.swe_trace_adapter \
  --traj path/to/run.traj \
  --out data/traces/<run_id>/<instance>_branch_0.jsonl

python -m agentweaver.tracing.mini_swe_trace_adapter \
  --traj path/to/run.traj.json \
  --out data/traces/<run_id>/<instance>_branch_0.jsonl
```

For direct LLM timing against an OpenAI-compatible local endpoint:

```bash
python -m agentweaver.tracing.llm_client_wrapper \
  --base-url http://localhost:8000/v1 \
  --model qwen2.5-7b \
  --prompt "Fix this SWE-bench issue" \
  --out-trace data/traces/manual/llm.jsonl
```

For tool timing:

```bash
python -m agentweaver.tracing.tool_wrapper --out-trace data/traces/manual/tool.jsonl pytest -q
```

The official SWE-bench harness can be attached by running the agent externally, preserving generated patches/test results in `.traj`, then converting with the adapters.

## Build Context Graph

```bash
python -m agentweaver.analysis.context_segment_graph \
  --trace-dir data/traces/<run_id> \
  --out data/processed/<run_id>
```

## GPU Cache Baselines

```bash
python -m agentweaver.simulator.gpu_cache_sim \
  --processed data/processed/<run_id> \
  --config configs/swebench_lite.yaml \
  --out data/results/gpu_cache.csv
```

Implemented simulated baselines:

- `GPU-LRU`
- `GPU-KVFlow-like`
- `GPU-Continuum-like`

Real GPU baselines (`vLLM no prefix`, `vLLM prefix`, `SGLang`) must be collected by running the servers and agent workloads; the artifact does not fabricate those measurements.

## Wafer Simulator

```bash
python -m agentweaver.simulator.replay \
  --processed data/processed/<run_id> \
  --wafer-config configs/wafer_6x6.yaml \
  --policy full_agentweaver \
  --out data/results/full_agentweaver.csv
```

Policies:

- `naive_wafer`
- `static_branch_pinning`
- `wafer_fcfs`
- `acd_only`
- `acd_bes`
- `acd_nisp`
- `full_agentweaver`

ACD can also be run directly:

```bash
python -m agentweaver.simulator.acd_mapping \
  --processed data/processed/<run_id> \
  --config configs/wafer_6x6.yaml \
  --out data/results/acd_mapping.csv
```

## Plot Generation

```bash
python -m agentweaver.plotting.plot_all \
  --results-dir data/results \
  --out-dir data/plots
```

The plotting code uses matplotlib only. It writes both PDF and PNG.

## What Is Real vs Simulated

- SWE-agent / mini-SWE-agent trajectories: real if converted from actual runs; synthetic if generated by `synthetic_fork_join.py`.
- Real agent-like trace: measured local vLLM LLM latency and measured local tool latency, but pseudo verifier correctness.
- Synthetic controlled trace: deterministic fork-join workload for mechanism sanity and negative controls.
- H100 latency model: measured only after successful `run_h100_profile.sh`; default analytic fallback otherwise.
- GPU vLLM/SGLang baselines: measured only when servers and real workloads are run.
- GPU cache policies and wafer policies: trace-driven simulation, not real wafer execution.

## Interpret Result Files

- `h100_profile_raw.csv`: measured vLLM profile rows; failed requests are retained.
- `h100_latency_model.json`: fitted H100 latency model used by replay; check `measured=true`.
- `pr2_h100_profile_report.md`: fixed-field gate report for H100 profile and latency fit quality.
- `real_agentlike_trace_summary.csv`: measured local vLLM/tool trace summary; not SWE-bench.
- `wafer_replay_summary.csv`: simulated wafer replay on traces.
- `ablation.csv`: real policy runs from the simulator, not copied full-policy output.

## Limitations

- Exact-prefix KV reuse only; no semantic KV reuse.
- No claim of exact Cerebras or commercial wafer simulation.
- Tool execution time is replayed from traces and not accelerated.
- Main scheduler does not alter agent policy or success probability; sibling cancellation occurs only after recorded verifier/test pass.
- The current multi-branch SWE script provides loader/adaptation scaffolding; full official harness orchestration should be attached externally for long runs.
- Default latency parameters are placeholders until H100 profiling data is collected.
- The artifact does not yet claim a complete real SWE-bench evaluation.
- The artifact does not execute on real wafer hardware.
- The mesh model is H100-calibrated trace-driven simulation, not exact Cerebras simulation.

## Artifact Checklist

- Trace schema and validation: `agentweaver/tracing/trace_schema.py`
- Prompt segmentation: `agentweaver/tracing/prompt_segmenter.py`
- Context Segment Graph: `agentweaver/analysis/context_segment_graph.py`
- ACD-Mapping: `agentweaver/simulator/acd_mapping.py`
- BES: `agentweaver/simulator/bes_scheduler.py`
- NISP: `agentweaver/simulator/nisp.py`
- Wafer replay: `agentweaver/simulator/replay.py`
- GPU baselines: `agentweaver/simulator/gpu_cache_sim.py`
- Synthetic workload: `agentweaver/workloads/synthetic_fork_join.py`
- Plotting: `agentweaver/plotting/plot_all.py`
- Tests: `tests/`

Run tests:

```bash
uv run pytest -q
```
