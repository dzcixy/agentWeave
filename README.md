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

- Local first: `/data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct`
- Fallback name: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Optional larger model: `Qwen/Qwen2.5-Coder-14B-Instruct`

## Installation

```bash
cd /home/duzc/data/agent_trace/AgentWeaver
uv sync
```

Optional profiling/SWE-bench dependencies:

```bash
uv sync --extra profiling --extra swebench --extra dev
```

## Small Synthetic Pipeline

This path requires no GPU and no SWE-bench download.

```bash
bash scripts/run_all_small.sh
```

It runs:

1. Synthetic fork-join trace generation.
2. Context Segment Graph and DAG export.
3. GPU cache baselines.
4. Wafer replay for `naive_wafer`, `acd_only`, `acd_bes`, `acd_nisp`, and `full_agentweaver`.
5. Plot generation.

Outputs:

- `data/traces/<run_id>/*.jsonl`
- `data/processed/<run_id>/context_segments.csv`
- `data/processed/<run_id>/context_graph.json`
- `data/results/*.csv`
- `data/plots/*.pdf` and `.png`

## H100 Profiling

Start vLLM manually, then profile through the OpenAI-compatible endpoint:

```bash
bash scripts/launch_vllm_noprefix.sh
bash scripts/run_h100_profile.sh --server http://localhost:8000/v1 --model qwen-coder-7b
```

The profiler writes:

- `data/profiles/h100_profile_raw.csv`
- `data/profiles/h100_latency_model.json`
- `data/plots/profile_fit_prefill.pdf`
- `data/plots/profile_fit_decode.pdf`

If a vLLM version changes server flags, keep the model/port/max-len settings and remove the incompatible flag. Prefix caching is launched with:

```bash
bash scripts/launch_vllm_prefix.sh
```

## vLLM Metrics

```bash
python -m agentweaver.profiling.collect_vllm_metrics \
  --metrics-url http://localhost:8000/metrics \
  --out data/profiles/vllm_metrics.csv
```

Metric names vary by vLLM version. Missing expected metrics are logged as warnings and do not stop collection.

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
  --model qwen-coder-7b \
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
- Tool latency and pass/fail: real when present in trajectories; synthetic only in small sanity traces.
- H100 latency model: measured after `run_h100_profile.sh`; default analytic fallback otherwise.
- GPU vLLM/SGLang baselines: measured only when servers and real workloads are run.
- GPU cache policies and wafer policies: trace-driven simulation.

## Limitations

- Exact-prefix KV reuse only; no semantic KV reuse.
- No claim of exact Cerebras or commercial wafer simulation.
- Tool execution time is replayed from traces and not accelerated.
- Main scheduler does not alter agent policy or success probability; sibling cancellation occurs only after recorded verifier/test pass.
- The current multi-branch SWE script provides loader/adaptation scaffolding; full official harness orchestration should be attached externally for long runs.
- Default latency parameters are placeholders until H100 profiling data is collected.

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
