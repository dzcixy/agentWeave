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

- Local first: `/data2/model_zoo/Qwen2.5-Coder-7B-Instruct`
- Served model name: `qwen-coder-7b`
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

## Launch vLLM Servers For PR2-v2

No-prefix server:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct \
PORT=8000 \
MAX_MODEL_LEN=32768 \
bash scripts/launch_vllm_noprefix.sh
```

Prefix-enabled server:

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_PATH=/data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct \
PORT=8001 \
MAX_MODEL_LEN=32768 \
bash scripts/launch_vllm_prefix.sh
```

## PR2 H100 Profiling

Start vLLM manually, then run:

```bash
bash scripts/run_h100_profile_pr2_v2.sh \
  --noprefix-server http://localhost:8000/v1 \
  --noprefix-metrics-url http://localhost:8000/metrics \
  --prefix-server http://localhost:8001/v1 \
  --prefix-metrics-url http://localhost:8001/metrics \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct \
  --run-id pr2_v2
```

The profile pipeline runs:

1. server health check
2. tokenizer-controlled sanity requests
3. length sweep
4. true async concurrency sweep
5. no-prefix vs prefix-server shared-prefix comparison
6. strict raw metrics audit
7. latency model fitting with interpolation fallback
8. PR2-v2 report generation

Outputs:

- `data/profiles/h100_profile_raw_pr2_v2.csv`
- `data/profiles/h100_profile_length_pr2_v2.csv`
- `data/profiles/h100_profile_concurrency_pr2_v2.csv`
- `data/profiles/h100_profile_prefix_pr2_v2.csv`
- `data/profiles/h100_latency_model_pr2_v2.json`
- `data/results/pr2_v2_report.md`
- `data/results/h100_latency_fit_report_pr2_v2.md`
- `data/results/vllm_metrics_source_audit_pr2_v2.md`
- `data/results/prefix_reuse_effect_pr2_v2.csv`
- `data/plots/profile_fit_prefill_pr2_v2.pdf`
- `data/plots/profile_fit_decode_pr2_v2.pdf`
- `data/plots/profile_fit_prefix_reuse_pr2_v2.pdf`
- `data/plots/profile_fit_concurrency_pr2_v2.pdf`

If either server is unavailable, the script writes `H100_PROFILE = FAIL` and does not create fake H100 measurements.

## vLLM Metrics

```bash
python -m agentweaver.profiling.collect_vllm_metrics \
  --metrics-url http://localhost:8000/metrics \
  --out data/profiles/vllm_metrics_profile.csv \
  --missing-out data/profiles/vllm_metrics_missing.json \
  --duration 10
```

PR2-v2 separates no-prefix and prefix-enabled servers, uses streaming TTFT for prefill fitting, records actual output tokens, and writes a strict metric source audit to `data/results/vllm_metrics_source_audit_pr2_v2.md`.

## Real Agent-Like vLLM Trace

Before connecting a full SWE-agent harness, PR2 provides a controlled real local vLLM trace:

```bash
bash scripts/run_real_agentlike_all_policies_pr2_v2.sh \
  --server http://localhost:8001/v1 \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct \
  --instances 5 \
  --branch-fanout 4 \
  --run-id real_agentlike_pr2_v2
```

This trace uses real vLLM latency for LLM0/LLM1 and real local subprocess timing for tools. The verifier is pseudo pass/fail for controlled replay only; it is not SWE-bench correctness.

Outputs:

- `data/results/real_agentlike_replay_all_policies_pr2_v2.csv`
- `data/results/real_agentlike_policy_comparison_pr2_v2.csv`
- `data/results/real_agentlike_trace_summary_pr2_v2.csv`
- `data/plots/real_agentlike_latency_breakdown_pr2_v2.pdf`
- `data/plots/real_agentlike_context_reuse_pr2_v2.pdf`

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

## PR3: mini-SWE-agent / SWE-agent real trace collection

PR3 collects small-scale real coding-agent traces from mini-SWE-agent/SWE-agent on SWE-bench Lite. The goal is workload characterization, not solved rate: shared context reuse, branch skew, and tool-stall/resume behavior. The PR2 controlled `real_agentlike_h100` workload is not SWE-bench and is not used for PR3 gates. The scripts never generate fake SWE traces, timestamps, patches, or verifier pass/fail labels.

Prerequisites:

- local OpenAI-compatible vLLM prefix server, normally `http://localhost:8001/v1`;
- mini-SWE-agent and LiteLLM;
- SWE-bench and HuggingFace `datasets`;
- Docker only for optional official SWE-bench harness evaluation;
- enough storage for SWE-bench images/repos. Start small before scaling.

Environment audit:

```bash
bash scripts/check_pr3_env.sh \
  --server http://localhost:8001/v1 \
  --metrics-url http://localhost:8001/metrics \
  --tokenizer-path /data2/model_zoo/Qwen2.5-Coder-7B-Instruct
```

Install/connect mini-SWE-agent, SWE-bench, datasets, and LiteLLM, and generate the local vLLM registry/config:

```bash
bash scripts/setup_pr3_miniswe_swebench.sh
```

The setup captures `mini-extra swebench --help` in `data/results/mini_extra_swebench_help.txt`. If the local mini-SWE-agent CLI changes, runner scripts fail clearly instead of fabricating trajectories. Advanced users can set `MINI_SWE_AGENT_RUN_CMD` with placeholders such as `{instance_id}`, `{run_out}`, `{server}`, `{hosted_model}`, `{max_steps}`, `{seed}`, and `{temperature}`.

The generated mini-SWE config uses the official `swebench_backticks` prompt format plus `litellm_textbased` for local vLLM. This keeps a real LLM-to-shell loop without requiring the vLLM server to be launched with OpenAI auto tool-call parsing.

Start the vLLM prefix server:

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_PATH=/data2/model_zoo/Qwen2.5-Coder-7B-Instruct \
PORT=8001 \
MAX_MODEL_LEN=32768 \
bash scripts/launch_vllm_prefix.sh
```

Select SWE-bench Lite instances:

```bash
python -m agentweaver.workloads.select_swebench_instances \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --num-instances 5 \
  --out data/results/mini_swe_lite5_instances.txt

python -m agentweaver.workloads.select_swebench_instances \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --num-instances 10 \
  --out data/results/mini_swe_lite10_instances.txt
```

Run Lite-5 single rollout:

```bash
bash scripts/run_mini_swe_trace_pr3.sh \
  --server http://localhost:8001/v1 \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen2.5-Coder-7B-Instruct \
  --instance-list data/results/mini_swe_lite5_instances.txt \
  --num-instances 5 \
  --max-steps 10 \
  --run-id mini_swe_lite5 \
  --run-real
```

Run Lite-10 with four independent rollouts per instance:

```bash
bash scripts/run_mini_swe_multibranch_pr3.sh \
  --server http://localhost:8001/v1 \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen2.5-Coder-7B-Instruct \
  --instance-list data/results/mini_swe_lite10_instances.txt \
  --num-instances 10 \
  --rollouts 4 \
  --max-steps 10 \
  --run-id mini_swe_lite10_r4 \
  --run-real
```

PR3-v2 timed collection records real wall-clock timing for mini-SWE-agent runs. The timed runner imports an AgentWeaver mini-SWE timing patch before invoking `mini-extra`; this writes LLM call timing, shell/tool timing when available, and rollout JCT into the raw trajectory. It also launches an OpenAI-compatible timing proxy at `http://localhost:8010/v1` when needed and stores the LLM timing sidecar under `data/logs/vllm_proxy_timing_<run_id>.jsonl`.

Run timed Lite-5:

```bash
bash scripts/run_mini_swe_trace_pr3_timed.sh \
  --server http://localhost:8010/v1 \
  --backend-server http://localhost:8001/v1 \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen2.5-Coder-7B-Instruct \
  --instance-list data/results/mini_swe_lite5_instances.txt \
  --num-instances 5 \
  --max-steps 10 \
  --run-id mini_swe_lite5_timed \
  --run-real
```

Run timed Lite-10 R4:

```bash
bash scripts/run_mini_swe_multibranch_pr3_timed.sh \
  --server http://localhost:8010/v1 \
  --backend-server http://localhost:8001/v1 \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen2.5-Coder-7B-Instruct \
  --instance-list data/results/mini_swe_lite10_instances.txt \
  --num-instances 10 \
  --rollouts 4 \
  --max-steps 10 \
  --run-id mini_swe_lite10_r4_timed \
  --run-real
```

If mini-SWE-agent was run externally, timed adapter-only conversion can merge a real LLM timing sidecar without inventing missing tool timing:

```bash
bash scripts/run_mini_swe_multibranch_pr3_timed.sh \
  --traj-root data/raw_trajs/mini_swe_lite10_r4_timed \
  --run-id mini_swe_lite10_r4_timed \
  --timing-sidecar data/logs/vllm_proxy_timing_mini_swe_lite10_r4_timed.jsonl \
  --rollouts 4
```

Adapter-only mode for externally collected real trajectories:

```bash
bash scripts/run_mini_swe_trace_pr3.sh \
  --traj-dir data/raw_trajs/mini_swe_lite5 \
  --run-id mini_swe_lite5 \
  --server http://localhost:8001/v1 \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen2.5-Coder-7B-Instruct
```

The script converts `.traj`, `.traj.json`, or JSON trajectory files into `data/traces/mini_swe_lite5/`, validates the trace schema, builds context reuse summaries, and runs all AgentWeaver replay policies with the PR2 H100 latency model.

For multi-rollout branch analysis, preserve this trajectory layout:

```text
data/raw_trajs/mini_swe_lite10_r4/<instance_id>/rollout_0.traj.json
data/raw_trajs/mini_swe_lite10_r4/<instance_id>/rollout_1.traj.json
data/raw_trajs/mini_swe_lite10_r4/<instance_id>/rollout_2.traj.json
data/raw_trajs/mini_swe_lite10_r4/<instance_id>/rollout_3.traj.json
```

Convert and replay:

```bash
bash scripts/run_mini_swe_multibranch_pr3.sh \
  --traj-root data/raw_trajs/mini_swe_lite10_r4 \
  --out-dir data/traces/mini_swe_lite10_r4 \
  --server http://localhost:8001/v1 \
  --model qwen-coder-7b \
  --tokenizer-path /data2/model_zoo/Qwen2.5-Coder-7B-Instruct \
  --rollouts 4 \
  --max-steps 10
```

Optional official SWE-bench harness evaluation uses only real generated patches:

```bash
bash scripts/run_swebench_eval_pr3.sh \
  --predictions data/results/mini_swe_lite5_timed_predictions.jsonl \
  --run-id mini_swe_lite5_timed_agentweaver \
  --max-workers 2
```

Expected PR3 outputs include:

- `data/results/pr3_env_report.md`
- `data/results/pr3_setup_report.md`
- `data/results/mini_swe_lite5_predictions.jsonl`
- `data/results/mini_swe_lite5_trace_summary.csv`
- `data/results/mini_swe_lite5_replay_all_policies.csv`
- `data/results/mini_swe_lite5_policy_comparison.csv`
- `data/results/mini_swe_lite10_r4_branch_summary.csv`
- `data/results/mini_swe_lite10_r4_replay_all_policies.csv`
- `data/results/pr3_report.md`
- `data/results/pr3_v2_report.md`

Unknown verifier results mean the trajectory did not contain an official pass/fail result. They are tracked separately and must not be reported as solved rate. Only official harness verifier output may be used for solved-rate claims.

PR3 limitations:

- no full SWE-bench Lite 300 run is launched by these scripts;
- missing timestamps are marked as `timing_missing=true` and are not fabricated;
- unknown verifier results stay unknown; they are not converted into pass/fail;
- replay uses the H100 latency model from token counts when measured agent timing is unavailable;
- solved rate is not reported unless the official SWE-bench harness actually runs;
- sample fixtures under `tests/fixtures/` are unit-test fixtures, not experimental data.

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
- H100 latency model: measured only after successful `run_h100_profile_pr2_v2.sh`; default analytic fallback otherwise.
- GPU vLLM/SGLang baselines: measured only when servers and real workloads are run.
- GPU cache policies and wafer policies: trace-driven simulation, not real wafer execution.

## Interpret Result Files

- `h100_profile_raw_pr2_v2.csv`: measured vLLM profile rows; failed requests are retained.
- `h100_latency_model_pr2_v2.json`: fitted H100 latency model used by replay; check `measured=true`.
- `pr2_v2_report.md`: fixed-field gate report for H100 profile and latency fit quality.
- `real_agentlike_trace_summary_pr2_v2.csv`: measured local vLLM/tool trace summary; not SWE-bench.
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
