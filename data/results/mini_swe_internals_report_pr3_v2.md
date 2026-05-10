# mini-SWE-agent Internals Report PR3-v2

package path = /home/duzc/data/agent_trace/AgentWeaver/.venv/lib/python3.12/site-packages/minisweagent
CLI command path = /home/duzc/data/agent_trace/AgentWeaver/.venv/bin/mini-extra
LLM call function/module = /home/duzc/data/agent_trace/AgentWeaver/.venv/lib/python3.12/site-packages/minisweagent/models/litellm_model.py: LitellmModel.query() / LitellmModel._query(); local vLLM config uses litellm_textbased, which inherits LitellmModel.query()
tool execution function/module = /home/duzc/data/agent_trace/AgentWeaver/.venv/lib/python3.12/site-packages/minisweagent/environments/docker.py: DockerEnvironment.execute()
trajectory output format = JSON trajectory with top-level messages/info; DefaultAgent.serialize() writes messages and info, DefaultAgent.save() writes config.output_path
direct timing patch possible = true

## Timing Plan

TIMING_INSTRUMENTATION = PATCHED_MINISWE
VLLM_PROXY_FALLBACK_AVAILABLE = true
TOOL_TIMING_AVAILABLE = true when DockerEnvironment.execute() is used

The PR3-v2 runner imports agentweaver.tracing.miniswe_timing_patch before invoking mini-extra. The patch records real wall-clock timestamp_start/timestamp_end/latency for each LitellmModel.query() result and each DockerEnvironment.execute() output. It also records rollout_start/rollout_end/measured_agent_jct in trajectory info. It does not alter prompts, commands, observations, return codes, model responses, or verifier results.

If the direct patch is unavailable for an upstream mini-SWE-agent version, scripts can route LLM calls through agentweaver.profiling.vllm_timing_proxy and merge the sidecar timing log by prompt_hash/request order. Tool timing remains missing in that fallback and is not fabricated.
