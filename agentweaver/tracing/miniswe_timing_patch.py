from __future__ import annotations

import time
from typing import Any


def apply_patch() -> None:
    """Install process-local mini-SWE-agent timing hooks.

    The hooks only add timing metadata to the trajectory. They do not change the
    prompt, action parsing, model response, shell command, return code, or
    observation text.
    """

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    if getattr(DefaultAgent, "_agentweaver_timing_patched", False):
        return

    orig_run = DefaultAgent.run
    orig_serialize = DefaultAgent.serialize
    orig_model_query = LitellmModel.query
    orig_execute = DockerEnvironment.execute

    def run_with_timing(self: Any, *args: Any, **kwargs: Any) -> dict:
        if not hasattr(self, "_agentweaver_rollout_start"):
            self._agentweaver_rollout_start = time.time()
        self._agentweaver_rollout_end = None
        try:
            return orig_run(self, *args, **kwargs)
        finally:
            self._agentweaver_rollout_end = time.time()
            try:
                self.save(self.config.output_path)
            except Exception:
                pass

    def serialize_with_timing(self: Any, *extra_dicts: Any) -> dict:
        data = orig_serialize(self, *extra_dicts)
        start = getattr(self, "_agentweaver_rollout_start", None)
        end = getattr(self, "_agentweaver_rollout_end", None) or time.time()
        if start:
            data.setdefault("info", {}).setdefault("timing", {})
            data["info"]["timing"].update(
                {
                    "timing_source": "patched_miniswe",
                    "rollout_start": start,
                    "rollout_end": end,
                    "measured_agent_jct": max(0.0, end - start),
                }
            )
        return data

    def query_with_timing(self: Any, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        start = time.time()
        try:
            message = orig_model_query(self, messages, **kwargs)
        finally:
            end = time.time()
        extra = message.setdefault("extra", {})
        extra.update(
            {
                "timing_source": "patched_miniswe",
                "timestamp_start": start,
                "timestamp_end": end,
                "latency": max(0.0, end - start),
            }
        )
        return message

    def execute_with_timing(self: Any, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        start = time.time()
        output = orig_execute(self, action, cwd=cwd, timeout=timeout)
        end = time.time()
        timing = {
            "timing_source": "patched_miniswe",
            "timestamp_start": start,
            "timestamp_end": end,
            "tool_latency": max(0.0, end - start),
            "latency": max(0.0, end - start),
        }
        output.update(timing)
        output.setdefault("extra", {}).update(timing)
        return output

    DefaultAgent.run = run_with_timing
    DefaultAgent.serialize = serialize_with_timing
    LitellmModel.query = query_with_timing
    DockerEnvironment.execute = execute_with_timing
    DefaultAgent._agentweaver_timing_patched = True


apply_patch()
