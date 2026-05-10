from __future__ import annotations

import os
import time
from typing import Any


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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

    def capture_git_diff(agent: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "agentweaver_patch_source": "git_diff_at_rollout_end",
            "agentweaver_patch_workspace_detected": False,
            "agentweaver_patch_bytes": 0,
            "agentweaver_patch_empty": True,
        }
        env = getattr(agent, "env", None)
        if env is None or not hasattr(env, "config"):
            info["agentweaver_patch_capture_error"] = "agent has no environment with config"
            return info
        configured_cwd = getattr(env.config, "cwd", "") or ""
        candidates = []
        for cwd in (configured_cwd, "/testbed", "."):
            if cwd and cwd not in candidates:
                candidates.append(cwd)
        workspace = ""
        last_error = ""
        for cwd in candidates:
            try:
                probe = orig_execute(env, {"command": "git rev-parse --show-toplevel"}, cwd=cwd, timeout=30)
            except Exception as exc:
                last_error = f"{cwd}: {exc}"
                continue
            if int(probe.get("returncode", -1)) == 0:
                workspace = str(probe.get("output", "")).strip().splitlines()[-1] if str(probe.get("output", "")).strip() else cwd
                break
            last_error = f"{cwd}: returncode={probe.get('returncode')}; {str(probe.get('output', '')).strip()[:200]}"
        if not workspace:
            info["agentweaver_patch_capture_error"] = f"git workspace not detected; {last_error}".strip()
            info["agentweaver_patch_empty_reason"] = "workspace not detected"
            return info

        info["agentweaver_patch_workspace_detected"] = True
        info["agentweaver_patch_workspace"] = workspace
        diff_result: dict[str, Any] = {}
        try:
            diff_result = orig_execute(env, {"command": "git diff --binary"}, cwd=workspace, timeout=120)
        except Exception as exc:
            info["agentweaver_patch_capture_error"] = f"git diff --binary failed: {exc}"
            info["agentweaver_patch_empty_reason"] = "git diff failed"
            info["agentweaver_patch_git_diff_exit_code"] = -1
            return info
        rc = int(diff_result.get("returncode", -1))
        patch = str(diff_result.get("output", "") or "")
        if rc != 0:
            try:
                fallback = orig_execute(env, {"command": "git diff"}, cwd=workspace, timeout=120)
                rc = int(fallback.get("returncode", -1))
                patch = str(fallback.get("output", "") or "")
            except Exception as exc:
                info["agentweaver_patch_capture_error"] = f"git diff fallback failed: {exc}"
                info["agentweaver_patch_empty_reason"] = "git diff failed"
                info["agentweaver_patch_git_diff_exit_code"] = -1
                return info
        info["agentweaver_patch_git_diff_exit_code"] = rc
        if rc != 0:
            info["agentweaver_patch_capture_error"] = f"git diff exit code {rc}"
            info["agentweaver_patch_empty_reason"] = "git diff failed"
            return info
        if patch.strip():
            info["agentweaver_patch"] = patch
            info["submission"] = patch
            info["agentweaver_patch_bytes"] = len(patch.encode("utf-8"))
            info["agentweaver_patch_empty"] = False
            info["agentweaver_patch_empty_reason"] = ""
        else:
            info["agentweaver_patch_empty"] = True
            info["agentweaver_patch_empty_reason"] = "git diff empty"
        return info

    def run_with_timing(self: Any, *args: Any, **kwargs: Any) -> dict:
        if not hasattr(self, "_agentweaver_rollout_start"):
            self._agentweaver_rollout_start = time.time()
        self._agentweaver_rollout_end = None
        try:
            return orig_run(self, *args, **kwargs)
        finally:
            self._agentweaver_rollout_end = time.time()
            if _env_enabled("AGENTWEAVER_CAPTURE_PATCH"):
                try:
                    self._agentweaver_patch_info = capture_git_diff(self)
                except Exception as exc:
                    self._agentweaver_patch_info = {
                        "agentweaver_patch_source": "git_diff_at_rollout_end",
                        "agentweaver_patch_workspace_detected": False,
                        "agentweaver_patch_bytes": 0,
                        "agentweaver_patch_empty": True,
                        "agentweaver_patch_empty_reason": "patch capture exception",
                        "agentweaver_patch_capture_error": str(exc),
                        "agentweaver_patch_git_diff_exit_code": -1,
                    }
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
        patch_info = getattr(self, "_agentweaver_patch_info", None)
        if isinstance(patch_info, dict):
            data.setdefault("info", {}).update(patch_info)
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
