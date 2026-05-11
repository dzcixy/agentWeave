from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from agentweaver.tracing.prompt_segmenter import segment_prompt
from agentweaver.tracing.tool_wrapper import classify_command
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace
from agentweaver.utils.hashing import prompt_hash, stable_hash
from agentweaver.utils.tokenization import count_tokens


TIMING_START_KEYS = ("timestamp_start", "start_time", "started_at", "start", "created_at")
TIMING_END_KEYS = ("timestamp_end", "end_time", "ended_at", "end", "completed_at")
LATENCY_KEYS = ("latency", "duration", "elapsed", "elapsed_seconds", "tool_latency")


def _load(path: str | Path) -> Any:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"text": line})
        return {"trajectory": rows, "source_format": "jsonl_or_text"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _first_text(obj: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return _text(obj[key])
    return ""


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _first_num(obj: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        val = _num(obj.get(key))
        if val is not None:
            return val
    return None


def _timing(obj: dict[str, Any]) -> tuple[float, float, float, bool]:
    info = obj.get("info")
    if isinstance(info, dict) and isinstance(info.get("timing"), dict):
        obj = {**obj, **info["timing"]}
    start = _first_num(obj, TIMING_START_KEYS)
    end = _first_num(obj, TIMING_END_KEYS)
    latency = _first_num(obj, LATENCY_KEYS)
    if start is not None and end is not None:
        return start, end, max(0.0, end - start), False
    if latency is not None:
        return 0.0, 0.0, max(0.0, latency), True
    return 0.0, 0.0, 0.0, True


def _load_timing_sidecar(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except Exception:
            continue
    return rows


def _steps(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [x if isinstance(x, dict) else {"text": x} for x in obj]
    if not isinstance(obj, dict):
        return [{"text": obj}]
    if isinstance(obj.get("messages"), list) and str(obj.get("trajectory_format", "")).startswith("mini-swe-agent"):
        return _mini_swe_message_steps(obj["messages"])
    for key in ("trajectory", "steps", "history", "turns", "messages_log", "events"):
        if isinstance(obj.get(key), list):
            return [x if isinstance(x, dict) else {"text": x} for x in obj[key]]
    if isinstance(obj.get("messages"), list):
        return [{"messages": obj["messages"], "response": obj.get("response") or obj.get("completion")}]
    return []


def _extract_text_command(content: str) -> str:
    patterns = [
        r"```mswea_bash_command\s*\n(.*?)\n```",
        r"<mswea_bash_command>(.*?)</mswea_bash_command>",
        r"```bash\s*\n(.*?)\n```",
    ]
    for pat in patterns:
        m = re.search(pat, content, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _parse_observation_content(content: str) -> tuple[str, int | None]:
    rc: int | None = None
    m_rc = re.search(r"<returncode>(.*?)</returncode>", content, flags=re.DOTALL | re.IGNORECASE)
    if m_rc:
        try:
            rc = int(m_rc.group(1).strip())
        except Exception:
            rc = None
    outputs: list[str] = []
    for tag in ("exception", "output", "output_head", "output_tail", "warning"):
        for m in re.finditer(fr"<{tag}>(.*?)</{tag}>", content, flags=re.DOTALL | re.IGNORECASE):
            txt = m.group(1).strip()
            if txt:
                outputs.append(txt)
    if outputs:
        return "\n".join(outputs), rc
    return content.strip(), rc


def _mini_msg(msg: Any) -> dict[str, Any]:
    if isinstance(msg, dict):
        content = msg.get("content")
        if content is None:
            content = msg.get("message") or msg.get("text") or msg.get("value")
        return {
            "role": str(msg.get("role") or msg.get("type") or msg.get("speaker") or "user"),
            "content": _text(content),
            "extra": msg.get("extra") if isinstance(msg.get("extra"), dict) else {},
        }
    return {"role": "user", "content": _text(msg), "extra": {}}


def _mini_swe_message_steps(messages: list[Any]) -> list[dict[str, Any]]:
    normalized_full = [_mini_msg(m) for m in messages]
    normalized = [{"role": m["role"], "content": m["content"]} for m in normalized_full]
    steps: list[dict[str, Any]] = []
    for i, msg in enumerate(normalized_full):
        if msg.get("role") != "assistant":
            continue
        assistant = msg.get("content", "")
        command = _extract_text_command(assistant)
        obs = ""
        rc: int | None = None
        tool_extra: dict[str, Any] = {}
        if i + 1 < len(normalized_full) and normalized_full[i + 1].get("role") == "user":
            obs, rc = _parse_observation_content(normalized_full[i + 1].get("content", ""))
            tool_extra = normalized_full[i + 1].get("extra") or {}
        assistant_extra = msg.get("extra") or {}
        step: dict[str, Any] = {
            "messages": normalized[:i],
            "assistant": assistant,
            "command": command,
            "observation": obs,
        }
        for key in ("timestamp_start", "timestamp_end", "latency", "ttft", "tpot"):
            if key in assistant_extra:
                step[key] = assistant_extra[key]
        if command and tool_extra:
            step["tool_timing"] = {
                k: tool_extra[k]
                for k in ("timestamp_start", "timestamp_end", "tool_latency", "latency")
                if k in tool_extra
            }
            for key in (
                "patch_snapshot_available",
                "modified_files_count",
                "untracked_files_count",
                "git_diff_stat_bytes",
                "patch_hash_prefix",
                "file_modification_seen",
            ):
                if key in tool_extra:
                    step[key] = tool_extra[key]
        if rc is not None:
            step["returncode"] = rc
        steps.append(step)
    return steps


def _messages(value: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in _as_list(value):
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type") or msg.get("speaker")
            content = msg.get("content")
            if content is None:
                content = msg.get("message") or msg.get("text") or msg.get("value")
            if content is not None:
                out.append({"role": str(role or "user"), "content": _text(content)})
        elif msg is not None:
            out.append({"role": "user", "content": _text(msg)})
    return out


def _context_prefix(obj: dict[str, Any], model: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = _first_text(obj, ("system_prompt", "system", "instructions"))
    if system:
        messages.append({"role": "system", "content": system})
    tools = _first_text(obj, ("tool_schema", "tools", "available_tools"))
    if tools:
        messages.append({"role": "system", "content": f"Available tools:\n{tools}"})
    task = _first_text(obj, ("problem_statement", "problem", "issue", "task", "prompt"))
    if task:
        messages.append({"role": "user", "content": f"Problem statement:\n{task}"})
    repo = _first_text(obj, ("repo", "repository", "repo_context", "file_context"))
    if repo:
        messages.append({"role": "user", "content": f"Repository context:\n{repo}"})
    return messages


def _llm_messages(step: dict[str, Any], prefix: list[dict[str, str]], history: list[dict[str, str]]) -> list[dict[str, str]]:
    if _command(step) and not any(
        key in step
        for key in (
            "messages",
            "model_messages",
            "prompt_messages",
            "prompt",
            "model_query",
            "query",
            "llm_input",
            "input",
            "thought",
            "reasoning",
            "response",
            "completion",
            "model_response",
            "assistant",
            "assistant_message",
        )
    ):
        return []
    explicit = _messages(step.get("messages") or step.get("model_messages") or step.get("prompt_messages"))
    if explicit:
        return prefix + explicit
    prompt = _first_text(step, ("prompt", "model_query", "query", "llm_input", "input"))
    if prompt:
        return prefix + history + [{"role": "user", "content": prompt}]
    thought = _first_text(step, ("thought", "reasoning"))
    action = _first_text(step, ("action", "assistant", "response", "completion", "model_response"))
    if thought or action:
        content = "\n".join(x for x in (thought, action) if x)
        return prefix + history + [{"role": "assistant", "content": content}]
    return []


def _assistant_output(step: dict[str, Any]) -> str:
    out = _first_text(step, ("response", "completion", "model_response", "assistant", "assistant_message"))
    if out:
        return out
    thought = _first_text(step, ("thought", "reasoning"))
    action = _first_text(step, ("action",))
    return "\n".join(x for x in (thought, action) if x)


def _command_from_tool_call(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("command", "cmd", "code", "arguments", "input"):
            if key in value and value[key] not in (None, ""):
                return _text(value[key])
        return _text(value)
    return _text(value)


def _looks_like_shell(action: str) -> bool:
    s = action.strip()
    if not s:
        return False
    if s.startswith(("pytest", "python", "uv ", "pip ", "git ", "grep", "rg ", "sed ", "cat ", "ls ", "find ")):
        return True
    if any(tok in s for tok in ("apply_patch", "bash ", "sh ", "make ", "npm ", "tox ", "mvn ", "cargo ")):
        return True
    return False


def _command(step: dict[str, Any]) -> str:
    for key in ("command", "cmd", "shell_command"):
        if step.get(key):
            return _text(step[key])
    tool_call = _command_from_tool_call(step.get("tool_call") or step.get("tool_calls"))
    if tool_call:
        return tool_call
    action = _first_text(step, ("action",))
    return action if _looks_like_shell(action) else ""


def _observation(step: dict[str, Any]) -> str:
    return _first_text(step, ("observation", "tool_output", "output", "stdout", "stderr", "result"))


def _exit_code(step: dict[str, Any]) -> int | None:
    for key in ("exit_code", "returncode", "return_code", "status_code"):
        value = step.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except Exception:
            return None
    return None


def _tool_timing(step: dict[str, Any]) -> tuple[float, float, float, bool]:
    timing = step.get("tool_timing")
    if isinstance(timing, dict):
        return _timing(timing)
    if "assistant" in step and any(k in step for k in ("timestamp_start", "timestamp_end", "latency")):
        return 0.0, 0.0, 0.0, True
    return _timing(step)


def _apply_timing_sidecar(events: list[Event], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    unused = set(range(len(rows)))
    by_hash: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        h = str(row.get("prompt_hash") or "")
        if h:
            by_hash.setdefault(h, []).append(i)
    order_indices = list(range(len(rows)))
    order_cursor = 0
    for ev in [e for e in events if e.node_type == "llm"]:
        if (not ev.timing_missing) and ev.timestamp_start and ev.timestamp_end:
            continue
        idx: int | None = None
        for cand in by_hash.get(ev.prompt_hash, []):
            if cand in unused:
                idx = cand
                break
        if idx is None:
            while order_cursor < len(order_indices) and order_indices[order_cursor] not in unused:
                order_cursor += 1
            if order_cursor < len(order_indices):
                idx = order_indices[order_cursor]
        if idx is None:
            continue
        unused.discard(idx)
        row = rows[idx]
        start = _num(row.get("timestamp_start"))
        end = _num(row.get("timestamp_end"))
        latency = _num(row.get("latency"))
        if start is None or end is None:
            continue
        ev.timestamp_ready = start
        ev.timestamp_start = start
        ev.timestamp_end = end
        ev.latency = max(0.0, latency if latency is not None else end - start)
        ev.ttft = _num(row.get("ttft"))
        ev.tpot = _num(row.get("tpot"))
        if row.get("input_tokens") not in (None, ""):
            try:
                ev.input_tokens = int(row["input_tokens"])
                ev.context_length = max(ev.context_length, ev.input_tokens)
            except Exception:
                pass
        if row.get("output_tokens") not in (None, ""):
            try:
                ev.output_tokens = int(row["output_tokens"])
            except Exception:
                pass
        ev.timing_missing = False


def _success(obj: dict[str, Any]) -> tuple[str, bool | None]:
    for key in ("success", "resolved", "passed", "is_resolved"):
        if key in obj and obj[key] is not None:
            val = obj[key]
            if isinstance(val, bool):
                return ("pass" if val else "fail"), val
            if isinstance(val, str):
                low = val.lower()
                if low in {"true", "pass", "passed", "resolved", "success", "successfully_resolved"}:
                    return "pass", True
                if low in {"false", "fail", "failed", "unresolved", "error"}:
                    return "fail", False
    status = _first_text(obj, ("status", "final_status", "verifier_result"))
    if status:
        low = status.lower()
        if low in {"pass", "passed", "resolved", "success", "successfully_resolved"}:
            return "pass", True
        if low in {"fail", "failed", "unresolved", "error"}:
            return "fail", False
    return "unknown", None


def _patch(obj: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    patch = _first_text(obj, ("patch", "diff", "final_patch"))
    if patch:
        return patch
    info = obj.get("info")
    if isinstance(info, dict):
        patch = _first_text(info, ("submission", "patch", "diff", "final_patch"))
        if patch:
            return patch
    for step in reversed(steps):
        patch = _first_text(step, ("patch", "diff"))
        if patch:
            return patch
    return ""


def _add_history(history: list[dict[str, str]], role: str, content: str) -> None:
    if content:
        history.append({"role": role, "content": content})


def _node_id(instance_id: str, branch_id: str, kind: str, index: int) -> str:
    return f"{instance_id}:{branch_id}:{kind}{index}"


def convert_swe_traj(
    path: str | Path,
    out: str | Path | None = None,
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    branch_id: str | None = None,
    parent_branch_id: str | None = None,
    rollout_id: str | None = None,
    instance_id: str | None = None,
    timing_sidecar: str | Path | None = None,
) -> Trace:
    loaded = _load(path)
    obj = loaded if isinstance(loaded, dict) else {"trajectory": loaded}
    steps = _steps(loaded)
    instance_id = instance_id or obj.get("instance_id") or obj.get("task_id") or Path(path).stem.replace(".traj", "")
    session_id = obj.get("run_id") or obj.get("session_id") or stable_hash(str(path))
    branch_id = branch_id or obj.get("branch_id") or "b0"
    parent_branch_id = parent_branch_id if parent_branch_id is not None else obj.get("parent_branch_id")
    rollout_id = rollout_id or obj.get("rollout_id")
    prefix = _context_prefix(obj, model)
    history: list[dict[str, str]] = []
    events: list[Event] = []
    step_id = 0
    for i, step in enumerate(steps):
        messages = _llm_messages(step, prefix, history)
        assistant_output = _assistant_output(step)
        if messages:
            rendered = "\n".join(str(m.get("content", "")) for m in messages)
            start, end, latency, timing_missing = _timing(step)
            segs = segment_prompt(messages, metadata={"model": model, "instance_id": instance_id, "branch_id": branch_id})
            events.append(
                Event(
                    event_id=f"{instance_id}:{branch_id}:llm{i}",
                    session_id=str(session_id),
                    instance_id=str(instance_id),
                    branch_id=str(branch_id),
                    parent_branch_id=parent_branch_id,
                    step_id=step_id,
                    node_id=_node_id(str(instance_id), str(branch_id), "llm", i),
                    node_type="llm",
                    role="worker",
                    model=model,
                    timestamp_ready=start,
                    timestamp_start=start,
                    timestamp_end=end,
                    latency=latency,
                    input_tokens=count_tokens(rendered, model),
                    output_tokens=count_tokens(assistant_output, model),
                    context_length=count_tokens(rendered, model),
                    prompt_hash=prompt_hash(rendered),
                    shared_prefix_id=segs[0].segment_id if segs else None,
                    context_segments=[ContextSegmentRef(s.segment_id, s.segment_type, s.start_pos, s.length) for s in segs],
                    context_segment_defs=segs,
                    timing_missing=timing_missing,
                    rollout_id=rollout_id,
                )
            )
            step_id += 1
            _add_history(history, "assistant", assistant_output)
        cmd = _command(step)
        if cmd:
            start, end, latency, timing_missing = _tool_timing(step)
            obs = _observation(step)
            events.append(
                Event(
                    event_id=f"{instance_id}:{branch_id}:tool{i}",
                    session_id=str(session_id),
                    instance_id=str(instance_id),
                    branch_id=str(branch_id),
                    parent_branch_id=parent_branch_id,
                    step_id=step_id,
                    node_id=_node_id(str(instance_id), str(branch_id), "tool", i),
                    node_type="tool",
                    timestamp_ready=start,
                    timestamp_start=start,
                    timestamp_end=end,
                    latency=latency,
                    tool_type=classify_command(cmd),
                    command=cmd,
                    tool_latency=latency if not timing_missing or latency else None,
                    observation_tokens=count_tokens(obs, model),
                    exit_code=_exit_code(step),
                    patch_snapshot_available=bool(step.get("patch_snapshot_available", False)),
                    modified_files_count=int(_num(step.get("modified_files_count")) or 0),
                    untracked_files_count=int(_num(step.get("untracked_files_count")) or 0),
                    git_diff_stat_bytes=int(_num(step.get("git_diff_stat_bytes")) or 0),
                    patch_hash_prefix=str(step.get("patch_hash_prefix") or "") or None,
                    file_modification_seen=bool(step.get("file_modification_seen", False)),
                    timing_missing=timing_missing,
                    rollout_id=rollout_id,
                )
            )
            step_id += 1
            if obs:
                label = "Test log" if "pytest" in cmd.lower() or "traceback" in obs.lower() else "Observation"
                _add_history(history, "tool", f"{label}:\n{obs}")
    verifier_result, success = _success(obj)
    patch = _patch(obj, steps)
    if events or verifier_result != "unknown" or patch:
        start, end, latency, timing_missing = _timing(obj)
        if start == 0.0 and end == 0.0 and events and not all(e.timing_missing for e in events):
            observed_ends = [e.timestamp_end for e in events if e.timestamp_end > 0]
            if observed_ends:
                start = end = max(observed_ends)
                latency = 0.0
                timing_missing = False
        patch_segments = (
            segment_prompt(rendered_prompt=patch, metadata={"model": model, "instance_id": instance_id, "branch_id": branch_id})
            if patch
            else []
        )
        events.append(
            Event(
                event_id=f"{instance_id}:{branch_id}:verify",
                session_id=str(session_id),
                instance_id=str(instance_id),
                branch_id=str(branch_id),
                parent_branch_id=parent_branch_id,
                step_id=9999,
                node_id=_node_id(str(instance_id), str(branch_id), "verify", 0),
                node_type="verifier",
                timestamp_ready=start,
                timestamp_start=start,
                timestamp_end=end,
                latency=latency,
                verifier_result=verifier_result,  # type: ignore[arg-type]
                success=success,
                context_segments=[
                    ContextSegmentRef(s.segment_id, s.segment_type, s.start_pos, s.length) for s in patch_segments
                ],
                context_segment_defs=patch_segments,
                patch_hash=stable_hash(patch) if patch else None,
                timing_missing=timing_missing,
                rollout_id=rollout_id,
            )
        )
    metadata = {
        "framework": obj.get("framework") or "swe-agent-adapter",
        "source": str(path),
        "timestamp": time.time(),
        "instance_id": str(instance_id),
        "branch_id": str(branch_id),
        "parent_branch_id": parent_branch_id,
        "rollout_id": rollout_id,
        "sample_fixture": bool(obj.get("sample_fixture", False)),
        "sample_fixture_note": obj.get("sample_fixture_note", ""),
    }
    _apply_timing_sidecar(events, _load_timing_sidecar(timing_sidecar))
    tr = Trace(metadata, sorted(events, key=lambda e: (e.branch_id, e.step_id, e.node_id)))
    if out:
        tr.to_jsonl(out)
    return tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--branch-id")
    ap.add_argument("--parent-branch-id")
    ap.add_argument("--rollout-id")
    ap.add_argument("--instance-id")
    ap.add_argument("--timing-sidecar")
    args = ap.parse_args()
    tr = convert_swe_traj(
        args.traj,
        args.out,
        model=args.model,
        branch_id=args.branch_id,
        parent_branch_id=args.parent_branch_id,
        rollout_id=args.rollout_id,
        instance_id=args.instance_id,
        timing_sidecar=args.timing_sidecar,
    )
    print(f"wrote {len(tr.events)} events to {args.out}")


if __name__ == "__main__":
    main()
