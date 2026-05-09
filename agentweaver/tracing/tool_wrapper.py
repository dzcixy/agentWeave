from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from agentweaver.tracing.trace_schema import Event, Trace
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.tokenization import count_tokens


def classify_command(command: str) -> str:
    low = command.lower()
    if "pytest" in low or "unittest" in low or "tox" in low:
        return "pytest"
    if "make" in low or "ninja" in low or "mvn" in low or "npm test" in low:
        return "build"
    if "grep" in low or "rg " in low:
        return "grep_search"
    if low.strip().startswith(("cat ", "sed ", "head ", "tail ")):
        return "file_read"
    if any(x in low for x in [">", "tee ", "apply_patch", "python - <<"]):
        return "file_write"
    return "shell_other"


def run_tool(command: str, cwd: str | None = None, out_trace: str | None = None, **meta: object) -> subprocess.CompletedProcess[str]:
    t0 = time.time()
    cp = subprocess.run(command, cwd=cwd, shell=True, text=True, capture_output=True)
    t1 = time.time()
    if out_trace:
        ev = Event(
            event_id=f"tool-{stable_hash([command, t0])}",
            session_id=str(meta.get("session_id", "tool-session")),
            instance_id=str(meta.get("instance_id", "unknown")),
            branch_id=str(meta.get("branch_id", "b0")),
            parent_branch_id=meta.get("parent_branch_id"),  # type: ignore[arg-type]
            step_id=int(meta.get("step_id", 0)),
            node_id=f"tool-{stable_hash([command, t0], 8)}",
            node_type="tool",
            timestamp_ready=t0,
            timestamp_start=t0,
            timestamp_end=t1,
            latency=t1 - t0,
            tool_type=classify_command(command),
            command=command,
            tool_latency=t1 - t0,
            observation_tokens=count_tokens((cp.stdout or "") + (cp.stderr or "")),
            exit_code=cp.returncode,
        )
        Trace({"framework": "tool_wrapper", "timestamp": t0}, [ev]).to_jsonl(out_trace)
    return cp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="+")
    ap.add_argument("--cwd")
    ap.add_argument("--out-trace")
    args = ap.parse_args()
    cp = run_tool(" ".join(shlex.quote(x) for x in args.command), cwd=args.cwd, out_trace=args.out_trace)
    print(cp.stdout, end="")
    print(cp.stderr, end="")
    raise SystemExit(cp.returncode)


if __name__ == "__main__":
    main()
