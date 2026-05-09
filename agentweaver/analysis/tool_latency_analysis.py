from __future__ import annotations

from collections import defaultdict

from agentweaver.tracing.trace_schema import Event


def tool_latency_cdf(events: list[Event]) -> dict[str, list[float]]:
    vals: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        if ev.node_type == "tool":
            vals[ev.tool_type or "shell_other"].append(ev.tool_latency if ev.tool_latency is not None else ev.latency)
    return {k: sorted(v) for k, v in vals.items()}
