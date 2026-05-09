from __future__ import annotations

from agentweaver.tracing.trace_schema import Event


def latency_breakdown(events: list[Event]) -> dict[str, float]:
    out = {"llm": 0.0, "tool": 0.0, "verifier": 0.0, "orchestration": 0.0}
    for ev in events:
        key = ev.node_type if ev.node_type in out else "orchestration"
        out[key] += ev.latency
    return out
