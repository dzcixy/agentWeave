from pathlib import Path

from agentweaver.tracing.trace_schema import Event, Trace, validate_trace


def test_trace_roundtrip(tmp_path: Path) -> None:
    tr = Trace(
        {"benchmark": "dummy"},
        [
            Event(
                event_id="e0",
                session_id="s",
                instance_id="i",
                branch_id="b0",
                parent_branch_id=None,
                step_id=0,
                node_id="n0",
                node_type="message",
                timestamp_ready=0,
                timestamp_start=0,
                timestamp_end=1,
                latency=1,
            )
        ],
    )
    assert validate_trace(tr) == []
    p = tmp_path / "t.jsonl"
    tr.to_jsonl(p)
    loaded = Trace.from_jsonl(p)
    assert loaded.events[0].event_id == "e0"


def test_trace_timestamp_validation() -> None:
    tr = Trace(
        {},
        [
            Event("e", "s", "i", "b", None, 0, "n", "llm", timestamp_ready=2, timestamp_start=1, timestamp_end=0, latency=1)
        ],
    )
    errors = validate_trace(tr)
    assert any("start before ready" in e for e in errors)
    assert any("end before start" in e for e in errors)
