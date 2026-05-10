from pathlib import Path

from agentweaver.tracing.mini_swe_trace_adapter import convert_mini_swe_traj
from agentweaver.tracing.trace_schema import Trace


def test_convert_sample_mini_swe_fixture_to_trace(tmp_path: Path):
    fixture = Path("tests/fixtures/sample_mini_swe_traj.json")
    out = tmp_path / "trace.jsonl"

    trace = convert_mini_swe_traj(fixture, out=out, model="adapter-test-tokenizer")

    assert out.exists()
    loaded = Trace.from_jsonl(out)
    assert len(loaded.events) == len(trace.events)
    assert loaded.metadata["sample_fixture"] is True

    llm_events = [e for e in trace.events if e.node_type == "llm"]
    tool_events = [e for e in trace.events if e.node_type == "tool"]
    verifier_events = [e for e in trace.events if e.node_type == "verifier"]

    assert llm_events
    assert tool_events
    assert verifier_events
    assert all(e.context_segments for e in llm_events)
    assert all(e.input_tokens > 0 for e in llm_events)
    assert all(e.output_tokens > 0 for e in llm_events)
    assert tool_events[0].command == "pytest -q"
    assert tool_events[0].observation_tokens and tool_events[0].observation_tokens > 0

    verifier = verifier_events[0]
    assert verifier.verifier_result == "unknown"
    assert verifier.success is None
    assert verifier.patch_hash
