from __future__ import annotations

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.pabb_online_replay import (
    BranchOnlineState,
    choose_next_branches,
    update_branch_state,
)
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event


def _event(
    event_id: str,
    branch_id: str,
    node_type: str,
    step_id: int,
    *,
    output_tokens: int = 0,
    command: str | None = None,
    exit_code: int | None = None,
    patch_hash: str | None = None,
    verifier_result: str | None = None,
) -> Event:
    segments = []
    if patch_hash:
        segments.append(ContextSegmentRef(segment_id=f"patch-{patch_hash}", segment_type="patch", start_pos=0, length=12))
    return Event(
        event_id=event_id,
        session_id="s0",
        instance_id="inst",
        branch_id=branch_id,
        parent_branch_id="root",
        step_id=step_id,
        node_id=event_id,
        node_type=node_type,  # type: ignore[arg-type]
        input_tokens=20 if node_type == "llm" else 0,
        output_tokens=output_tokens,
        context_length=20,
        command=command,
        exit_code=exit_code,
        patch_hash=patch_hash,
        verifier_result=verifier_result,  # type: ignore[arg-type]
        context_segments=segments,
    )


def test_online_cannot_see_future_patch_before_patch_event() -> None:
    no_patch = BranchOnlineState("a_no_patch", [_event("b0_llm", "a_no_patch", "llm", 0, output_tokens=5)])
    future_patch = BranchOnlineState(
        "z_future_patch",
        [
            _event("a0_llm", "z_future_patch", "llm", 0, output_tokens=5),
            _event("a1_patch", "z_future_patch", "verifier", 1, patch_hash="patch-a"),
        ],
    )
    states = {s.branch_id: s for s in [no_patch, future_patch]}

    online = choose_next_branches(states, "pabb_online", max_active=1)[0]
    oracle = choose_next_branches(states, "pabb_oracle_upper_bound", max_active=1)[0]

    assert online.branch_id == "a_no_patch"
    assert oracle.branch_id == "z_future_patch"
    assert not future_patch.patch_candidate_seen


def test_patch_signal_becomes_visible_only_after_patch_event() -> None:
    lm = LatencyModel()
    seen: set[str] = set()
    st = BranchOnlineState(
        "b0",
        [
            _event("e0", "b0", "llm", 0, output_tokens=8),
            _event("e1", "b0", "verifier", 1, patch_hash="patch-a"),
        ],
    )

    update_branch_state(st, st.next_event(), lm, seen)
    assert not st.patch_candidate_seen
    assert st.patch_hash_seen == ""

    update_branch_state(st, st.next_event(), lm, seen)
    assert st.patch_candidate_seen
    assert st.patch_hash_seen == "patch-a"


def test_duplicate_patch_hash_visible_only_after_second_patch_event() -> None:
    lm = LatencyModel()
    seen: set[str] = set()
    first = BranchOnlineState("b0", [_event("p0", "b0", "verifier", 0, patch_hash="same")])
    second = BranchOnlineState("b1", [_event("p1", "b1", "verifier", 0, patch_hash="same")])

    update_branch_state(first, first.next_event(), lm, seen)
    assert not first.duplicate_patch_seen_so_far
    assert not second.duplicate_patch_seen_so_far

    update_branch_state(second, second.next_event(), lm, seen)
    assert second.duplicate_patch_seen_so_far


def test_official_verifier_visible_only_after_verifier_event() -> None:
    lm = LatencyModel()
    seen: set[str] = set()
    st = BranchOnlineState(
        "b0",
        [
            _event("e0", "b0", "tool", 0, command="pytest tests/test_x.py", exit_code=0),
            _event("e1", "b0", "verifier", 1, verifier_result="pass"),
        ],
    )

    assert st.official_verifier_seen == "unknown"
    update_branch_state(st, st.next_event(), lm, seen)
    assert st.official_verifier_seen == "unknown"
    update_branch_state(st, st.next_event(), lm, seen)
    assert st.official_verifier_seen == "pass"
