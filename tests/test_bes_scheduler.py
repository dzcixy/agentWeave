from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.bes_scheduler import BESScheduler, BranchState
from agentweaver.simulator.context_arena import ContextArena
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event


def _ev(branch: str, seg: str) -> Event:
    return Event(
        f"e-{branch}",
        "s",
        "i",
        branch,
        None,
        1,
        f"n-{branch}",
        "llm",
        timestamp_ready=0,
        timestamp_start=0,
        timestamp_end=1,
        latency=1,
        input_tokens=100,
        output_tokens=10,
        context_length=100,
        context_segments=[ContextSegmentRef(seg, "repo", 0, 80)],
    )


def test_locality_scheduled_earlier_and_tool_releases() -> None:
    arena = ContextArena(10**9)
    arena.resident["hot"] = (1000, 0)
    sched = BESScheduler(LatencyModel(), arena, free_regions=[(0, 0)])
    ordered = sched.order_ready([_ev("b0", "cold"), _ev("b1", "hot")], now=1)
    assert ordered[0].branch_id == "b1"
    region = sched.allocate(ordered[0], now=1)
    assert region == (0, 0)
    sched.enter_tool("b1", 5, now=2)
    assert sched.records["b1"].state == BranchState.BLOCKED_TOOL
    assert (0, 0) in sched.free_regions
