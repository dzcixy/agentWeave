from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.context_arena import ContextArena
from agentweaver.simulator.wafer_mesh import Coord
from agentweaver.tracing.trace_schema import Event


class BranchState(str, Enum):
    READY_LLM = "READY_LLM"
    RUNNING_LLM = "RUNNING_LLM"
    BLOCKED_TOOL = "BLOCKED_TOOL"
    READY_VERIFY = "READY_VERIFY"
    ACCEPTED = "ACCEPTED"
    CANCELLED = "CANCELLED"
    DONE_FAIL = "DONE_FAIL"


@dataclass
class BranchRecord:
    branch_id: str
    state: BranchState = BranchState.READY_LLM
    waiting_since: float = 0.0
    region: Optional[Coord] = None
    cancelled: bool = False


@dataclass
class BESScheduler:
    latency_model: LatencyModel
    arena: ContextArena
    alpha: float = 1.0e-12
    beta: float = 1.0e-12
    gamma: float = 0.01
    records: dict[str, BranchRecord] = field(default_factory=dict)
    free_regions: list[Coord] = field(default_factory=list)
    branch_wait_time: dict[str, float] = field(default_factory=dict)
    blocked_compute_time_avoided: float = 0.0

    def ensure_branch(self, branch_id: str, now: float) -> BranchRecord:
        if branch_id not in self.records:
            self.records[branch_id] = BranchRecord(branch_id, waiting_since=now)
        return self.records[branch_id]

    def score(self, ev: Event, now: float, avg_hops: float = 1.0, private_kv_bytes: float = 0.0) -> float:
        rec = self.ensure_branch(ev.branch_id, now)
        prefix_hits = self.arena.match(ev.context_segments)
        delta_tokens = max(0, ev.input_tokens - prefix_hits)
        locality_gain = sum(self.latency_model.predict_prefill(s.length) for s in ev.context_segments if s.segment_id in self.arena.resident)
        join_impact = 1.0 / (1.0 + max(1, 3 - ev.step_id))
        age_boost = self.gamma * max(0.0, now - rec.waiting_since)
        predicted_service = self.latency_model.predict_prefill(delta_tokens) + self.latency_model.predict_decode(
            ev.context_length, ev.output_tokens
        )
        noc_cost = max(0, ev.input_tokens - prefix_hits) * 28672 * avg_hops
        sram_cost = private_kv_bytes
        return (locality_gain + join_impact + age_boost) / (
            predicted_service + self.alpha * noc_cost + self.beta * sram_cost + 1e-9
        )

    def order_ready(self, events: list[Event], now: float) -> list[Event]:
        ready = [e for e in events if self.ensure_branch(e.branch_id, now).state == BranchState.READY_LLM]
        return sorted(ready, key=lambda e: self.score(e, now), reverse=True)

    def allocate(self, ev: Event, now: float, preferred: Optional[Coord] = None) -> Optional[Coord]:
        rec = self.ensure_branch(ev.branch_id, now)
        if rec.state != BranchState.READY_LLM or not self.free_regions:
            return None
        if preferred in self.free_regions:
            region = preferred
        else:
            region = self.free_regions.pop(0)
            rec.region = region
            rec.state = BranchState.RUNNING_LLM
            self.branch_wait_time[ev.branch_id] = self.branch_wait_time.get(ev.branch_id, 0.0) + max(
                0.0, now - rec.waiting_since
            )
            return region
        self.free_regions.remove(region)
        rec.region = region
        rec.state = BranchState.RUNNING_LLM
        self.branch_wait_time[ev.branch_id] = self.branch_wait_time.get(ev.branch_id, 0.0) + max(0.0, now - rec.waiting_since)
        return region

    def llm_done(self, branch_id: str, region: Coord | None, now: float) -> None:
        rec = self.ensure_branch(branch_id, now)
        if region is not None and region not in self.free_regions:
            self.free_regions.append(region)
        rec.region = None
        rec.state = BranchState.BLOCKED_TOOL

    def enter_tool(self, branch_id: str, measured_tool_latency: float, now: float) -> None:
        # Branch Elasticity:
        # Compute regions should be allocated only when a branch is READY_LLM.
        # When a branch is BLOCKED_TOOL, it releases compute resources but its
        # state may be parked by NISP. This avoids dead silicon caused by static
        # branch pinning.
        rec = self.ensure_branch(branch_id, now)
        if rec.region is not None and rec.region not in self.free_regions:
            self.free_regions.append(rec.region)
            rec.region = None
        rec.state = BranchState.BLOCKED_TOOL
        self.blocked_compute_time_avoided += measured_tool_latency

    def tool_done(self, branch_id: str, now: float) -> None:
        rec = self.ensure_branch(branch_id, now)
        rec.state = BranchState.READY_VERIFY

    def verify_done(self, branch_id: str, success: bool, sibling_branches: list[str], now: float) -> list[str]:
        rec = self.ensure_branch(branch_id, now)
        if success:
            rec.state = BranchState.ACCEPTED
            cancelled = []
            for sib in sibling_branches:
                if sib == branch_id:
                    continue
                srec = self.ensure_branch(sib, now)
                if srec.state not in {BranchState.ACCEPTED, BranchState.DONE_FAIL}:
                    srec.state = BranchState.CANCELLED
                    srec.cancelled = True
                    cancelled.append(sib)
            return cancelled
        rec.state = BranchState.DONE_FAIL
        return []
