from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.wafer_mesh import Coord, WaferMesh


class ParkingState(str, Enum):
    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"


@dataclass
class BranchKVState:
    branch_id: str
    shared_prefix_tokens: int
    private_suffix_tokens: int
    observation_tokens: int
    full_context_tokens: int
    shared_prefix_kv_bytes: int
    private_suffix_kv_bytes: int
    region: Coord | None = None


@dataclass
class NISPDecision:
    state: ParkingState
    ttl: float
    cached_tokens: int
    resume_prefill_tokens: int
    migration_bytes: int = 0
    exposed_migration_latency: float = 0.0
    noc_background_bytes: int = 0


@dataclass
class NISP:
    latency_model: LatencyModel
    mesh: WaferMesh | None = None
    ttl_candidates: list[float] = field(default_factory=lambda: [1, 2, 5, 10, 30, 60, 120, 300])
    pressure_weight: float = 1.0e-12
    hot_pressure_threshold: float = 0.85
    warm_pressure_threshold: float = 0.98
    chunk_bytes: int = 4 * 1024 * 1024
    hot_count: int = 0
    warm_count: int = 0
    cold_count: int = 0
    migration_bytes: int = 0
    exposed_migration_latency: float = 0.0
    noc_background_bytes: int = 0
    resume_prefill_tokens: int = 0
    parked: dict[str, NISPDecision] = field(default_factory=dict)

    def _cdf(self, tool_latency_samples: list[float], ttl: float) -> float:
        if not tool_latency_samples:
            return min(1.0, ttl / 60.0)
        return sum(1 for x in tool_latency_samples if x <= ttl) / len(tool_latency_samples)

    def decide(
        self,
        tool_latency_samples: list[float],
        state: BranchKVState,
        kv_bytes_per_token: int,
        current_sram_usage: float,
        capacity: float,
        src: Coord | None = None,
        dst: Coord | None = None,
    ) -> NISPDecision:
        # Non-Invasive State Parking:
        # State migration is allowed only when foreground NoC traffic has slack.
        # If migration would interfere with foreground LLM execution, NISP either
        # delays migration or chooses recomputation. This prevents state parking
        # overhead from appearing on the critical path.
        pressure = min(1.0, current_sram_usage / max(1.0, capacity))
        recompute_cost = self.latency_model.predict_prefill(state.private_suffix_tokens)
        queue_cost = 0.02 * pressure
        sram_bytes = state.shared_prefix_kv_bytes + state.private_suffix_kv_bytes
        best_ttl, best_utility = 0.0, -1e30
        for ttl in self.ttl_candidates:
            benefit = self._cdf(tool_latency_samples, ttl) * (recompute_cost + queue_cost)
            cost = ttl * sram_bytes * self.pressure_weight
            utility = benefit - cost
            if utility > best_utility:
                best_ttl, best_utility = ttl, utility

        shared_prefix_value = self.latency_model.predict_prefill(state.shared_prefix_tokens)
        private_value = self.latency_model.predict_prefill(max(1, state.private_suffix_tokens))
        short_or_light_resume = best_ttl <= 30 or state.observation_tokens <= 320
        if best_utility > 0 and pressure < self.hot_pressure_threshold and short_or_light_resume:
            parking = ParkingState.HOT
            cached = state.shared_prefix_tokens + state.private_suffix_tokens
            resume = state.observation_tokens
            self.hot_count += 1
        elif shared_prefix_value >= 0.75 * private_value and pressure < self.warm_pressure_threshold:
            parking = ParkingState.WARM
            cached = state.shared_prefix_tokens
            resume = state.private_suffix_tokens + state.observation_tokens
            self.warm_count += 1
        else:
            parking = ParkingState.COLD
            cached = 0
            resume = state.full_context_tokens
            self.cold_count += 1

        migration_bytes = state.private_suffix_kv_bytes if parking == ParkingState.WARM else 0
        exposed = 0.0
        background = 0
        if migration_bytes and self.mesh and src and dst:
            remaining = migration_bytes
            admitted = True
            while remaining > 0:
                chunk = min(self.chunk_bytes, remaining)
                if not self.mesh.schedule_background_migration_if_slack_available(src, dst, chunk):
                    admitted = False
                    exposed += self.mesh.transfer_latency(max(1, self.mesh.manhattan(src, dst)), remaining)
                    break
                background += chunk
                remaining -= chunk
            if admitted:
                self.migration_bytes += migration_bytes
                self.noc_background_bytes += background
            else:
                parking = ParkingState.COLD
                cached = 0
                resume = state.full_context_tokens
                self.cold_count += 1
                if self.warm_count:
                    self.warm_count -= 1
        self.exposed_migration_latency += exposed
        self.resume_prefill_tokens += resume
        return NISPDecision(parking, best_ttl, cached, resume, migration_bytes, exposed, background)

    def park_branch(
        self,
        branch_state: BranchKVState,
        tool_latency_samples: list[float],
        kv_bytes_per_token: int,
        current_sram_usage: float,
        capacity: float,
        src: Coord | None = None,
        dst: Coord | None = None,
    ) -> NISPDecision:
        decision = self.decide(tool_latency_samples, branch_state, kv_bytes_per_token, current_sram_usage, capacity, src, dst)
        self.parked[branch_state.branch_id] = decision
        return decision

    def restore_state(self, branch_id: str) -> NISPDecision | None:
        return self.parked.pop(branch_id, None)

    def park_state(
        self,
        tool_type: str,
        tool_latency_samples: list[float],
        private_suffix_tokens: int,
        shared_prefix_tokens: int,
        kv_bytes_per_token: int,
        current_sram_usage: float,
        capacity: float,
        src: Coord | None = None,
        dst: Coord | None = None,
        observation_tokens: int = 0,
        full_context_tokens: int | None = None,
        branch_id: str = "compat",
    ) -> NISPDecision:
        full_context_tokens = full_context_tokens or (private_suffix_tokens + shared_prefix_tokens + observation_tokens)
        state = BranchKVState(
            branch_id=branch_id,
            shared_prefix_tokens=shared_prefix_tokens,
            private_suffix_tokens=private_suffix_tokens,
            observation_tokens=observation_tokens,
            full_context_tokens=full_context_tokens,
            shared_prefix_kv_bytes=shared_prefix_tokens * kv_bytes_per_token,
            private_suffix_kv_bytes=private_suffix_tokens * kv_bytes_per_token,
            region=src,
        )
        return self.park_branch(state, tool_latency_samples, kv_bytes_per_token, current_sram_usage, capacity, src, dst)

    def metrics(self) -> dict[str, float]:
        total = self.hot_count + self.warm_count + self.cold_count
        return {
            "hot_count": self.hot_count,
            "warm_count": self.warm_count,
            "cold_count": self.cold_count,
            "migration_bytes": self.migration_bytes,
            "exposed_migration_latency": self.exposed_migration_latency,
            "state_migration_exposed_latency": self.exposed_migration_latency,
            "noc_background_bytes": self.noc_background_bytes,
            "resume_prefill_tokens": self.resume_prefill_tokens,
            "state_parking_hit_rate": (self.hot_count + self.warm_count) / total if total else 0.0,
        }
