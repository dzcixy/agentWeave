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
class NISPDecision:
    state: ParkingState
    ttl: float
    resume_prefill_tokens: int
    migration_bytes: int = 0
    exposed_migration_latency: float = 0.0


@dataclass
class NISP:
    latency_model: LatencyModel
    mesh: WaferMesh | None = None
    ttl_candidates: list[float] = field(default_factory=lambda: [1, 2, 5, 10, 30, 60, 120, 300])
    pressure_weight: float = 1.0e-12
    chunk_bytes: int = 4 * 1024 * 1024
    hot_count: int = 0
    warm_count: int = 0
    cold_count: int = 0
    migration_bytes: int = 0
    exposed_migration_latency: float = 0.0

    def _cdf(self, tool_latency_samples: list[float], ttl: float) -> float:
        if not tool_latency_samples:
            return min(1.0, ttl / 60.0)
        return sum(1 for x in tool_latency_samples if x <= ttl) / len(tool_latency_samples)

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
    ) -> NISPDecision:
        # Non-Invasive State Parking:
        # State migration is allowed only when foreground NoC traffic has slack.
        # If migration would interfere with foreground LLM execution, NISP either
        # delays migration or chooses recomputation. This prevents state parking
        # overhead from appearing on the critical path.
        pressure = min(1.0, current_sram_usage / max(1.0, capacity))
        recompute = self.latency_model.predict_prefill(private_suffix_tokens)
        queue_cost = 0.02 * pressure
        sram_cost = (private_suffix_tokens + shared_prefix_tokens) * kv_bytes_per_token
        best_ttl, best_utility = 0.0, -1e30
        for ttl in self.ttl_candidates:
            benefit = self._cdf(tool_latency_samples, ttl) * (recompute + queue_cost)
            cost = ttl * sram_cost * self.pressure_weight * max(0.1, pressure)
            utility = benefit - cost
            if utility > best_utility:
                best_ttl, best_utility = ttl, utility
        if best_utility > 0 and pressure < 0.85:
            state = ParkingState.HOT
            resume = 0
            self.hot_count += 1
        elif shared_prefix_tokens > private_suffix_tokens // 2 and pressure < 0.98:
            state = ParkingState.WARM
            resume = private_suffix_tokens
            self.warm_count += 1
        else:
            state = ParkingState.COLD
            resume = private_suffix_tokens + shared_prefix_tokens
            self.cold_count += 1
        mig_bytes = private_suffix_tokens * kv_bytes_per_token if state == ParkingState.WARM else 0
        exposed = 0.0
        if mig_bytes and self.mesh and src and dst:
            admitted = True
            remaining = mig_bytes
            while remaining > 0:
                chunk = min(self.chunk_bytes, remaining)
                if not self.mesh.schedule_background_migration_if_slack_available(src, dst, chunk):
                    admitted = False
                    break
                remaining -= chunk
            if admitted:
                self.migration_bytes += mig_bytes
            else:
                exposed = 0.0
                state = ParkingState.COLD
                resume = private_suffix_tokens + shared_prefix_tokens
        self.exposed_migration_latency += exposed
        return NISPDecision(state, best_ttl, resume, mig_bytes, exposed)

    def metrics(self) -> dict[str, float]:
        total = self.hot_count + self.warm_count + self.cold_count
        return {
            "hot_count": self.hot_count,
            "warm_count": self.warm_count,
            "cold_count": self.cold_count,
            "migration_bytes": self.migration_bytes,
            "exposed_migration_latency": self.exposed_migration_latency,
            "state_parking_hit_rate": (self.hot_count + self.warm_count) / total if total else 0.0,
        }
