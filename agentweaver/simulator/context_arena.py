from __future__ import annotations

from dataclasses import dataclass, field

from agentweaver.tracing.trace_schema import ContextSegmentRef


@dataclass
class ContextArena:
    capacity_bytes: int
    resident: dict[str, tuple[int, float]] = field(default_factory=dict)
    now: float = 0.0

    def match(self, segments: list[ContextSegmentRef]) -> int:
        hits = 0
        for seg in segments:
            if seg.segment_id in self.resident:
                hits += seg.length
        return hits

    def insert(self, segments: list[ContextSegmentRef], bytes_per_token: int = 28672) -> int:
        evictions = 0
        for seg in segments:
            size = seg.kv_bytes or seg.length * bytes_per_token
            self.resident[seg.segment_id] = (size, self.now)
            while self.occupancy() > self.capacity_bytes and self.resident:
                victim = min(self.resident.items(), key=lambda kv: kv[1][1])[0]
                self.resident.pop(victim, None)
                evictions += 1
        return evictions

    def touch(self, segment_id: str) -> None:
        if segment_id in self.resident:
            size, _ = self.resident[segment_id]
            self.resident[segment_id] = (size, self.now)

    def occupancy(self) -> int:
        return sum(size for size, _ in self.resident.values())
