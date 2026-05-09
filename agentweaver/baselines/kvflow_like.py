from __future__ import annotations


class KVFlowLikeCache:
    def __init__(self, capacity_bytes: int, future_scores: dict[str, float]):
        self.capacity_bytes = capacity_bytes
        self.future_scores = future_scores
        self.cache: dict[str, int] = {}
        self.evictions = 0

    def get(self, key: str) -> bool:
        return key in self.cache

    def put(self, key: str, size: int) -> None:
        self.cache[key] = size
        while self.occupancy() > self.capacity_bytes and self.cache:
            victim = min(self.cache, key=lambda k: self.future_scores.get(k, 0.0) / max(1, self.cache[k]))
            self.cache.pop(victim)
            self.evictions += 1

    def occupancy(self) -> int:
        return sum(self.cache.values())
