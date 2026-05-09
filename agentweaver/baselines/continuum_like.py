from __future__ import annotations

from agentweaver.baselines.lru_cache import LRUCache


class ContinuumLikeCache(LRUCache):
    def __init__(self, capacity_bytes: int, ttl_seconds: float = 30.0):
        super().__init__(capacity_bytes)
        self.ttl_seconds = ttl_seconds
        self.insert_time: dict[str, float] = {}

    def expire(self, now: float) -> None:
        for k, t in list(self.insert_time.items()):
            if now - t > self.ttl_seconds:
                self.cache.pop(k, None)
                self.insert_time.pop(k, None)

    def put_at(self, key: str, size: int, now: float) -> None:
        self.put(key, size)
        self.insert_time[key] = now
