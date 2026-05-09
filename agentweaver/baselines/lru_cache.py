from __future__ import annotations

from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.cache: OrderedDict[str, int] = OrderedDict()
        self.evictions = 0

    def get(self, key: str) -> bool:
        if key in self.cache:
            self.cache.move_to_end(key)
            return True
        return False

    def put(self, key: str, size: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = size
        else:
            self.cache[key] = size
        while self.occupancy() > self.capacity_bytes and self.cache:
            self.cache.popitem(last=False)
            self.evictions += 1

    def occupancy(self) -> int:
        return sum(self.cache.values())
