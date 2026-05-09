from __future__ import annotations

from agentweaver.baselines.lru_cache import LRUCache


class VLLMLikeCache(LRUCache):
    """Request-level exact prefix cache approximation."""
