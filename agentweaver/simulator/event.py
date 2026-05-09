from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class SimEvent:
    time: float
    kind: str
    priority: int = 0
    payload: dict[str, Any] = field(default_factory=dict, compare=False)
