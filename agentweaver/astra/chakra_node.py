from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ChakraNodeType = Literal["compute", "communication", "memory", "delay"]


@dataclass
class ChakraNode:
    id: int
    name: str
    type: ChakraNodeType
    npu_id: int
    duration_cycles: int = 0
    size_bytes: int = 0
    deps: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

