from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LatencyModel:
    prefill_a: float = 0.00009
    prefill_b: float = 0.0
    prefill_c: float = 0.010
    decode_d: float = 0.0015
    decode_e: float = 0.00000008
    queue_factors: dict[str, float] = field(default_factory=dict)
    measured: bool = False
    source: str = "default_placeholder"
    quality: str = "PLACEHOLDER"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LatencyModel":
        if path is None or not Path(path).exists():
            return cls()
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**fields)

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, sort_keys=True)

    def predict_prefill(self, tokens: int | float) -> float:
        n = max(0.0, float(tokens))
        return self.prefill_a * n + self.prefill_b * n * n + self.prefill_c

    def predict_decode(self, context_len: int | float, output_tokens: int | float) -> float:
        return max(0.0, float(output_tokens)) * (self.decode_d + self.decode_e * max(0.0, float(context_len)))

    def queue_factor(self, concurrency: int | float = 1, input_tokens: int | float = 0, output_tokens: int | float = 0) -> float:
        key = str(int(concurrency))
        return float(self.queue_factors.get(key, 1.0))

    def predict_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        concurrency: int = 1,
    ) -> float:
        delta = max(0, int(input_tokens) - int(cached_tokens))
        base = self.predict_prefill(delta) + self.predict_decode(input_tokens, output_tokens)
        return base * self.queue_factor(concurrency, input_tokens, output_tokens)
