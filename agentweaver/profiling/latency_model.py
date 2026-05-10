from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math


@dataclass
class LatencyModel:
    mode: str = "parametric"
    prefill_a: float = 0.00009
    prefill_b: float = 0.0
    prefill_c: float = 0.010
    decode_d: float = 0.0015
    decode_e: float = 0.00000008
    queue_factors: dict[str, float] = field(default_factory=dict)
    interpolation_points: list[dict[str, float]] = field(default_factory=list)
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
        if self.mode == "interpolation" and self.interpolation_points:
            return self._interpolate_prefill(n)
        return self.prefill_a * n + self.prefill_b * n * n + self.prefill_c

    def predict_decode(self, context_len: int | float, output_tokens: int | float) -> float:
        if self.mode == "interpolation" and self.interpolation_points:
            return self._interpolate_decode(max(0.0, float(context_len)), max(0.0, float(output_tokens)))
        return max(0.0, float(output_tokens)) * (self.decode_d + self.decode_e * max(0.0, float(context_len)))

    def _idw(self, x: float, y: float | None, value_key: str) -> float:
        pts = [p for p in self.interpolation_points if p.get(value_key, 0.0) >= 0]
        if not pts:
            return 0.0
        scale_x = max(1.0, max(float(p.get("input", 0.0)) for p in pts))
        scale_y = max(1.0, max(float(p.get("output", 0.0)) for p in pts)) if y is not None else 1.0
        weighted = 0.0
        weights = 0.0
        exact: list[float] = []
        for p in pts:
            dx = (x - float(p.get("input", 0.0))) / scale_x
            dy = 0.0 if y is None else (y - float(p.get("output", 0.0))) / scale_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                exact.append(float(p.get(value_key, 0.0)))
                continue
            w = 1.0 / (dist * dist)
            weighted += w * float(p.get(value_key, 0.0))
            weights += w
        if exact:
            xs = sorted(exact)
            return xs[len(xs) // 2]
        return weighted / max(1e-12, weights)

    def _interpolate_prefill(self, tokens: float) -> float:
        return max(0.0, self._idw(tokens, None, "prefill"))

    def _interpolate_decode(self, context_len: float, output_tokens: float) -> float:
        if output_tokens <= 0:
            return 0.0
        tpot = self._idw(context_len, output_tokens, "tpot")
        if tpot <= 0:
            nearest = self._idw(context_len, output_tokens, "decode")
            return max(0.0, nearest)
        return max(0.0, output_tokens * tpot)

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
