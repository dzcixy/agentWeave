from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentweaver.profiling.latency_model import LatencyModel as AnalyticLatencyModel


@dataclass
class LatencyComponents:
    prefill_latency: float
    decode_latency: float
    local_memory_latency: float
    noC_latency: float
    state_prefetch_latency: float
    tool_latency: float
    queueing_latency: float

    @property
    def model_side_latency(self) -> float:
        return self.prefill_latency + self.decode_latency + self.local_memory_latency + self.noC_latency + self.state_prefetch_latency + self.queueing_latency

    @property
    def end_to_end_latency(self) -> float:
        return self.model_side_latency + self.tool_latency


class AgentWeaverLatencyModel:
    def __init__(
        self,
        *,
        h100_fit: dict[str, Any] | None = None,
        link_bandwidth: float = 2.0e11,
        link_latency: float = 2.0e-6,
        local_memory_bandwidth: float = 1.0e12,
        remote_memory_bandwidth: float = 2.0e11,
        noc_slack_bandwidth: float = 1.0e11,
    ) -> None:
        self.h100_fit = h100_fit if h100_fit and h100_fit.get("H100_CALIBRATION_STATUS") == "OK" else None
        self.analytic = AnalyticLatencyModel()
        self.link_bandwidth = link_bandwidth
        self.link_latency = link_latency
        self.local_memory_bandwidth = local_memory_bandwidth
        self.remote_memory_bandwidth = remote_memory_bandwidth
        self.noc_slack_bandwidth = noc_slack_bandwidth

    @classmethod
    def from_h100_json(cls, path: str | Path | None = None) -> "AgentWeaverLatencyModel":
        if path and Path(path).exists():
            with Path(path).open("r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(h100_fit=data)
        return cls()

    @property
    def uses_h100(self) -> bool:
        return self.h100_fit is not None

    def prefill_latency(self, tokens: int | float) -> float:
        n = max(0.0, float(tokens))
        if self.h100_fit:
            coeff = self.h100_fit.get("prefill_latency", {})
            return max(0.0, float(coeff.get("a0", 0.0)) + float(coeff.get("a1", 0.0)) * n + float(coeff.get("a2", 0.0)) * n * n)
        return self.analytic.predict_prefill(n)

    def decode_latency(self, context_length: int | float, output_tokens: int | float, batch_size: int | float = 1) -> float:
        ctx = max(0.0, float(context_length))
        out = max(0.0, float(output_tokens))
        batch = max(1.0, float(batch_size))
        if self.h100_fit:
            coeff = self.h100_fit.get("decode_latency_per_token", {})
            per = float(coeff.get("b0", 0.0)) + float(coeff.get("b1", 0.0)) * ctx + float(coeff.get("b2", 0.0)) * batch
            return max(0.0, out * per)
        return self.analytic.predict_decode(ctx, out)

    def components(
        self,
        *,
        prefill_compute_tokens: int | float,
        decode_tokens: int | float,
        context_length: int | float,
        local_context_bytes: int | float,
        remote_context_bytes: int | float,
        remote_kv_bytes: int | float,
        avg_context_hops: float,
        effective_regions: int | float,
        region_utilization: float,
        prefetch_bytes: int | float = 0.0,
        foreground_prefetch_conflict: bool = False,
        tool_latency: float = 0.0,
        ready_queue_wait: float = 0.0,
        batch_size: int | float = 1,
    ) -> LatencyComponents:
        util = max(0.0, min(0.99, float(region_utilization)))
        effective_bw = self.link_bandwidth * max(0.10, 1.0 - 0.65 * util)
        remote_bytes = max(float(remote_context_bytes), float(remote_kv_bytes), 0.0)
        noc_transfer = remote_bytes / max(1.0, effective_bw)
        contention = util * remote_bytes / max(1.0, self.remote_memory_bandwidth)
        noc_latency = noc_transfer + max(0.0, avg_context_hops) * self.link_latency + contention
        local_latency = max(0.0, float(local_context_bytes)) / max(1.0, self.local_memory_bandwidth)
        prefetch_latency = (max(0.0, float(prefetch_bytes)) / max(1.0, self.noc_slack_bandwidth)) if foreground_prefetch_conflict else 0.0
        return LatencyComponents(
            prefill_latency=self.prefill_latency(prefill_compute_tokens),
            decode_latency=self.decode_latency(context_length, decode_tokens, batch_size),
            local_memory_latency=local_latency,
            noC_latency=noc_latency,
            state_prefetch_latency=prefetch_latency,
            tool_latency=max(0.0, float(tool_latency)),
            queueing_latency=max(0.0, float(ready_queue_wait)),
        )
