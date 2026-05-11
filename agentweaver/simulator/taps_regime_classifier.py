from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal


Regime = Literal["ADMISSION_STARVED", "DOMAIN_HOT", "TAIL_RISK", "MEMORY_PRESSURE", "BALANCED"]


@dataclass
class RegimeSnapshot:
    now: float = 0.0
    ready_queue_len: int = 0
    effective_regions: int = 1
    backlog_len: int = 0
    active_sessions: int = 0
    blocked_session_fraction: float = 0.0
    domain_cache_hit_rate: float = 0.0
    ready_domain_share: float = 0.0
    remote_kv_pressure: float = 0.0
    memory_pressure: float = 0.0
    p95_risk: float = 0.0
    max_ready_wait: float = 0.0
    session_pressure: float = 1.0
    region_pressure: float = 1.0


@dataclass
class RegimeThresholds:
    admission_blocked_fraction: float = 0.35
    domain_cache_hit_rate: float = 0.85
    ready_domain_share: float = 0.45
    remote_kv_pressure: float = 0.15
    tail_risk: float = 0.25
    ready_wait_seconds: float = 15.0
    memory_pressure: float = 0.85
    eviction_pressure: float = 0.01


class RegimeClassifier:
    def __init__(self, thresholds: RegimeThresholds | None = None) -> None:
        self.thresholds = thresholds or RegimeThresholds()
        self.regime_counts: Counter[str] = Counter()
        self.profile_counts: Counter[str] = Counter()
        self.transition_counts: Counter[str] = Counter()
        self.decision_counts: Counter[str] = Counter()
        self.last_regime: str | None = None

    def classify(self, s: RegimeSnapshot) -> Regime:
        t = self.thresholds
        if s.memory_pressure >= t.memory_pressure:
            regime: Regime = "MEMORY_PRESSURE"
        elif s.ready_queue_len < s.effective_regions and s.backlog_len > 0 and s.blocked_session_fraction >= t.admission_blocked_fraction:
            regime = "ADMISSION_STARVED"
        elif s.p95_risk >= t.tail_risk or s.max_ready_wait >= t.ready_wait_seconds:
            regime = "TAIL_RISK"
        elif (
            (s.domain_cache_hit_rate >= t.domain_cache_hit_rate or s.ready_domain_share >= t.ready_domain_share)
            and s.remote_kv_pressure >= t.remote_kv_pressure
        ):
            regime = "DOMAIN_HOT"
        else:
            regime = "BALANCED"
        self.regime_counts[regime] += 1
        if self.last_regime is not None and self.last_regime != regime:
            self.transition_counts[f"{self.last_regime}->{regime}"] += 1
        self.last_regime = regime
        return regime

    def record_profile(self, regime: str, profile: str, decision: str) -> None:
        self.profile_counts[profile] += 1
        self.decision_counts[f"{regime}:{decision}"] += 1

