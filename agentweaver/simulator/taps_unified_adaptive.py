from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.taps_regime_classifier import RegimeClassifier, RegimeSnapshot, RegimeThresholds
from agentweaver.simulator.taps_unified import (
    POLICIES as BASE_POLICIES,
    SessionState,
    TAPSUnifiedConfig,
    TAPSUnifiedReplay,
    _f,
    _fill_gains,
    _load_traces,
    _shared_tokens,
    _tool_latency,
)
from agentweaver.tracing.trace_schema import Event, Trace
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv


BASELINE_POLICIES = [
    "reactive_admission",
    "acd_nisp",
    "taps_domain_v4",
    "taps_admission_v4",
    "taps_unified_v5",
]
ADAPTIVE_POLICY = "taps_unified_adaptive_v6"
V6_POLICIES = BASELINE_POLICIES + [ADAPTIVE_POLICY]


@dataclass
class AdaptiveProfiles:
    balanced: TAPSUnifiedConfig
    admission_starved: TAPSUnifiedConfig
    domain_hot: TAPSUnifiedConfig
    tail_risk: TAPSUnifiedConfig
    memory_pressure: TAPSUnifiedConfig
    thresholds: RegimeThresholds
    tail_slo_percentile: int = 90

    @classmethod
    def default(cls) -> "AdaptiveProfiles":
        balanced = TAPSUnifiedConfig(
            w_tail=4.0,
            w_domain=1.0,
            w_batch=0.0,
            w_resume=0.5,
            w_age=0.05,
            w_short=1.0,
            w_remote=0.1,
            w_switch=0.1,
            w_mem=0.5,
            ready_depth_factor=1.0,
            memory_pressure_threshold=0.85,
        )
        admission = TAPSUnifiedConfig(
            **{**asdict(balanced), "ready_depth_factor": 4.0, "admission_tail": 2.0, "admission_stall": 4.0}
        )
        domain = TAPSUnifiedConfig(
            **{**asdict(balanced), "w_domain": 4.0, "w_batch": 1.0, "w_remote": 0.5, "w_switch": 0.5, "w_short": 0.5}
        )
        tail = TAPSUnifiedConfig(**{**asdict(balanced), "w_tail": 8.0, "w_age": 0.1, "w_short": 0.0})
        memory = TAPSUnifiedConfig(**{**asdict(balanced), "w_mem": 2.0, "memory_pressure_threshold": 0.7, "ready_depth_factor": 1.0})
        return cls(balanced, admission, domain, tail, memory, RegimeThresholds())

    def for_regime(self, regime: str) -> TAPSUnifiedConfig:
        if regime == "ADMISSION_STARVED":
            return self.admission_starved
        if regime == "DOMAIN_HOT":
            return self.domain_hot
        if regime == "TAIL_RISK":
            return self.tail_risk
        if regime == "MEMORY_PRESSURE":
            return self.memory_pressure
        return self.balanced

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "balanced": asdict(self.balanced),
            "admission_starved": asdict(self.admission_starved),
            "domain_hot": asdict(self.domain_hot),
            "tail_risk": asdict(self.tail_risk),
            "memory_pressure": asdict(self.memory_pressure),
            "thresholds": asdict(self.thresholds),
            "tail_slo_percentile": self.tail_slo_percentile,
        }


class TAPSAdaptiveReplay(TAPSUnifiedReplay):
    def __init__(
        self,
        traces: list[Trace],
        total_sessions: int,
        active_session_limit: int,
        effective_regions: int,
        arrival_pattern: str,
        memory_budget_gb: int,
        latency_model: LatencyModel,
        profiles: AdaptiveProfiles | None = None,
        seed: int = 121,
    ) -> None:
        self.profiles = profiles or AdaptiveProfiles.default()
        self.classifier = RegimeClassifier(self.profiles.thresholds)
        self.current_regime = "BALANCED"
        self.current_profile = self.profiles.balanced
        super().__init__(
            traces,
            total_sessions,
            active_session_limit,
            effective_regions,
            arrival_pattern,
            memory_budget_gb,
            "taps_unified",
            latency_model,
            config=self.profiles.balanced,
            seed=seed,
        )
        self.policy = ADAPTIVE_POLICY
        self.slo_target = self.config.slo_target or self._historical_slo_percentile(traces, self.profiles.tail_slo_percentile)

    def _historical_slo_percentile(self, traces: list[Trace], pct: int) -> float:
        vals: list[float] = []
        for trace in traces:
            evs = [e for e in trace.events if e.node_type in {"llm", "tool", "verifier"} and e.timestamp_start and e.timestamp_end]
            if evs:
                vals.append(max(e.timestamp_end for e in evs) - min(e.timestamp_start for e in evs))
        if not vals:
            vals = [sum(self._event_ground_duration(e) for e in trace.events if e.node_type in {"llm", "tool", "verifier"}) for trace in traces]
        vals = sorted(v for v in vals if v >= 0)
        if not vals:
            return self.slo_target
        idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
        return vals[idx]

    def _ready_domain_share(self) -> float:
        if not self.ready:
            return 0.0
        counts: Counter[str] = Counter()
        for sid, _ in self.ready:
            st = self.active.get(sid)
            if st is not None:
                counts[st.context_domain_id] += 1
                counts[st.repo_domain_id] += 1
        return max(counts.values() or [0]) / max(1, len(self.ready))

    def _max_tail_risk(self, now: float) -> float:
        risks: list[float] = []
        for sid, _ in self.ready:
            st = self.active.get(sid)
            if st is not None:
                risks.append(self._tail_risk(st, now) / max(1e-9, self.slo_target))
        return max(risks or [0.0])

    def _blocked_fraction_now(self) -> float:
        if not self.active:
            return 0.0
        return sum(1 for st in self.active.values() if st.state in {"BLOCKED_TOOL", "RESUME_SOON"}) / max(1, len(self.active))

    def _snapshot(self, now: float) -> RegimeSnapshot:
        domain_queries = self.cache_hit_tokens + self.recompute_tokens
        remote_pressure = self.remote_kv_bytes / max(1.0, max(1, domain_queries) * self.bpt)
        max_ready = max(
            [max(0.0, now - self.ready_since.get((sid, ev.event_id), now)) for sid, ev in self.ready] or [0.0]
        )
        return RegimeSnapshot(
            now=now,
            ready_queue_len=len(self.ready),
            effective_regions=self.regions,
            backlog_len=len(self.backlog),
            active_sessions=len(self.active),
            blocked_session_fraction=self._blocked_fraction_now(),
            domain_cache_hit_rate=self.cache_hit_tokens / max(1, domain_queries),
            ready_domain_share=self._ready_domain_share(),
            remote_kv_pressure=remote_pressure,
            memory_pressure=self._memory_pressure(),
            p95_risk=self._max_tail_risk(now),
            max_ready_wait=max_ready,
            session_pressure=self.total_sessions / max(1, self.active_limit),
            region_pressure=self.active_limit / max(1, self.regions),
        )

    def _set_regime(self, now: float, decision: str) -> str:
        regime = self.classifier.classify(self._snapshot(now))
        self.current_regime = regime
        self.current_profile = self.profiles.for_regime(regime)
        self.config = self.current_profile
        self.classifier.record_profile(regime, regime.lower(), decision)
        return regime

    def _candidate_score(self, cand: tuple[str, list[Event], str, str, float], now: float) -> float:
        self._set_regime(now, "admit_score")
        return super()._candidate_score(cand, now)

    def _maybe_admit(self, now: float) -> None:
        if not self.backlog:
            return
        regime = self._set_regime(now, "admit")
        if regime == "MEMORY_PRESSURE" and self._memory_pressure() >= self.current_profile.memory_pressure_threshold:
            return
        target = max(1, int(math.ceil(self.current_profile.ready_depth_factor * self.regions)))
        while (
            self.backlog
            and len(self.active) < self.active_limit
            and len(self.ready) < target
            and self._memory_pressure() < self.current_profile.memory_pressure_threshold
        ):
            if not self._admit_one(now):
                break

    def _admit_one(self, now: float, force_first: bool = False) -> bool:
        candidates = [c for c in self.backlog if c[4] <= now or force_first]
        if not candidates or len(self.active) >= self.active_limit:
            return False
        cand = max(candidates, key=lambda c: self._candidate_score(c, now))
        self.backlog.remove(cand)
        sid, evs, domain, repo, arrival = cand
        first = evs[0]
        st = SessionState(
            session_id=sid,
            instance_id=first.instance_id,
            context_domain_id=domain,
            repo_domain_id=repo,
            state="READY_LLM",
            arrival_time=max(now, arrival),
            events=evs,
            shared_context_tokens=_shared_tokens(first),
            private_suffix_tokens=0,
            last_progress_time=now,
        )
        self.active[sid] = st
        self.admission_count += 1
        self._push(st.arrival_time, "SESSION_ARRIVAL", sid)
        return True

    def _score_ready(self, now: float, sid: str, ev: Event) -> float:
        self._set_regime(now, "schedule_score")
        return super()._score_ready(now, sid, ev)

    def _schedule(self, now: float) -> None:
        while self.ready and self.free_regions:
            selected = max(self.ready, key=lambda pair: self._score_ready(now, pair[0], pair[1]))
            self.ready.remove(selected)
            sid, ev = selected
            st = self.active[sid]
            region = self._choose_region(st)
            wait = max(0.0, now - self.ready_since.pop((sid, ev.event_id), now))
            self.ready_wait += wait
            if wait > max(2.0 * self.slo_target, 60.0):
                self.starved_ready_events += 1
            st.state = "RUNNING_LLM"
            st.current_region = region
            self._push(now, "LLM_START", sid, ev, region)

    def _choose_region(self, st: Any) -> int:
        home = self.domain_home_region.setdefault(st.context_domain_id, int(stable_hash(st.context_domain_id), 16) % self.regions)
        region = min(self.free_regions, key=lambda r: abs(r - home))
        self.free_regions.remove(region)
        return region

    def _llm_duration(self, st: Any, ev: Event, region: int) -> float:
        return super()._llm_duration(st, ev, region)

    def run(self) -> dict[str, Any]:
        row = super().run()
        row["policy"] = ADAPTIVE_POLICY
        row["regime_counts"] = json.dumps(dict(self.classifier.regime_counts), sort_keys=True)
        row["policy_profile_counts"] = json.dumps(dict(self.classifier.profile_counts), sort_keys=True)
        row["regime_transition_counts"] = json.dumps(dict(self.classifier.transition_counts), sort_keys=True)
        row["regime_switches"] = sum(self.classifier.transition_counts.values())
        row["per_regime_policy_decisions"] = json.dumps(dict(self.classifier.decision_counts), sort_keys=True)
        return row


def _run_policy(
    traces: list[Trace],
    lm: LatencyModel,
    policy: str,
    total: int,
    limit: int,
    regions: int,
    arrival: str,
    memory: int,
    profiles: AdaptiveProfiles | None,
    v5_config: TAPSUnifiedConfig | None,
) -> dict[str, Any]:
    if policy == ADAPTIVE_POLICY:
        return TAPSAdaptiveReplay(traces, total, limit, regions, arrival, memory, lm, profiles=profiles, seed=307 + total + regions).run()
    mapped = "taps_unified" if policy == "taps_unified_v5" else policy
    config = v5_config if mapped == "taps_unified" and v5_config is not None else None
    row = TAPSUnifiedReplay(traces, total, limit, regions, arrival, memory, mapped, lm, config=config, seed=307 + total + regions).run()
    row["policy"] = policy
    return row


def _pressure_points() -> list[tuple[int, int, int, str, int]]:
    points: set[tuple[int, int, int, str, int]] = set()
    for total in [16, 32, 64, 128]:
        points.add((total, min(16, total), 4, "bursty", 32))
    for limit in [4, 8, 16, 32]:
        points.add((64, limit, 4, "bursty", 32))
    for regions in [1, 2, 4, 8, 16]:
        points.add((64, 16, regions, "bursty", 32))
    for arrival in ["closed_loop", "poisson", "bursty"]:
        points.add((64, 16, 4, arrival, 32))
    for memory in [8, 16, 32, 64]:
        points.add((64, 16, 4, "bursty", memory))
    points.update(
        {
            (128, 32, 16, "bursty", 64),
            (128, 32, 8, "poisson", 32),
            (32, 8, 2, "closed_loop", 16),
            (16, 4, 1, "poisson", 8),
        }
    )
    return sorted(points)


def _fill_adaptive_gains(rows: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[int, int, int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["total_sessions"]),
            int(row["active_session_limit"]),
            int(row["effective_regions"]),
            str(row["arrival_pattern"]),
            int(row["memory_budget_gb"]),
        )
        by_key[key][row["policy"]] = row
    for group in by_key.values():
        adaptive = group.get(ADAPTIVE_POLICY)
        reactive = group.get("reactive_admission")
        baselines = {k: v for k, v in group.items() if k != ADAPTIVE_POLICY}
        if not adaptive:
            continue
        strongest_p95_policy, strongest_p95_row = min(baselines.items(), key=lambda kv: _f(kv[1], "p95_jct"))
        strongest_thr_policy, strongest_thr_row = max(baselines.items(), key=lambda kv: _f(kv[1], "throughput"))
        strongest_wait_policy, strongest_wait_row = min(baselines.items(), key=lambda kv: _f(kv[1], "ready_queue_wait", float("inf")))
        for row in group.values():
            row["strongest_non_oracle_baseline_p95_policy"] = strongest_p95_policy
            row["strongest_non_oracle_baseline_throughput_policy"] = strongest_thr_policy
            row["strongest_non_oracle_baseline_ready_wait_policy"] = strongest_wait_policy
            row["strongest_baseline_p95"] = _f(strongest_p95_row, "p95_jct")
            row["strongest_baseline_throughput"] = _f(strongest_thr_row, "throughput")
            row["strongest_baseline_ready_wait"] = _f(strongest_wait_row, "ready_queue_wait")
            row["p95_gain_over_reactive"] = (
                (_f(reactive, "p95_jct") - _f(adaptive, "p95_jct")) / max(1e-9, _f(reactive, "p95_jct"))
                if reactive
                else 0.0
            )
            row["p95_gain_over_strongest_baseline"] = (
                (_f(strongest_p95_row, "p95_jct") - _f(adaptive, "p95_jct")) / max(1e-9, _f(strongest_p95_row, "p95_jct"))
            )
            row["throughput_gain_over_strongest_baseline"] = (
                (_f(adaptive, "throughput") - _f(strongest_thr_row, "throughput")) / max(1e-9, _f(strongest_thr_row, "throughput"))
            )
            row["ready_wait_gain_over_strongest_baseline"] = (
                (_f(strongest_wait_row, "ready_queue_wait") - _f(adaptive, "ready_queue_wait"))
                / max(1e-9, _f(strongest_wait_row, "ready_queue_wait"))
            )


def run_adaptive_sweep(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/taps_unified_adaptive_pr4_v6.csv",
    profiles: AdaptiveProfiles | None = None,
    v5_config: TAPSUnifiedConfig | None = None,
    points: list[tuple[int, int, int, str, int]] | None = None,
    policies: list[str] | None = None,
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces = _load_traces(trace_dirs)
    lm = LatencyModel.load(model_json)
    policies = policies or V6_POLICIES
    rows: list[dict[str, Any]] = []
    for total, limit, regions, arrival, memory in points or _pressure_points():
        for policy in policies:
            rows.append(_run_policy(traces, lm, policy, total, limit, regions, arrival, memory, profiles, v5_config))
    _fill_adaptive_gains(rows)
    write_csv(out_csv, rows)
    plot_adaptive(rows)
    return rows


def plot_adaptive(rows: list[dict[str, Any]], prefix: str = "data/plots/taps_adaptive") -> None:
    ensure_dir(Path(prefix).parent)
    sub = [
        r
        for r in rows
        if int(r.get("active_session_limit", 0) or 0) == 16
        and int(r.get("effective_regions", 0) or 0) == 4
        and int(r.get("memory_budget_gb", 0) or 0) == 32
        and r.get("arrival_pattern") == "bursty"
    ]
    totals = sorted({int(r["total_sessions"]) for r in sub})
    for metric, suffix, ylabel in [
        ("p95_jct", "p95_pr4_v6.pdf", "p95 JCT"),
        ("throughput", "throughput_pr4_v6.pdf", "throughput"),
    ]:
        plt.figure(figsize=(6.4, 3.8))
        for policy in V6_POLICIES:
            vals = [next((_f(r, metric) for r in sub if r["policy"] == policy and int(r["total_sessions"]) == t), 0.0) for t in totals]
            plt.plot(totals, vals, marker="o", label=policy)
        plt.xlabel("total sessions")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{prefix}_{suffix}")
        plt.close()
    regime_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("policy") == ADAPTIVE_POLICY:
            try:
                regime_counts.update(json.loads(str(row.get("regime_counts", "{}"))))
            except Exception:
                pass
    plt.figure(figsize=(6.4, 3.8))
    keys = sorted(regime_counts)
    plt.bar(keys, [regime_counts[k] for k in keys])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("scheduler epochs")
    plt.tight_layout()
    plt.savefig("data/plots/taps_adaptive_regime_breakdown_pr4_v6.pdf")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/taps_unified_adaptive_pr4_v6.csv")
    args = ap.parse_args()
    rows = run_adaptive_sweep(out_csv=args.out)
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
