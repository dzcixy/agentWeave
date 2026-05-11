from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.metrics import percentile
from agentweaver.simulator.multisession_replay import ToolLatencyPredictor
from agentweaver.simulator.taps_domain_scheduler import context_domain_id, repo_domain_id
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace, load_trace_dir
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv


POLICIES = [
    "static_admission",
    "reactive_admission",
    "acd_nisp",
    "taps_admission_v4",
    "taps_domain_v4",
    "taps_unified",
]


@dataclass
class TAPSUnifiedConfig:
    w_tail: float = 2.0
    w_domain: float = 1.0
    w_batch: float = 0.5
    w_resume: float = 0.5
    w_age: float = 0.05
    w_short: float = 0.5
    w_remote: float = 0.1
    w_switch: float = 0.1
    w_mem: float = 0.1
    ready_depth_factor: float = 2.0
    memory_pressure_threshold: float = 0.85
    admission_domain: float = 1.0
    admission_llm: float = 0.5
    admission_stall: float = 1.0
    admission_tail: float = 1.0
    admission_tool_penalty: float = 0.5
    admission_mem: float = 0.5
    slo_target: float | None = None


@dataclass(order=True)
class SimEvent:
    time: float
    seq: int
    kind: str = field(compare=False)
    session_id: str = field(compare=False)
    event: Event | None = field(default=None, compare=False)
    region_id: int | None = field(default=None, compare=False)


@dataclass
class CacheObject:
    key: str
    domain: str
    tokens: int
    bytes: int
    value: float
    last_access: float
    object_class: str


@dataclass
class SessionState:
    session_id: str
    instance_id: str
    context_domain_id: str
    repo_domain_id: str
    state: str
    arrival_time: float
    events: list[Event]
    elapsed_time: float = 0.0
    predicted_remaining_time: float = 0.0
    predicted_tool_latency: float = 0.0
    actual_tool_latency: float = 0.0
    shared_context_tokens: int = 0
    private_suffix_tokens: int = 0
    resident_context_tokens: int = 0
    parked_state_bytes: int = 0
    last_progress_time: float = 0.0
    age: float = 0.0
    next_index: int = 0
    done_time: float = 0.0
    current_region: int | None = None
    previous_tool_latency: float | None = None
    parked_cached_tokens: int = 0


def _session_events(trace: Trace) -> list[Event]:
    return sorted(
        [e for e in trace.events if e.branch_id != "root" and e.node_type in {"llm", "tool", "verifier"}],
        key=lambda e: (e.step_id, e.timestamp_start or e.timestamp_ready or 0.0, e.node_id),
    )


def _arrival(i: int, pattern: str, regions: int, rng: random.Random) -> float:
    if pattern == "closed_loop":
        return 0.0
    if pattern == "bursty":
        return (i // max(1, regions)) * 0.45 + (i % max(1, regions)) * 0.01
    rate = max(0.35, min(10.0, regions / 1.25))
    return sum(rng.expovariate(rate) for _ in range(i + 1))


def _tool_latency(ev: Event | None) -> float:
    if ev is None:
        return 0.0
    return max(0.0, float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0))


def _shared_tokens(ev: Event | None) -> int:
    if ev is None:
        return 0
    return sum(ref.length for ref in ev.context_segments if ref.segment_type in {"system", "tool_schema", "task", "repo", "history"})


def _private_tokens(ev: Event | None) -> int:
    if ev is None:
        return 0
    return sum(ref.length for ref in ev.context_segments if ref.segment_type not in {"system", "tool_schema", "task", "repo", "history"})


def _cache_key(domain: str, ref: ContextSegmentRef) -> str:
    return f"{domain}:{ref.segment_id}"


class TAPSUnifiedReplay:
    def __init__(
        self,
        traces: list[Trace],
        total_sessions: int,
        active_session_limit: int,
        effective_regions: int,
        arrival_pattern: str,
        memory_budget_gb: int,
        policy: str,
        latency_model: LatencyModel,
        config: TAPSUnifiedConfig | None = None,
        seed: int = 101,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy}; expected {POLICIES}")
        if not traces:
            raise ValueError("no traces available for TAPS-U replay")
        self.traces = traces
        self.total_sessions = total_sessions
        self.active_limit = max(1, active_session_limit)
        self.regions = max(1, effective_regions)
        self.arrival_pattern = arrival_pattern
        self.memory_budget_gb = memory_budget_gb
        self.memory_budget_bytes = max(1, memory_budget_gb) * 1024**3
        self.policy = policy
        self.lm = latency_model
        self.config = config or TAPSUnifiedConfig()
        self.rng = random.Random(seed)
        self.bpt = kv_bytes_per_token()
        self.seq = 0
        self.eventq: list[SimEvent] = []
        self.ready: deque[tuple[str, Event]] = deque()
        self.ready_since: dict[tuple[str, str], float] = {}
        self.free_regions = list(range(self.regions))
        self.backlog: list[tuple[str, list[Event], str, str, float]] = []
        self.active: dict[str, SessionState] = {}
        self.done: list[SessionState] = []
        self.predictor = ToolLatencyPredictor()
        all_events = [e for tr in traces for e in tr.events]
        self.predictor.train(all_events)
        self.predictor_median_abs_error, self.predictor_p95_abs_error = self.predictor.error_summary(all_events)
        self.global_llm_service = self._median_llm_service(all_events)
        self.slo_target = self.config.slo_target or self._historical_slo(traces)
        self.domain_home_region: dict[str, int] = {}
        self.domain_hotness: Counter[str] = Counter()
        self.last_domain: str | None = None
        self.cache: dict[str, CacheObject] = {}
        self.domain_resident_tokens: Counter[str] = Counter()
        self.domain_resident_bytes: Counter[str] = Counter()
        self.memory_occupancy = 0
        self.eviction_count = 0
        self.evicted_tokens = 0
        self.recompute_tokens = 0
        self.cache_hit_tokens = 0
        self.remote_kv_bytes = 0.0
        self.hop_weighted_bytes = 0.0
        self.domain_switches = 0
        self.llm_busy = 0.0
        self.tool_time = 0.0
        self.ready_wait = 0.0
        self.ready_empty_time = 0.0
        self.blocked_time_area = 0.0
        self.last_event_time = 0.0
        self.admission_count = 0
        self.starvation_count = 0
        self.starved_ready_events = 0
        self.max_ready_age = 0.0

    def _median_llm_service(self, events: list[Event]) -> float:
        vals = [self.lm.predict_llm(e.input_tokens, e.output_tokens) for e in events if e.node_type == "llm"]
        vals = sorted(vals)
        return vals[len(vals) // 2] if vals else 1.0

    def _historical_slo(self, traces: list[Trace]) -> float:
        jcts: list[float] = []
        for trace in traces:
            evs = [e for e in _session_events(trace) if e.timestamp_start and e.timestamp_end]
            if evs:
                jcts.append(max(e.timestamp_end for e in evs) - min(e.timestamp_start for e in evs))
        if not jcts:
            jcts = [sum(self._event_ground_duration(e) for e in _session_events(t)) for t in traces]
        return percentile(jcts, 90) if jcts else 60.0

    def _event_ground_duration(self, ev: Event) -> float:
        if ev.node_type == "tool":
            return _tool_latency(ev)
        if ev.node_type == "llm":
            return self.lm.predict_llm(ev.input_tokens, ev.output_tokens)
        return max(0.0, float(ev.latency or 0.0))

    def _push(self, t: float, kind: str, sid: str, event: Event | None = None, region_id: int | None = None) -> None:
        self.seq += 1
        heapq.heappush(self.eventq, SimEvent(t, self.seq, kind, sid, event, region_id))

    def _prepare_backlog(self) -> None:
        for i in range(self.total_sessions):
            trace = self.traces[i % len(self.traces)]
            evs = _session_events(trace)
            if not evs:
                continue
            first = next((e for e in evs if e.node_type == "llm"), evs[0])
            domain = context_domain_id(first)
            repo = repo_domain_id(first.instance_id)
            arrival = _arrival(i, self.arrival_pattern, self.regions, self.rng)
            self.domain_home_region.setdefault(domain, int(stable_hash(domain), 16) % self.regions)
            self.backlog.append((f"s{i}", evs, domain, repo, arrival))
        self.backlog.sort(key=lambda x: x[4])

    def _peek(self, st: SessionState) -> Event | None:
        return st.events[st.next_index] if st.next_index < len(st.events) else None

    def _next(self, st: SessionState) -> Event | None:
        if st.next_index >= len(st.events):
            return None
        ev = st.events[st.next_index]
        st.next_index += 1
        return ev

    def _ready_count_in_domain(self, domain: str, repo: str) -> int:
        return sum(
            1
            for sid, _ in self.ready
            if sid in self.active
            and (self.active[sid].context_domain_id == domain or self.active[sid].repo_domain_id == repo)
        )

    def _resident_tokens_for_event(self, st: SessionState, ev: Event) -> int:
        if self.policy == "static_admission":
            return 0
        return sum(ref.length for ref in ev.context_segments if _cache_key(st.context_domain_id, ref) in self.cache)

    def _memory_pressure(self) -> float:
        return self.memory_occupancy / max(1, self.memory_budget_bytes)

    def _candidate_score(self, cand: tuple[str, list[Event], str, str, float], now: float) -> float:
        sid, evs, domain, repo, arrival = cand
        first_llm = next((e for e in evs if e.node_type == "llm"), None)
        first_tool = next((e for e in evs if e.node_type == "tool"), None)
        resident = self.domain_resident_tokens[domain]
        expected_reuse = resident / max(1, first_llm.input_tokens if first_llm else 1)
        expected_llm_work = min(1.0, (first_llm.input_tokens + first_llm.output_tokens) / 25000.0) if first_llm else 0.2
        predicted_tool = self.predictor.predict(first_tool) if first_tool else self.predictor.global_median
        stall_overlap = min(1.0, predicted_tool / max(1e-9, self.slo_target))
        urgency = 1.0 if len(self.ready) < self.regions else 0.0
        tool_dom = predicted_tool / max(1e-9, predicted_tool + self.global_llm_service)
        return (
            self.config.admission_domain * expected_reuse
            + self.config.admission_llm * expected_llm_work
            + self.config.admission_stall * stall_overlap
            + self.config.admission_tail * urgency
            - self.config.admission_tool_penalty * tool_dom
            - self.config.admission_mem * self._memory_pressure()
            - 0.001 * max(0.0, now - arrival)
            + 1e-6 * int(stable_hash(sid), 16) % 1000
        )

    def _admit_one(self, now: float, force_first: bool = False) -> bool:
        candidates = [c for c in self.backlog if c[4] <= now or force_first]
        if not candidates or len(self.active) >= self.active_limit:
            return False
        if self.policy in {"taps_admission_v4", "taps_unified"}:
            cand = max(candidates, key=lambda c: self._candidate_score(c, now))
        else:
            cand = min(candidates, key=lambda c: c[4])
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
            private_suffix_tokens=_private_tokens(first),
            last_progress_time=now,
        )
        self.active[sid] = st
        self.admission_count += 1
        self._push(st.arrival_time, "SESSION_ARRIVAL", sid)
        return True

    def _maybe_admit(self, now: float) -> None:
        if self.policy == "static_admission":
            if not self.active and self.backlog:
                for _ in range(min(self.active_limit, len(self.backlog))):
                    self._admit_one(now, force_first=True)
            return
        if self.policy in {"reactive_admission", "acd_nisp", "taps_admission_v4", "taps_domain_v4"}:
            target = self.regions
        else:
            target = max(1, int(math.ceil(self.config.ready_depth_factor * self.regions)))
        while (
            self.backlog
            and len(self.active) < self.active_limit
            and len(self.ready) < target
            and (self.policy != "taps_unified" or self._memory_pressure() < self.config.memory_pressure_threshold)
        ):
            if not self._admit_one(now):
                break

    def _initial_admit(self) -> None:
        self._prepare_backlog()
        for _ in range(min(self.active_limit, len(self.backlog))):
            self._admit_one(0.0, force_first=True)

    def _predicted_remaining(self, st: SessionState, now: float) -> float:
        # Online scheduler uses structural depth plus historical medians; it does not use exact future tool latencies.
        remaining = st.events[st.next_index :]
        llm_count = sum(1 for e in remaining if e.node_type == "llm")
        tool_count = sum(1 for e in remaining if e.node_type == "tool")
        verifier_count = sum(1 for e in remaining if e.node_type == "verifier")
        current_block = max(0.0, st.predicted_tool_latency - max(0.0, now - st.last_progress_time)) if st.state == "BLOCKED_TOOL" else 0.0
        return llm_count * self.global_llm_service + tool_count * self.predictor.global_median + verifier_count * 0.1 + current_block

    def _tail_risk(self, st: SessionState, now: float) -> float:
        st.elapsed_time = max(0.0, now - st.arrival_time)
        st.predicted_remaining_time = self._predicted_remaining(st, now)
        return max(0.0, st.elapsed_time + st.predicted_remaining_time - self.slo_target)

    def _remote_cost(self, st: SessionState, region: int, cached_tokens: int) -> tuple[float, float]:
        home = self.domain_home_region.setdefault(st.context_domain_id, int(stable_hash(st.context_domain_id), 16) % self.regions)
        hops = abs(region - home)
        return float(hops), cached_tokens * self.bpt

    def _eviction_penalty(self, st: SessionState) -> float:
        if self.memory_occupancy <= self.memory_budget_bytes:
            return 0.0
        domain_bytes = self.domain_resident_bytes[st.context_domain_id]
        return domain_bytes / max(1, self.memory_budget_bytes)

    def _score_ready(self, now: float, sid: str, ev: Event) -> float:
        st = self.active[sid]
        cached = min(ev.input_tokens, self._resident_tokens_for_event(st, ev) + st.parked_cached_tokens)
        domain_hit = cached / max(1, ev.input_tokens)
        batch = self._ready_count_in_domain(st.context_domain_id, st.repo_domain_id)
        resume = 0.0
        if st.predicted_tool_latency:
            resume = math.exp(-max(0.0, st.predicted_tool_latency - max(0.0, now - st.last_progress_time)) / max(1e-9, self.slo_target / 8.0))
        age = max(0.0, now - self.ready_since.get((sid, ev.event_id), now))
        service = self.lm.predict_llm(ev.input_tokens, ev.output_tokens, cached)
        short = 1.0 / max(1e-9, service)
        home = self.domain_home_region.setdefault(st.context_domain_id, int(stable_hash(st.context_domain_id), 16) % self.regions)
        nearest = min(self.free_regions or [home], key=lambda r: abs(r - home))
        hops, bytes_ = self._remote_cost(st, nearest, cached)
        switch = 1.0 if self.last_domain and self.last_domain != st.context_domain_id else 0.0
        return (
            self.config.w_tail * self._tail_risk(st, now)
            + self.config.w_domain * domain_hit
            + self.config.w_batch * batch
            + self.config.w_resume * resume
            + self.config.w_age * age
            + self.config.w_short * short
            - self.config.w_remote * (bytes_ * hops / 1e9)
            - self.config.w_switch * switch
            - self.config.w_mem * self._eviction_penalty(st)
        )

    def _choose_region(self, st: SessionState) -> int:
        if self.policy in {"taps_domain_v4", "taps_unified"}:
            home = self.domain_home_region.setdefault(st.context_domain_id, int(stable_hash(st.context_domain_id), 16) % self.regions)
            region = min(self.free_regions, key=lambda r: abs(r - home))
        else:
            region = self.free_regions[0]
        self.free_regions.remove(region)
        return region

    def _schedule(self, now: float) -> None:
        while self.ready and self.free_regions:
            if self.policy == "taps_unified":
                selected = max(self.ready, key=lambda pair: self._score_ready(now, pair[0], pair[1]))
            elif self.policy == "taps_domain_v4":
                selected = max(
                    self.ready,
                    key=lambda pair: (
                        self._resident_tokens_for_event(self.active[pair[0]], pair[1]) / max(1, pair[1].input_tokens),
                        self._ready_count_in_domain(self.active[pair[0]].context_domain_id, self.active[pair[0]].repo_domain_id),
                        now - self.ready_since.get((pair[0], pair[1].event_id), now),
                    ),
                )
            else:
                selected = self.ready[0]
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

    def _enqueue_next(self, now: float, st: SessionState) -> None:
        nxt = self._next(st)
        if nxt is None:
            st.state = "DONE"
            st.done_time = now
            self.done.append(st)
            self.active.pop(st.session_id, None)
            self._maybe_admit(now)
            return
        if nxt.node_type == "llm":
            st.state = "READY_LLM"
            st.shared_context_tokens = _shared_tokens(nxt)
            st.private_suffix_tokens = _private_tokens(nxt)
            self.ready.append((st.session_id, nxt))
            self.ready_since[(st.session_id, nxt.event_id)] = now
            self._schedule(now)
        elif nxt.node_type == "tool":
            st.state = "BLOCKED_TOOL"
            self._push(now, "TOOL_START", st.session_id, nxt)
        elif nxt.node_type == "verifier":
            self._push(now + max(0.0, float(nxt.latency or 0.0)), "SESSION_DONE", st.session_id, nxt)

    def _cache_value(self, st: SessionState, ref: ContextSegmentRef, now: float) -> float:
        saved = self.lm.predict_prefill(ref.length)
        tail = 1.0 + min(4.0, self._tail_risk(st, now) / max(1.0, self.slo_target))
        reuse = 1.5 if ref.segment_type in {"system", "tool_schema", "task", "repo", "history"} else 0.5
        return reuse * saved * tail / max(1, ref.length * self.bpt)

    def _insert_cache(self, st: SessionState, ev: Event, now: float) -> None:
        if self.policy == "static_admission":
            return
        for ref in ev.context_segments:
            key = _cache_key(st.context_domain_id, ref)
            bytes_ = int(ref.length * self.bpt)
            if key in self.cache:
                obj = self.cache[key]
                obj.last_access = now
                obj.value += self._cache_value(st, ref, now) * 0.1
                continue
            self.cache[key] = CacheObject(
                key=key,
                domain=st.context_domain_id,
                tokens=ref.length,
                bytes=bytes_,
                value=self._cache_value(st, ref, now),
                last_access=now,
                object_class=ref.segment_type,
            )
            self.memory_occupancy += bytes_
            self.domain_resident_tokens[st.context_domain_id] += ref.length
            self.domain_resident_bytes[st.context_domain_id] += bytes_
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while self.memory_occupancy > self.memory_budget_bytes and self.cache:
            victim = min(self.cache.values(), key=lambda o: (o.value, o.last_access))
            self.memory_occupancy -= victim.bytes
            self.domain_resident_tokens[victim.domain] -= victim.tokens
            self.domain_resident_bytes[victim.domain] -= victim.bytes
            self.evicted_tokens += victim.tokens
            self.eviction_count += 1
            self.cache.pop(victim.key, None)

    def _llm_duration(self, st: SessionState, ev: Event, region: int) -> float:
        if self.policy == "static_admission":
            cached = 0
        else:
            cached = min(ev.input_tokens, self._resident_tokens_for_event(st, ev) + st.parked_cached_tokens)
        self.cache_hit_tokens += cached
        self.recompute_tokens += max(0, ev.input_tokens - cached)
        hops, bytes_ = self._remote_cost(st, region, cached)
        self.remote_kv_bytes += bytes_ * hops
        self.hop_weighted_bytes += bytes_ * hops
        remote_penalty = (bytes_ * hops) / 2e11
        st.parked_cached_tokens = 0
        return self.lm.predict_llm(ev.input_tokens, ev.output_tokens, cached, concurrency=max(1, self.regions)) + remote_penalty

    def _account_idle(self, now: float) -> None:
        dt = max(0.0, now - self.last_event_time)
        if not self.ready and self.free_regions:
            self.ready_empty_time += dt
        if self.active:
            blocked = sum(1 for st in self.active.values() if st.state in {"BLOCKED_TOOL", "RESUME_SOON"})
            self.blocked_time_area += blocked * dt
        for sid, ev in self.ready:
            age = max(0.0, now - self.ready_since.get((sid, ev.event_id), now))
            self.max_ready_age = max(self.max_ready_age, age)
        self.last_event_time = now

    def run(self) -> dict[str, Any]:
        self._initial_admit()
        last = 0.0
        while self.eventq:
            item = heapq.heappop(self.eventq)
            now = item.time
            self._account_idle(now)
            last = max(last, now)
            # Allow poisson/bursty arrivals to enter when clock reaches their arrival.
            self._maybe_admit(now)
            st = self.active.get(item.session_id)
            if item.kind == "SESSION_ARRIVAL" and st is not None:
                self._enqueue_next(now, st)
            elif item.kind == "LLM_START" and st is not None and item.event is not None and item.region_id is not None:
                if self.last_domain and self.last_domain != st.context_domain_id:
                    self.domain_switches += 1
                self.last_domain = st.context_domain_id
                dur = self._llm_duration(st, item.event, item.region_id)
                self.llm_busy += dur
                self._push(now + dur, "LLM_DONE", item.session_id, item.event, item.region_id)
            elif item.kind == "LLM_DONE" and st is not None:
                if item.event is not None:
                    self.domain_hotness[st.context_domain_id] += 1
                    self._insert_cache(st, item.event, now)
                if item.region_id is not None:
                    self.free_regions.append(item.region_id)
                    self.free_regions.sort()
                st.last_progress_time = now
                self._enqueue_next(now, st)
                self._maybe_admit(now)
                self._schedule(now)
            elif item.kind == "TOOL_START" and st is not None:
                actual = _tool_latency(item.event)
                pred = self.predictor.predict(item.event, st)
                st.predicted_tool_latency = pred
                st.actual_tool_latency = actual
                st.state = "BLOCKED_TOOL"
                self.tool_time += actual
                nxt = self._peek(st)
                if nxt is not None and nxt.node_type == "llm" and self.policy != "static_admission":
                    st.parked_cached_tokens = _shared_tokens(nxt)
                    st.parked_state_bytes = int(st.parked_cached_tokens * self.bpt)
                self._maybe_admit(now)
                self._schedule(now)
                self._push(now + actual, "TOOL_DONE", item.session_id, item.event)
            elif item.kind == "TOOL_DONE" and st is not None:
                st.previous_tool_latency = st.actual_tool_latency
                st.state = "RESUME_SOON"
                st.last_progress_time = now
                self._enqueue_next(now, st)
                self._schedule(now)
            elif item.kind == "SESSION_DONE" and st is not None:
                st.state = "DONE"
                st.done_time = now
                self.done.append(st)
                self.active.pop(st.session_id, None)
                self._maybe_admit(now)
                self._schedule(now)

        jcts = [st.done_time - st.arrival_time for st in self.done]
        makespan = max([st.done_time for st in self.done] or [last, 1e-9])
        total_region_time = max(1e-9, makespan * self.regions)
        self.starvation_count = self.starved_ready_events + max(0, self.total_sessions - len(self.done))
        domain_queries = self.cache_hit_tokens + self.recompute_tokens
        return {
            "total_sessions": self.total_sessions,
            "active_session_limit": self.active_limit,
            "effective_regions": self.regions,
            "arrival_pattern": self.arrival_pattern,
            "memory_budget_gb": self.memory_budget_gb,
            "policy": self.policy,
            "throughput": len(self.done) / max(1e-9, makespan),
            "mean_jct": sum(jcts) / max(1, len(jcts)),
            "p50_jct": percentile(jcts, 50),
            "p95_jct": percentile(jcts, 95),
            "p99_jct": percentile(jcts, 99),
            "ready_queue_wait": self.ready_wait,
            "ready_queue_empty_time": self.ready_empty_time,
            "region_utilization": self.llm_busy / total_region_time,
            "blocked_session_fraction": self.blocked_time_area / max(1e-9, makespan * max(1, self.active_limit)),
            "domain_cache_hit_rate": self.cache_hit_tokens / max(1, domain_queries),
            "remote_kv_bytes": self.remote_kv_bytes,
            "avg_context_hops": self.hop_weighted_bytes / max(1.0, self.remote_kv_bytes),
            "domain_switches": self.domain_switches,
            "memory_occupancy": self.memory_occupancy,
            "eviction_count": self.eviction_count,
            "evicted_tokens": self.evicted_tokens,
            "recompute_tokens": self.recompute_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "completed_sessions": len(self.done),
            "admission_count": self.admission_count,
            "starvation_count": self.starvation_count,
            "predictor_median_abs_error": self.predictor_median_abs_error,
            "predictor_p95_abs_error": self.predictor_p95_abs_error,
            "slo_target": self.slo_target,
        }


def _load_traces(trace_dirs: list[str | Path]) -> list[Trace]:
    traces: list[Trace] = []
    for d in trace_dirs:
        p = Path(d)
        if p.exists():
            traces.extend(load_trace_dir(p))
    return traces


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _fill_gains(rows: list[dict[str, Any]]) -> None:
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
        taps = group.get("taps_unified")
        reactive = group.get("reactive_admission")
        acd = group.get("acd_nisp")
        if not taps:
            continue
        strongest_p95 = min([_f(r, "p95_jct", float("inf")) for p, r in group.items() if p != "taps_unified"] or [0.0])
        strongest_thr = max([_f(r, "throughput") for p, r in group.items() if p != "taps_unified"] or [0.0])
        for row in group.values():
            row["taps_u_p95_gain_over_reactive"] = (
                (_f(reactive, "p95_jct") - _f(taps, "p95_jct")) / max(1e-9, _f(reactive, "p95_jct"))
                if reactive
                else 0.0
            )
            row["taps_u_p95_gain_over_acd_nisp"] = (
                (_f(acd, "p95_jct") - _f(taps, "p95_jct")) / max(1e-9, _f(acd, "p95_jct"))
                if acd
                else 0.0
            )
            row["taps_u_throughput_gain_over_reactive"] = (
                (_f(taps, "throughput") - _f(reactive, "throughput")) / max(1e-9, _f(reactive, "throughput"))
                if reactive
                else 0.0
            )
            row["taps_u_ready_wait_gain_over_reactive"] = (
                (_f(reactive, "ready_queue_wait") - _f(taps, "ready_queue_wait")) / max(1e-9, _f(reactive, "ready_queue_wait"))
                if reactive
                else 0.0
            )
            row["taps_u_region_util_gain_over_reactive"] = (
                _f(taps, "region_utilization") - _f(reactive, "region_utilization") if reactive else 0.0
            )
            row["strongest_baseline_p95"] = strongest_p95
            row["strongest_baseline_throughput"] = strongest_thr
            row["taps_u_p95_gain_over_strongest"] = (
                (strongest_p95 - _f(taps, "p95_jct")) / max(1e-9, strongest_p95) if strongest_p95 < float("inf") else 0.0
            )
            row["taps_u_throughput_gain_over_strongest"] = (
                (_f(taps, "throughput") - strongest_thr) / max(1e-9, strongest_thr) if strongest_thr else 0.0
            )


def run_taps_unified_sweep(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/taps_unified_pr4_v5.csv",
    total_sessions_list: list[int] | None = None,
    active_limits: list[int] | None = None,
    effective_regions_list: list[int] | None = None,
    arrival_patterns: list[str] | None = None,
    memory_budgets_gb: list[int] | None = None,
    policies: list[str] | None = None,
    config: TAPSUnifiedConfig | None = None,
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces = _load_traces(trace_dirs)
    lm = LatencyModel.load(model_json)
    total_sessions_list = total_sessions_list or [16, 32, 64, 128]
    active_limits = active_limits or [4, 8, 16, 32]
    effective_regions_list = effective_regions_list or [1, 2, 4, 8, 16]
    arrival_patterns = arrival_patterns or ["closed_loop", "poisson", "bursty"]
    memory_budgets_gb = memory_budgets_gb or [8, 16, 32, 64]
    policies = policies or POLICIES
    rows: list[dict[str, Any]] = []
    for total, limit, regions, arrival, mem in itertools.product(
        total_sessions_list, active_limits, effective_regions_list, arrival_patterns, memory_budgets_gb
    ):
        if limit > total:
            continue
        for policy in policies:
            rows.append(
                TAPSUnifiedReplay(
                    traces,
                    total,
                    limit,
                    regions,
                    arrival,
                    mem,
                    policy,
                    lm,
                    config=config,
                    seed=211 + total + limit * 3 + regions * 7 + mem,
                ).run()
            )
    _fill_gains(rows)
    write_csv(out_csv, rows)
    plot_taps_unified(rows)
    return rows


def plot_taps_unified(rows: list[dict[str, Any]], out: str | Path = "data/plots/taps_unified_pr4_v5.pdf") -> None:
    ensure_dir(Path(out).parent)
    sub = [
        r
        for r in rows
        if int(r.get("active_session_limit", 0) or 0) == 16
        and int(r.get("effective_regions", 0) or 0) == 4
        and int(r.get("memory_budget_gb", 0) or 0) == 32
        and r.get("arrival_pattern") == "bursty"
    ]
    totals = sorted({int(r["total_sessions"]) for r in sub})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for policy in POLICIES:
        p95 = [next((_f(r, "p95_jct") for r in sub if r["policy"] == policy and int(r["total_sessions"]) == total), 0.0) for total in totals]
        thr = [next((_f(r, "throughput") for r in sub if r["policy"] == policy and int(r["total_sessions"]) == total), 0.0) for total in totals]
        axes[0].plot(totals, p95, marker="o", label=policy)
        axes[1].plot(totals, thr, marker="o", label=policy)
    axes[0].set_xlabel("total sessions")
    axes[0].set_ylabel("p95 JCT (s)")
    axes[1].set_xlabel("total sessions")
    axes[1].set_ylabel("throughput (sessions/s)")
    axes[0].legend(fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/taps_unified_pr4_v5.csv")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--trace-dirs", default="data/traces/mini_swe_lite10_r4_timed,data/traces/mini_swe_lite5_patchcap_verified")
    args = ap.parse_args()
    rows = run_taps_unified_sweep([x for x in args.trace_dirs.split(",") if x], args.model_json, args.out)
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
