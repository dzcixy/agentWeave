from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.context_domain_factorization import selected_segment_ids
from agentweaver.simulator.metrics import percentile
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


POLICIES = ["naive_wafer", "acd_nisp", "acd_cdf_nisp", "taps", "taps_oracle", "taps_predictive"]


@dataclass(order=True)
class SimEvent:
    time: float
    seq: int
    kind: str = field(compare=False)
    session_id: str = field(compare=False)
    event: Event | None = field(default=None, compare=False)
    region_id: int | None = field(default=None, compare=False)


@dataclass
class SessionState:
    session_id: str
    instance_id: str
    trace_name: str
    arrival_time: float
    events: list[Event]
    next_index: int = 0
    done_time: float = 0.0
    last_tool_return_time: float = 0.0
    predicted_tool_return_time: float = 0.0
    previous_tool_latency: float | None = None
    parked_cached_tokens: int = 0
    parked_resume_tokens: int = 0


def command_class(command: str | None) -> str:
    cmd = (command or "").strip().lower()
    if not cmd:
        return "none"
    if any(x in cmd for x in ("pytest", "tox", "manage.py test", "unittest")):
        return "test"
    if any(x in cmd for x in ("sed ", "python - <<", "cat >", "perl ", "apply_patch")):
        return "file_write"
    if any(x in cmd for x in ("cat ", "grep", "rg ", "sed -n", "ls ", "find ")):
        return "file_read"
    if any(x in cmd for x in ("git diff", "git status", "git ")):
        return "git"
    return "shell_other"


class ToolLatencyPredictor:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, str, float]] = []
        self.global_median = 0.0

    @staticmethod
    def _median(vals: list[float]) -> float:
        vals = sorted(v for v in vals if v >= 0)
        if not vals:
            return 0.0
        return vals[len(vals) // 2]

    def train(self, events: list[Event]) -> None:
        self.records = []
        for ev in events:
            if ev.node_type != "tool":
                continue
            lat = ev.tool_latency if ev.tool_latency is not None else ev.latency
            if lat is None:
                continue
            self.records.append((ev.instance_id, ev.tool_type or "shell_other", command_class(ev.command), max(0.0, float(lat))))
        self.global_median = self._median([r[3] for r in self.records])

    def predict(self, ev: Event | None, previous_latency: float | None = None) -> float:
        if ev is None:
            return self.global_median
        inst = ev.instance_id
        typ = ev.tool_type or "shell_other"
        cls = command_class(ev.command)
        candidates = [lat for i, t, c, lat in self.records if i != inst and t == typ and c == cls]
        if not candidates:
            candidates = [lat for i, t, _, lat in self.records if i != inst and t == typ]
        if not candidates:
            candidates = [lat for i, _, c, lat in self.records if i != inst and c == cls]
        pred = self._median(candidates) if candidates else self.global_median
        if previous_latency is not None and previous_latency >= 0:
            pred = 0.7 * pred + 0.3 * previous_latency
        return max(0.0, pred)

    def error_summary(self, events: list[Event]) -> tuple[float, float]:
        errs: list[float] = []
        prev_by_session: dict[str, float] = {}
        for ev in sorted([e for e in events if e.node_type == "tool"], key=lambda e: (e.session_id, e.timestamp_start or e.timestamp_ready or 0.0)):
            actual = ev.tool_latency if ev.tool_latency is not None else ev.latency
            if actual is None:
                continue
            pred = self.predict(ev, prev_by_session.get(ev.session_id))
            errs.append(abs(pred - max(0.0, float(actual))))
            prev_by_session[ev.session_id] = max(0.0, float(actual))
        if not errs:
            return 0.0, 0.0
        errs = sorted(errs)
        p95_idx = min(len(errs) - 1, max(0, int(round(0.95 * (len(errs) - 1)))))
        return errs[len(errs) // 2], errs[p95_idx]


class MultiSessionReplay:
    def __init__(
        self,
        traces: list[Trace],
        sessions: int,
        policy: str,
        latency_model: LatencyModel,
        effective_regions: int = 4,
        arrival_pattern: str = "closed_loop",
        run_id: str = "pr4_algo",
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy}; expected {POLICIES}")
        if not traces:
            raise ValueError("no traces available for multisession replay")
        self.traces = traces
        self.sessions_n = sessions
        self.policy = policy
        self.lm = latency_model
        self.effective_regions = max(1, int(effective_regions))
        self.arrival_pattern = arrival_pattern
        self.run_id = run_id
        self.bpt = kv_bytes_per_token()
        self.seq = 0
        self.eventq: list[SimEvent] = []
        self.ready: deque[tuple[str, Event]] = deque()
        self.ready_since: dict[tuple[str, str], float] = {}
        self.free_regions = list(range(self.effective_regions))
        self.sessions: dict[str, SessionState] = {}
        self.arena_resident: set[str] = set()
        self.cdf_selected = selected_segment_ids([ev for tr in traces for ev in tr.events], latency_model)
        self.predictor = ToolLatencyPredictor()
        self.predictor.train([ev for tr in traces for ev in tr.events])
        self.predictor_median_abs_error, self.predictor_p95_abs_error = self.predictor.error_summary([ev for tr in traces for ev in tr.events])
        self.cdf_seen: set[str] = set()
        self.llm_busy_time = 0.0
        self.model_side_latency = 0.0
        self.tool_time = 0.0
        self.ready_queue_wait = 0.0
        self.tool_blocked_compute_waste = 0.0
        self.resume_latency = 0.0
        self.prefill_tokens_avoided = 0
        self.resume_prefill_tokens = 0
        self.state_prefetch_bytes = 0
        self.exposed_state_migration_latency = 0.0
        self.prefill_tokens = 0

    def _push(self, time: float, kind: str, session_id: str, event: Event | None = None, region_id: int | None = None) -> None:
        self.seq += 1
        heapq.heappush(self.eventq, SimEvent(time, self.seq, kind, session_id, event, region_id))

    def _session_events(self, trace: Trace) -> list[Event]:
        return sorted(
            [e for e in trace.events if e.branch_id != "root" and e.node_type in {"llm", "tool", "verifier"}],
            key=lambda e: (e.step_id, e.timestamp_start or e.timestamp_ready or 0.0),
        )

    def _init_sessions(self) -> None:
        for i in range(self.sessions_n):
            tr = self.traces[i % len(self.traces)]
            evs = self._session_events(tr)
            if not evs:
                continue
            arrival = 0.0 if self.arrival_pattern == "closed_loop" else i * 0.5
            sid = f"s{i}"
            first_instance = evs[0].instance_id
            self.sessions[sid] = SessionState(sid, first_instance, str(tr.metadata.get("source", "")), arrival, evs)
            self._push(arrival, "SESSION_ARRIVAL", sid)

    def _next_node(self, st: SessionState) -> Event | None:
        if st.next_index >= len(st.events):
            return None
        ev = st.events[st.next_index]
        st.next_index += 1
        return ev

    def _peek_next_node(self, st: SessionState) -> Event | None:
        return st.events[st.next_index] if st.next_index < len(st.events) else None

    def _shared_private_tokens(self, ev: Event) -> tuple[int, int]:
        shared = 0
        private = 0
        for seg in ev.context_segments:
            if seg.segment_type in {"system", "tool_schema", "task", "repo", "history"}:
                shared += seg.length
            else:
                private += seg.length
        return shared, max(0, ev.input_tokens - shared) if private == 0 else private

    def _resident_match(self, refs: list[ContextSegmentRef]) -> int:
        return sum(ref.length for ref in refs if ref.segment_id in self.arena_resident)

    def _cached_tokens(self, st: SessionState, ev: Event) -> tuple[int, int, int]:
        natural = self._resident_match(ev.context_segments) if self.policy in {"acd_nisp", "acd_cdf_nisp", "taps"} else 0
        cdf_added = 0
        if self.policy in {"acd_cdf_nisp", "taps", "taps_oracle", "taps_predictive"}:
            cdf_refs = [ref for ref in ev.context_segments if ref.segment_id in self.cdf_selected and ref.segment_id in self.cdf_seen]
            eligible = sum(ref.length for ref in cdf_refs)
            resident = self._resident_match(cdf_refs)
            cdf_added = max(0, eligible - resident)
        nisp_cached = 0
        if self.policy in {"acd_nisp", "acd_cdf_nisp", "taps", "taps_oracle", "taps_predictive"} and st.parked_cached_tokens:
            nisp_cached = st.parked_cached_tokens
            st.parked_cached_tokens = 0
            self.resume_prefill_tokens += st.parked_resume_tokens
            st.parked_resume_tokens = 0
        cached = min(ev.input_tokens, max(natural + cdf_added, nisp_cached))
        cdf_added = min(cdf_added, cached)
        return cached, natural, cdf_added

    def _llm_service(self, st: SessionState, ev: Event) -> tuple[float, int, int, int]:
        cached, natural, cdf_added = self._cached_tokens(st, ev)
        delta = max(0, ev.input_tokens - cached)
        self.prefill_tokens += delta
        self.prefill_tokens_avoided += cached
        return self.lm.predict_prefill(delta) + self.lm.predict_decode(ev.context_length or ev.input_tokens, ev.output_tokens), cached, natural, cdf_added

    def _tool_latency(self, ev: Event | None) -> float:
        if ev is None:
            return 0.0
        if ev.tool_latency is not None:
            return max(0.0, float(ev.tool_latency))
        return max(0.0, float(ev.latency or 0.0))

    def _score(self, now: float, st: SessionState, ev: Event) -> float:
        cached = self._resident_match(ev.context_segments)
        locality_gain = self.lm.predict_prefill(cached)
        return_time = st.predicted_tool_return_time if self.policy in {"taps_predictive", "taps"} else st.last_tool_return_time
        resume_urgency = 1.0 / (1.0 + max(0.0, return_time - now)) if return_time else 0.0
        remaining = max(1, sum(1 for e in st.events[st.next_index :] if e.node_type in {"llm", "tool", "verifier"}))
        criticality = 1.0 / (1.0 + remaining)
        age = now - self.ready_since.get((st.session_id, ev.event_id), now)
        predicted = self.lm.predict_prefill(max(0, ev.input_tokens - cached)) + self.lm.predict_decode(ev.context_length or ev.input_tokens, ev.output_tokens)
        estimated_fetch_bytes = sum((seg.kv_bytes or seg.length * self.bpt) for seg in ev.context_segments if seg.segment_id in self.arena_resident)
        noc_cost = estimated_fetch_bytes / 1e12
        eviction_penalty = 0.0
        return (locality_gain + resume_urgency + criticality + 0.05 * age) / (predicted + 0.2 * noc_cost + 0.1 * eviction_penalty + 1e-9)

    def _try_schedule(self, now: float) -> None:
        while self.ready and self.free_regions:
            if self.policy in {"taps", "taps_oracle", "taps_predictive"}:
                selected = max(self.ready, key=lambda pair: self._score(now, self.sessions[pair[0]], pair[1]))
            else:
                selected = self.ready[0]
            self.ready.remove(selected)
            sid, ev = selected
            region = self.free_regions.pop(0)
            self.ready_queue_wait += max(0.0, now - self.ready_since.pop((sid, ev.event_id), now))
            self._push(now, "LLM_START", sid, ev, region)

    def _enqueue_next(self, now: float, st: SessionState) -> None:
        nxt = self._next_node(st)
        if nxt is None:
            st.done_time = now
        elif nxt.node_type == "llm":
            self.ready.append((st.session_id, nxt))
            self.ready_since[(st.session_id, nxt.event_id)] = now
            self._try_schedule(now)
        elif nxt.node_type == "tool":
            self._push(now, "TOOL_START", st.session_id, nxt)
        elif nxt.node_type == "verifier":
            self._push(now + max(0.0, float(nxt.latency or 0.0)), "SESSION_DONE", st.session_id, nxt)

    def run(self) -> dict[str, Any]:
        self._init_sessions()
        last_time = 0.0
        while self.eventq:
            event = heapq.heappop(self.eventq)
            now = event.time
            last_time = max(last_time, now)
            st = self.sessions.get(event.session_id)
            if st is None:
                continue
            if event.kind == "SESSION_ARRIVAL":
                self._enqueue_next(now, st)
            elif event.kind == "LLM_START":
                if event.event is None or event.region_id is None:
                    continue
                duration, _, _, _ = self._llm_service(st, event.event)
                self.llm_busy_time += duration
                self.model_side_latency += duration
                self._push(now + duration, "LLM_DONE", event.session_id, event.event, event.region_id)
            elif event.kind == "LLM_DONE":
                if event.event is not None:
                    for ref in event.event.context_segments:
                        self.arena_resident.add(ref.segment_id)
                        if ref.segment_id in self.cdf_selected:
                            self.cdf_seen.add(ref.segment_id)
                if event.region_id is not None:
                    self.free_regions.append(event.region_id)
                self._enqueue_next(now, st)
                self._try_schedule(now)
            elif event.kind == "TOOL_START":
                latency = self._tool_latency(event.event)
                self.tool_time += latency
                if self.policy == "naive_wafer":
                    if not self.free_regions:
                        self._push(now + latency, "TOOL_DONE", event.session_id, event.event, None)
                    else:
                        region = self.free_regions.pop(0)
                        self.tool_blocked_compute_waste += latency
                        self._push(now + latency, "TOOL_DONE", event.session_id, event.event, region)
                else:
                    next_llm = self._peek_next_node(st)
                    if next_llm and next_llm.node_type == "llm":
                        shared, private = self._shared_private_tokens(next_llm)
                        if self.policy in {"taps", "taps_oracle", "taps_predictive"}:
                            st.parked_cached_tokens = shared + private
                            st.parked_resume_tokens = event.event.observation_tokens if event.event else 0
                            self.state_prefetch_bytes += shared * self.bpt
                        else:
                            st.parked_cached_tokens = shared
                            st.parked_resume_tokens = private + (event.event.observation_tokens if event.event else 0)
                    predicted_latency = latency if self.policy == "taps_oracle" else self.predictor.predict(event.event, st.previous_tool_latency)
                    st.predicted_tool_return_time = now + predicted_latency
                    st.last_tool_return_time = now + latency
                    st.previous_tool_latency = latency
                    self._push(now + latency, "TOOL_DONE", event.session_id, event.event, None)
                    self._try_schedule(now)
            elif event.kind == "TOOL_DONE":
                if event.region_id is not None:
                    self.free_regions.append(event.region_id)
                self._enqueue_next(now, st)
                self._try_schedule(now)
            elif event.kind == "SESSION_DONE":
                st.done_time = now
        completed = [st for st in self.sessions.values() if st.done_time > 0]
        jcts = [st.done_time - st.arrival_time for st in completed]
        makespan = max([st.done_time for st in completed] or [last_time, 1e-9])
        total_region_time = max(1e-9, makespan * self.effective_regions)
        return {
            "run_id": self.run_id,
            "sessions": self.sessions_n,
            "arrival_pattern": self.arrival_pattern,
            "policy": self.policy,
            "throughput_sessions_per_sec": len(completed) / max(1e-9, makespan),
            "mean_jct": sum(jcts) / max(1, len(jcts)),
            "p50_jct": percentile(jcts, 50),
            "p95_jct": percentile(jcts, 95),
            "p99_jct": percentile(jcts, 99),
            "model_side_latency": self.model_side_latency,
            "tool_time": self.tool_time,
            "ready_queue_wait": self.ready_queue_wait,
            "region_utilization": self.llm_busy_time / total_region_time,
            "tool_blocked_compute_waste": self.tool_blocked_compute_waste,
            "resume_latency": self.resume_latency,
            "prefill_tokens_avoided": self.prefill_tokens_avoided,
            "resume_prefill_tokens": self.resume_prefill_tokens,
            "state_prefetch_bytes": self.state_prefetch_bytes,
            "exposed_state_migration_latency": self.exposed_state_migration_latency,
            "completed_sessions": len(completed),
            "predictor_median_abs_error": self.predictor_median_abs_error,
            "predictor_p95_abs_error": self.predictor_p95_abs_error,
        }


def run_multisession(
    trace_dir: str | Path = "data/traces/mini_swe_lite10_r4_timed",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/multisession_taps_pr4_algo.csv",
    sessions_list: list[int] | None = None,
    effective_regions: int = 4,
    arrival_pattern: str = "closed_loop",
    run_id: str = "pr4_algo",
    plot_prefix: str = "data/plots/multisession_taps",
    plot_suffix: str = "pr4_algo",
) -> list[dict[str, Any]]:
    sessions_list = sessions_list or [1, 2, 4, 8, 16]
    traces = load_trace_dir(trace_dir)
    lm = LatencyModel.load(model_json)
    rows: list[dict[str, Any]] = []
    for sessions in sessions_list:
        for policy in POLICIES:
            rows.append(MultiSessionReplay(traces, sessions, policy, lm, effective_regions, arrival_pattern, run_id).run())
    _enrich_predictive_metrics(rows)
    write_csv(out_csv, rows)
    plot_multisession(rows, plot_prefix=plot_prefix, plot_suffix=plot_suffix)
    return rows


def _enrich_predictive_metrics(rows: list[dict[str, Any]]) -> None:
    for sessions in sorted({int(r["sessions"]) for r in rows}):
        by_policy = {r["policy"]: r for r in rows if int(r["sessions"]) == sessions}
        base = by_policy.get("acd_nisp", {})
        pred = by_policy.get("taps_predictive", {})
        oracle = by_policy.get("taps_oracle", {})
        base_thr = float(base.get("throughput_sessions_per_sec", 0.0) or 0.0)
        pred_thr = float(pred.get("throughput_sessions_per_sec", 0.0) or 0.0)
        oracle_thr = float(oracle.get("throughput_sessions_per_sec", 0.0) or 0.0)
        gain = (pred_thr - base_thr) / base_thr if base_thr > 0 else 0.0
        gap = (oracle_thr - pred_thr) / oracle_thr if oracle_thr > 0 else 0.0
        for row in rows:
            if int(row["sessions"]) == sessions:
                row["taps_predictive_gain_vs_acd_nisp"] = gain
                row["taps_oracle_gap"] = gap


def _series(rows: list[dict[str, Any]], metric: str) -> tuple[list[int], dict[str, list[float]]]:
    sessions = sorted({int(r["sessions"]) for r in rows})
    by_policy = {p: [] for p in POLICIES}
    for s in sessions:
        for p in POLICIES:
            row = next((r for r in rows if int(r["sessions"]) == s and r["policy"] == p), {})
            by_policy[p].append(float(row.get(metric, 0.0) or 0.0))
    return sessions, by_policy


def _line_plot(rows: list[dict[str, Any]], metric: str, ylabel: str, out: str | Path) -> None:
    ensure_dir(Path(out).parent)
    sessions, by_policy = _series(rows, metric)
    plt.figure(figsize=(6.2, 3.8))
    for policy, vals in by_policy.items():
        plt.plot(sessions, vals, marker="o", label=policy)
    plt.xlabel("sessions")
    plt.ylabel(ylabel)
    plt.xticks(sessions)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_multisession(rows: list[dict[str, Any]], plot_prefix: str = "data/plots/multisession_taps", plot_suffix: str = "pr4_algo") -> None:
    _line_plot(rows, "throughput_sessions_per_sec", "throughput (sessions/s)", f"{plot_prefix}_throughput_{plot_suffix}.pdf")
    _line_plot(rows, "p95_jct", "p95 JCT (s)", f"{plot_prefix}_p95_jct_{plot_suffix}.pdf")
    _line_plot(rows, "region_utilization", "region utilization", f"{plot_prefix}_region_util_{plot_suffix}.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="data/traces/mini_swe_lite10_r4_timed")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--out", default="data/results/multisession_taps_pr4_algo.csv")
    ap.add_argument("--sessions", default="1,2,4,8,16")
    ap.add_argument("--effective-regions", type=int, default=4)
    ap.add_argument("--arrival-pattern", choices=("closed_loop", "poisson"), default="closed_loop")
    ap.add_argument("--run-id", default="pr4_algo")
    ap.add_argument("--plot-prefix", default="data/plots/multisession_taps")
    ap.add_argument("--plot-suffix", default="pr4_algo")
    args = ap.parse_args()
    rows = run_multisession(
        args.trace_dir,
        args.model_json,
        args.out,
        [int(x) for x in args.sessions.split(",") if x.strip()],
        args.effective_regions,
        args.arrival_pattern,
        args.run_id,
        args.plot_prefix,
        args.plot_suffix,
    )
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
