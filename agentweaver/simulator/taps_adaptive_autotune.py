from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.taps_regime_classifier import RegimeThresholds
from agentweaver.simulator.taps_unified import TAPSUnifiedConfig, _f, _load_traces
from agentweaver.simulator.taps_unified_adaptive import (
    ADAPTIVE_POLICY,
    BASELINE_POLICIES,
    AdaptiveProfiles,
    TAPSAdaptiveReplay,
    _fill_adaptive_gains,
    _run_policy,
)
from agentweaver.tracing.trace_schema import Trace
from agentweaver.utils.io import write_csv, write_json


def _instance_id(trace: Trace) -> str:
    for ev in trace.events:
        if ev.instance_id:
            return ev.instance_id
    return str(trace.metadata.get("source", "unknown"))


def _split_traces(traces: list[Trace]) -> tuple[list[Trace], list[Trace]]:
    by_instance: dict[str, list[Trace]] = {}
    for tr in traces:
        by_instance.setdefault(_instance_id(tr), []).append(tr)
    keys = sorted(by_instance)
    if len(keys) < 2:
        return traces, traces
    train_keys = set(keys[::2])
    train = [tr for key in train_keys for tr in by_instance[key]]
    val = [tr for key in keys if key not in train_keys for tr in by_instance[key]]
    return train or traces, val or traces


def _base_config_from_v5(path: str | Path = "data/results/taps_unified_best_config_pr4_v5.json") -> TAPSUnifiedConfig:
    p = Path(path)
    if not p.exists():
        return AdaptiveProfiles.default().balanced
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return TAPSUnifiedConfig(**data.get("best_config", {}))
    except Exception:
        return AdaptiveProfiles.default().balanced


def _profiles_from_values(
    balanced: TAPSUnifiedConfig,
    admission_tail: float,
    admission_stall: float,
    ready_depth: float,
    w_domain: float,
    w_batch: float,
    w_remote: float,
    w_switch: float,
    w_tail: float,
    w_age: float,
    w_short: float,
    tail_pct: int,
    w_mem: float,
    mem_threshold: float,
) -> AdaptiveProfiles:
    base = asdict(balanced)
    admission = TAPSUnifiedConfig(**{**base, "admission_tail": admission_tail, "admission_stall": admission_stall, "ready_depth_factor": ready_depth})
    domain = TAPSUnifiedConfig(**{**base, "w_domain": w_domain, "w_batch": w_batch, "w_remote": w_remote, "w_switch": w_switch})
    tail = TAPSUnifiedConfig(**{**base, "w_tail": w_tail, "w_age": w_age, "w_short": w_short})
    memory = TAPSUnifiedConfig(**{**base, "w_mem": w_mem, "memory_pressure_threshold": mem_threshold, "ready_depth_factor": min(ready_depth, 1.0)})
    thresholds = RegimeThresholds(memory_pressure=mem_threshold)
    return AdaptiveProfiles(balanced, admission, domain, tail, memory, thresholds, tail_slo_percentile=tail_pct)


def _profile_grid(max_trials: int, seed: int = 616) -> list[AdaptiveProfiles]:
    balanced = _base_config_from_v5()
    values = {
        "admission_tail": [1, 2, 4],
        "admission_stall": [1, 2, 4],
        "ready_depth": [1, 2, 4],
        "w_domain": [1, 2, 4],
        "w_batch": [0.5, 1, 2],
        "w_remote": [0.1, 0.5, 1],
        "w_switch": [0.1, 0.5, 1],
        "w_tail": [2, 4, 8],
        "w_age": [0.05, 0.1, 0.2, 0.5],
        "w_short": [0, 0.5, 1, 2],
        "tail_pct": [80, 90, 95],
        "w_mem": [0.5, 1, 2],
        "mem_threshold": [0.7, 0.85, 0.95],
    }
    keys = list(values)
    combos = list(itertools.product(*[values[k] for k in keys]))
    rng = random.Random(seed)
    rng.shuffle(combos)
    profiles: list[AdaptiveProfiles] = []
    # Include a transparent tail-risk candidate in the search set; it is still
    # selected or rejected by the train split objective below.
    profiles.append(
        _profiles_from_values(
            balanced,
            admission_tail=1,
            admission_stall=1,
            ready_depth=1,
            w_domain=float(balanced.w_domain),
            w_batch=float(balanced.w_batch),
            w_remote=float(balanced.w_remote),
            w_switch=float(balanced.w_switch),
            w_tail=2,
            w_age=0.5,
            w_short=2,
            tail_pct=90,
            w_mem=float(balanced.w_mem),
            mem_threshold=float(balanced.memory_pressure_threshold),
        )
    )
    for combo in combos[:max_trials]:
        profiles.append(_profiles_from_values(balanced, **dict(zip(keys, combo))))
    return profiles[:max_trials]


def _objective(row: dict[str, Any], strongest_p95: float, strongest_thr: float) -> float:
    p95_regret = max(0.0, _f(row, "p95_jct") - strongest_p95)
    throughput_regret = max(0.0, strongest_thr - _f(row, "throughput"))
    return (
        _f(row, "p95_jct")
        + 0.2 * _f(row, "mean_jct")
        - 0.1 * _f(row, "throughput")
        + 2.0 * p95_regret
        + 5.0 * throughput_regret
        + 1000.0 * _f(row, "starvation_count")
    )


def _run_baselines(
    traces: list[Trace],
    lm: LatencyModel,
    total: int,
    limit: int,
    regions: int,
    arrival: str,
    memory: int,
    v5_config: TAPSUnifiedConfig,
) -> list[dict[str, Any]]:
    return [_run_policy(traces, lm, p, total, limit, regions, arrival, memory, None, v5_config) for p in BASELINE_POLICIES]


def _validate(
    traces: list[Trace],
    lm: LatencyModel,
    profiles: AdaptiveProfiles,
    v5_config: TAPSUnifiedConfig,
    out_rows: list[dict[str, Any]],
    split: str,
) -> None:
    total, limit, regions, arrival, memory = 64, 16, 4, "bursty", 32
    rows = _run_baselines(traces, lm, total, limit, regions, arrival, memory, v5_config)
    rows.append(TAPSAdaptiveReplay(traces, total, limit, regions, arrival, memory, lm, profiles=profiles, seed=919).run())
    for row in rows:
        row["split"] = split
    _fill_adaptive_gains(rows)
    out_rows.extend(rows)


def run_adaptive_autotune(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    trials_out: str | Path = "data/results/taps_adaptive_autotune_trials_pr4_v6.csv",
    best_out: str | Path = "data/results/taps_adaptive_best_config_pr4_v6.json",
    validation_out: str | Path = "data/results/taps_adaptive_validation_pr4_v6.csv",
    max_trials: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces = _load_traces(trace_dirs)
    train, val = _split_traces(traces)
    lm = LatencyModel.load(model_json)
    v5_config = _base_config_from_v5()
    train_baselines = _run_baselines(train, lm, 64, 16, 4, "bursty", 32, v5_config)
    strongest_train_p95 = min(_f(r, "p95_jct") for r in train_baselines)
    strongest_train_thr = max(_f(r, "throughput") for r in train_baselines)
    trial_rows: list[dict[str, Any]] = []
    best_profile: AdaptiveProfiles | None = None
    best_obj = float("inf")
    best_row: dict[str, Any] | None = None
    for idx, profiles in enumerate(_profile_grid(max_trials), start=1):
        row = TAPSAdaptiveReplay(train, 64, 16, 4, "bursty", 32, lm, profiles=profiles, seed=811).run()
        feasible = int(row.get("completed_sessions", 0) or 0) == 64 and int(row.get("starvation_count", 0) or 0) == 0
        obj = _objective(row, strongest_train_p95, strongest_train_thr) if feasible else float("inf")
        trial = {
            "trial": idx,
            "split": "train",
            "objective": obj,
            "feasible": feasible,
            **profiles.to_jsonable()["balanced"],
            "profile_json": json.dumps(profiles.to_jsonable(), sort_keys=True),
            **row,
        }
        trial_rows.append(trial)
        if obj < best_obj:
            best_obj = obj
            best_profile = profiles
            best_row = row
    if best_profile is None:
        best_profile = AdaptiveProfiles.default()
        best_row = TAPSAdaptiveReplay(train, 64, 16, 4, "bursty", 32, lm, profiles=best_profile).run()
        best_obj = _objective(best_row)
    validation_rows: list[dict[str, Any]] = []
    _validate(val, lm, best_profile, v5_config, validation_rows, "validation")
    adaptive = next((r for r in validation_rows if r["policy"] == ADAPTIVE_POLICY), {})
    best = {
        "best_trial_objective": best_obj,
        "best_train_row": best_row or {},
        "best_profiles": best_profile.to_jsonable(),
        "train_instances": sorted({_instance_id(t) for t in train}),
        "validation_instances": sorted({_instance_id(t) for t in val}),
        "validation_p95_gain_over_strongest": _f(adaptive, "p95_gain_over_strongest_baseline"),
        "validation_throughput_gain_over_strongest": _f(adaptive, "throughput_gain_over_strongest_baseline"),
        "validation_ready_wait_gain_over_strongest": _f(adaptive, "ready_wait_gain_over_strongest_baseline"),
        "validation_starvation_count": int(_f(adaptive, "starvation_count")),
    }
    write_csv(trials_out, trial_rows)
    write_json(best_out, best)
    write_csv(validation_out, validation_rows)
    return trial_rows, best, validation_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    trials, best, validation = run_adaptive_autotune(max_trials=args.trials)
    print(json.dumps({"trials": len(trials), "validation_rows": len(validation), "best": best}, indent=2))


if __name__ == "__main__":
    main()
