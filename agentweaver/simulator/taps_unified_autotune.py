from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.taps_unified import POLICIES, TAPSUnifiedConfig, TAPSUnifiedReplay, _fill_gains, _load_traces
from agentweaver.tracing.trace_schema import Trace
from agentweaver.utils.io import write_csv, write_json


def _instance_id(trace: Trace) -> str:
    for ev in trace.events:
        if ev.instance_id:
            return ev.instance_id
    return str(trace.metadata.get("source", "unknown"))


def _split_traces(traces: list[Trace]) -> tuple[list[Trace], list[Trace]]:
    by_instance: dict[str, list[Trace]] = {}
    for trace in traces:
        by_instance.setdefault(_instance_id(trace), []).append(trace)
    keys = sorted(by_instance)
    if len(keys) < 2:
        return traces, traces
    train_keys = set(keys[::2])
    train = [tr for key in train_keys for tr in by_instance[key]]
    val = [tr for key in keys if key not in train_keys for tr in by_instance[key]]
    return train or traces, val or traces


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _objective(row: dict[str, Any]) -> float:
    return _f(row, "p95_jct") + 0.2 * _f(row, "mean_jct") - 0.1 * _f(row, "throughput")


def _config_grid(max_trials: int, seed: int = 515) -> list[TAPSUnifiedConfig]:
    values = {
        "w_tail": [0.5, 1.0, 2.0, 4.0],
        "w_domain": [0, 0.5, 1, 2],
        "w_batch": [0, 0.5, 1],
        "w_resume": [0, 0.5, 1],
        "w_age": [0.01, 0.05, 0.1],
        "w_short": [0, 0.5, 1],
        "w_remote": [0, 0.1, 0.5],
        "w_switch": [0, 0.1, 0.5],
        "w_mem": [0, 0.1, 0.5],
        "ready_depth_factor": [1, 2, 4],
        "memory_pressure_threshold": [0.7, 0.85, 0.95],
    }
    keys = list(values)
    combos = list(itertools.product(*[values[k] for k in keys]))
    rng = random.Random(seed)
    rng.shuffle(combos)
    configs: list[TAPSUnifiedConfig] = []
    for combo in combos[:max_trials]:
        configs.append(TAPSUnifiedConfig(**dict(zip(keys, combo))))
    return configs


def _run_one(
    traces: list[Trace],
    lm: LatencyModel,
    config: TAPSUnifiedConfig,
    policy: str,
    total_sessions: int = 64,
    active_limit: int = 16,
    regions: int = 4,
    arrival: str = "bursty",
    memory_gb: int = 32,
) -> dict[str, Any]:
    return TAPSUnifiedReplay(
        traces,
        total_sessions,
        active_limit,
        regions,
        arrival,
        memory_gb,
        policy,
        lm,
        config=config,
        seed=733,
    ).run()


def run_taps_unified_autotune(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    trials_out: str | Path = "data/results/taps_unified_autotune_trials_pr4_v5.csv",
    best_out: str | Path = "data/results/taps_unified_best_config_pr4_v5.json",
    validation_out: str | Path = "data/results/taps_unified_validation_pr4_v5.csv",
    max_trials: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces = _load_traces(trace_dirs)
    train, val = _split_traces(traces)
    lm = LatencyModel.load(model_json)
    trial_rows: list[dict[str, Any]] = []
    best_config: TAPSUnifiedConfig | None = None
    best_obj = float("inf")
    best_row: dict[str, Any] | None = None
    for idx, config in enumerate(_config_grid(max_trials), start=1):
        row = _run_one(train, lm, config, "taps_unified")
        completed = int(row.get("completed_sessions", 0) or 0)
        starvation = int(row.get("starvation_count", 0) or 0)
        memory_ok = _f(row, "memory_occupancy") <= 32 * 1024**3
        feasible = completed == 64 and starvation <= 1 and memory_ok
        obj = _objective(row) if feasible else float("inf")
        trial = {"trial": idx, "split": "train", "objective": obj, "feasible": feasible, **asdict(config), **row}
        trial_rows.append(trial)
        if obj < best_obj:
            best_obj = obj
            best_config = config
            best_row = row
    if best_config is None:
        best_config = TAPSUnifiedConfig()
        best_row = _run_one(train, lm, best_config, "taps_unified")
        best_obj = _objective(best_row)

    validation_rows: list[dict[str, Any]] = []
    for policy in ["reactive_admission", "acd_nisp", "taps_unified"]:
        row = _run_one(val, lm, best_config, policy)
        row["split"] = "validation"
        validation_rows.append(row)
    _fill_gains(validation_rows)
    taps = next((r for r in validation_rows if r["policy"] == "taps_unified"), {})
    reactive = next((r for r in validation_rows if r["policy"] == "reactive_admission"), {})
    acd = next((r for r in validation_rows if r["policy"] == "acd_nisp"), {})
    best = {
        "best_trial_objective": best_obj,
        "best_train_row": best_row or {},
        "best_config": asdict(best_config),
        "train_instances": sorted({_instance_id(t) for t in train}),
        "validation_instances": sorted({_instance_id(t) for t in val}),
        "validation_p95_gain_over_reactive": (
            (_f(reactive, "p95_jct") - _f(taps, "p95_jct")) / max(1e-9, _f(reactive, "p95_jct")) if reactive else 0.0
        ),
        "validation_p95_gain_over_acd_nisp": (
            (_f(acd, "p95_jct") - _f(taps, "p95_jct")) / max(1e-9, _f(acd, "p95_jct")) if acd else 0.0
        ),
        "validation_throughput_gain_over_reactive": (
            (_f(taps, "throughput") - _f(reactive, "throughput")) / max(1e-9, _f(reactive, "throughput")) if reactive else 0.0
        ),
    }
    write_csv(trials_out, trial_rows)
    write_json(best_out, best)
    write_csv(validation_out, validation_rows)
    return trial_rows, best, validation_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--trace-dirs", default="data/traces/mini_swe_lite10_r4_timed,data/traces/mini_swe_lite5_patchcap_verified")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    args = ap.parse_args()
    trials, best, validation = run_taps_unified_autotune(
        [x for x in args.trace_dirs.split(",") if x],
        args.model_json,
        max_trials=args.trials,
    )
    print(json.dumps({"trials": len(trials), "validation_rows": len(validation), "best": best}, indent=2))


if __name__ == "__main__":
    main()
