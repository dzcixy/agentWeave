from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.multisession_replay import MultiSessionReplay, TAPSConfig
from agentweaver.tracing.trace_schema import Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


GRID = {
    "w_l": [0, 0.5, 1.0, 2.0],
    "w_r": [0, 0.5, 1.0, 2.0],
    "w_c": [0, 0.5, 1.0],
    "w_a": [0.01, 0.05, 0.1],
    "w_s": [0, 0.5, 1.0],
    "lambda_s": [0, 0.1, 0.5],
    "lambda_n": [0, 0.1, 0.5],
    "tau": [1, 5, 10, 30],
}


def _trace_instance(trace: Trace) -> str:
    for ev in trace.events:
        if ev.instance_id:
            return ev.instance_id
    return str(trace.metadata.get("source", "unknown"))


def split_traces(traces: list[Trace]) -> tuple[list[Trace], list[Trace]]:
    ordered = sorted(traces, key=_trace_instance)
    train = [tr for i, tr in enumerate(ordered) if i % 2 == 0]
    valid = [tr for i, tr in enumerate(ordered) if i % 2 == 1]
    if not valid and train:
        valid = train[-1:]
        train = train[:-1] or valid
    return train or ordered, valid or ordered


def _configs(max_configs: int) -> list[TAPSConfig]:
    keys = list(GRID)
    all_values = list(itertools.product(*(GRID[k] for k in keys)))
    seed_configs = [
        TAPSConfig(),
        TAPSConfig(w_l=0.5, w_r=0.5, w_c=0.5, w_a=0.05, w_s=0.5, lambda_s=0, lambda_n=0, tau=5),
        TAPSConfig(w_l=1.0, w_r=0.5, w_c=0.0, w_a=0.05, w_s=1.0, lambda_s=0, lambda_n=0, tau=10),
        TAPSConfig(w_l=0.0, w_r=1.0, w_c=0.5, w_a=0.1, w_s=1.0, lambda_s=0.1, lambda_n=0.1, tau=5),
        TAPSConfig(w_l=2.0, w_r=1.0, w_c=1.0, w_a=0.05, w_s=0.5, lambda_s=0.1, lambda_n=0.1, tau=30),
    ]
    if max_configs and len(all_values) > max_configs:
        remaining = max(0, max_configs - len(seed_configs))
        stride = max(1, len(all_values) // max(1, remaining))
        sampled = all_values[::stride][:remaining]
        if all_values[-1] not in sampled:
            if sampled:
                sampled[-1] = all_values[-1]
            else:
                sampled = [all_values[-1]]
        all_values = sampled
    configs = seed_configs + [TAPSConfig(**dict(zip(keys, vals))) for vals in all_values]
    dedup: dict[tuple[tuple[str, float], ...], TAPSConfig] = {}
    for cfg in configs:
        vals = tuple(sorted((k, float(v)) for k, v in asdict(cfg).items() if k in GRID))
        dedup.setdefault(vals, cfg)
    return list(dedup.values())[:max_configs] if max_configs else list(dedup.values())


def _evaluate(
    traces: list[Trace],
    lm: LatencyModel,
    cfg: TAPSConfig,
    sessions_list: list[int],
    effective_regions: int,
    arrival_pattern: str,
    run_id: str,
) -> dict[str, Any]:
    rows = []
    base_rows = []
    for sessions in sessions_list:
        base_rows.append(
            MultiSessionReplay(
                traces,
                sessions,
                "acd_nisp",
                lm,
                effective_regions,
                arrival_pattern,
                run_id,
                cfg,
                seed=101 + sessions + effective_regions,
            ).run()
        )
        rows.append(
            MultiSessionReplay(
                traces,
                sessions,
                "taps_v3",
                lm,
                effective_regions,
                arrival_pattern,
                run_id,
                cfg,
                seed=101 + sessions + effective_regions,
            ).run()
        )
    expected = sum(sessions_list)
    completed = sum(int(r.get("completed_sessions", 0) or 0) for r in rows)
    mean_jct = sum(float(r.get("mean_jct", 0.0) or 0.0) for r in rows) / max(1, len(rows))
    p95_jct = sum(float(r.get("p95_jct", 0.0) or 0.0) for r in rows) / max(1, len(rows))
    mean_ratio = sum(
        float(r.get("mean_jct", 0.0) or 0.0) / max(1e-9, float(b.get("mean_jct", 0.0) or 0.0))
        for r, b in zip(rows, base_rows)
    ) / max(1, len(rows))
    p95_ratio = sum(
        float(r.get("p95_jct", 0.0) or 0.0) / max(1e-9, float(b.get("p95_jct", 0.0) or 0.0))
        for r, b in zip(rows, base_rows)
    ) / max(1, len(rows))
    objective = p95_ratio + 0.1 * mean_ratio
    if completed < expected:
        objective += 1e6
    return {
        "objective": objective,
        "mean_ratio_vs_acd_nisp": mean_ratio,
        "p95_ratio_vs_acd_nisp": p95_ratio,
        "mean_gain_vs_acd_nisp": 1.0 - mean_ratio,
        "p95_gain_vs_acd_nisp": 1.0 - p95_ratio,
        "mean_jct": mean_jct,
        "p95_jct": p95_jct,
        "completed_sessions": completed,
        "expected_sessions": expected,
        "no_starvation": completed == expected,
        "rows": rows,
    }


def run_taps_autotune(
    trace_dir: str | Path = "data/traces/mini_swe_lite10_r4_timed",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/taps_v3_autotune.csv",
    best_json: str | Path = "data/results/taps_v3_best_config.json",
    validation_csv: str | Path = "data/results/taps_v3_validation.csv",
    max_configs: int = 96,
    sessions_list: list[int] | None = None,
    effective_regions: int = 4,
    arrival_pattern: str = "bursty",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    sessions_list = sessions_list or [16, 32]
    traces = load_trace_dir(trace_dir)
    train, valid = split_traces(traces)
    lm = LatencyModel.load(model_json)
    rows: list[dict[str, Any]] = []
    best: tuple[float, TAPSConfig, dict[str, Any]] | None = None
    for idx, cfg in enumerate(_configs(max_configs)):
        result = _evaluate(train, lm, cfg, sessions_list, effective_regions, arrival_pattern, "pr4_algo_v3_autotune_train")
        row = {"config_id": idx, **asdict(cfg), **{k: v for k, v in result.items() if k != "rows"}, "split": "train"}
        rows.append(row)
        if best is None or result["objective"] < best[0]:
            best = (float(result["objective"]), cfg, result)
    if best is None:
        raise ValueError("no TAPS configs evaluated")
    best_cfg = best[1]
    validation = _evaluate(valid, lm, best_cfg, sessions_list, effective_regions, arrival_pattern, "pr4_algo_v3_autotune_validation")
    validation_rows = [
        {"split": "validation", "config_id": "best", **asdict(best_cfg), **{k: v for k, v in validation.items() if k != "rows"}}
    ]
    write_csv(out_csv, rows)
    write_csv(validation_csv, validation_rows)
    p = Path(best_json)
    ensure_dir(p.parent)
    p.write_text(json.dumps({"config": asdict(best_cfg), "train": best[2], "validation": validation}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return rows, {"config": asdict(best_cfg), "train": best[2], "validation": validation}, validation_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="data/traces/mini_swe_lite10_r4_timed")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--out", default="data/results/taps_v3_autotune.csv")
    ap.add_argument("--best-json", default="data/results/taps_v3_best_config.json")
    ap.add_argument("--validation-out", default="data/results/taps_v3_validation.csv")
    ap.add_argument("--max-configs", type=int, default=96)
    ap.add_argument("--sessions", default="16,32")
    ap.add_argument("--effective-regions", type=int, default=4)
    ap.add_argument("--arrival-pattern", default="bursty")
    args = ap.parse_args()
    rows, best, validation = run_taps_autotune(
        args.trace_dir,
        args.model_json,
        args.out,
        args.best_json,
        args.validation_out,
        args.max_configs,
        [int(x) for x in args.sessions.split(",") if x.strip()],
        args.effective_regions,
        args.arrival_pattern,
    )
    print(json.dumps({"train_rows": len(rows), "validation_rows": len(validation), "best_config": best["config"]}, indent=2))


if __name__ == "__main__":
    main()
