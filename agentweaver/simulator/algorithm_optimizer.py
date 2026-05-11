from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.pabb_online_replay import (
    _branch_events_from_dirs,
    _run_instance_snapshot_policy,
    _as_float,
    pabb_weights,
)
from agentweaver.simulator.taps_admission_control import AdmissionConfig, AdmissionReplay
from agentweaver.simulator.taps_domain_scheduler import DomainConfig, DomainReplay
from agentweaver.tracing.trace_schema import Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


def _trace_instance(trace: Trace) -> str:
    for ev in trace.events:
        if ev.instance_id:
            return ev.instance_id
    return str(trace.metadata.get("source", "unknown"))


def _split_traces(traces: list[Trace]) -> tuple[list[Trace], list[Trace]]:
    ordered = sorted(traces, key=_trace_instance)
    train = [tr for i, tr in enumerate(ordered) if i % 2 == 0]
    valid = [tr for i, tr in enumerate(ordered) if i % 2 == 1]
    return train or ordered, valid or ordered


def _objective(row: dict[str, Any]) -> float:
    return float(row.get("p95_jct", 0.0) or 0.0) + 0.1 * float(row.get("mean_jct", 0.0) or 0.0) - 0.05 * float(row.get("throughput", row.get("throughput_sessions_per_sec", 0.0)) or 0.0)


def _domain_configs(max_trials: int) -> list[DomainConfig]:
    vals = []
    for w_domain, w_batch, w_hot, w_resume, w_short in itertools.product([0.5, 1.0, 2.0, 4.0], [0, 0.5, 1.0], [0, 0.3, 0.8], [0, 0.5, 1.0], [0, 0.3, 1.0]):
        vals.append(DomainConfig(w_domain=w_domain, w_batch=w_batch, w_hot=w_hot, w_resume=w_resume, w_short=w_short))
    return vals[:max_trials]


def _admission_configs(max_trials: int) -> list[AdmissionConfig]:
    vals = []
    for w_reuse, w_llm, w_stall, w_sram, w_tool in itertools.product([0.5, 1.0, 2.0, 4.0], [0, 0.5, 1.0], [0, 1.0, 2.0], [0, 0.5], [0, 0.5, 1.0]):
        vals.append(AdmissionConfig(w_reuse=w_reuse, w_llm=w_llm, w_stall=w_stall, w_sram=w_sram, w_tool_penalty=w_tool))
    return vals[:max_trials]


def _pabb_weight_configs(base: dict[str, float], max_trials: int) -> list[dict[str, float]]:
    configs: list[dict[str, float]] = []
    for patch, file_w, test, token in itertools.product([2, 4, 6], [1, 2, 4], [0.5, 1.5, 3.0], [0.5, 1.0, 2.0]):
        cfg = dict(base)
        cfg.update({"w_patch": patch, "w_file": file_w, "w_test": test, "w_token": token})
        configs.append(cfg)
    return configs[:max_trials]


def _evaluate_domain(traces: list[Trace], lm: LatencyModel, cfg: DomainConfig, split: str) -> dict[str, Any]:
    row = DomainReplay(traces, 32, "taps_domain", lm, 4, "bursty", cfg).run()
    base = DomainReplay(traces, 32, "acd_nisp", lm, 4, "bursty", cfg).run()
    row["algorithm"] = "taps_domain"
    row["split"] = split
    row["objective"] = _objective(row)
    row["p95_gain_vs_base"] = (float(base["p95_jct"]) - float(row["p95_jct"])) / max(1e-9, float(base["p95_jct"]))
    row["throughput_gain_vs_base"] = (float(row["throughput_sessions_per_sec"]) - float(base["throughput_sessions_per_sec"])) / max(1e-9, float(base["throughput_sessions_per_sec"]))
    row.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
    return row


def _evaluate_admission(traces: list[Trace], lm: LatencyModel, cfg: AdmissionConfig, split: str) -> dict[str, Any]:
    row = AdmissionReplay(traces, 64, 16, 4, "taps_admission", lm, cfg).run()
    base = AdmissionReplay(traces, 64, 16, 4, "static_admission", lm, cfg).run()
    row["algorithm"] = "taps_admission"
    row["split"] = split
    row["objective"] = _objective(row)
    row["p95_gain_vs_base"] = (float(base["p95_jct"]) - float(row["p95_jct"])) / max(1e-9, float(base["p95_jct"]))
    row["throughput_gain_vs_base"] = (float(row["throughput"]) - float(base["throughput"])) / max(1e-9, float(base["throughput"]))
    row.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
    return row


def _evaluate_pabb(
    by_instance: dict[str, dict[str, list[Any]]],
    instances: set[str],
    weights: dict[str, float],
    split: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    lm = LatencyModel.load("data/profiles/h100_latency_model_pr2_v2.json")
    for inst in sorted(instances):
        branches = by_instance.get(inst)
        if not branches:
            continue
        rows.append(_run_instance_snapshot_policy(inst, branches, "pabb_snapshot_online", 4, 15, lm, weights))
    costs = [_as_float(r.get("cost_to_patch")) for r in rows]
    costs = [c for c in costs if c is not None]
    tokens = [_as_float(r.get("tokens_to_first_nonempty_patch")) for r in rows]
    tokens = [t for t in tokens if t is not None]
    obj = (sum(costs) / len(costs) if costs else 1e6) + 0.001 * (sum(tokens) / len(tokens) if tokens else 1e6)
    return {
        "algorithm": "pabb_snapshot",
        "split": split,
        "objective": obj,
        "mean_cost_to_patch": sum(costs) / len(costs) if costs else "",
        "mean_tokens_to_patch": sum(tokens) / len(tokens) if tokens else "",
        "patch_instances": len(costs),
        **{f"cfg_{k}": v for k, v in weights.items()},
    }


def run_optimizer(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    trials_csv: str | Path = "data/results/pr4_v4_optimizer_trials.csv",
    best_json: str | Path = "data/results/pr4_v4_best_configs.json",
    validation_csv: str | Path = "data/results/pr4_v4_validation_results.csv",
    taps_trials: int = 50,
    pabb_trials: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces: list[Trace] = []
    for d in trace_dirs:
        if Path(d).exists():
            traces.extend(load_trace_dir(d))
    train, valid = _split_traces(traces)
    lm = LatencyModel.load(model_json)
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    validation: list[dict[str, Any]] = []

    domain_train: list[tuple[float, DomainConfig, dict[str, Any]]] = []
    for cfg in _domain_configs(max(1, taps_trials // 2)):
        row = _evaluate_domain(train, lm, cfg, "train")
        trials.append(row)
        domain_train.append((float(row["objective"]), cfg, row))
    admission_train: list[tuple[float, AdmissionConfig, dict[str, Any]]] = []
    for cfg in _admission_configs(max(1, taps_trials - len(domain_train))):
        row = _evaluate_admission(train, lm, cfg, "train")
        trials.append(row)
        admission_train.append((float(row["objective"]), cfg, row))

    if domain_train:
        _, cfg, train_row = min(domain_train, key=lambda x: x[0])
        val = _evaluate_domain(valid, lm, cfg, "validation")
        validation.append(val)
        best["taps_domain"] = {"config": asdict(cfg), "train": train_row, "validation": val}
    if admission_train:
        _, cfg, train_row = min(admission_train, key=lambda x: x[0])
        val = _evaluate_admission(valid, lm, cfg, "validation")
        validation.append(val)
        best["taps_admission"] = {"config": asdict(cfg), "train": train_row, "validation": val}

    by_instance = _branch_events_from_dirs(trace_dirs)
    insts = sorted(by_instance)
    train_insts = {x for i, x in enumerate(insts) if i % 2 == 0}
    valid_insts = set(insts) - train_insts or set(insts)
    pabb_train: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    for cfg in _pabb_weight_configs(pabb_weights(), pabb_trials):
        row = _evaluate_pabb(by_instance, train_insts, cfg, "train")
        trials.append(row)
        pabb_train.append((float(row["objective"]), cfg, row))
    if pabb_train:
        _, cfg, train_row = min(pabb_train, key=lambda x: x[0])
        val = _evaluate_pabb(by_instance, valid_insts, cfg, "validation")
        validation.append(val)
        best["pabb_snapshot"] = {"config": cfg, "train": train_row, "validation": val}

    write_csv(trials_csv, trials)
    write_csv(validation_csv, validation)
    p = Path(best_json)
    ensure_dir(p.parent)
    p.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    return trials, best, validation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-out", default="data/results/pr4_v4_optimizer_trials.csv")
    ap.add_argument("--best-out", default="data/results/pr4_v4_best_configs.json")
    ap.add_argument("--validation-out", default="data/results/pr4_v4_validation_results.csv")
    ap.add_argument("--taps-trials", type=int, default=50)
    ap.add_argument("--pabb-trials", type=int, default=20)
    args = ap.parse_args()
    trials, best, validation = run_optimizer(
        trials_csv=args.trials_out,
        best_json=args.best_out,
        validation_csv=args.validation_out,
        taps_trials=args.taps_trials,
        pabb_trials=args.pabb_trials,
    )
    print(json.dumps({"trials": len(trials), "validation": len(validation), "best": list(best)}, indent=2))


if __name__ == "__main__":
    main()
