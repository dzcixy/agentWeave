from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agentweaver.simulator.aligned_policy_sweep import ALIGNED_POLICIES
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv, write_json


TARGETS = ["p95_jct", "mean_jct", "throughput", "ready_queue_wait", "region_utilization"]
OBJECTIVES = ["p95_opt", "throughput_opt", "balanced"]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        v = row.get(key)
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def _objective(row: dict[str, Any], objective: str = "balanced") -> float:
    if objective == "p95_opt":
        return _f(row, "p95_jct")
    if objective == "throughput_opt":
        return -_f(row, "throughput")
    return _f(row, "p95_jct") + 0.2 * _f(row, "mean_jct") - 10.0 * _f(row, "throughput")


def _pred_objective(pred: dict[str, float], objective: str = "balanced") -> float:
    if objective == "p95_opt":
        return pred["p95_jct"]
    if objective == "throughput_opt":
        return -pred["throughput"]
    return pred["p95_jct"] + 0.2 * pred["mean_jct"] - 10.0 * pred["throughput"]


def _safe_gain(base: float, new: float, lower_better: bool = True) -> float:
    if base <= 0:
        return 0.0
    return (base - new) / base if lower_better else (new - base) / base


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[str(row["config_id"])][str(row["policy"])] = row
    return groups


def _split_random(configs: list[str]) -> tuple[set[str], set[str]]:
    train, val = set(), set()
    for cid in sorted(configs):
        (val if int(stable_hash(cid), 16) % 5 == 0 else train).add(cid)
    if not train or not val:
        for i, cid in enumerate(sorted(configs)):
            (val if i % 5 == 0 else train).add(cid)
    return train, val


def build_splits(groups: dict[str, dict[str, dict[str, Any]]]) -> dict[str, tuple[set[str], set[str]]]:
    configs = sorted(groups)
    splits: dict[str, tuple[set[str], set[str]]] = {"random": _split_random(configs)}
    arrivals = sorted({next(iter(groups[c].values())).get("arrival_pattern", "") for c in configs})
    for arrival in arrivals:
        val = {c for c in configs if next(iter(groups[c].values())).get("arrival_pattern") == arrival}
        splits[f"leave_arrival_{arrival}"] = (set(configs) - val, val)
    totals = sorted({int(_f(next(iter(groups[c].values())), "total_sessions")) for c in configs})
    for total in totals:
        val = {c for c in configs if int(_f(next(iter(groups[c].values())), "total_sessions")) == total}
        splits[f"leave_session_{total}"] = (set(configs) - val, val)
    memories = sorted({int(_f(next(iter(groups[c].values())), "memory_budget_gb")) for c in configs})
    for mem in memories:
        val = {c for c in configs if int(_f(next(iter(groups[c].values())), "memory_budget_gb")) == mem}
        splits[f"leave_memory_{mem}"] = (set(configs) - val, val)
    return {k: v for k, v in splits.items() if v[0] and v[1]}


@dataclass
class PriorStats:
    global_means: dict[str, float] = field(default_factory=dict)
    by_arrival: dict[str, dict[str, float]] = field(default_factory=dict)
    by_scale: dict[str, dict[str, float]] = field(default_factory=dict)

    def estimate(self, row: dict[str, Any], key: str) -> float:
        arrival = str(row.get("arrival_pattern", ""))
        total = int(_f(row, "total_sessions"))
        scale = "small" if total <= 32 else ("medium" if total <= 64 else "large")
        vals = [
            self.by_arrival.get(arrival, {}).get(key),
            self.by_scale.get(scale, {}).get(key),
            self.global_means.get(key),
        ]
        valid = [float(v) for v in vals if v is not None]
        return sum(valid) / max(1, len(valid))


def _median(vals: list[float]) -> float:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return 0.0
    return vals[len(vals) // 2]


def fit_priors(groups: dict[str, dict[str, dict[str, Any]]], train_ids: set[str]) -> PriorStats:
    keys = ["blocked_session_fraction", "domain_cache_hit_rate", "ready_queue_wait", "remote_kv_bytes", "memory_occupancy", "region_utilization"]
    buckets_global: dict[str, list[float]] = defaultdict(list)
    buckets_arrival: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    buckets_scale: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cid in train_ids:
        for row in groups[cid].values():
            arrival = str(row.get("arrival_pattern", ""))
            total = int(_f(row, "total_sessions"))
            scale = "small" if total <= 32 else ("medium" if total <= 64 else "large")
            for key in keys:
                buckets_global[key].append(_f(row, key))
                buckets_arrival[arrival][key].append(_f(row, key))
                buckets_scale[scale][key].append(_f(row, key))
    return PriorStats(
        global_means={k: _median(v) for k, v in buckets_global.items()},
        by_arrival={a: {k: _median(v) for k, v in d.items()} for a, d in buckets_arrival.items()},
        by_scale={s: {k: _median(v) for k, v in d.items()} for s, d in buckets_scale.items()},
    )


FEATURE_NAMES = [
    "bias",
    "total_sessions_log",
    "active_session_limit_log",
    "effective_regions_log",
    "memory_budget_log",
    "session_pressure",
    "region_pressure",
    "memory_pressure",
    "blocked_fraction",
    "tool_latency_median",
    "tool_latency_p95",
    "domain_cache_hit_rate",
    "shared_context_tokens_est",
    "remote_kv_pressure",
    "ready_queue_pressure",
    "arrival_closed_loop",
    "arrival_poisson",
    "arrival_bursty",
    "predicted_slo_risk",
]


def feature_vector(row: dict[str, Any], priors: PriorStats) -> np.ndarray:
    total = max(1.0, _f(row, "total_sessions"))
    limit = max(1.0, _f(row, "active_session_limit"))
    regions = max(1.0, _f(row, "effective_regions"))
    mem = max(1.0, _f(row, "memory_budget_gb"))
    session_pressure = total / limit
    region_pressure = limit / regions
    mem_occ = priors.estimate(row, "memory_occupancy")
    memory_pressure = mem_occ / max(1.0, mem * 1024**3)
    blocked = priors.estimate(row, "blocked_session_fraction")
    cache_hit = priors.estimate(row, "domain_cache_hit_rate")
    ready_wait = priors.estimate(row, "ready_queue_wait")
    remote_bytes = priors.estimate(row, "remote_kv_bytes")
    remote_pressure = remote_bytes / max(1.0, total * 1024**3)
    ready_pressure = ready_wait / max(1.0, total)
    # Train-derived tool priors are deliberately coarse; exact future tool duration is not used.
    tool_median = blocked * 10.0
    tool_p95 = blocked * 30.0
    shared_context_tokens = cache_hit * 4096.0
    arrival = row.get("arrival_pattern", "")
    slo_risk = session_pressure * region_pressure * (1.0 + blocked)
    vals = [
        1.0,
        math.log1p(total),
        math.log1p(limit),
        math.log1p(regions),
        math.log1p(mem),
        session_pressure,
        region_pressure,
        memory_pressure,
        blocked,
        tool_median,
        tool_p95,
        cache_hit,
        shared_context_tokens,
        remote_pressure,
        ready_pressure,
        1.0 if arrival == "closed_loop" else 0.0,
        1.0 if arrival == "poisson" else 0.0,
        1.0 if arrival == "bursty" else 0.0,
        slo_risk,
    ]
    return np.asarray(vals, dtype=float)


@dataclass
class RidgeModel:
    coef: list[float]
    residual_median: float
    residual_p95: float
    method: str = "numpy_ridge"

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(np.asarray(self.coef), x))


class TAPSCostModel:
    def __init__(self, alpha: float = 0.2, beta: float = 10.0, gamma: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.models: dict[str, dict[str, RidgeModel]] = {}
        self.priors = PriorStats()
        self.best_fixed: dict[str, str] = {}
        self.policy_residual: dict[str, float] = {}

    def fit(self, groups: dict[str, dict[str, dict[str, Any]]], train_ids: set[str], objective: str = "balanced") -> list[dict[str, Any]]:
        self.priors = fit_priors(groups, train_ids)
        train_rows: list[dict[str, Any]] = []
        objective_by_policy: dict[str, list[float]] = defaultdict(list)
        for policy in ALIGNED_POLICIES:
            policy_rows = [groups[cid][policy] for cid in train_ids if policy in groups[cid]]
            for row in policy_rows:
                objective_by_policy[policy].append(_objective(row, objective))
            self.models[policy] = {}
            if len(policy_rows) < 2:
                continue
            x = np.vstack([feature_vector(row, self.priors) for row in policy_rows])
            xtx = x.T @ x + 1e-3 * np.eye(x.shape[1])
            inv = np.linalg.pinv(xtx) @ x.T
            residuals_for_policy: list[float] = []
            for target in TARGETS:
                y = np.asarray([_f(row, target) for row in policy_rows], dtype=float)
                coef = inv @ y
                pred = x @ coef
                abs_err = np.abs(pred - y)
                model = RidgeModel(coef=[float(c) for c in coef], residual_median=_median(abs_err.tolist()), residual_p95=float(np.percentile(abs_err, 95)), method="numpy_ridge")
                self.models[policy][target] = model
                residuals_for_policy.extend(abs_err.tolist())
                train_rows.append(
                    {
                        "policy": policy,
                        "target": target,
                        "train_rows": len(policy_rows),
                        "median_abs_error": model.residual_median,
                        "p95_abs_error": model.residual_p95,
                        "model_method": model.method,
                    }
                )
            self.policy_residual[policy] = _median(residuals_for_policy)
        self.best_fixed[objective] = min(objective_by_policy, key=lambda p: sum(objective_by_policy[p]) / max(1, len(objective_by_policy[p]))) if objective_by_policy else "reactive_admission"
        return train_rows

    def predict_policy(self, row: dict[str, Any], policy: str) -> dict[str, float]:
        x = feature_vector(row, self.priors)
        out: dict[str, float] = {}
        for target in TARGETS:
            model = self.models.get(policy, {}).get(target)
            out[target] = max(0.0, model.predict(x)) if model else 0.0
        return out

    def select(self, row: dict[str, Any], objective: str = "balanced") -> dict[str, Any]:
        preds = {p: self.predict_policy(row, p) for p in ALIGNED_POLICIES}
        if objective == "throughput_opt":
            fallback = self.best_fixed.get(objective, self.best_fixed.get("balanced", "reactive_admission"))
            p95_guard = preds[fallback]["p95_jct"] * 1.05 if fallback in preds else float("inf")
            feasible = [p for p in ALIGNED_POLICIES if preds[p]["p95_jct"] <= p95_guard]
            ranked = sorted(feasible or [fallback], key=lambda p: -preds[p]["throughput"])
        else:
            ranked = sorted(ALIGNED_POLICIES, key=lambda p: _pred_objective(preds[p], objective))
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else best
        best_obj = _pred_objective(preds[best], objective)
        second_obj = _pred_objective(preds[second], objective)
        residual = max(self.policy_residual.get(best, 0.0), self.policy_residual.get(second, 0.0), 1e-9)
        gap = abs(second_obj - best_obj)
        fallback = self.best_fixed.get(objective, self.best_fixed.get("balanced", "reactive_admission"))
        fallback_used = gap < residual
        selected = fallback if fallback_used else best
        confidence = min(1.0, gap / max(residual, 1e-9))
        return {
            "selected_policy": selected,
            "predicted_p95": preds[selected]["p95_jct"],
            "predicted_throughput": preds[selected]["throughput"],
            "predicted_confidence": confidence,
            "fallback_used": str(fallback_used).lower(),
            "selection_reason": f"objective={objective}; predicted_best={best}; second={second}; gap={gap:.4f}; residual={residual:.4f}; fallback={fallback if fallback_used else 'none'}",
            "predictions": preds,
        }

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "feature_names": FEATURE_NAMES,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "best_fixed": self.best_fixed,
            "policy_residual": self.policy_residual,
            "priors": {
                "global_means": self.priors.global_means,
                "by_arrival": self.priors.by_arrival,
                "by_scale": self.priors.by_scale,
            },
            "models": {
                p: {
                    t: {
                        "coef": m.coef,
                        "residual_median": m.residual_median,
                        "residual_p95": m.residual_p95,
                        "method": m.method,
                    }
                    for t, m in targets.items()
                }
                for p, targets in self.models.items()
            },
            "uses_validation_labels_at_runtime": False,
            "uses_future_jct": False,
            "uses_future_tool_completion": False,
        }


def _oracle_p95(group: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    return min(group.items(), key=lambda kv: _f(kv[1], "p95_jct", float("inf")))


def evaluate_model(
    groups: dict[str, dict[str, dict[str, Any]]],
    train_ids: set[str],
    val_ids: set[str],
    split_type: str,
    objective: str = "balanced",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], TAPSCostModel]:
    model = TAPSCostModel()
    training_rows = model.fit(groups, train_ids, objective)
    out: list[dict[str, Any]] = []
    for cid in sorted(val_ids):
        group = groups[cid]
        rep = next(iter(group.values()))
        sel = model.select(rep, objective)
        selected_policy = str(sel["selected_policy"])
        selected = group[selected_policy]
        best_fixed_policy = model.best_fixed.get(objective, "reactive_admission")
        best_fixed = group[best_fixed_policy]
        oracle_policy, oracle_row = _oracle_p95(group)
        reactive = group.get("reactive_admission", best_fixed)
        out.append(
            {
                "config_id": cid,
                "selected_policy": selected_policy,
                "selected_p95": _f(selected, "p95_jct"),
                "selected_throughput": _f(selected, "throughput"),
                "best_fixed_policy": best_fixed_policy,
                "best_fixed_p95": _f(best_fixed, "p95_jct"),
                "best_fixed_throughput": _f(best_fixed, "throughput"),
                "oracle_policy": oracle_policy,
                "oracle_p95": _f(oracle_row, "p95_jct"),
                "gain_over_best_fixed_p95": _safe_gain(_f(best_fixed, "p95_jct"), _f(selected, "p95_jct")),
                "gain_over_reactive_p95": _safe_gain(_f(reactive, "p95_jct"), _f(selected, "p95_jct")),
                "regret_to_oracle_p95": (_f(selected, "p95_jct") - _f(oracle_row, "p95_jct")) / max(1e-9, _f(oracle_row, "p95_jct")),
                "throughput_gain_over_best_fixed": _safe_gain(_f(best_fixed, "throughput"), _f(selected, "throughput"), lower_better=False),
                "confidence": sel["predicted_confidence"],
                "fallback_used": sel["fallback_used"],
                "selection_reason": sel["selection_reason"],
                "split_type": split_type,
                "objective": objective,
                "total_sessions": rep.get("total_sessions"),
                "active_session_limit": rep.get("active_session_limit"),
                "effective_regions": rep.get("effective_regions"),
                "arrival_pattern": rep.get("arrival_pattern"),
                "memory_budget_gb": rep.get("memory_budget_gb"),
            }
        )
    return training_rows, out, model


def run_taps_compiler_validation(
    grid_csv: str | Path = "data/results/aligned_policy_grid_pr4_v8.csv",
    training_out: str | Path = "data/results/taps_cost_model_training_pr4_v8.csv",
    cost_validation_out: str | Path = "data/results/taps_cost_model_validation_pr4_v8.csv",
    validation_out: str | Path = "data/results/taps_compiler_validation_pr4_v8.csv",
    params_out: str | Path = "data/results/taps_cost_model_params_pr4_v8.json",
    objectives_out: str | Path = "data/results/taps_compiler_objectives_pr4_v8.csv",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dict(r) for r in _read_csv(grid_csv)]
    groups = {cid: g for cid, g in _group_rows(rows).items() if all(p in g for p in ALIGNED_POLICIES)}
    splits = build_splits(groups)
    all_training: list[dict[str, Any]] = []
    all_cost_validation: list[dict[str, Any]] = []
    all_validation: list[dict[str, Any]] = []
    saved_model: TAPSCostModel | None = None
    for split_name, (train_ids, val_ids) in splits.items():
        training, validation, model = evaluate_model(groups, train_ids, val_ids, split_name, "balanced")
        all_training.extend({"split_type": split_name, **r} for r in training)
        all_validation.extend(validation)
        for policy in ALIGNED_POLICIES:
            by_target: dict[str, list[float]] = defaultdict(list)
            for cid in val_ids:
                if policy not in groups[cid]:
                    continue
                actual = groups[cid][policy]
                pred = model.predict_policy(actual, policy)
                for target in TARGETS:
                    by_target[target].append(abs(pred[target] - _f(actual, target)))
            for target, errs in by_target.items():
                all_cost_validation.append(
                    {
                        "split_type": split_name,
                        "policy": policy,
                        "target": target,
                        "validation_rows": len(errs),
                        "median_abs_error": _median(errs),
                        "p95_abs_error": float(np.percentile(errs, 95)) if errs else 0.0,
                    }
                )
        if split_name == "random":
            saved_model = model
    write_csv(training_out, all_training)
    write_csv(cost_validation_out, all_cost_validation)
    write_csv(validation_out, all_validation)
    write_json(params_out, saved_model.to_jsonable() if saved_model else {})

    objective_rows: list[dict[str, Any]] = []
    train_ids, val_ids = splits.get("random", next(iter(splits.values())))
    for objective in OBJECTIVES:
        _, validation, model = evaluate_model(groups, train_ids, val_ids, "random", objective)
        summary = summarize_validation(validation)
        selected_counts = Counter(r["selected_policy"] for r in validation)
        objective_rows.append(
            {
                "objective": objective,
                "best_fixed_policy": model.best_fixed.get(objective),
                "selected_policy_distribution": json.dumps(dict(selected_counts), sort_keys=True),
                "gain_over_best_fixed_p95": summary["mean_gain_over_best_fixed_p95"],
                "throughput_gain_over_best_fixed": summary["mean_throughput_gain_over_best_fixed"],
                "regret_to_oracle_p95": summary["mean_regret_to_oracle_p95"],
                "validation_rows": len(validation),
            }
        )
    write_csv(objectives_out, objective_rows)
    return all_training, all_validation


def summarize_validation(rows: list[dict[str, Any]], split_prefix: str | None = None) -> dict[str, float]:
    sub = [r for r in rows if split_prefix is None or str(r.get("split_type", "")).startswith(split_prefix)]
    if not sub:
        return {
            "mean_gain_over_best_fixed_p95": 0.0,
            "mean_throughput_gain_over_best_fixed": 0.0,
            "mean_regret_to_oracle_p95": 0.0,
            "mean_gain_over_reactive_p95": 0.0,
            "failure_configs": 0.0,
        }
    return {
        "mean_gain_over_best_fixed_p95": sum(_f(r, "gain_over_best_fixed_p95") for r in sub) / len(sub),
        "mean_throughput_gain_over_best_fixed": sum(_f(r, "throughput_gain_over_best_fixed") for r in sub) / len(sub),
        "mean_regret_to_oracle_p95": sum(_f(r, "regret_to_oracle_p95") for r in sub) / len(sub),
        "mean_gain_over_reactive_p95": sum(_f(r, "gain_over_reactive_p95") for r in sub) / len(sub),
        "failure_configs": float(sum(1 for r in sub if _f(r, "gain_over_best_fixed_p95") < 0)),
    }


def write_plots(validation_csv: str | Path = "data/results/taps_compiler_validation_pr4_v8.csv") -> None:
    rows = [r for r in _read_csv(validation_csv) if r.get("split_type") == "random"]
    if not rows:
        return
    ensure_dir("data/plots")
    rows = sorted(rows, key=lambda r: r["config_id"])
    x = list(range(len(rows)))
    plt.figure(figsize=(8.0, 3.8))
    plt.plot(x, [_f(r, "selected_p95") for r in rows], marker="o", label="TAPS-C")
    plt.plot(x, [_f(r, "best_fixed_p95") for r in rows], marker="o", label="best fixed")
    plt.plot(x, [_f(r, "oracle_p95") for r in rows], marker="o", label="oracle envelope")
    plt.ylabel("p95 JCT")
    plt.xlabel("random validation config")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/plots/taps_compiler_p95_pr4_v8.pdf")
    plt.close()

    plt.figure(figsize=(8.0, 3.8))
    plt.bar(x, [_f(r, "regret_to_oracle_p95") for r in rows])
    plt.ylabel("regret to oracle p95")
    plt.xlabel("random validation config")
    plt.tight_layout()
    plt.savefig("data/plots/taps_compiler_oracle_regret_pr4_v8.pdf")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/results/aligned_policy_grid_pr4_v8.csv")
    args = ap.parse_args()
    training, validation = run_taps_compiler_validation(args.grid)
    write_plots()
    print(json.dumps({"training_rows": len(training), "validation_rows": len(validation)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
