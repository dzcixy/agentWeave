from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agentweaver.simulator.aligned_policy_sweep import ALIGNED_POLICIES
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv, write_json


OBJECTIVES = ["p95_opt", "throughput_opt", "balanced"]
EPS = 1e-9


FEATURE_KEYS = [
    "bias",
    "total_sessions_log",
    "active_session_limit_log",
    "effective_regions_log",
    "memory_budget_log",
    "session_pressure",
    "region_pressure",
    "memory_budget_per_active_session",
    "predicted_arrival_burstiness",
    "trace_llm_time_mean",
    "trace_llm_time_p95",
    "trace_tool_time_mean",
    "trace_tool_time_p95",
    "tool_time_share",
    "llm_time_share",
    "branch_jct_cv",
    "tool_latency_cv",
    "shared_context_ratio",
    "estimated_domain_fanout",
    "estimated_context_entropy",
    "predicted_ready_depth",
    "predicted_blocked_fraction",
    "estimated_remote_kv_pressure",
    "arrival_closed_loop",
    "arrival_poisson",
    "arrival_bursty",
]


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
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _median(vals: list[float]) -> float:
    vals = sorted(v for v in vals if math.isfinite(v) and v > 0)
    if not vals:
        return 1.0
    return vals[len(vals) // 2]


def _safe_gain(base: float, new: float, lower_better: bool = True) -> float:
    if base <= 0:
        return 0.0
    return (base - new) / base if lower_better else (new - base) / base


def _group(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[str(row["config_id"])][str(row["policy"])] = row
    return groups


def _split_random(configs: list[str]) -> tuple[set[str], set[str]]:
    train, val = set(), set()
    for cid in sorted(configs):
        (val if int(stable_hash(("v9", cid)), 16) % 5 == 0 else train).add(cid)
    if not train or not val:
        for i, cid in enumerate(sorted(configs)):
            (val if i % 5 == 0 else train).add(cid)
    return train, val


def build_splits(groups: dict[str, dict[str, dict[str, Any]]]) -> dict[str, tuple[set[str], set[str]]]:
    configs = sorted(groups)
    splits = {"random": _split_random(configs)}
    arrivals = sorted({next(iter(groups[c].values())).get("arrival_pattern", "") for c in configs})
    for arrival in arrivals:
        val = {c for c in configs if next(iter(groups[c].values())).get("arrival_pattern") == arrival}
        if val and len(val) < len(configs):
            splits[f"leave_arrival_{arrival}"] = (set(configs) - val, val)
    totals = sorted({int(_f(next(iter(groups[c].values())), "total_sessions")) for c in configs})
    for total in totals:
        val = {c for c in configs if int(_f(next(iter(groups[c].values())), "total_sessions")) == total}
        if val and len(val) < len(configs):
            splits[f"leave_session_{total}"] = (set(configs) - val, val)
    memories = sorted({int(_f(next(iter(groups[c].values())), "memory_budget_gb")) for c in configs})
    for mem in memories:
        val = {c for c in configs if int(_f(next(iter(groups[c].values())), "memory_budget_gb")) == mem}
        if val and len(val) < len(configs):
            splits[f"leave_memory_{mem}"] = (set(configs) - val, val)
    return splits


def _train_medians(groups: dict[str, dict[str, dict[str, Any]]], train_ids: set[str]) -> dict[str, float]:
    rows = [r for cid in train_ids for r in groups[cid].values()]
    return {
        "p95_jct": _median([_f(r, "p95_jct") for r in rows]),
        "mean_jct": _median([_f(r, "mean_jct") for r in rows]),
        "throughput": _median([_f(r, "throughput") for r in rows]),
    }


def normalized_objective(row: dict[str, Any], med: dict[str, float], objective: str) -> float:
    if objective == "p95_opt":
        return _f(row, "p95_jct") / max(EPS, med["p95_jct"])
    if objective == "throughput_opt":
        # Constraint-aware objective: prioritize throughput but retain p95 in the score.
        return -(_f(row, "throughput") / max(EPS, med["throughput"])) + 0.05 * (_f(row, "p95_jct") / max(EPS, med["p95_jct"]))
    p95 = _f(row, "p95_jct") / max(EPS, med["p95_jct"])
    mean = _f(row, "mean_jct") / max(EPS, med["mean_jct"])
    throughput = _f(row, "throughput") / max(EPS, med["throughput"])
    return p95 + 0.2 * mean - 0.2 * throughput


def _features(row: dict[str, Any]) -> np.ndarray:
    arrival = row.get("arrival_pattern", "")
    vals = [
        1.0,
        math.log1p(_f(row, "total_sessions")),
        math.log1p(_f(row, "active_session_limit")),
        math.log1p(_f(row, "effective_regions")),
        math.log1p(_f(row, "memory_budget_gb")),
        _f(row, "session_pressure"),
        _f(row, "region_pressure"),
        _f(row, "memory_budget_per_active_session"),
        _f(row, "predicted_arrival_burstiness"),
        _f(row, "trace_llm_time_mean"),
        _f(row, "trace_llm_time_p95"),
        _f(row, "trace_tool_time_mean"),
        _f(row, "trace_tool_time_p95"),
        _f(row, "tool_time_share"),
        _f(row, "llm_time_share"),
        _f(row, "branch_jct_cv"),
        _f(row, "tool_latency_cv"),
        _f(row, "shared_context_ratio"),
        _f(row, "estimated_domain_fanout"),
        _f(row, "estimated_context_entropy"),
        _f(row, "predicted_ready_depth"),
        _f(row, "predicted_blocked_fraction"),
        _f(row, "estimated_remote_kv_pressure"),
        1.0 if arrival == "closed_loop" else 0.0,
        1.0 if arrival == "poisson" else 0.0,
        1.0 if arrival == "bursty" else 0.0,
    ]
    return np.asarray(vals, dtype=float)


@dataclass
class DeltaModel:
    coef: list[float]
    median_abs_error: float
    p95_abs_error: float
    train_pairs: int

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(np.asarray(self.coef), x))


class InvalidityRiskModel:
    def __init__(self) -> None:
        self.policy_rate: dict[str, float] = {}
        self.bucket_rate: dict[tuple[str, int, int], float] = {}

    @staticmethod
    def _bucket(feature: dict[str, Any]) -> tuple[int, int]:
        return (min(10, int(_f(feature, "session_pressure"))), min(10, int(_f(feature, "region_pressure"))))

    def fit(self, audit_rows: list[dict[str, Any]], features: dict[str, dict[str, Any]], train_ids: set[str]) -> None:
        policy_counts: Counter[str] = Counter()
        policy_invalid: Counter[str] = Counter()
        bucket_counts: Counter[tuple[str, int, int]] = Counter()
        bucket_invalid: Counter[tuple[str, int, int]] = Counter()
        for row in audit_rows:
            cid = str(row.get("config_id", ""))
            if cid not in train_ids:
                continue
            policy = str(row.get("policy", ""))
            feat = features.get(cid, row)
            key = (policy, *self._bucket(feat))
            invalid = str(row.get("validity", "")).lower() == "false"
            policy_counts[policy] += 1
            bucket_counts[key] += 1
            if invalid:
                policy_invalid[policy] += 1
                bucket_invalid[key] += 1
        self.policy_rate = {
            policy: policy_invalid[policy] / max(1, count)
            for policy, count in policy_counts.items()
        }
        self.bucket_rate = {
            key: bucket_invalid[key] / max(1, count)
            for key, count in bucket_counts.items()
        }

    def predict(self, policy: str, feature: dict[str, Any]) -> float:
        key = (policy, *self._bucket(feature))
        return max(self.policy_rate.get(policy, 0.0), self.bucket_rate.get(key, 0.0))


class PairwiseCompiler:
    def __init__(self, invalid_risk_threshold: float = 0.0, confidence_threshold: float = 0.10) -> None:
        self.models: dict[str, dict[tuple[str, str], DeltaModel]] = defaultdict(dict)
        self.invalidity = InvalidityRiskModel()
        self.best_fixed: dict[str, str] = {}
        self.medians: dict[str, float] = {}
        self.invalid_risk_threshold = invalid_risk_threshold
        self.confidence_threshold = confidence_threshold

    def fit(
        self,
        groups: dict[str, dict[str, dict[str, Any]]],
        features: dict[str, dict[str, Any]],
        audit_rows: list[dict[str, Any]],
        train_ids: set[str],
    ) -> list[dict[str, Any]]:
        self.medians = _train_medians(groups, train_ids)
        training_rows: list[dict[str, Any]] = []
        self.invalidity.fit(audit_rows, features, train_ids)
        for objective in OBJECTIVES:
            obj_by_policy: dict[str, list[float]] = defaultdict(list)
            for cid in train_ids:
                for policy, row in groups[cid].items():
                    obj_by_policy[policy].append(normalized_objective(row, self.medians, objective))
            safe_policies = [
                p
                for p in ALIGNED_POLICIES
                if obj_by_policy.get(p) and self.invalidity.policy_rate.get(p, 0.0) <= self.invalid_risk_threshold
            ]
            if not safe_policies:
                safe_policies = [p for p in ALIGNED_POLICIES if obj_by_policy.get(p)]
            self.best_fixed[objective] = min(
                safe_policies,
                key=lambda p: sum(obj_by_policy[p]) / len(obj_by_policy[p]),
                default="reactive_admission",
            )

        for i, a in enumerate(ALIGNED_POLICIES):
            for b in ALIGNED_POLICIES[i + 1 :]:
                pair_rows = [cid for cid in train_ids if a in groups[cid] and b in groups[cid] and cid in features]
                if not pair_rows:
                    continue
                x = np.vstack([_features(features[cid]) for cid in pair_rows])
                xtx = x.T @ x + 1e-3 * np.eye(x.shape[1])
                inv = np.linalg.pinv(xtx) @ x.T
                targets: dict[str, np.ndarray] = {}
                targets["p95_opt"] = np.asarray([_f(groups[cid][a], "p95_jct") - _f(groups[cid][b], "p95_jct") for cid in pair_rows])
                targets["throughput_opt"] = np.asarray([_f(groups[cid][a], "throughput") - _f(groups[cid][b], "throughput") for cid in pair_rows])
                targets["balanced"] = np.asarray(
                    [
                        normalized_objective(groups[cid][a], self.medians, "balanced")
                        - normalized_objective(groups[cid][b], self.medians, "balanced")
                        for cid in pair_rows
                    ]
                )
                for objective, y in targets.items():
                    coef = inv @ y
                    pred = x @ coef
                    err = np.abs(pred - y)
                    model = DeltaModel(
                        coef=[float(c) for c in coef],
                        median_abs_error=_median(err.tolist()),
                        p95_abs_error=float(np.percentile(err, 95)) if len(err) else 0.0,
                        train_pairs=len(pair_rows),
                    )
                    self.models[objective][(a, b)] = model
                    training_rows.append(
                        {
                            "objective": objective,
                            "policy_a": a,
                            "policy_b": b,
                            "train_pairs": len(pair_rows),
                            "median_abs_error": model.median_abs_error,
                            "p95_abs_error": model.p95_abs_error,
                        }
                    )
        return training_rows

    def _pair_delta(self, objective: str, a: str, b: str, feature: dict[str, Any]) -> tuple[float, float]:
        if (a, b) in self.models[objective]:
            model = self.models[objective][(a, b)]
            return model.predict(_features(feature)), model.median_abs_error
        if (b, a) in self.models[objective]:
            model = self.models[objective][(b, a)]
            return -model.predict(_features(feature)), model.median_abs_error
        return 0.0, 1e9

    def select(self, feature: dict[str, Any], objective: str) -> tuple[str, float, float, bool, str]:
        risks = {policy: self.invalidity.predict(policy, feature) for policy in ALIGNED_POLICIES}
        candidates = [p for p in ALIGNED_POLICIES if risks[p] <= self.invalid_risk_threshold]
        fallback = self.best_fixed.get(objective, "reactive_admission")
        if fallback not in candidates:
            candidates.append(fallback)
        if not candidates:
            candidates = [fallback]
        votes: Counter[str] = Counter()
        uncertainty: dict[str, list[float]] = defaultdict(list)
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                delta, err = self._pair_delta(objective, a, b, feature)
                if objective == "throughput_opt":
                    winner = a if delta > 0 else b
                    margin = abs(delta)
                else:
                    winner = a if delta < 0 else b
                    margin = abs(delta)
                votes[winner] += 1
                uncertainty[a].append(err)
                uncertainty[b].append(err)
                if margin <= err:
                    # Low-confidence pair: give both a weak vote so fallback can win via confidence.
                    votes[a] += 0
                    votes[b] += 0
        if not votes:
            return fallback, risks.get(fallback, 0.0), 0.0, True, "fallback:no_pairwise_votes"
        ranked = votes.most_common()
        selected = ranked[0][0]
        top = ranked[0][1]
        second = ranked[1][1] if len(ranked) > 1 else 0
        confidence = (top - second) / max(1.0, len(candidates) - 1)
        avg_uncertainty = sum(uncertainty.get(selected, [1.0])) / max(1, len(uncertainty.get(selected, [1.0])))
        confidence = max(0.0, min(1.0, confidence * (1.0 - risks.get(selected, 0.0)) / (1.0 + avg_uncertainty)))
        if confidence < self.confidence_threshold:
            return fallback, risks.get(fallback, 0.0), confidence, True, f"fallback:low_confidence:{confidence:.4f}"
        if risks.get(selected, 0.0) > self.invalid_risk_threshold:
            return fallback, risks.get(selected, 0.0), confidence, True, f"fallback:invalidity_risk:{risks.get(selected, 0.0):.4f}"
        return selected, risks.get(selected, 0.0), confidence, False, f"pairwise_votes:{dict(votes)}"


def evaluate_compiler(
    valid_grid: str | Path = "data/results/aligned_policy_grid_valid_pr4_v9.csv",
    audit_csv: str | Path = "data/results/aligned_policy_grid_audit_pr4_v9.csv",
    features_csv: str | Path = "data/results/workload_features_pr4_v9.csv",
    training_out: str | Path = "data/results/taps_cost_model_v2_training_pr4_v9.csv",
    validation_out: str | Path = "data/results/taps_compiler_v2_validation_pr4_v9.csv",
    params_out: str | Path = "data/results/taps_compiler_v2_params_pr4_v9.json",
    objectives_out: str | Path = "data/results/taps_compiler_v2_objectives_pr4_v9.csv",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_csv(valid_grid)
    audit_rows = _read_csv(audit_csv)
    features = {r["config_id"]: r for r in _read_csv(features_csv)}
    groups = _group(rows)
    splits = build_splits(groups)
    all_training: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    objective_summary: list[dict[str, Any]] = []
    params: dict[str, Any] = {"feature_keys": FEATURE_KEYS, "splits": sorted(splits), "objectives": OBJECTIVES}

    for split_name, (train_ids, val_ids) in splits.items():
        compiler = PairwiseCompiler()
        training = compiler.fit(groups, features, audit_rows, train_ids)
        for row in training:
            row["split_type"] = split_name
        all_training.extend(training)
        params[f"{split_name}_best_fixed"] = compiler.best_fixed
        for objective in OBJECTIVES:
            selected_policies: Counter[str] = Counter()
            invalid_selection_count = 0
            failure_configs: list[str] = []
            for cid in sorted(val_ids):
                if cid not in features or cid not in groups:
                    continue
                group = groups[cid]
                selected, risk, confidence, fallback, reason = compiler.select(features[cid], objective)
                selected_row = group.get(selected)
                invalid_selected = selected_row is None
                if invalid_selected:
                    invalid_selection_count += 1
                    failure_configs.append(cid)
                    # Metric reporting falls back to best fixed only for observability; invalid_selection_rate still records the failed choice.
                    selected = compiler.best_fixed.get(objective, "reactive_admission")
                    selected_row = group.get(selected)
                    fallback = True
                    reason = f"invalid_selection_blocked:{reason}"
                if selected_row is None:
                    continue
                selected_policies[selected] += 1
                best_fixed = compiler.best_fixed.get(objective, "reactive_admission")
                best_fixed_row = group.get(best_fixed)
                if best_fixed_row is None:
                    best_fixed_row = min(group.values(), key=lambda r: normalized_objective(r, compiler.medians, objective))
                    best_fixed = str(best_fixed_row["policy"])
                oracle_row = min(group.values(), key=lambda r: _f(r, "p95_jct", float("inf")))
                reactive = group.get("reactive_admission")
                validation_rows.append(
                    {
                        "config_id": cid,
                        "split_type": split_name,
                        "objective": objective,
                        "selected_policy": selected,
                        "best_fixed_policy": best_fixed,
                        "oracle_policy": oracle_row["policy"],
                        "selected_p95": _f(selected_row, "p95_jct"),
                        "best_fixed_p95": _f(best_fixed_row, "p95_jct"),
                        "oracle_p95": _f(oracle_row, "p95_jct"),
                        "selected_throughput": _f(selected_row, "throughput"),
                        "best_fixed_throughput": _f(best_fixed_row, "throughput"),
                        "gain_over_best_fixed_p95": _safe_gain(_f(best_fixed_row, "p95_jct"), _f(selected_row, "p95_jct")),
                        "throughput_gain_over_best_fixed": _safe_gain(_f(best_fixed_row, "throughput"), _f(selected_row, "throughput"), lower_better=False),
                        "gain_over_reactive_p95": _safe_gain(_f(reactive, "p95_jct"), _f(selected_row, "p95_jct")) if reactive else 0.0,
                        "regret_to_oracle_p95": max(0.0, (_f(selected_row, "p95_jct") - _f(oracle_row, "p95_jct")) / max(EPS, _f(oracle_row, "p95_jct"))),
                        "selected_policy_invalid_risk": risk,
                        "confidence": confidence,
                        "fallback_used": str(fallback).lower(),
                        "invalid_selection": str(invalid_selected).lower(),
                        "selection_reason": reason,
                    }
                )
            vals = [r for r in validation_rows if r["split_type"] == split_name and r["objective"] == objective]
            objective_summary.append(
                {
                    "split_type": split_name,
                    "objective": objective,
                    "best_fixed_policy": compiler.best_fixed.get(objective, ""),
                    "selected_policy_distribution": json.dumps(dict(selected_policies), sort_keys=True),
                    "validation_rows": len(vals),
                    "invalid_selection_rate": invalid_selection_count / max(1, len(val_ids)),
                    "gain_over_best_fixed_p95": sum(_f(r, "gain_over_best_fixed_p95") for r in vals) / max(1, len(vals)),
                    "throughput_gain_over_best_fixed": sum(_f(r, "throughput_gain_over_best_fixed") for r in vals) / max(1, len(vals)),
                    "regret_to_oracle_p95": sum(_f(r, "regret_to_oracle_p95") for r in vals) / max(1, len(vals)),
                    "worst_case_regret": max([_f(r, "regret_to_oracle_p95") for r in vals] or [0.0]),
                    "failure_configs": ";".join(failure_configs[:20]),
                }
            )

    write_csv(training_out, all_training)
    write_csv(validation_out, validation_rows)
    write_csv(objectives_out, objective_summary)
    write_json(params_out, params)
    summary = summarize_validation(validation_rows)
    return all_training, validation_rows, summary


def summarize_validation(rows: list[dict[str, Any]], split_type: str | None = None, objective: str = "balanced") -> dict[str, float]:
    vals = [
        r
        for r in rows
        if (split_type is None or r.get("split_type") == split_type)
        and r.get("objective") == objective
    ]
    def avg(key: str) -> float:
        return sum(_f(r, key) for r in vals) / max(1, len(vals))
    invalid_rate = sum(1 for r in vals if str(r.get("invalid_selection", "")).lower() == "true") / max(1, len(vals))
    return {
        "rows": float(len(vals)),
        "mean_gain_over_best_fixed_p95": avg("gain_over_best_fixed_p95"),
        "mean_throughput_gain_over_best_fixed": avg("throughput_gain_over_best_fixed"),
        "mean_gain_over_reactive_p95": avg("gain_over_reactive_p95"),
        "mean_regret_to_oracle_p95": avg("regret_to_oracle_p95"),
        "invalid_selection_rate": invalid_rate,
        "worst_case_regret": max([_f(r, "regret_to_oracle_p95") for r in vals] or [0.0]),
        "failure_configs": float(sum(1 for r in vals if _f(r, "gain_over_best_fixed_p95") < 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-grid", default="data/results/aligned_policy_grid_valid_pr4_v9.csv")
    ap.add_argument("--audit", default="data/results/aligned_policy_grid_audit_pr4_v9.csv")
    ap.add_argument("--features", default="data/results/workload_features_pr4_v9.csv")
    args = ap.parse_args()
    training, validation, summary = evaluate_compiler(args.valid_grid, args.audit, args.features)
    print(json.dumps({"training_rows": len(training), "validation_rows": len(validation), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
