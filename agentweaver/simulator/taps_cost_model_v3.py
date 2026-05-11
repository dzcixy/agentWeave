from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agentweaver.simulator.aligned_policy_sweep import ALIGNED_POLICIES
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import write_csv, write_json


OBJECTIVES = ["p95_opt", "throughput_opt", "balanced", "wafer_efficiency"]
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
    "trace_mix_instance_entropy",
    "trace_llm_time_mean",
    "trace_llm_time_p95",
    "trace_tool_time_mean",
    "trace_tool_time_p95",
    "tool_time_share",
    "llm_time_share",
    "branch_jct_cv",
    "tool_latency_cv",
    "context_reuse_tokens_log",
    "shared_context_ratio",
    "estimated_context_domain_count_log",
    "estimated_context_entropy",
    "predicted_ready_depth",
    "predicted_tool_blocked_fraction",
    "estimated_remote_context_bytes_log",
    "context_hotspot_score",
    "tool_events_per_session",
    "llm_events_per_session",
    "arrival_closed_loop",
    "arrival_poisson",
    "arrival_bursty",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        (val if int(stable_hash(("v10", cid)), 16) % 5 == 0 else train).add(cid)
    if not train or not val:
        for i, cid in enumerate(sorted(configs)):
            (val if i % 5 == 0 else train).add(cid)
    return train, val


def build_splits(groups: dict[str, dict[str, dict[str, Any]]]) -> dict[str, tuple[set[str], set[str]]]:
    configs = sorted(groups)
    splits = {"random": _split_random(configs)}
    def first(cid: str) -> dict[str, Any]:
        return next(iter(groups[cid].values()))
    for arrival in sorted({first(c).get("arrival_pattern", "") for c in configs}):
        val = {c for c in configs if first(c).get("arrival_pattern") == arrival}
        if val and len(val) < len(configs):
            splits[f"leave_arrival_{arrival}"] = (set(configs) - val, val)
    for total in sorted({int(_f(first(c), "total_sessions")) for c in configs}):
        val = {c for c in configs if int(_f(first(c), "total_sessions")) == total}
        if val and len(val) < len(configs):
            splits[f"leave_session_{total}"] = (set(configs) - val, val)
    for mem in sorted({int(_f(first(c), "memory_budget_gb")) for c in configs}):
        val = {c for c in configs if int(_f(first(c), "memory_budget_gb")) == mem}
        if val and len(val) < len(configs):
            splits[f"leave_memory_{mem}"] = (set(configs) - val, val)
    for regions in sorted({int(_f(first(c), "effective_regions")) for c in configs}):
        val = {c for c in configs if int(_f(first(c), "effective_regions")) == regions}
        if val and len(val) < len(configs):
            splits[f"leave_region_{regions}"] = (set(configs) - val, val)
    return splits


def _train_medians(groups: dict[str, dict[str, dict[str, Any]]], train_ids: set[str]) -> dict[str, float]:
    rows = [r for cid in train_ids for r in groups[cid].values()]
    return {
        "p95_jct": _median([_f(r, "p95_jct") for r in rows]),
        "mean_jct": _median([_f(r, "mean_jct") for r in rows]),
        "throughput": _median([_f(r, "throughput") for r in rows]),
        "ready_queue_wait": _median([_f(r, "ready_queue_wait") for r in rows]),
        "remote_kv_bytes": _median([_f(r, "remote_kv_bytes") for r in rows]),
    }


def normalized_objective(row: dict[str, Any], med: dict[str, float], objective: str, best_fixed_norm_p95: float = 1.0) -> float:
    norm_p95 = _f(row, "p95_jct") / max(EPS, med["p95_jct"])
    norm_mean = _f(row, "mean_jct") / max(EPS, med["mean_jct"])
    norm_thr = _f(row, "throughput") / max(EPS, med["throughput"])
    norm_wait = _f(row, "ready_queue_wait") / max(EPS, med["ready_queue_wait"])
    norm_remote = _f(row, "remote_kv_bytes") / max(EPS, med["remote_kv_bytes"])
    if objective == "p95_opt":
        return norm_p95 + 0.1 * norm_mean
    if objective == "throughput_opt":
        return -norm_thr + 0.2 * max(0.0, norm_p95 - best_fixed_norm_p95 * 1.05)
    if objective == "wafer_efficiency":
        return norm_p95 - 0.2 * _f(row, "region_utilization") + 0.2 * norm_remote
    return norm_p95 + 0.2 * norm_mean - 0.3 * norm_thr + 0.1 * norm_wait


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
        _f(row, "trace_mix_instance_entropy"),
        _f(row, "trace_llm_time_mean"),
        _f(row, "trace_llm_time_p95"),
        _f(row, "trace_tool_time_mean"),
        _f(row, "trace_tool_time_p95"),
        _f(row, "tool_time_share"),
        _f(row, "llm_time_share"),
        _f(row, "branch_jct_cv"),
        _f(row, "tool_latency_cv"),
        math.log1p(_f(row, "context_reuse_tokens")),
        _f(row, "shared_context_ratio"),
        math.log1p(_f(row, "estimated_context_domain_count")),
        _f(row, "estimated_context_entropy"),
        _f(row, "predicted_ready_depth"),
        _f(row, "predicted_tool_blocked_fraction", _f(row, "predicted_blocked_fraction")),
        math.log1p(_f(row, "estimated_remote_context_bytes")),
        _f(row, "context_hotspot_score"),
        _f(row, "tool_events_per_session"),
        _f(row, "llm_events_per_session"),
        1.0 if arrival == "closed_loop" else 0.0,
        1.0 if arrival == "poisson" else 0.0,
        1.0 if arrival == "bursty" else 0.0,
    ]
    return np.asarray(vals, dtype=float)


@dataclass
class LinearModel:
    coef: list[float]
    median_abs_error: float
    p95_abs_error: float
    train_rows: int

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(np.asarray(self.coef), x))


class InvalidityRiskModel:
    def __init__(self) -> None:
        self.policy_rate: dict[str, float] = {}
        self.bucket_rate: dict[tuple[str, int, int, int], float] = {}

    @staticmethod
    def _bucket(feature: dict[str, Any]) -> tuple[int, int, int]:
        return (
            min(12, int(_f(feature, "session_pressure"))),
            min(12, int(_f(feature, "region_pressure"))),
            min(12, int(_f(feature, "predicted_ready_depth"))),
        )

    def fit(self, audit_rows: list[dict[str, Any]], features: dict[str, dict[str, Any]], train_ids: set[str]) -> None:
        policy_counts: Counter[str] = Counter()
        policy_invalid: Counter[str] = Counter()
        bucket_counts: Counter[tuple[str, int, int, int]] = Counter()
        bucket_invalid: Counter[tuple[str, int, int, int]] = Counter()
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
        self.policy_rate = {p: policy_invalid[p] / max(1, n) for p, n in policy_counts.items()}
        self.bucket_rate = {k: bucket_invalid[k] / max(1, n) for k, n in bucket_counts.items()}

    def predict(self, policy: str, feature: dict[str, Any]) -> float:
        key = (policy, *self._bucket(feature))
        return max(self.policy_rate.get(policy, 0.0), self.bucket_rate.get(key, 0.0))


class TAPSCompilerV3:
    def __init__(self, lambda_pairwise: float = 0.35, lambda_invalid: float = 4.0, lambda_regime: float = 0.25) -> None:
        self.lambda_pairwise = lambda_pairwise
        self.lambda_invalid = lambda_invalid
        self.lambda_regime = lambda_regime
        self.medians: dict[str, float] = {}
        self.best_fixed: dict[str, str] = {}
        self.best_fixed_norm_p95: dict[str, float] = {}
        self.invalidity = InvalidityRiskModel()
        self.listwise: dict[str, dict[str, LinearModel]] = defaultdict(dict)
        self.pairwise: dict[str, dict[tuple[str, str], LinearModel]] = defaultdict(dict)
        self.train_policy_invalid_free: dict[str, bool] = {}
        self.train_arrivals: set[str] = set()
        self.train_sessions: set[int] = set()
        self.train_memories: set[int] = set()
        self.train_regions: set[int] = set()

    def _fit_linear(self, xs: list[np.ndarray], ys: list[float]) -> LinearModel:
        if not xs:
            return LinearModel([0.0] * len(FEATURE_KEYS), 1e9, 1e9, 0)
        x = np.vstack(xs)
        y = np.asarray(ys, dtype=float)
        xtx = x.T @ x + 1e-3 * np.eye(x.shape[1])
        coef = np.linalg.pinv(xtx) @ x.T @ y
        pred = x @ coef
        err = np.abs(pred - y)
        return LinearModel(
            coef=[float(c) for c in coef],
            median_abs_error=float(np.median(err)) if len(err) else 0.0,
            p95_abs_error=float(np.percentile(err, 95)) if len(err) else 0.0,
            train_rows=len(xs),
        )

    def fit(
        self,
        groups: dict[str, dict[str, dict[str, Any]]],
        features: dict[str, dict[str, Any]],
        audit_rows: list[dict[str, Any]],
        train_ids: set[str],
    ) -> list[dict[str, Any]]:
        self.medians = _train_medians(groups, train_ids)
        for cid in train_ids:
            if cid not in groups:
                continue
            row = next(iter(groups[cid].values()))
            self.train_arrivals.add(str(row.get("arrival_pattern", "")))
            self.train_sessions.add(int(_f(row, "total_sessions")))
            self.train_memories.add(int(_f(row, "memory_budget_gb")))
            self.train_regions.add(int(_f(row, "effective_regions")))
        self.invalidity.fit(audit_rows, features, train_ids)
        self.train_policy_invalid_free = {p: self.invalidity.policy_rate.get(p, 0.0) == 0.0 for p in ALIGNED_POLICIES}
        training_rows: list[dict[str, Any]] = []
        for objective in OBJECTIVES:
            obj_by_policy: dict[str, list[float]] = defaultdict(list)
            for cid in train_ids:
                for policy, row in groups[cid].items():
                    obj_by_policy[policy].append(normalized_objective(row, self.medians, objective))
            safe_policies = [p for p in ALIGNED_POLICIES if obj_by_policy.get(p) and self.train_policy_invalid_free.get(p, False)]
            if not safe_policies:
                safe_policies = [p for p in ALIGNED_POLICIES if obj_by_policy.get(p)]
            self.best_fixed[objective] = min(safe_policies, key=lambda p: sum(obj_by_policy[p]) / len(obj_by_policy[p]), default="reactive_admission")
            bf_vals = obj_by_policy.get(self.best_fixed[objective], [1.0])
            self.best_fixed_norm_p95[objective] = sum(bf_vals) / max(1, len(bf_vals))

            for policy in ALIGNED_POLICIES:
                xs: list[np.ndarray] = []
                ys: list[float] = []
                for cid in train_ids:
                    if cid in features and policy in groups[cid]:
                        xs.append(_features(features[cid]))
                        ys.append(normalized_objective(groups[cid][policy], self.medians, objective, self.best_fixed_norm_p95[objective]))
                model = self._fit_linear(xs, ys)
                self.listwise[objective][policy] = model
                training_rows.append(
                    {
                        "model": "listwise",
                        "objective": objective,
                        "policy": policy,
                        "train_rows": model.train_rows,
                        "median_abs_error": model.median_abs_error,
                        "p95_abs_error": model.p95_abs_error,
                    }
                )

            for i, a in enumerate(ALIGNED_POLICIES):
                for b in ALIGNED_POLICIES[i + 1 :]:
                    xs = []
                    ys = []
                    for cid in train_ids:
                        if cid in features and a in groups[cid] and b in groups[cid]:
                            xs.append(_features(features[cid]))
                            ys.append(
                                normalized_objective(groups[cid][a], self.medians, objective, self.best_fixed_norm_p95[objective])
                                - normalized_objective(groups[cid][b], self.medians, objective, self.best_fixed_norm_p95[objective])
                            )
                    model = self._fit_linear(xs, ys)
                    self.pairwise[objective][(a, b)] = model
                    training_rows.append(
                        {
                            "model": "pairwise",
                            "objective": objective,
                            "policy_a": a,
                            "policy_b": b,
                            "train_rows": model.train_rows,
                            "median_abs_error": model.median_abs_error,
                            "p95_abs_error": model.p95_abs_error,
                        }
                    )
        return training_rows

    def _pair_delta(self, objective: str, a: str, b: str, feature: dict[str, Any]) -> tuple[float, float]:
        if (a, b) in self.pairwise[objective]:
            m = self.pairwise[objective][(a, b)]
            return m.predict(_features(feature)), m.median_abs_error
        if (b, a) in self.pairwise[objective]:
            m = self.pairwise[objective][(b, a)]
            return -m.predict(_features(feature)), m.median_abs_error
        return 0.0, 1e9

    def _regime_prior(self, policy: str, feature: dict[str, Any]) -> float:
        blocked = _f(feature, "predicted_tool_blocked_fraction", _f(feature, "predicted_blocked_fraction"))
        ready = _f(feature, "predicted_ready_depth")
        regions = max(1.0, _f(feature, "effective_regions"))
        remote = math.log1p(_f(feature, "estimated_remote_context_bytes"))
        hotspot = _f(feature, "context_hotspot_score")
        memory_per_active = _f(feature, "memory_budget_per_active_session")
        tail = _f(feature, "branch_jct_cv") + _f(feature, "tool_latency_cv")
        prior = 0.0
        if blocked > 0.45 and ready <= 1.5 * regions and policy in {"reactive_admission", "taps_admission_v4", "acd_nisp"}:
            prior += 1.0
        if hotspot > 0.25 and remote > 12.0 and policy in {"taps_domain_v4", "taps_unified_v5", "taps_unified_adaptive_v6"}:
            prior += 1.0
        if tail > 1.0 and policy in {"taps_unified_v5", "taps_unified_adaptive_v6", "taps_admission_v4"}:
            prior += 0.8
        if memory_per_active < 1.0 and policy in {"acd_nisp", "reactive_admission"}:
            prior += 0.8
        return prior

    def select(self, feature: dict[str, Any], valid_policies: set[str], objective: str) -> tuple[str, dict[str, Any]]:
        fallback = self.best_fixed.get(objective, "reactive_admission")
        candidates = [p for p in ALIGNED_POLICIES if p in valid_policies]
        if not candidates:
            return fallback, {"fallback_used": True, "reason": "no_valid_candidates", "confidence": 0.0, "invalidity_risk": 1.0}
        ood_reasons: list[str] = []
        if str(feature.get("arrival_pattern", "")) not in self.train_arrivals:
            ood_reasons.append("arrival")
        if int(_f(feature, "total_sessions")) not in self.train_sessions:
            ood_reasons.append("total_sessions")
        if int(_f(feature, "memory_budget_gb")) not in self.train_memories:
            ood_reasons.append("memory_budget_gb")
        if int(_f(feature, "effective_regions")) not in self.train_regions:
            ood_reasons.append("effective_regions")
        if ood_reasons and fallback in valid_policies:
            if (
                "taps_domain_v4" in valid_policies
                and _f(feature, "total_sessions") <= _f(feature, "active_session_limit") + 1e-9
                and _f(feature, "effective_regions") <= 4
            ):
                risk = self.invalidity.predict("taps_domain_v4", feature)
                return "taps_domain_v4", {
                    "fallback_used": True,
                    "reason": "fallback:ood_regime_domain_full_admission_" + ",".join(ood_reasons),
                    "confidence": 0.0,
                    "invalidity_risk": risk,
                    "score_breakdown": "{}",
                }
            risk = self.invalidity.predict(fallback, feature)
            return fallback, {
                "fallback_used": True,
                "reason": "fallback:ood_" + ",".join(ood_reasons),
                "confidence": 0.0,
                "invalidity_risk": risk,
                "score_breakdown": "{}",
            }
        risks = {p: self.invalidity.predict(p, feature) for p in candidates}
        invalid_free = [p for p in candidates if risks[p] == 0.0]
        if invalid_free:
            candidates = invalid_free
        if fallback not in candidates and fallback in valid_policies and (not invalid_free or risks.get(fallback, 0.0) == 0.0):
            candidates.append(fallback)
        x = _features(feature)
        pair_votes: Counter[str] = Counter()
        pair_uncertainty: dict[str, list[float]] = defaultdict(list)
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                delta, err = self._pair_delta(objective, a, b, feature)
                winner = a if delta < 0 else b
                pair_votes[winner] += 1
                pair_uncertainty[a].append(err)
                pair_uncertainty[b].append(err)
        pair_score = {p: -pair_votes[p] / max(1.0, len(candidates) - 1) for p in candidates}
        final_scores: dict[str, float] = {}
        pieces: dict[str, dict[str, float]] = {}
        for p in candidates:
            list_model = self.listwise[objective].get(p)
            list_score = list_model.predict(x) if list_model else 0.0
            invalid = risks.get(p, 0.0)
            regime = self._regime_prior(p, feature)
            final = list_score + self.lambda_pairwise * pair_score[p] + self.lambda_invalid * invalid - self.lambda_regime * regime
            final_scores[p] = final
            pieces[p] = {"listwise": list_score, "pairwise": pair_score[p], "invalidity": invalid, "regime_prior": regime, "final": final}
        ranked = sorted(final_scores, key=final_scores.get)
        selected = ranked[0]
        gap = (final_scores[ranked[1]] - final_scores[selected]) if len(ranked) > 1 else 1.0
        uncertainty = sum(pair_uncertainty.get(selected, [0.0])) / max(1, len(pair_uncertainty.get(selected, [0.0])))
        vote_margin = (pair_votes[selected] - (pair_votes[ranked[1]] if len(ranked) > 1 else 0)) / max(1.0, len(candidates) - 1)
        confidence = max(0.0, min(1.0, 0.5 * gap + 0.5 * vote_margin))
        confidence = confidence / (1.0 + max(0.0, uncertainty))
        fallback_used = False
        reason = "hybrid"
        if confidence < 0.05 and fallback in valid_policies and risks.get(fallback, 0.0) <= risks.get(selected, 0.0):
            selected = fallback
            fallback_used = True
            reason = f"fallback:low_confidence:{confidence:.4f}"
        return selected, {
            "fallback_used": fallback_used,
            "reason": reason,
            "confidence": confidence,
            "invalidity_risk": risks.get(selected, 0.0),
            "score_breakdown": json.dumps(pieces, sort_keys=True),
        }


def evaluate_compiler(
    valid_grid: str | Path = "data/results/aligned_policy_grid_valid_pr4_v10.csv",
    audit_csv: str | Path = "data/results/aligned_policy_grid_audit_pr4_v10.csv",
    features_csv: str | Path = "data/results/workload_features_pr4_v10.csv",
    training_out: str | Path = "data/results/taps_cost_model_v3_training_pr4_v10.csv",
    validation_out: str | Path = "data/results/taps_compiler_v3_validation_pr4_v10.csv",
    params_out: str | Path = "data/results/taps_compiler_v3_params_pr4_v10.json",
    objectives_out: str | Path = "data/results/taps_compiler_v3_objectives_pr4_v10.csv",
    integrity_out: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_csv(valid_grid)
    audit_rows = _read_csv(audit_csv)
    features = {r["config_id"]: r for r in _read_csv(features_csv)}
    groups = _group(rows)
    splits = build_splits(groups)
    all_training: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    objective_summary: list[dict[str, Any]] = []
    params: dict[str, Any] = {
        "feature_keys": FEATURE_KEYS,
        "splits": sorted(splits),
        "objectives": OBJECTIVES,
        "policies": ALIGNED_POLICIES,
        "no_validation_labels_at_runtime": True,
        "no_future_jct_or_tool_completion_at_runtime": True,
    }
    for split_name, (train_ids, val_ids) in splits.items():
        compiler = TAPSCompilerV3()
        training = compiler.fit(groups, features, audit_rows, train_ids)
        for row in training:
            row["split_type"] = split_name
        all_training.extend(training)
        params[f"{split_name}_best_fixed"] = compiler.best_fixed
        params[f"{split_name}_train_medians"] = compiler.medians
        for objective in OBJECTIVES:
            selected_policies: Counter[str] = Counter()
            fallback_count = 0
            invalid_selection_count = 0
            failure_configs: list[str] = []
            vals_start = len(validation_rows)
            for cid in sorted(val_ids):
                if cid not in features or cid not in groups:
                    continue
                group = groups[cid]
                selected, meta = compiler.select(features[cid], set(group), objective)
                selected_row = group.get(selected)
                invalid_selected = selected_row is None
                if invalid_selected:
                    invalid_selection_count += 1
                    failure_configs.append(cid)
                    fallback = compiler.best_fixed.get(objective, "reactive_admission")
                    selected = fallback if fallback in group else next(iter(group))
                    selected_row = group[selected]
                    meta["fallback_used"] = True
                    meta["reason"] = f"invalid_selection_blocked:{meta.get('reason', '')}"
                selected_policies[selected] += 1
                fallback_count += int(bool(meta.get("fallback_used")))
                best_fixed = compiler.best_fixed.get(objective, "reactive_admission")
                best_fixed_row = group.get(best_fixed)
                if best_fixed_row is None:
                    best_fixed_row = min(group.values(), key=lambda r: normalized_objective(r, compiler.medians, objective, compiler.best_fixed_norm_p95.get(objective, 1.0)))
                    best_fixed = str(best_fixed_row["policy"])
                oracle_row = min(group.values(), key=lambda r: normalized_objective(r, compiler.medians, objective, compiler.best_fixed_norm_p95.get(objective, 1.0)))
                p95_oracle = min(group.values(), key=lambda r: _f(r, "p95_jct", float("inf")))
                reactive = group.get("reactive_admission")
                gain_best = _safe_gain(_f(best_fixed_row, "p95_jct"), _f(selected_row, "p95_jct"))
                if gain_best < 0:
                    failure_configs.append(cid)
                validation_rows.append(
                    {
                        "config_id": cid,
                        "split_type": split_name,
                        "objective": objective,
                        "selected_policy": selected,
                        "best_fixed_policy": best_fixed,
                        "oracle_policy": oracle_row["policy"],
                        "p95_oracle_policy": p95_oracle["policy"],
                        "selected_p95": _f(selected_row, "p95_jct"),
                        "selected_ready_wait": _f(selected_row, "ready_queue_wait"),
                        "best_fixed_p95": _f(best_fixed_row, "p95_jct"),
                        "best_fixed_ready_wait": _f(best_fixed_row, "ready_queue_wait"),
                        "oracle_p95": _f(p95_oracle, "p95_jct"),
                        "selected_mean_jct": _f(selected_row, "mean_jct"),
                        "best_fixed_mean_jct": _f(best_fixed_row, "mean_jct"),
                        "selected_throughput": _f(selected_row, "throughput"),
                        "best_fixed_throughput": _f(best_fixed_row, "throughput"),
                        "reactive_p95": _f(reactive, "p95_jct") if reactive else 0.0,
                        "acd_nisp_p95": _f(group.get("acd_nisp"), "p95_jct"),
                        "gain_over_best_fixed_p95": gain_best,
                        "throughput_gain_over_best_fixed": _safe_gain(_f(best_fixed_row, "throughput"), _f(selected_row, "throughput"), lower_better=False),
                        "gain_over_reactive_p95": _safe_gain(_f(reactive, "p95_jct"), _f(selected_row, "p95_jct")) if reactive else 0.0,
                        "regret_to_oracle_p95": max(0.0, (_f(selected_row, "p95_jct") - _f(p95_oracle, "p95_jct")) / max(EPS, _f(p95_oracle, "p95_jct"))),
                        "selected_policy_invalid_risk": meta.get("invalidity_risk", 0.0),
                        "confidence": meta.get("confidence", 0.0),
                        "fallback_used": str(bool(meta.get("fallback_used"))).lower(),
                        "invalid_selection": str(invalid_selected).lower(),
                        "selection_reason": meta.get("reason", ""),
                        "score_breakdown": meta.get("score_breakdown", "{}"),
                    }
                )
            vals = validation_rows[vals_start:]
            objective_summary.append(
                {
                    "split_type": split_name,
                    "objective": objective,
                    "best_fixed_policy": compiler.best_fixed.get(objective, ""),
                    "selected_policy_distribution": json.dumps(dict(selected_policies), sort_keys=True),
                    "validation_rows": len(vals),
                    "invalid_selection_rate": invalid_selection_count / max(1, len(vals)),
                    "fallback_rate": fallback_count / max(1, len(vals)),
                    "gain_over_best_fixed_p95": sum(_f(r, "gain_over_best_fixed_p95") for r in vals) / max(1, len(vals)),
                    "throughput_gain_over_best_fixed": sum(_f(r, "throughput_gain_over_best_fixed") for r in vals) / max(1, len(vals)),
                    "gain_over_reactive_p95": sum(_f(r, "gain_over_reactive_p95") for r in vals) / max(1, len(vals)),
                    "regret_to_oracle_p95": sum(_f(r, "regret_to_oracle_p95") for r in vals) / max(1, len(vals)),
                    "worst_case_regret": max([_f(r, "regret_to_oracle_p95") for r in vals] or [0.0]),
                    "failure_config_count": len(set(failure_configs)),
                    "failure_configs": ";".join(sorted(set(failure_configs))[:50]),
                }
            )
    write_csv(training_out, all_training)
    write_csv(validation_out, validation_rows)
    write_csv(objectives_out, objective_summary)
    write_json(params_out, params)
    persisted = _read_csv(validation_out)
    if not persisted:
        raise RuntimeError(f"compiler validation CSV is empty after write: {validation_out}")
    if integrity_out:
        lines = [
            "# TAPS Compiler v3 Validation Integrity",
            "",
            f"VALIDATION_CSV = {validation_out}",
            f"ROW_COUNT = {len(persisted)}",
            f"SHA256 = {_file_sha256(validation_out)}",
            f"OBJECTIVES_CSV = {objectives_out}",
            f"OBJECTIVES_SHA256 = {_file_sha256(objectives_out)}",
        ]
        Path(integrity_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_training, validation_rows, summarize_validation(validation_rows)


def summarize_validation(rows: list[dict[str, Any]], split_type: str | None = None, objective: str = "balanced") -> dict[str, float]:
    vals = [r for r in rows if (split_type is None or r.get("split_type") == split_type) and r.get("objective") == objective]
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
        "fallback_rate": sum(1 for r in vals if str(r.get("fallback_used", "")).lower() == "true") / max(1, len(vals)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-grid", default="data/results/aligned_policy_grid_valid_pr4_v10.csv")
    ap.add_argument("--audit", default="data/results/aligned_policy_grid_audit_pr4_v10.csv")
    ap.add_argument("--features", default="data/results/workload_features_pr4_v10.csv")
    ap.add_argument("--training-out", default="data/results/taps_cost_model_v3_training_pr4_v10.csv")
    ap.add_argument("--validation-out", default="data/results/taps_compiler_v3_validation_pr4_v10.csv")
    ap.add_argument("--params-out", default="data/results/taps_compiler_v3_params_pr4_v10.json")
    ap.add_argument("--objectives-out", default="data/results/taps_compiler_v3_objectives_pr4_v10.csv")
    ap.add_argument("--integrity-out")
    args = ap.parse_args()
    training, validation, summary = evaluate_compiler(
        args.valid_grid,
        args.audit,
        args.features,
        args.training_out,
        args.validation_out,
        args.params_out,
        args.objectives_out,
        args.integrity_out,
    )
    print(json.dumps({"training_rows": len(training), "validation_rows": len(validation), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
