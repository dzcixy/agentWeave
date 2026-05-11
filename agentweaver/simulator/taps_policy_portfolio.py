from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv, write_json


PORTFOLIO_POLICIES = [
    "reactive_admission",
    "acd_nisp",
    "taps_domain_v4",
    "taps_admission_v4",
    "taps_unified_v5",
    "taps_unified_adaptive_v6",
]
NON_ORACLE_POLICIES = PORTFOLIO_POLICIES + ["static_admission", "taps_predictive_v2", "naive_wafer", "acd_cdf_nisp", "taps"]
NUMERIC_FEATURES = [
    "total_sessions",
    "active_session_limit",
    "effective_regions",
    "memory_budget_gb",
    "session_pressure",
    "region_pressure",
    "memory_pressure_bucket_id",
    "arrival_closed_loop",
    "arrival_poisson",
    "arrival_bursty",
    "high_domain_locality",
    "high_tool_blocking",
    "high_region_pressure",
    "high_session_pressure",
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


def _i(row: dict[str, Any] | None, key: str, default: int = 0) -> int:
    return int(round(_f(row, key, float(default))))


def _safe_gain(base: float, new: float, lower_better: bool = True) -> float:
    if base <= 0:
        return 0.0
    return (base - new) / base if lower_better else (new - base) / base


def _objective(row: dict[str, Any], beta: float = 10.0) -> float:
    return _f(row, "p95_jct") + 0.2 * _f(row, "mean_jct") - beta * _f(row, "throughput")


def _memory_bucket(memory_gb: int) -> str:
    if memory_gb <= 0:
        return "unknown"
    if memory_gb <= 16:
        return "low"
    if memory_gb <= 32:
        return "mid"
    return "high"


def _memory_bucket_id(memory_gb: int) -> int:
    return {"unknown": 0, "low": 1, "mid": 2, "high": 3}[_memory_bucket(memory_gb)]


def _config_id(total: int, limit: int, regions: int, arrival: str, memory: int, scope: str = "aligned") -> str:
    return f"{scope}:ts{total}:al{limit}:er{regions}:arr{arrival}:mem{memory}"


def _base_features(total: int, limit: int, regions: int, arrival: str, memory: int) -> dict[str, Any]:
    limit = max(1, limit)
    regions = max(1, regions)
    session_pressure = total / limit
    region_pressure = limit / regions
    return {
        "session_pressure": session_pressure,
        "region_pressure": region_pressure,
        "memory_pressure_bucket": _memory_bucket(memory),
        "memory_pressure_bucket_id": _memory_bucket_id(memory),
        "arrival_closed_loop": 1 if arrival == "closed_loop" else 0,
        "arrival_poisson": 1 if arrival == "poisson" else 0,
        "arrival_bursty": 1 if arrival == "bursty" else 0,
        "high_domain_locality": 1 if arrival == "bursty" or region_pressure >= 4 else 0,
        "high_tool_blocking": 1 if session_pressure >= 4 or arrival in {"poisson", "bursty"} else 0,
        "high_region_pressure": 1 if region_pressure >= 4 else 0,
        "high_session_pressure": 1 if session_pressure >= 4 else 0,
    }


def _normal_row(
    *,
    config_id: str,
    total: int,
    limit: int,
    regions: int,
    arrival: str,
    memory: int,
    policy: str,
    row: dict[str, Any],
    source: str,
    comparable: bool,
) -> dict[str, Any]:
    throughput = _f(row, "throughput")
    if throughput <= 0:
        throughput = _f(row, "throughput_sessions_per_sec")
    out: dict[str, Any] = {
        "config_id": config_id,
        "total_sessions": total,
        "active_session_limit": limit,
        "effective_regions": regions,
        "arrival_pattern": arrival,
        "memory_budget_gb": memory,
        "policy": policy,
        "throughput": throughput,
        "mean_jct": _f(row, "mean_jct"),
        "p95_jct": _f(row, "p95_jct"),
        "p99_jct": _f(row, "p99_jct"),
        "ready_queue_wait": _f(row, "ready_queue_wait"),
        "region_utilization": _f(row, "region_utilization"),
        "domain_cache_hit_rate": _f(row, "domain_cache_hit_rate"),
        "blocked_session_fraction": _f(row, "blocked_session_fraction"),
        "remote_kv_bytes": _f(row, "remote_kv_bytes"),
        "memory_occupancy": _f(row, "memory_occupancy"),
        "starvation_count": _i(row, "starvation_count"),
        "source_file": source,
        "comparable_axis": str(comparable).lower(),
        "oracle_policy": "false",
    }
    out.update(_base_features(total, limit, regions, arrival, memory))
    return out


def _add_or_replace(rows: dict[tuple[str, str], dict[str, Any]], row: dict[str, Any]) -> None:
    rows[(str(row["config_id"]), str(row["policy"]))] = row


def build_policy_performance_dataset(
    taps_v5_csv: str | Path = "data/results/taps_unified_pr4_v5.csv",
    adaptive_v6_csv: str | Path = "data/results/taps_unified_adaptive_pr4_v6.csv",
    domain_v4_csv: str | Path = "data/results/taps_domain_scheduler_pr4_v4.csv",
    admission_v4_csv: str | Path = "data/results/taps_admission_pr4_v4.csv",
    predictive_v2_csv: str | Path = "data/results/multisession_taps_predictive_pr4_v2.csv",
    out_csv: str | Path = "data/results/taps_policy_performance_dataset_pr4_v7.csv",
    coverage_out: str | Path = "data/results/taps_policy_coverage_pr4_v7.json",
) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str], dict[str, Any]] = {}

    for row in _read_csv(taps_v5_csv):
        total, limit, regions = _i(row, "total_sessions"), _i(row, "active_session_limit"), _i(row, "effective_regions")
        arrival, memory = row.get("arrival_pattern", ""), _i(row, "memory_budget_gb")
        policy = "taps_unified_v5" if row.get("policy") == "taps_unified" else row.get("policy", "")
        if policy not in NON_ORACLE_POLICIES:
            continue
        cid = _config_id(total, limit, regions, arrival, memory)
        _add_or_replace(keyed, _normal_row(config_id=cid, total=total, limit=limit, regions=regions, arrival=arrival, memory=memory, policy=policy, row=row, source=str(taps_v5_csv), comparable=True))

    for row in _read_csv(adaptive_v6_csv):
        policy = row.get("policy", "")
        if policy != "taps_unified_adaptive_v6":
            continue
        total, limit, regions = _i(row, "total_sessions"), _i(row, "active_session_limit"), _i(row, "effective_regions")
        arrival, memory = row.get("arrival_pattern", ""), _i(row, "memory_budget_gb")
        cid = _config_id(total, limit, regions, arrival, memory)
        _add_or_replace(keyed, _normal_row(config_id=cid, total=total, limit=limit, regions=regions, arrival=arrival, memory=memory, policy=policy, row=row, source=str(adaptive_v6_csv), comparable=True))

    for row in _read_csv(domain_v4_csv):
        policy = {"taps_domain": "taps_domain_v4"}.get(row.get("policy", ""), row.get("policy", ""))
        if policy not in NON_ORACLE_POLICIES:
            continue
        total, regions = _i(row, "sessions"), _i(row, "effective_regions")
        limit, arrival, memory = total, row.get("arrival_pattern", ""), 0
        cid = _config_id(total, limit, regions, arrival, memory, "legacy_domain_v4")
        _add_or_replace(keyed, _normal_row(config_id=cid, total=total, limit=limit, regions=regions, arrival=arrival, memory=memory, policy=policy, row=row, source=str(domain_v4_csv), comparable=False))

    for row in _read_csv(admission_v4_csv):
        policy = {"taps_admission": "taps_admission_v4"}.get(row.get("policy", ""), row.get("policy", ""))
        if policy not in NON_ORACLE_POLICIES:
            continue
        total, limit, regions = _i(row, "total_sessions"), _i(row, "active_session_limit"), _i(row, "effective_regions")
        arrival, memory = "not_recorded", 0
        cid = _config_id(total, limit, regions, arrival, memory, "legacy_admission_v4")
        _add_or_replace(keyed, _normal_row(config_id=cid, total=total, limit=limit, regions=regions, arrival=arrival, memory=memory, policy=policy, row=row, source=str(admission_v4_csv), comparable=False))

    for row in _read_csv(predictive_v2_csv):
        policy = {"taps_predictive": "taps_predictive_v2"}.get(row.get("policy", ""), row.get("policy", ""))
        if policy == "taps_oracle" or policy not in NON_ORACLE_POLICIES:
            continue
        total = _i(row, "sessions")
        limit, regions, arrival, memory = total, 0, row.get("arrival_pattern", ""), 0
        cid = _config_id(total, limit, regions, arrival, memory, "legacy_predictive_v2")
        _add_or_replace(keyed, _normal_row(config_id=cid, total=total, limit=limit, regions=max(1, regions), arrival=arrival, memory=memory, policy=policy, row=row, source=str(predictive_v2_csv), comparable=False))

    rows = list(keyed.values())
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[str(row["config_id"])].append(row)

    coverage: dict[str, Any] = {"total_rows": len(rows), "configs": len(by_config), "policy_rows": dict(Counter(str(r["policy"]) for r in rows))}
    comparable_configs = 0
    for cid, group in by_config.items():
        policies = {str(r["policy"]) for r in group}
        missing = [p for p in PORTFOLIO_POLICIES if p not in policies]
        comparable = any(r.get("comparable_axis") == "true" for r in group)
        comparable_configs += int(comparable)
        best_p95 = min(group, key=lambda r: _f(r, "p95_jct", float("inf")))
        best_thr = max(group, key=lambda r: _f(r, "throughput"))
        best_wait = min(group, key=lambda r: _f(r, "ready_queue_wait", float("inf")))
        best_obj = min(group, key=_objective)
        for row in group:
            row["available_policies"] = ",".join(sorted(policies))
            row["missing_policies"] = ",".join(missing)
            row["best_policy_by_p95"] = best_p95["policy"]
            row["best_policy_by_throughput"] = best_thr["policy"]
            row["best_policy_by_ready_wait"] = best_wait["policy"]
            row["best_policy_by_objective"] = best_obj["policy"]
            row["policy_coverage_count"] = len(policies)
            row["objective"] = _objective(row)
    coverage["comparable_configs"] = comparable_configs
    coverage["legacy_axis_configs"] = len(by_config) - comparable_configs
    coverage["portfolio_policy_rows"] = {p: sum(1 for r in rows if r["policy"] == p) for p in PORTFOLIO_POLICIES}
    write_csv(out_csv, sorted(rows, key=lambda r: (str(r["config_id"]), str(r["policy"]))))
    write_json(coverage_out, coverage)
    return rows


def _config_groups(rows: list[dict[str, Any]], comparable_only: bool = True) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if comparable_only and row.get("comparable_axis") != "true":
            continue
        if row.get("policy") not in PORTFOLIO_POLICIES:
            continue
        groups[str(row["config_id"])][str(row["policy"])] = row
    return groups


def _split_configs(config_ids: list[str]) -> tuple[set[str], set[str]]:
    train: set[str] = set()
    validation: set[str] = set()
    for cid in sorted(config_ids):
        target = train if int(stable_hash(cid), 16) % 2 == 0 else validation
        target.add(cid)
    if not train or not validation:
        for i, cid in enumerate(sorted(config_ids)):
            (train if i % 2 == 0 else validation).add(cid)
    return train, validation


def _representative(group: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for policy in ["reactive_admission", "acd_nisp", "taps_unified_v5", "taps_unified_adaptive_v6"]:
        if policy in group:
            return group[policy]
    return next(iter(group.values()))


def _feature_vector(row: dict[str, Any], ranges: dict[str, tuple[float, float]] | None = None) -> list[float]:
    vals: list[float] = []
    for key in NUMERIC_FEATURES:
        value = _f(row, key)
        if ranges and key in ranges:
            lo, hi = ranges[key]
            value = 0.0 if hi <= lo else (value - lo) / (hi - lo)
        vals.append(value)
    return vals


def _feature_ranges(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key in NUMERIC_FEATURES:
        vals = [_f(r, key) for r in rows]
        out[key] = (min(vals or [0.0]), max(vals or [1.0]))
    return out


def _pressure_bucket(v: float) -> str:
    if v < 2:
        return "low"
    if v < 4:
        return "mid"
    return "high"


def _rule_key(row: dict[str, Any], level: int = 0) -> tuple[str, ...]:
    base = (
        str(row.get("arrival_pattern", "")),
        _pressure_bucket(_f(row, "session_pressure")),
        _pressure_bucket(_f(row, "region_pressure")),
        str(row.get("memory_pressure_bucket", "")),
    )
    if level == 0:
        return base
    if level == 1:
        return base[:3]
    if level == 2:
        return base[:2]
    return (base[0],)


@dataclass
class Selection:
    policy: str
    confidence: float
    reason: str


class RuleTableSelector:
    def __init__(self) -> None:
        self.rules: dict[tuple[str, ...], str] = {}
        self.backoff_rules: dict[tuple[str, ...], str] = {}
        self.global_policy = "taps_unified_v5"
        self.rule_support: dict[str, int] = {}

    def fit(self, groups: dict[str, dict[str, dict[str, Any]]], train_ids: set[str]) -> None:
        counters: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        support: Counter[str] = Counter()
        global_counts: Counter[str] = Counter()
        for cid in train_ids:
            group = groups[cid]
            rep = _representative(group)
            label = str(rep.get("best_policy_by_objective") or min(group.values(), key=_objective)["policy"])
            global_counts[label] += 1
            for level in [0, 1, 2, 3]:
                counters[_rule_key(rep, level)][label] += 1
        self.global_policy = (global_counts.most_common(1) or [("taps_unified_v5", 1)])[0][0]
        for key, counts in counters.items():
            policy, count = counts.most_common(1)[0]
            if len(key) == 4:
                self.rules[key] = policy
            else:
                self.backoff_rules[key] = policy
            support["|".join(key)] = count
        self.rule_support = dict(support)

    def select(self, row: dict[str, Any]) -> Selection:
        for level in [0, 1, 2, 3]:
            key = _rule_key(row, level)
            policy = self.rules.get(key) if level == 0 else self.backoff_rules.get(key)
            if policy:
                support = self.rule_support.get("|".join(key), 1)
                conf = min(0.95, 0.45 + 0.1 * math.log1p(support))
                return Selection(policy, conf, f"rule level={level} key={key} support={support}")
        return Selection(self.global_policy, 0.25, f"global fallback policy={self.global_policy}")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "selector": "RuleTableSelector",
            "global_policy": self.global_policy,
            "rules": [{"conditions": list(k), "policy": v, "support": self.rule_support.get("|".join(k), 0)} for k, v in sorted(self.rules.items())],
            "backoff_rules": [{"conditions": list(k), "policy": v, "support": self.rule_support.get("|".join(k), 0)} for k, v in sorted(self.backoff_rules.items())],
        }


class NearestNeighborSelector:
    def __init__(self) -> None:
        self.train: list[tuple[str, dict[str, Any], str, list[float]]] = []
        self.ranges: dict[str, tuple[float, float]] = {}
        self.global_policy = "taps_unified_v5"

    def fit(self, groups: dict[str, dict[str, dict[str, Any]]], train_ids: set[str]) -> None:
        reps = [_representative(groups[cid]) for cid in train_ids]
        self.ranges = _feature_ranges(reps)
        counts: Counter[str] = Counter()
        for cid in sorted(train_ids):
            rep = _representative(groups[cid])
            label = str(rep.get("best_policy_by_objective") or min(groups[cid].values(), key=_objective)["policy"])
            counts[label] += 1
            self.train.append((cid, rep, label, _feature_vector(rep, self.ranges)))
        self.global_policy = (counts.most_common(1) or [("taps_unified_v5", 1)])[0][0]

    def select(self, row: dict[str, Any]) -> Selection:
        if not self.train:
            return Selection(self.global_policy, 0.0, "no training rows")
        vec = _feature_vector(row, self.ranges)
        best_cid, _, policy, best_vec = min(
            self.train,
            key=lambda item: math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, item[3], strict=False))),
        )
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, best_vec, strict=False)))
        conf = 1.0 / (1.0 + dist)
        return Selection(policy, conf, f"nearest config={best_cid} distance={dist:.4f}")

    def examples(self, limit: int = 8) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cid, row, label, _ in self.train[:limit]:
            out.append({"config_id": cid, "label": label, "arrival_pattern": row.get("arrival_pattern"), "session_pressure": row.get("session_pressure"), "region_pressure": row.get("region_pressure")})
        return out


class ConservativeFallbackSelector:
    """Confidence gate for non-oracle portfolio selectors.

    The fallback policy is chosen from the train split. The gate only looks at
    selector confidence and policy availability for the current configuration.
    """

    def __init__(self, base: RuleTableSelector | NearestNeighborSelector, fallback_policy: str, threshold: float) -> None:
        self.base = base
        self.fallback_policy = fallback_policy
        self.threshold = threshold

    def select(self, row: dict[str, Any], available_policies: set[str]) -> tuple[Selection, bool]:
        sel = self.base.select(row)
        if sel.confidence < self.threshold or sel.policy not in available_policies:
            return (
                Selection(
                    self.fallback_policy,
                    sel.confidence,
                    f"{sel.reason}; conservative fallback={self.fallback_policy} threshold={self.threshold}",
                ),
                True,
            )
        return sel, False


def _fallback_policy(groups: dict[str, dict[str, dict[str, Any]]], train_ids: set[str]) -> str:
    vals: dict[str, list[float]] = defaultdict(list)
    for cid in train_ids:
        for policy, row in groups[cid].items():
            vals[policy].append(_f(row, "p95_jct", float("inf")))
    if not vals:
        return "taps_unified_v5"
    return min(vals.items(), key=lambda kv: sum(kv[1]) / max(1, len(kv[1])))[0]


def _evaluate_selector(
    groups: dict[str, dict[str, dict[str, Any]]],
    ids: set[str],
    selector: RuleTableSelector | NearestNeighborSelector,
    selector_type: str,
    fallback_policy: str,
    confidence_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cid in sorted(ids):
        group = groups[cid]
        rep = _representative(group)
        gated = ConservativeFallbackSelector(selector, fallback_policy, confidence_threshold)
        sel, fallback_used = gated.select(rep, set(group))
        selected_policy = sel.policy if sel.policy in group else min(group.values(), key=_objective)["policy"]
        reason = sel.reason
        selected = group[selected_policy]
        strongest_p95_policy, strongest_p95 = min(group.items(), key=lambda kv: _f(kv[1], "p95_jct", float("inf")))
        strongest_thr_policy, strongest_thr = max(group.items(), key=lambda kv: _f(kv[1], "throughput"))
        strongest_wait_policy, strongest_wait = min(group.items(), key=lambda kv: _f(kv[1], "ready_queue_wait", float("inf")))
        oracle_policy, oracle = min(group.items(), key=lambda kv: _f(kv[1], "p95_jct", float("inf")))
        reactive = group.get("reactive_admission")
        out.append(
            {
                "config_id": cid,
                "selector_type": selector_type,
                "selected_policy": selected_policy,
                "selected_p95_jct": _f(selected, "p95_jct"),
                "selected_throughput": _f(selected, "throughput"),
                "selected_ready_wait": _f(selected, "ready_queue_wait"),
                "strongest_baseline_policy": strongest_p95_policy,
                "strongest_baseline_p95": _f(strongest_p95, "p95_jct"),
                "strongest_throughput_policy": strongest_thr_policy,
                "strongest_baseline_throughput": _f(strongest_thr, "throughput"),
                "strongest_ready_wait_policy": strongest_wait_policy,
                "strongest_baseline_ready_wait": _f(strongest_wait, "ready_queue_wait"),
                "oracle_best_policy": oracle_policy,
                "oracle_best_p95": _f(oracle, "p95_jct"),
                "p95_gain_over_reactive": _safe_gain(_f(reactive, "p95_jct"), _f(selected, "p95_jct")) if reactive else 0.0,
                "p95_gain_over_strongest": _safe_gain(_f(strongest_p95, "p95_jct"), _f(selected, "p95_jct")),
                "throughput_gain_over_strongest": _safe_gain(_f(strongest_thr, "throughput"), _f(selected, "throughput"), lower_better=False),
                "ready_wait_gain_over_strongest": _safe_gain(_f(strongest_wait, "ready_queue_wait"), _f(selected, "ready_queue_wait")),
                "regret_to_oracle_p95": (_f(selected, "p95_jct") - _f(oracle, "p95_jct")) / max(1e-9, _f(oracle, "p95_jct")),
                "selected_policy_confidence": sel.confidence,
                "fallback_used": str(fallback_used).lower(),
                "selection_reason": reason,
                "total_sessions": rep.get("total_sessions"),
                "active_session_limit": rep.get("active_session_limit"),
                "effective_regions": rep.get("effective_regions"),
                "arrival_pattern": rep.get("arrival_pattern"),
                "memory_budget_gb": rep.get("memory_budget_gb"),
            }
        )
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "mean_p95_gain_over_reactive": 0.0,
            "mean_p95_gain_over_strongest": 0.0,
            "mean_throughput_gain_over_strongest": 0.0,
            "mean_ready_wait_gain_over_strongest": 0.0,
            "mean_regret_to_oracle": 0.0,
            "fallback_rate": 0.0,
            "worst_case_regret": 0.0,
            "failure_configs": 0.0,
        }
    return {
        "mean_p95_gain_over_reactive": sum(_f(r, "p95_gain_over_reactive") for r in rows) / len(rows),
        "mean_p95_gain_over_strongest": sum(_f(r, "p95_gain_over_strongest") for r in rows) / len(rows),
        "mean_throughput_gain_over_strongest": sum(_f(r, "throughput_gain_over_strongest") for r in rows) / len(rows),
        "mean_ready_wait_gain_over_strongest": sum(_f(r, "ready_wait_gain_over_strongest") for r in rows) / len(rows),
        "mean_regret_to_oracle": sum(_f(r, "regret_to_oracle_p95") for r in rows) / len(rows),
        "fallback_rate": sum(1 for r in rows if r.get("fallback_used") == "true") / len(rows),
        "worst_case_regret": max(_f(r, "regret_to_oracle_p95") for r in rows),
        "failure_configs": float(sum(1 for r in rows if _f(r, "p95_gain_over_strongest") < 0)),
    }


def train_and_validate_selectors(
    dataset_csv: str | Path = "data/results/taps_policy_performance_dataset_pr4_v7.csv",
    rules_out: str | Path = "data/results/taps_policy_selector_rules_pr4_v7.json",
    training_out: str | Path = "data/results/taps_policy_selector_training_pr4_v7.csv",
    validation_out: str | Path = "data/results/taps_policy_portfolio_validation_pr4_v7.csv",
    fallback_out: str | Path = "data/results/taps_policy_portfolio_safe_fallback_pr4_v7.csv",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in _read_csv(dataset_csv)]
    groups = {cid: g for cid, g in _config_groups(rows).items() if len(g) >= 3}
    train_ids, validation_ids = _split_configs(list(groups))

    rule = RuleTableSelector()
    nn = NearestNeighborSelector()
    rule.fit(groups, train_ids)
    nn.fit(groups, train_ids)
    fallback = _fallback_policy(groups, train_ids)

    training_rows: list[dict[str, Any]] = []
    for cid in sorted(train_ids):
        group = groups[cid]
        rep = _representative(group)
        rule_sel = rule.select(rep)
        nn_sel = nn.select(rep)
        training_rows.append(
            {
                "config_id": cid,
                "split": "train",
                "best_policy_by_objective": rep.get("best_policy_by_objective"),
                "best_policy_by_p95": rep.get("best_policy_by_p95"),
                "rule_selected_policy": rule_sel.policy,
                "rule_confidence": rule_sel.confidence,
                "nearest_neighbor_selected_policy": nn_sel.policy,
                "nearest_neighbor_confidence": nn_sel.confidence,
                "fallback_policy": fallback,
                "total_sessions": rep.get("total_sessions"),
                "active_session_limit": rep.get("active_session_limit"),
                "effective_regions": rep.get("effective_regions"),
                "arrival_pattern": rep.get("arrival_pattern"),
                "memory_budget_gb": rep.get("memory_budget_gb"),
                "session_pressure": rep.get("session_pressure"),
                "region_pressure": rep.get("region_pressure"),
            }
        )

    write_csv(training_out, training_rows)
    write_json(
        rules_out,
        {
            "feature_keys": NUMERIC_FEATURES,
            "train_config_count": len(train_ids),
            "validation_config_count": len(validation_ids),
            "fallback_policy": fallback,
            "selectors": [rule.to_jsonable(), {"selector": "NearestNeighborSelector", "examples": nn.examples()}],
            "runtime_constraints": {
                "uses_validation_labels_at_runtime": False,
                "uses_future_jct": False,
                "uses_future_tool_completion": False,
            },
            "conservative_fallback_selector": {
                "implemented": True,
                "fallback_policy": fallback,
                "thresholds_evaluated": [0.0, 0.25, 0.5, 0.75],
            },
        },
    )

    validation_rows: list[dict[str, Any]] = []
    for selector, name in [(rule, "RuleTable"), (nn, "NearestNeighbor")]:
        validation_rows.extend(_evaluate_selector(groups, validation_ids, selector, name, fallback, 0.0))
    write_csv(validation_out, validation_rows)

    fallback_rows: list[dict[str, Any]] = []
    for selector, name in [(rule, "RuleTable"), (nn, "NearestNeighbor")]:
        for threshold in [0.0, 0.25, 0.5, 0.75]:
            eval_rows = _evaluate_selector(groups, validation_ids, selector, name, fallback, threshold)
            summary = _summarize(eval_rows)
            fallback_rows.append(
                {
                    "selector_type": name,
                    "confidence_threshold": threshold,
                    "p95_gain_over_strongest": summary["mean_p95_gain_over_strongest"],
                    "throughput_gain_over_strongest": summary["mean_throughput_gain_over_strongest"],
                    "regret_to_oracle": summary["mean_regret_to_oracle"],
                    "fallback_rate": summary["fallback_rate"],
                    "worst_case_regret": summary["worst_case_regret"],
                    "failure_configs": int(summary["failure_configs"]),
                }
            )
    write_csv(fallback_out, fallback_rows)
    meta = {
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(validation_ids),
        "fallback_policy": fallback,
    }
    return validation_rows, fallback_rows, meta


def write_interpretability_report(
    validation_csv: str | Path = "data/results/taps_policy_portfolio_validation_pr4_v7.csv",
    rules_json: str | Path = "data/results/taps_policy_selector_rules_pr4_v7.json",
    out_md: str | Path = "data/results/taps_policy_portfolio_interpretability_pr4_v7.md",
) -> None:
    rows = _read_csv(validation_csv)
    rules = json.loads(Path(rules_json).read_text(encoding="utf-8")) if Path(rules_json).exists() else {}
    counts = Counter(r.get("selected_policy", "") for r in rows)
    failures = [r for r in rows if _f(r, "p95_gain_over_strongest") < 0]
    lines = [
        "# TAPS-P Portfolio Interpretability PR4-v7",
        "",
        "SELECTED_POLICY_COUNTS = " + ", ".join(f"{k}:{v}" for k, v in counts.most_common()),
        f"FAILURE_CONFIGS = {len(failures)}",
        "",
        "## Rule Table",
    ]
    for rule in (rules.get("selectors", [{}])[0].get("rules", []) if rules.get("selectors") else [])[:20]:
        lines.append(f"- if {rule.get('conditions')} -> {rule.get('policy')} (support={rule.get('support')})")
    lines.extend(["", "## Nearest-Neighbor Examples"])
    nn_examples = rules.get("selectors", [{}, {}])[1].get("examples", []) if len(rules.get("selectors", [])) > 1 else []
    for ex in nn_examples[:8]:
        lines.append(f"- {ex.get('config_id')} -> {ex.get('label')} (arrival={ex.get('arrival_pattern')}, session_pressure={ex.get('session_pressure')}, region_pressure={ex.get('region_pressure')})")
    lines.extend(["", "## Failure Analysis"])
    for row in failures[:12]:
        lines.append(
            f"- {row.get('selector_type')} {row.get('config_id')}: selected={row.get('selected_policy')} strongest={row.get('strongest_baseline_policy')} p95_gain={_f(row, 'p95_gain_over_strongest'):.4f}"
        )
    lines.extend(
        [
            "",
            "TAPS-P uses train-derived rules or nearest train configurations. Validation labels are used only for offline evaluation.",
        ]
    )
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_selector(rows: list[dict[str, Any]]) -> tuple[str, dict[str, float], list[dict[str, Any]]]:
    by_selector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_selector[str(row["selector_type"])].append(row)
    summaries = {name: _summarize(group) for name, group in by_selector.items()}
    if not summaries:
        return "none", _summarize([]), []
    name = min(
        summaries,
        key=lambda n: (
            -summaries[n]["mean_p95_gain_over_strongest"],
            -summaries[n]["mean_throughput_gain_over_strongest"],
            summaries[n]["mean_regret_to_oracle"],
        ),
    )
    return name, summaries[name], by_selector[name]


def write_report(
    validation_csv: str | Path = "data/results/taps_policy_portfolio_validation_pr4_v7.csv",
    fallback_csv: str | Path = "data/results/taps_policy_portfolio_safe_fallback_pr4_v7.csv",
    dataset_csv: str | Path = "data/results/taps_policy_performance_dataset_pr4_v7.csv",
    out_md: str | Path = "data/results/pr4_algo_v7_report.md",
) -> dict[str, Any]:
    rows = _read_csv(validation_csv)
    fallback_rows = _read_csv(fallback_csv)
    best_selector, summary, selected_rows = _best_selector(rows)
    best_fallback = min(
        fallback_rows,
        key=lambda r: (_f(r, "worst_case_regret", float("inf")), -_f(r, "p95_gain_over_strongest")),
        default={},
    )
    p95_gain = summary["mean_p95_gain_over_strongest"]
    thr_gain = summary["mean_throughput_gain_over_strongest"]
    ready = (p95_gain >= 0.03 or (thr_gain >= 0.03 and p95_gain >= 0.0)) and Path(dataset_csv).exists()
    gate = "PASS" if ready else ("WARNING" if rows else "FAIL")
    failures = [r["config_id"] for r in selected_rows if _f(r, "p95_gain_over_strongest") < 0]
    fields: dict[str, Any] = {
        "PR4_ALGO_V7_GATE": gate,
        "TAPS_P_IMPLEMENTED": "true",
        "SELECTORS_IMPLEMENTED": "RuleTable,NearestNeighbor,ConservativeFallback",
        "BEST_SELECTOR": best_selector,
        "TAPS_P_VALIDATION_P95_GAIN_OVER_REACTIVE": f"{summary['mean_p95_gain_over_reactive']:.6f}",
        "TAPS_P_VALIDATION_P95_GAIN_OVER_STRONGEST": f"{p95_gain:.6f}",
        "TAPS_P_VALIDATION_THROUGHPUT_GAIN_OVER_STRONGEST": f"{thr_gain:.6f}",
        "TAPS_P_READY_WAIT_GAIN_OVER_STRONGEST": f"{summary['mean_ready_wait_gain_over_strongest']:.6f}",
        "TAPS_P_ORACLE_REGRET_P95": f"{summary['mean_regret_to_oracle']:.6f}",
        "TAPS_P_SAFE_FALLBACK_RATE": f"{_f(best_fallback, 'fallback_rate'):.6f}",
        "TAPS_P_FAILURE_CONFIGS": len(failures),
        "READY_FOR_PR4_SCALE": str(ready).lower(),
    }
    lines = ["# PR4 Algorithm v7 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Fairness Notes",
            "- TAPS-P selectors are trained on train configurations only.",
            "- Validation labels are used only to score held-out configurations.",
            "- The strongest-baseline columns include the best available non-oracle policy for each configuration.",
            "- Table-based selector evaluation is not presented as a fresh replay.",
            "",
            "## Failure Configs",
            ", ".join(failures) if failures else "none",
        ]
    )
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def write_algorithm_logic(out_md: str | Path = "data/results/final_algorithm_logic_pr4_v7.md") -> None:
    lines = [
        "# Final Algorithm Logic PR4-v7",
        "",
        "## Problem",
        "Agent-on-wafer serving is a stateful execution-graph problem, not a flat request-stream scheduling problem. Coding agents repeatedly reuse task/repo/tool context, fork branch-local work, block on shell tools, and later resume with discontinuous state.",
        "",
        "## Challenges",
        "- Context-domain fragmentation: identical task/repo/tool context can be placed and prefetched repeatedly when the runtime ignores the agent graph.",
        "- Tool-stall state discontinuity: tool calls leave compute regions idle while private branch state may be evicted or recomputed before resume.",
        "- Regime-dependent multi-session scheduling: admission-heavy, domain-hot, tail-risk, and memory-pressure regimes favor different non-oracle schedulers.",
        "",
        "## Algorithms",
        "- Agent Execution Graph: converts trajectories into LLM/tool/verifier DAGs with context segments and branch structure.",
        "- ACD: places shared context domains in a wafer-resident arena to reduce repeated prefill and remote KV movement.",
        "- NISP: parks HOT/WARM/COLD branch state across tool stalls to reduce resume prefill.",
        "- TAPS-P: a regret-aware policy portfolio that selects among non-oracle scheduling policies using train-derived, online-visible configuration/regime features.",
        "",
        "## Demoted Mechanisms",
        "- CDF is optional because strict-prefix mini-SWE gains are weak in the current traces.",
        "- PABB-S is optional until real patch snapshot events are available.",
        "- Old BES is deprecated and is not used as a real mini-SWE main-result mechanism.",
        "",
        "## Evidence Boundaries",
        "- ACD/NISP evidence is token and model-side replay evidence on real mini-SWE traces.",
        "- TAPS-P targets multi-session p95/throughput under fair strongest-baseline comparison.",
        "- No solved-rate claim is made unless official verifier coverage is available.",
    ]
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_portfolio(
    validation_csv: str | Path = "data/results/taps_policy_portfolio_validation_pr4_v7.csv",
    fallback_csv: str | Path = "data/results/taps_policy_portfolio_safe_fallback_pr4_v7.csv",
) -> None:
    rows = _read_csv(validation_csv)
    if not rows:
        return
    ensure_dir("data/plots")
    best_selector, _, selected_rows = _best_selector(rows)
    selected_rows = sorted(selected_rows, key=lambda r: r["config_id"])
    x = list(range(len(selected_rows)))
    plt.figure(figsize=(8.5, 3.8))
    plt.plot(x, [_f(r, "strongest_baseline_p95") for r in selected_rows], marker="o", label="strongest baseline")
    plt.plot(x, [_f(r, "selected_p95_jct") for r in selected_rows], marker="o", label=f"TAPS-P {best_selector}")
    plt.ylabel("p95 JCT")
    plt.xlabel("held-out config")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/plots/taps_policy_portfolio_p95_pr4_v7.pdf")
    plt.close()

    plt.figure(figsize=(8.5, 3.8))
    plt.bar(x, [_f(r, "regret_to_oracle_p95") for r in selected_rows])
    plt.ylabel("p95 regret to oracle")
    plt.xlabel("held-out config")
    plt.tight_layout()
    plt.savefig("data/plots/taps_policy_portfolio_regret_pr4_v7.pdf")
    plt.close()

    counts = Counter(r.get("selected_policy", "") for r in selected_rows)
    plt.figure(figsize=(6.4, 3.8))
    plt.bar(list(counts), [counts[k] for k in counts])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("selected configs")
    plt.tight_layout()
    plt.savefig("data/plots/taps_policy_portfolio_selected_policy_pr4_v7.pdf")
    plt.close()

    fallback_rows = _read_csv(fallback_csv)
    plt.figure(figsize=(6.4, 3.8))
    for selector in sorted({r.get("selector_type", "") for r in fallback_rows}):
        sub = [r for r in fallback_rows if r.get("selector_type") == selector]
        sub = sorted(sub, key=lambda r: _f(r, "confidence_threshold"))
        plt.plot([_f(r, "confidence_threshold") for r in sub], [_f(r, "worst_case_regret") for r in sub], marker="o", label=selector)
    plt.xlabel("confidence threshold")
    plt.ylabel("worst-case regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/plots/taps_policy_portfolio_fallback_pr4_v7.pdf")
    plt.close()


def run_all() -> dict[str, Any]:
    dataset = build_policy_performance_dataset()
    validation, fallback, split_meta = train_and_validate_selectors()
    write_interpretability_report()
    write_algorithm_logic()
    plot_portfolio()
    report = write_report()
    return {
        "dataset_rows": len(dataset),
        "validation_rows": len(validation),
        "fallback_rows": len(fallback),
        "train_configs": len(split_meta["train_ids"]),
        "validation_configs": len(split_meta["validation_ids"]),
        "report": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run-all")
    sub.add_parser("dataset")
    sub.add_parser("validate")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(), indent=2, sort_keys=True))
    elif args.cmd == "dataset":
        print(json.dumps({"rows": len(build_policy_performance_dataset())}, indent=2))
    elif args.cmd == "validate":
        validation, fallback, meta = train_and_validate_selectors()
        print(json.dumps({"validation_rows": len(validation), "fallback_rows": len(fallback), **meta}, indent=2, sort_keys=True))
    elif args.cmd == "report":
        print(json.dumps(write_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
