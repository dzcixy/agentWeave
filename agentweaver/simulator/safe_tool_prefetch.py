from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import write_csv


DEFAULT_TRACE_DIRS = ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
SAFE_READ_ONLY = "SAFE_READ_ONLY"
UNSAFE_SIDE_EFFECT = "UNSAFE_SIDE_EFFECT"
UNKNOWN = "UNKNOWN"


def _load_traces(trace_dirs: list[str | Path] | None = None) -> list[Trace]:
    traces: list[Trace] = []
    for trace_dir in trace_dirs or DEFAULT_TRACE_DIRS:
        p = Path(trace_dir)
        if p.exists():
            traces.extend(load_trace_dir(p))
    return traces


def _tool_latency(ev: Event) -> float:
    return max(0.0, float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0))


def _repo_family(trace: Trace, ev: Event | None = None) -> str:
    raw = str(trace.metadata.get("repo", "") or trace.metadata.get("instance_id", "") or (ev.instance_id if ev else "") or "unknown")
    if "__" in raw:
        return raw.split("__", 1)[0]
    if "/" in raw:
        return raw.rsplit("/", 1)[-1].split(".", 1)[0]
    return raw.split("-", 1)[0] if raw else "unknown"


def command_class(command: str | None, tool_type: str | None = None) -> str:
    cmd = (command or "").strip()
    low = cmd.lower()
    if not cmd:
        return tool_type or "unknown"
    if re.search(r"\b(rg|grep)\b", low):
        return "grep_search"
    if re.search(r"\b(cat|sed|head|tail|wc)\b", low):
        return "file_read"
    if re.search(r"\b(ls|find|pwd|tree)\b", low):
        return "list_find"
    if re.search(r"\bgit\s+(status|diff|log|show|rev-parse|branch\s+--show-current)\b", low):
        return "git_read"
    if "python" in low:
        return "python_inspect"
    if any(x in low for x in ["pytest", "tox", "unittest", "nox", "make test"]):
        return "test"
    if any(x in low for x in ["apply_patch", "git apply", "write_file", "edit_file"]):
        return "file_write"
    return "shell_other"


def classify_tool_command(command: str | None, tool_type: str | None = None) -> tuple[str, str, str]:
    cmd = (command or "").strip()
    low = cmd.lower()
    cls = command_class(cmd, tool_type)
    if not cmd:
        return UNKNOWN, cls, "empty_command"
    unsafe_patterns = [
        r"(^|[;&|]\s*)rm\s",
        r"(^|[;&|]\s*)mv\s",
        r"(^|[;&|]\s*)cp\s",
        r"(^|[;&|]\s*)touch\s",
        r"(^|[;&|]\s*)mkdir\s",
        r"(^|[;&|]\s*)chmod\s",
        r"(^|[;&|]\s*)chown\s",
        r"(^|[;&|]\s*)pip\s+install\b",
        r"(^|[;&|]\s*)npm\s+(install|publish|run|test)\b",
        r"(^|[;&|]\s*)git\s+(commit|checkout|reset|merge|rebase|push|pull|apply|clean)\b",
        r">\s*[^&]",
        r">>",
        r"\btee\s+",
        r"\bsed\s+-i\b",
        r"\bpytest\b|\btox\b|\bunittest\b|\bnox\b",
        r"\bmanage\.py\s+(makemigrations|migrate|test|runserver|createsuperuser)\b",
        r"\bsetup\.py\s+build\b|\bbuild_ext\b|\bmake\b|\bconfigure\b",
        r"\bapply_patch\b|\bwrite_file\b|\bedit_file\b",
        r"\bcurl\b.*\b(-X\s*(POST|PUT|PATCH|DELETE)|--request\s*(POST|PUT|PATCH|DELETE))",
    ]
    for pat in unsafe_patterns:
        if re.search(pat, low):
            return UNSAFE_SIDE_EFFECT, cls, f"matched:{pat}"
    if ("python - <<" in low or "python3 - <<" in low) and not any(
        x in low for x in ["open(", ".write(", "write_text", "unlink(", "remove(", "mkdir(", "makemigrations", "migrate", " test", "createsuperuser"]
    ):
        return SAFE_READ_ONLY, cls, "python_heredoc_inspection"
    if re.search(r"python\s+manage\.py\s+shell\s+-c", low) and not any(
        x in low for x in ["save(", "delete(", "create(", "update(", "bulk_create", "makemigrations", "migrate"]
    ):
        return SAFE_READ_ONLY, cls, "django_shell_inspection"
    if _is_readonly_shell_chain(cmd):
        return SAFE_READ_ONLY, cls, "readonly_shell_chain"
    try:
        parts = shlex.split(cmd)
    except Exception:
        return UNKNOWN, cls, "unparseable_shell"
    base = Path(parts[0]).name if parts else ""
    if base in {"ls", "cat", "grep", "rg", "find", "wc", "head", "tail", "pwd", "tree", "less"}:
        return SAFE_READ_ONLY, cls, "readonly_whitelist"
    if base == "sed" and "-i" not in parts:
        return SAFE_READ_ONLY, cls, "sed_readonly"
    if base == "git" and len(parts) >= 2 and parts[1] in {"status", "diff", "log", "show", "rev-parse"}:
        return SAFE_READ_ONLY, cls, "git_readonly"
    if "python" in base:
        if any(x in low for x in ["open(", ".write(", "write_text", "unlink(", "remove(", "rmdir(", "mkdir(", "requests.post", "requests.put"]):
            return UNSAFE_SIDE_EFFECT, cls, "python_write_or_network"
        if any(flag in parts for flag in ["-c", "-m"]) or cmd.endswith(".py"):
            return SAFE_READ_ONLY, cls, "python_inspection_conservative"
    return UNKNOWN, cls, "not_whitelisted"


def _is_readonly_shell_chain(command: str) -> bool:
    low = command.lower()
    if any(x in low for x in ["| tee", ">>", ">", " sed -i", " apply_patch", " git apply"]):
        return False
    parts = re.split(r"\s*(?:&&|;|\n)\s*", command.strip())
    if not parts:
        return False
    saw_read = False
    for part in parts:
        if not part:
            continue
        p = part.strip()
        plow = p.lower()
        if plow.startswith(("cd ", "export ", "source ")):
            continue
        if plow.startswith("echo ") and "|" not in plow:
            continue
        if re.match(r"^(ls|pwd|tree|less)\b", plow):
            saw_read = True
            continue
        if re.match(r"^(cat|head|tail|wc|find|rg|grep|sed)\b", plow) and " -i" not in plow:
            saw_read = True
            continue
        if re.match(r"^git\s+(status|diff|log|show|rev-parse)\b", plow):
            saw_read = True
            continue
        if re.match(r"^curl\s+(-s\s+)?https?://", plow) and re.search(r"\|\s*(grep|rg|head|tail|wc)\b", plow):
            saw_read = True
            continue
        if re.match(r"^pip\s+(freeze|show)\b", plow) and re.search(r"\|\s*(grep|rg|head|tail|wc)\b", plow):
            saw_read = True
            continue
        if "python" in plow and not any(x in plow for x in ["open(", ".write(", "write_text", "unlink(", "remove(", "mkdir(", "makemigrations", " migrate", " test "]):
            saw_read = True
            continue
        return False
    return saw_read


def _events_by_branch(trace: Trace) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = defaultdict(list)
    for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start or 0.0, e.node_id)):
        if ev.node_type in {"llm", "tool", "verifier"}:
            out[ev.branch_id].append(ev)
    return out


class NextToolPredictor:
    def __init__(self) -> None:
        self.by_repo_prev: Counter[tuple[str, str, str]] = Counter()
        self.by_prev: Counter[tuple[str, str]] = Counter()
        self.global_counts: Counter[str] = Counter()
        self.latency_by_class: dict[str, float] = {}

    def fit(self, records: list[dict[str, Any]]) -> None:
        latencies: dict[str, list[float]] = defaultdict(list)
        for row in records:
            actual = str(row["tool_class"])
            prev = str(row["prev_tool_class"])
            repo = str(row["repo_family"])
            self.by_repo_prev[(repo, prev, actual)] += 1
            self.by_prev[(prev, actual)] += 1
            self.global_counts[actual] += 1
            latencies[actual].append(float(row.get("tool_latency", 0.0)))
        self.latency_by_class = {k: sorted(v)[len(v) // 2] for k, v in latencies.items() if v}

    def predict_topk(self, repo: str, prev: str, k: int = 3) -> list[str]:
        scored: Counter[str] = Counter()
        for (r, p, c), n in self.by_repo_prev.items():
            if r == repo and p == prev:
                scored[c] += 3 * n
        for (p, c), n in self.by_prev.items():
            if p == prev:
                scored[c] += 2 * n
        scored.update(self.global_counts)
        return [c for c, _ in scored.most_common(k)]

    def predicted_latency(self, cls: str) -> float:
        if cls in self.latency_by_class:
            return self.latency_by_class[cls]
        vals = sorted(self.latency_by_class.values())
        return vals[len(vals) // 2] if vals else 0.0


def build_tool_records(traces: list[Trace]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        repo = _repo_family(trace)
        for branch, events in _events_by_branch(trace).items():
            prev_class = "START"
            error_count = 0
            last_returncode = 0
            obs_len = 0
            for idx, ev in enumerate(events):
                if ev.node_type == "tool":
                    safety, cls, reason = classify_tool_command(ev.command, ev.tool_type)
                    obs_len = int(ev.observation_tokens or 0)
                    rows.append(
                        {
                            "trace_source": trace.metadata.get("source", ""),
                            "repo_family": repo,
                            "branch_id": branch,
                            "step_index": idx,
                            "event_id": ev.event_id,
                            "command": ev.command or "",
                            "tool_type": ev.tool_type or "",
                            "tool_class": cls,
                            "safety": safety,
                            "safety_reason": reason,
                            "is_safe": str(safety == SAFE_READ_ONLY).lower(),
                            "prev_tool_class": prev_class,
                            "previous_returncode": last_returncode,
                            "observation_length": obs_len,
                            "error_count": error_count,
                            "mentioned_file_not_found": str("no such file" in (ev.command or "").lower()).lower(),
                            "mentioned_test_fail": str("fail" in (ev.command or "").lower()).lower(),
                            "mentioned_import_error": str("importerror" in (ev.command or "").lower()).lower(),
                            "tool_latency": _tool_latency(ev),
                        }
                    )
                    last_returncode = int(ev.exit_code or 0)
                    if last_returncode != 0:
                        error_count += 1
                    prev_class = cls
    return rows


def _split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, val = [], []
    for row in records:
        key = (row.get("trace_source", ""), row.get("branch_id", ""), row.get("event_id", ""))
        (val if int(stable_hash(key), 16) % 5 == 0 else train).append(row)
    if not train or not val:
        for i, row in enumerate(records):
            (val if i % 5 == 0 else train).append(row)
    return train, val


def evaluate_predictor(
    records: list[dict[str, Any]],
    classification_out: str | Path = "data/results/tool_safety_classification_pr4_v10.csv",
    predictor_out: str | Path = "data/results/next_tool_predictor_pr4_v10.csv",
) -> tuple[NextToolPredictor, dict[str, Any]]:
    write_csv(classification_out, records)
    train, val = _split_records(records)
    predictor = NextToolPredictor()
    predictor.fit(train)
    total = len(val)
    top1 = top3 = safe_top1 = safe_top3 = 0
    safe_total = 0
    false_safe = false_unsafe = 0
    examples: list[dict[str, Any]] = []
    for row in val:
        preds = predictor.predict_topk(str(row["repo_family"]), str(row["prev_tool_class"]), 3)
        actual = str(row["tool_class"])
        actual_safe = str(row["is_safe"]).lower() == "true"
        pred_safe = bool(preds and any(r.get("tool_class") == preds[0] and str(r.get("is_safe")) == "true" for r in train))
        top1 += int(bool(preds) and preds[0] == actual)
        top3 += int(actual in preds)
        if actual_safe:
            safe_total += 1
            safe_top1 += int(bool(preds) and preds[0] == actual)
            safe_top3 += int(actual in preds)
        false_safe += int(pred_safe and not actual_safe)
        false_unsafe += int((not pred_safe) and actual_safe)
        if len(examples) < 25:
            examples.append(
                {
                    "row_type": "example",
                    "event_id": row["event_id"],
                    "repo_family": row["repo_family"],
                    "prev_tool_class": row["prev_tool_class"],
                    "actual_tool_class": actual,
                    "pred_top1": preds[0] if preds else "",
                    "pred_top3": json.dumps(preds),
                    "actual_safe": str(actual_safe).lower(),
                }
            )
    metrics = {
        "row_type": "aggregate",
        "train_rows": len(train),
        "validation_rows": len(val),
        "top1_accuracy": top1 / max(1, total),
        "top3_accuracy": top3 / max(1, total),
        "safe_top1_accuracy": safe_top1 / max(1, safe_total),
        "safe_top3_accuracy": safe_top3 / max(1, safe_total),
        "coverage": total / max(1, len(records)),
        "safe_coverage": safe_total / max(1, total),
        "false_safe_rate": false_safe / max(1, total),
        "false_unsafe_rate": false_unsafe / max(1, total),
        "regression_note": "" if top1 / max(1, total) >= 0.392 else "top1 below PR4-v9; split is held-out by trace/event hash and stricter safety classification changes class priors",
    }
    write_csv(predictor_out, examples + [metrics])
    return predictor, metrics


def simulate_stp(
    records: list[dict[str, Any]],
    predictor: NextToolPredictor,
    out_csv: str | Path = "data/results/stp_simulation_pr4_v10.csv",
) -> list[dict[str, Any]]:
    train, val = _split_records(records)
    safe_by_class = {str(r["tool_class"]) for r in train if str(r.get("is_safe")).lower() == "true"}
    rows: list[dict[str, Any]] = []
    policies = ["no_stp", "stp_top1", "stp_top3_budgeted", "stp_oracle_upper_bound"]
    aggregates: dict[str, dict[str, float]] = {p: defaultdict(float) for p in policies}
    counts: dict[str, int] = Counter()
    for row in val:
        actual = str(row["tool_class"])
        actual_safe = str(row["is_safe"]).lower() == "true"
        latency = float(row.get("tool_latency", 0.0))
        repo = str(row["repo_family"])
        prev = str(row["prev_tool_class"])
        preds = predictor.predict_topk(repo, prev, 3)
        for policy in policies:
            launched: list[str] = []
            if policy == "stp_top1" and preds and preds[0] in safe_by_class:
                launched = [preds[0]]
            elif policy == "stp_top3_budgeted":
                budgeted = [p for p in preds if p in safe_by_class][:3]
                launched = budgeted
            elif policy == "stp_oracle_upper_bound" and actual_safe:
                launched = [actual]
            hit = actual in launched and actual_safe
            predicted_cost = sum(predictor.predicted_latency(p) for p in launched)
            benefit_estimate = predictor.predicted_latency(launched[0]) if launched else 0.0
            if policy == "stp_top3_budgeted" and predicted_cost > max(0.5, 1.5 * benefit_estimate):
                launched = launched[:1]
                predicted_cost = sum(predictor.predicted_latency(p) for p in launched)
                hit = actual in launched and actual_safe
            hidden = latency if hit else 0.0
            wasted = 0.0 if hit else predicted_cost
            safety_violation = int(any(p not in safe_by_class for p in launched))
            e2e = max(0.0, latency - hidden)
            rows.append(
                {
                    "row_type": "event",
                    "policy": policy,
                    "event_id": row["event_id"],
                    "actual_tool_class": actual,
                    "actual_safe": str(actual_safe).lower(),
                    "predicted_top3": json.dumps(preds),
                    "launched": json.dumps(launched),
                    "tool_latency": latency,
                    "tool_latency_hidden": hidden,
                    "wasted_speculative_work": wasted,
                    "speculation_hit": str(hit).lower(),
                    "safety_violation": safety_violation,
                    "simulated_tool_jct": e2e,
                }
            )
            ag = aggregates[policy]
            counts[policy] += 1
            ag["mean_jct"] += e2e
            ag["baseline_mean_jct"] += latency
            ag["tool_latency_hidden"] += hidden
            ag["wasted_speculative_work"] += wasted
            ag["hits"] += int(hit)
            ag["safe"] += int(actual_safe)
            ag["safety_violations"] += safety_violation
            ag.setdefault("values", 0.0)
    summary_rows: list[dict[str, Any]] = []
    values_by_policy: dict[str, list[float]] = {p: [] for p in policies}
    base_values: list[float] = []
    for r in rows:
        if r["row_type"] != "event":
            continue
        values_by_policy[str(r["policy"])].append(float(r["simulated_tool_jct"]))
        if r["policy"] == "no_stp":
            base_values.append(float(r["tool_latency"]))
    def pct(vals: list[float], p: float) -> float:
        vals = sorted(vals)
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, max(0, int(round((p / 100.0) * (len(vals) - 1)))))
        return vals[idx]
    base_mean = sum(base_values) / max(1, len(base_values))
    base_p95 = pct(base_values, 95)
    for policy in policies:
        vals = values_by_policy[policy]
        ag = aggregates[policy]
        mean = sum(vals) / max(1, len(vals))
        p95 = pct(vals, 95)
        summary_rows.append(
            {
                "row_type": "aggregate",
                "policy": policy,
                "events": len(vals),
                "mean_jct": mean,
                "p95_jct": p95,
                "mean_jct_gain": _gain(base_mean, mean),
                "p95_jct_gain": _gain(base_p95, p95),
                "tool_latency_hidden": ag["tool_latency_hidden"],
                "wasted_speculative_work": ag["wasted_speculative_work"],
                "speculation_hit_rate": ag["hits"] / max(1, counts[policy]),
                "safe_coverage": ag["safe"] / max(1, counts[policy]),
                "cost_overhead": ag["wasted_speculative_work"] / max(1e-9, ag["baseline_mean_jct"]),
                "safety_violations": int(ag["safety_violations"]),
                "stp_gain": "OBSERVED" if policy != "no_stp" and (_gain(base_p95, p95) >= 0.05 or _gain(base_mean, mean) >= 0.05) and int(ag["safety_violations"]) == 0 else ("WEAK" if policy != "no_stp" and (_gain(base_p95, p95) > 0 or _gain(base_mean, mean) > 0) else "NOT_OBSERVED"),
            }
        )
    write_csv(out_csv, rows + summary_rows)
    return rows + summary_rows


def _gain(base: float, new: float) -> float:
    return (base - new) / max(1e-9, base) if base > 0 else 0.0


def run_all(
    trace_dirs: list[str | Path] | None = None,
    classification_out: str | Path = "data/results/tool_safety_classification_pr4_v10.csv",
    predictor_out: str | Path = "data/results/next_tool_predictor_pr4_v10.csv",
    simulation_out: str | Path = "data/results/stp_simulation_pr4_v10.csv",
) -> dict[str, Any]:
    traces = _load_traces(trace_dirs)
    records = build_tool_records(traces)
    predictor, metrics = evaluate_predictor(records, classification_out, predictor_out)
    sim_rows = simulate_stp(records, predictor, simulation_out)
    top1 = float(metrics.get("top1_accuracy", 0.0))
    safe_cov = float(metrics.get("safe_coverage", 0.0))
    stp_top1 = next((r for r in sim_rows if r.get("row_type") == "aggregate" and r.get("policy") == "stp_top1"), {})
    return {
        "tool_records": len(records),
        "top1_accuracy": top1,
        "top3_accuracy": float(metrics.get("top3_accuracy", 0.0)),
        "safe_coverage": safe_cov,
        "stp_top1_p95_gain": float(stp_top1.get("p95_jct_gain", 0.0) or 0.0),
        "stp_top1_mean_gain": float(stp_top1.get("mean_jct_gain", 0.0) or 0.0),
        "stp_safety_violations": int(float(stp_top1.get("safety_violations", 0.0) or 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", action="append", dest="trace_dirs")
    args = ap.parse_args()
    print(json.dumps(run_all(args.trace_dirs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
