from __future__ import annotations

import argparse
import csv
import json
import math
import posixpath
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import write_csv


DEFAULT_TRACE_DIRS = ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]

SAFE_READ_ONLY_EXACT = "SAFE_READ_ONLY_EXACT"
SAFE_SANDBOXED = "SAFE_SANDBOXED"
UNSAFE = "UNSAFE"
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


def _normalize_ws(command: str) -> str:
    return " ".join(command.replace("\\\n", " ").split())


def _strip_cd_prefix(command: str) -> str:
    cmd = _normalize_ws(command)
    while True:
        match = re.match(r"^cd\s+([A-Za-z0-9_./-]+)\s*&&\s*(.+)$", cmd)
        if not match:
            return cmd
        cmd = match.group(2).strip()


def _norm_path(arg: str) -> str:
    if arg in {"-", "--"} or "://" in arg:
        return arg
    if arg.startswith("$"):
        return arg
    normalized = posixpath.normpath(arg)
    if normalized == ".":
        return "."
    return normalized[2:] if normalized.startswith("./") else normalized


def command_class(command: str | None, tool_type: str | None = None) -> str:
    cmd = (command or "").strip()
    low = cmd.lower()
    if not cmd:
        return tool_type or "unknown"
    if re.search(r"\b(rg|grep)\b", low):
        return "grep_search"
    if re.search(r"\b(cat|sed|head|tail|wc|less)\b", low):
        return "file_read"
    if re.search(r"\b(ls|find|pwd|tree)\b", low):
        return "list_find"
    if re.search(r"\bgit\s+(status|diff|log|show|rev-parse|branch\s+--show-current)\b", low):
        return "git_read"
    if re.search(r"\bpython(3)?\b", low):
        return "python_inspect"
    if re.search(r"\b(pytest|tox|nox|unittest)\b|\bmake\s+(test|check)\b", low):
        return "test"
    if re.search(r"\b(apply_patch|write_file|edit_file)\b", low):
        return "file_write"
    return "shell_other"


def canonicalize_command(command: str | None) -> str:
    cmd = _strip_cd_prefix(command or "")
    cls = command_class(cmd)
    if not cmd:
        return f"{cls}:empty:{stable_hash('')[:16]}"
    low = cmd.lower()
    if re.search(r"\s(&&|;|\|)\s", cmd):
        normalized_chain = _normalize_ws(cmd)
        return f"{cls}:shell-chain:{stable_hash(normalized_chain)[:16]}"
    if re.search(r"\bpython(3)?\b.*<<", low):
        code_hash = stable_hash(_normalize_ws(cmd))[:16]
        return f"{cls}:python-heredoc:{code_hash}"
    try:
        parts = shlex.split(cmd)
    except Exception:
        normalized = _normalize_ws(cmd)
        return f"{cls}:unparseable:{stable_hash(normalized)[:16]}"
    if not parts:
        return f"{cls}:empty:{stable_hash('')[:16]}"
    exe = Path(parts[0]).name
    rest = parts[1:]
    if exe in {"python", "python3"} and "-c" in rest:
        idx = rest.index("-c")
        code = rest[idx + 1] if idx + 1 < len(rest) else ""
        tail = [_norm_path(x) for x in rest[idx + 2 :]]
        normalized = f"{exe} -c {stable_hash(code)[:16]} {' '.join(tail)}".strip()
        return f"{cls}:{normalized}:{stable_hash(normalized)[:16]}"
    if exe in {"rg", "grep"}:
        flags: list[str] = []
        positional: list[str] = []
        i = 0
        while i < len(rest):
            arg = rest[i]
            if arg.startswith("-"):
                if arg in {"-e", "-f", "--glob", "-g", "--type", "-t"} and i + 1 < len(rest):
                    flags.append(f"{arg}={rest[i + 1]}")
                    i += 2
                    continue
                flags.append(arg)
            else:
                positional.append(_norm_path(arg))
            i += 1
        normalized = " ".join([exe, *sorted(flags), *positional]).strip()
        return f"{cls}:{normalized}:{stable_hash(normalized)[:16]}"
    if exe in {"ls", "find", "cat", "head", "tail", "wc", "sed", "git", "pwd", "tree", "less"}:
        flags = sorted([x for x in rest if x.startswith("-")])
        args = [_norm_path(x) for x in rest if not x.startswith("-")]
        normalized = " ".join([exe, *flags, *args]).strip()
        return f"{cls}:{normalized}:{stable_hash(normalized)[:16]}"
    normalized = " ".join([exe, *(_norm_path(x) for x in rest)]).strip()
    return f"{cls}:{normalized}:{stable_hash(normalized)[:16]}"


def _readonly_shell_chain(command: str) -> bool:
    cmd = _strip_cd_prefix(command)
    if any(x in cmd.lower() for x in [">", ">>", "| tee", " sed -i", " apply_patch", " git apply"]):
        return False
    parts = re.split(r"\s*(?:&&|;|\n)\s*", cmd)
    if not parts:
        return False
    saw_read = False
    for part in parts:
        p = part.strip()
        if not p:
            continue
        low = p.lower()
        if low.startswith(("cd ", "export ", "source ")):
            continue
        if re.match(r"^(ls|pwd|tree|less)\b", low):
            saw_read = True
            continue
        if re.match(r"^(cat|head|tail|wc|find|rg|grep)\b", low):
            saw_read = True
            continue
        if re.match(r"^sed\b", low) and " -i" not in low:
            saw_read = True
            continue
        if re.match(r"^git\s+(status|diff|log|show|rev-parse|branch\s+--show-current)\b", low):
            saw_read = True
            continue
        if re.match(r"^python(3)?\s+(-c|- <<)", low) and not _python_looks_mutating(low):
            saw_read = True
            continue
        return False
    return saw_read


def _python_looks_mutating(low_command: str) -> bool:
    mutating = [
        "open(",
        ".write(",
        "write_text",
        "unlink(",
        "remove(",
        "rmdir(",
        "mkdir(",
        "rename(",
        "replace(",
        "requests.post",
        "requests.put",
        "requests.patch",
        "requests.delete",
        "subprocess.run",
        "os.system",
    ]
    return any(x in low_command for x in mutating)


def classify_command_safety(command: str | None, tool_type: str | None = None) -> tuple[str, str, str]:
    cmd = _strip_cd_prefix(command or "")
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
        r"(^|[;&|]\s*)git\s+(commit|checkout|reset|merge|rebase|push|pull|apply|clean|stash)\b",
        r"\b(apply_patch|write_file|edit_file)\b",
        r">\s*[^&]",
        r">>",
        r"\btee\s+",
        r"\bsed\s+-i\b",
        r"\b(pip|uv|poetry|npm|pnpm|yarn)\s+(install|add|remove|publish|update|upgrade)\b",
        r"\bcurl\b.*\b(-X\s*(POST|PUT|PATCH|DELETE)|--request\s*(POST|PUT|PATCH|DELETE))",
        r"\b(wget|curl)\b.*\b(upload|post|put|delete)\b",
    ]
    for pat in unsafe_patterns:
        if re.search(pat, low):
            return UNSAFE, cls, f"matched:{pat}"
    sandbox_patterns = [
        r"\bpytest\b",
        r"\btox\b",
        r"\bnox\b",
        r"\bunittest\b",
        r"\bpython(3)?\s+-m\s+pytest\b",
        r"\bmake\s+(test|check)\b",
        r"\bnpm\s+(test|run\s+test)\b",
        r"\bsetup\.py\s+(test|build|build_ext)\b",
        r"\bpython(3)?\s+setup\.py\s+(build|build_ext)\b",
    ]
    for pat in sandbox_patterns:
        if re.search(pat, low):
            return SAFE_SANDBOXED, cls, f"sandboxed:{pat}"
    if re.search(r"\bpython(3)?\b", low):
        if _python_looks_mutating(low):
            return UNSAFE, cls, "python_write_or_network"
        if "-c" in low or "<< " in low or "<<'" in low or '<<"' in low or low.endswith(".py"):
            return SAFE_READ_ONLY_EXACT, cls, "python_inspection_readonly"
    if _readonly_shell_chain(cmd):
        return SAFE_READ_ONLY_EXACT, cls, "readonly_shell_chain"
    try:
        parts = shlex.split(cmd)
    except Exception:
        return UNKNOWN, cls, "unparseable_shell"
    base = Path(parts[0]).name if parts else ""
    if base in {"ls", "cat", "grep", "rg", "find", "wc", "head", "tail", "pwd", "tree", "less"}:
        return SAFE_READ_ONLY_EXACT, cls, "readonly_whitelist"
    if base == "sed" and "-i" not in parts:
        return SAFE_READ_ONLY_EXACT, cls, "sed_readonly"
    if base == "git" and len(parts) >= 2 and parts[1] in {"status", "diff", "log", "show", "rev-parse"}:
        return SAFE_READ_ONLY_EXACT, cls, "git_readonly"
    return UNKNOWN, cls, "not_whitelisted"


def _events_by_branch(trace: Trace) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = defaultdict(list)
    for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start or 0.0, e.node_id)):
        if ev.node_type in {"llm", "tool", "verifier"}:
            out[ev.branch_id].append(ev)
    return out


def build_tool_records(traces: list[Trace]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        repo = _repo_family(trace)
        for branch, events in _events_by_branch(trace).items():
            prev_class = "START"
            prev_key = "START"
            error_count = 0
            last_returncode = 0
            obs_len = 0
            recent_modified = 0
            for idx, ev in enumerate(events):
                if ev.node_type != "tool":
                    continue
                safety, cls, reason = classify_command_safety(ev.command, ev.tool_type)
                canonical = canonicalize_command(ev.command)
                command_low = (ev.command or "").lower()
                rows.append(
                    {
                        "trace_source": trace.metadata.get("source", ""),
                        "repo_family": repo,
                        "branch_id": branch,
                        "step_index": idx,
                        "event_id": ev.event_id,
                        "command": ev.command or "",
                        "canonical_tool_key": canonical,
                        "tool_type": ev.tool_type or "",
                        "command_class": cls,
                        "safety_level": safety,
                        "safety_reason": reason,
                        "is_read_only_exact": str(safety == SAFE_READ_ONLY_EXACT).lower(),
                        "is_sandboxed": str(safety == SAFE_SANDBOXED).lower(),
                        "prev_canonical_tool_key": prev_key,
                        "prev_command_class": prev_class,
                        "previous_returncode": last_returncode,
                        "observation_token_length": int(ev.observation_tokens or obs_len or 0),
                        "error_count": error_count,
                        "mentioned_file_not_found": str("no such file" in command_low or "file not found" in command_low).lower(),
                        "mentioned_import_error": str("importerror" in command_low or "modulenotfounderror" in command_low).lower(),
                        "mentioned_traceback": str("traceback" in command_low).lower(),
                        "mentioned_test_fail": str("fail" in command_low or "pytest" in command_low).lower(),
                        "recent_modified_files": recent_modified,
                        "tool_latency": _tool_latency(ev),
                    }
                )
                last_returncode = int(ev.exit_code or 0)
                if last_returncode != 0:
                    error_count += 1
                recent_modified = int(ev.modified_files_count or 0)
                obs_len = int(ev.observation_tokens or 0)
                prev_class = cls
                prev_key = canonical
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


class NextCommandPredictor:
    def __init__(self) -> None:
        self.exact_repo_prev_key: Counter[tuple[str, str, str]] = Counter()
        self.exact_repo_prev_class: Counter[tuple[str, str, str]] = Counter()
        self.exact_prev_class: Counter[tuple[str, str]] = Counter()
        self.exact_global: Counter[str] = Counter()
        self.class_repo_prev_class: Counter[tuple[str, str, str]] = Counter()
        self.class_prev_class: Counter[tuple[str, str]] = Counter()
        self.class_global: Counter[str] = Counter()
        self.latency_by_key: dict[str, float] = {}
        self.latency_by_class: dict[str, float] = {}
        self.safety_by_key: dict[str, str] = {}

    def fit(self, records: list[dict[str, Any]]) -> None:
        key_latencies: dict[str, list[float]] = defaultdict(list)
        class_latencies: dict[str, list[float]] = defaultdict(list)
        safety_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in records:
            repo = str(row["repo_family"])
            prev_key = str(row["prev_canonical_tool_key"])
            prev_class = str(row["prev_command_class"])
            actual_key = str(row["canonical_tool_key"])
            actual_class = str(row["command_class"])
            latency = float(row.get("tool_latency", 0.0))
            self.exact_repo_prev_key[(repo, prev_key, actual_key)] += 1
            self.exact_repo_prev_class[(repo, prev_class, actual_key)] += 1
            self.exact_prev_class[(prev_class, actual_key)] += 1
            self.exact_global[actual_key] += 1
            self.class_repo_prev_class[(repo, prev_class, actual_class)] += 1
            self.class_prev_class[(prev_class, actual_class)] += 1
            self.class_global[actual_class] += 1
            key_latencies[actual_key].append(latency)
            class_latencies[actual_class].append(latency)
            safety_counts[actual_key][str(row.get("safety_level", UNKNOWN))] += 1
        self.latency_by_key = {k: sorted(v)[len(v) // 2] for k, v in key_latencies.items() if v}
        self.latency_by_class = {k: sorted(v)[len(v) // 2] for k, v in class_latencies.items() if v}
        self.safety_by_key = {k: c.most_common(1)[0][0] for k, c in safety_counts.items() if c}

    def predict_exact_topk(self, repo: str, prev_key: str, prev_class: str, k: int = 3) -> list[tuple[str, float]]:
        scored: Counter[str] = Counter()
        for (r, p, key), n in self.exact_repo_prev_key.items():
            if r == repo and p == prev_key:
                scored[key] += 6 * n
        for (r, p, key), n in self.exact_repo_prev_class.items():
            if r == repo and p == prev_class:
                scored[key] += 3 * n
        for (p, key), n in self.exact_prev_class.items():
            if p == prev_class:
                scored[key] += 2 * n
        scored.update(self.exact_global)
        total = sum(scored.values()) or 1
        return [(key, count / total) for key, count in scored.most_common(k)]

    def predict_class_topk(self, repo: str, prev_class: str, k: int = 3) -> list[tuple[str, float]]:
        scored: Counter[str] = Counter()
        for (r, p, cls), n in self.class_repo_prev_class.items():
            if r == repo and p == prev_class:
                scored[cls] += 3 * n
        for (p, cls), n in self.class_prev_class.items():
            if p == prev_class:
                scored[cls] += 2 * n
        scored.update(self.class_global)
        total = sum(scored.values()) or 1
        return [(cls, count / total) for cls, count in scored.most_common(k)]

    def predicted_key_latency(self, key: str) -> float:
        if key in self.latency_by_key:
            return self.latency_by_key[key]
        vals = sorted(self.latency_by_key.values())
        return vals[len(vals) // 2] if vals else 0.0

    def predicted_class_latency(self, cls: str) -> float:
        if cls in self.latency_by_class:
            return self.latency_by_class[cls]
        vals = sorted(self.latency_by_class.values())
        return vals[len(vals) // 2] if vals else 0.0

    def key_safety(self, key: str) -> str:
        return self.safety_by_key.get(key, UNKNOWN)


def evaluate_predictor(
    records: list[dict[str, Any]],
    classification_out: str | Path = "data/results/tool_safety_classification_pr4_v11.csv",
    predictor_out: str | Path = "data/results/next_tool_predictor_v2_pr4_v11.csv",
) -> tuple[NextCommandPredictor, dict[str, Any]]:
    write_csv(classification_out, records)
    train, val = _split_records(records)
    predictor = NextCommandPredictor()
    predictor.fit(train)
    metrics = Counter()
    examples: list[dict[str, Any]] = []
    for row in val:
        actual_key = str(row["canonical_tool_key"])
        actual_class = str(row["command_class"])
        repo = str(row["repo_family"])
        prev_key = str(row["prev_canonical_tool_key"])
        prev_class = str(row["prev_command_class"])
        safety = str(row["safety_level"])
        exact = [k for k, _ in predictor.predict_exact_topk(repo, prev_key, prev_class, 3)]
        classes = [c for c, _ in predictor.predict_class_topk(repo, prev_class, 3)]
        safe = safety in {SAFE_READ_ONLY_EXACT, SAFE_SANDBOXED}
        metrics["total"] += 1
        metrics["exact_top1"] += int(bool(exact) and exact[0] == actual_key)
        metrics["exact_top3"] += int(actual_key in exact)
        metrics["class_top1"] += int(bool(classes) and classes[0] == actual_class)
        metrics["class_top3"] += int(actual_class in classes)
        metrics["read_only"] += int(safety == SAFE_READ_ONLY_EXACT)
        metrics["sandboxed"] += int(safety == SAFE_SANDBOXED)
        if safe:
            metrics["safe_total"] += 1
            metrics["safe_exact_top1"] += int(bool(exact) and exact[0] == actual_key)
            metrics["safe_exact_top3"] += int(actual_key in exact)
        if len(examples) < 30:
            examples.append(
                {
                    "row_type": "example",
                    "event_id": row["event_id"],
                    "repo_family": repo,
                    "prev_command_class": prev_class,
                    "actual_command_class": actual_class,
                    "actual_canonical_tool_key": actual_key,
                    "exact_top1": exact[0] if exact else "",
                    "exact_top3": json.dumps(exact),
                    "class_top1": classes[0] if classes else "",
                    "class_top3": json.dumps(classes),
                    "safety_level": safety,
                }
            )
    total = max(1, metrics["total"])
    safe_total = max(1, metrics["safe_total"])
    aggregate = {
        "row_type": "aggregate",
        "train_rows": len(train),
        "validation_rows": len(val),
        "exact_top1_accuracy": metrics["exact_top1"] / total,
        "exact_top3_accuracy": metrics["exact_top3"] / total,
        "class_top1_accuracy": metrics["class_top1"] / total,
        "class_top3_accuracy": metrics["class_top3"] / total,
        "safe_exact_top1_accuracy": metrics["safe_exact_top1"] / safe_total,
        "safe_exact_top3_accuracy": metrics["safe_exact_top3"] / safe_total,
        "sandboxed_coverage": metrics["sandboxed"] / total,
        "read_only_coverage": metrics["read_only"] / total,
        "coverage": len(val) / max(1, len(records)),
        "class_hits_are_potential_only": "true",
    }
    write_csv(predictor_out, examples + [aggregate])
    return predictor, aggregate


def _pct(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round((p / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def _gain(base: float, new: float) -> float:
    return (base - new) / max(1e-9, base) if base > 0 else 0.0


def _launch_exact(
    predictions: list[tuple[str, float]],
    predictor: NextCommandPredictor,
    allowed_safety: set[str],
    *,
    budget: int,
    latency_threshold: float,
    worker_cost: float,
    contention_penalty: float,
) -> list[str]:
    launched: list[str] = []
    for key, prob in predictions:
        if len(launched) >= budget:
            break
        safety = predictor.key_safety(key)
        if safety not in allowed_safety:
            continue
        latency = predictor.predicted_key_latency(key)
        if latency < latency_threshold:
            continue
        benefit = prob * latency
        cost = worker_cost * latency + contention_penalty
        if benefit > cost:
            launched.append(key)
    return launched


def simulate_stp_v2(
    records: list[dict[str, Any]],
    predictor: NextCommandPredictor,
    out_csv: str | Path = "data/results/stp_v2_simulation_pr4_v11.csv",
    latency_threshold: float = 0.5,
    worker_cost: float = 0.25,
    contention_penalty: float = 0.02,
) -> list[dict[str, Any]]:
    _train, val = _split_records(records)
    policies = [
        "no_stp",
        "stp_exact_top1",
        "stp_exact_top3_budgeted",
        "stp_sandbox_top1",
        "stp_sandbox_top3_budgeted",
        "stp_class_upper_bound",
        "stp_oracle_upper_bound",
    ]
    rows: list[dict[str, Any]] = []
    vals_by_policy: dict[str, list[float]] = {p: [] for p in policies}
    base_vals: list[float] = []
    aggregate: dict[str, Counter[str]] = {p: Counter() for p in policies}
    aggregate_float: dict[str, defaultdict[str, float]] = {p: defaultdict(float) for p in policies}
    for row in val:
        actual_key = str(row["canonical_tool_key"])
        actual_class = str(row["command_class"])
        safety = str(row["safety_level"])
        actual_exact_safe = safety == SAFE_READ_ONLY_EXACT
        actual_sandboxed = safety == SAFE_SANDBOXED
        actual_safe = actual_exact_safe or actual_sandboxed
        latency = float(row.get("tool_latency", 0.0))
        repo = str(row["repo_family"])
        prev_key = str(row["prev_canonical_tool_key"])
        prev_class = str(row["prev_command_class"])
        exact_preds_scored = predictor.predict_exact_topk(repo, prev_key, prev_class, 3)
        class_preds_scored = predictor.predict_class_topk(repo, prev_class, 3)
        exact_preds = [k for k, _ in exact_preds_scored]
        class_preds = [c for c, _ in class_preds_scored]
        for policy in policies:
            launched_keys: list[str] = []
            launched_classes: list[str] = []
            if policy == "stp_exact_top1":
                launched_keys = _launch_exact(
                    exact_preds_scored[:1],
                    predictor,
                    {SAFE_READ_ONLY_EXACT},
                    budget=1,
                    latency_threshold=latency_threshold,
                    worker_cost=worker_cost,
                    contention_penalty=contention_penalty,
                )
            elif policy == "stp_exact_top3_budgeted":
                launched_keys = _launch_exact(
                    exact_preds_scored,
                    predictor,
                    {SAFE_READ_ONLY_EXACT},
                    budget=3,
                    latency_threshold=latency_threshold,
                    worker_cost=worker_cost,
                    contention_penalty=contention_penalty,
                )
            elif policy == "stp_sandbox_top1":
                launched_keys = _launch_exact(
                    exact_preds_scored[:1],
                    predictor,
                    {SAFE_SANDBOXED},
                    budget=1,
                    latency_threshold=latency_threshold,
                    worker_cost=worker_cost,
                    contention_penalty=contention_penalty,
                )
            elif policy == "stp_sandbox_top3_budgeted":
                launched_keys = _launch_exact(
                    exact_preds_scored,
                    predictor,
                    {SAFE_SANDBOXED},
                    budget=3,
                    latency_threshold=latency_threshold,
                    worker_cost=worker_cost,
                    contention_penalty=contention_penalty,
                )
            elif policy == "stp_class_upper_bound":
                for cls, prob in class_preds_scored:
                    pred_latency = predictor.predicted_class_latency(cls)
                    if pred_latency < latency_threshold:
                        continue
                    if prob * pred_latency > worker_cost * pred_latency + contention_penalty:
                        launched_classes.append(cls)
                        break
            elif policy == "stp_oracle_upper_bound" and actual_safe and latency >= latency_threshold:
                launched_keys = [actual_key]

            exact_hit = actual_key in launched_keys and actual_exact_safe and policy.startswith("stp_exact")
            sandbox_hit = actual_key in launched_keys and actual_sandboxed and policy.startswith("stp_sandbox")
            class_hit = actual_class in launched_classes and actual_safe and policy == "stp_class_upper_bound"
            oracle_hit = actual_key in launched_keys and actual_safe and policy == "stp_oracle_upper_bound"
            hit = exact_hit or sandbox_hit or class_hit or oracle_hit
            predicted_cost = sum(predictor.predicted_key_latency(k) for k in launched_keys)
            predicted_cost += sum(predictor.predicted_class_latency(c) for c in launched_classes)
            hidden = latency if hit else 0.0
            wasted = 0.0 if hit else predicted_cost
            safety_violation = 0
            for key in launched_keys:
                allowed = {SAFE_READ_ONLY_EXACT} if policy.startswith("stp_exact") else {SAFE_SANDBOXED}
                if policy == "stp_oracle_upper_bound":
                    allowed = {SAFE_READ_ONLY_EXACT, SAFE_SANDBOXED}
                safety_violation += int(predictor.key_safety(key) not in allowed)
            e2e = max(0.0, latency - hidden)
            vals_by_policy[policy].append(e2e)
            if policy == "no_stp":
                base_vals.append(latency)
            ag = aggregate[policy]
            agf = aggregate_float[policy]
            ag["events"] += 1
            ag["hits"] += int(hit)
            ag["exact_hits"] += int(exact_hit)
            ag["safe_events"] += int(actual_safe)
            ag["safety_violations"] += safety_violation
            agf["tool_latency_hidden"] += hidden
            agf["wasted_speculative_work"] += wasted
            agf["baseline_tool_time"] += latency
            agf["speculation_cost"] += predicted_cost * worker_cost
            rows.append(
                {
                    "row_type": "event",
                    "policy": policy,
                    "event_id": row["event_id"],
                    "actual_command_class": actual_class,
                    "actual_canonical_tool_key": actual_key,
                    "actual_safety_level": safety,
                    "predicted_exact_top3": json.dumps(exact_preds),
                    "predicted_class_top3": json.dumps(class_preds),
                    "launched_exact_keys": json.dumps(launched_keys),
                    "launched_classes": json.dumps(launched_classes),
                    "tool_latency": latency,
                    "tool_latency_hidden": hidden,
                    "wasted_speculative_work": wasted,
                    "speculation_hit": str(hit).lower(),
                    "exact_hit": str(exact_hit).lower(),
                    "safety_violation": safety_violation,
                    "simulated_tool_jct": e2e,
                }
            )
    base_mean = sum(base_vals) / max(1, len(base_vals))
    base_p95 = _pct(base_vals, 95)
    summary_rows: list[dict[str, Any]] = []
    for policy in policies:
        vals = vals_by_policy[policy]
        mean = sum(vals) / max(1, len(vals))
        p95 = _pct(vals, 95)
        ag = aggregate[policy]
        agf = aggregate_float[policy]
        p95_gain = _gain(base_p95, p95)
        mean_gain = _gain(base_mean, mean)
        safety_violations = int(ag["safety_violations"])
        gain = "NOT_OBSERVED"
        if policy not in {"no_stp", "stp_class_upper_bound", "stp_oracle_upper_bound"}:
            if safety_violations == 0 and p95_gain >= 0.05:
                gain = "OBSERVED"
            elif safety_violations == 0 and (p95_gain > 0 or mean_gain > 0):
                gain = "WEAK"
        elif policy in {"stp_class_upper_bound", "stp_oracle_upper_bound"}:
            gain = "UPPER_BOUND_ONLY"
        summary_rows.append(
            {
                "row_type": "aggregate",
                "policy": policy,
                "events": len(vals),
                "mean_jct": mean,
                "p95_jct": p95,
                "tool_latency_hidden": agf["tool_latency_hidden"],
                "speculation_hit_rate": ag["hits"] / max(1, ag["events"]),
                "exact_hit_rate": ag["exact_hits"] / max(1, ag["events"]),
                "wasted_speculative_work": agf["wasted_speculative_work"],
                "safety_violations": safety_violations,
                "sandbox_overhead": agf["speculation_cost"] / max(1e-9, agf["baseline_tool_time"]) if "sandbox" in policy else 0.0,
                "cost_overhead": agf["speculation_cost"] / max(1e-9, agf["baseline_tool_time"]),
                "p95_jct_gain": p95_gain,
                "mean_jct_gain": mean_gain,
                "stp_gain": gain,
                "class_upper_bound_only": str(policy == "stp_class_upper_bound").lower(),
                "oracle_upper_bound_only": str(policy == "stp_oracle_upper_bound").lower(),
            }
        )
    write_csv(out_csv, rows + summary_rows)
    return rows + summary_rows


def run_all(
    trace_dirs: list[str | Path] | None = None,
    classification_out: str | Path = "data/results/tool_safety_classification_pr4_v11.csv",
    predictor_out: str | Path = "data/results/next_tool_predictor_v2_pr4_v11.csv",
    simulation_out: str | Path = "data/results/stp_v2_simulation_pr4_v11.csv",
) -> dict[str, Any]:
    traces = _load_traces(trace_dirs)
    records = build_tool_records(traces)
    predictor, metrics = evaluate_predictor(records, classification_out, predictor_out)
    sim_rows = simulate_stp_v2(records, predictor, simulation_out)
    def agg(policy: str) -> dict[str, Any]:
        return next((r for r in sim_rows if r.get("row_type") == "aggregate" and r.get("policy") == policy), {})
    exact = agg("stp_exact_top1")
    sandbox = agg("stp_sandbox_top1")
    return {
        "tool_records": len(records),
        "exact_top1_accuracy": float(metrics.get("exact_top1_accuracy", 0.0)),
        "exact_top3_accuracy": float(metrics.get("exact_top3_accuracy", 0.0)),
        "class_top1_accuracy": float(metrics.get("class_top1_accuracy", 0.0)),
        "class_top3_accuracy": float(metrics.get("class_top3_accuracy", 0.0)),
        "read_only_coverage": float(metrics.get("read_only_coverage", 0.0)),
        "sandboxed_coverage": float(metrics.get("sandboxed_coverage", 0.0)),
        "stp_exact_p95_gain": float(exact.get("p95_jct_gain", 0.0) or 0.0),
        "stp_sandbox_p95_gain": float(sandbox.get("p95_jct_gain", 0.0) or 0.0),
        "stp_safety_violations": int(float(exact.get("safety_violations", 0.0) or 0.0))
        + int(float(sandbox.get("safety_violations", 0.0) or 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", action="append", dest="trace_dirs")
    ap.add_argument("--classification-out", default="data/results/tool_safety_classification_pr4_v11.csv")
    ap.add_argument("--predictor-out", default="data/results/next_tool_predictor_v2_pr4_v11.csv")
    ap.add_argument("--simulation-out", default="data/results/stp_v2_simulation_pr4_v11.csv")
    args = ap.parse_args()
    print(json.dumps(run_all(args.trace_dirs, args.classification_out, args.predictor_out, args.simulation_out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
