from __future__ import annotations

import argparse
import csv
import json
import math
import posixpath
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentweaver.simulator.safe_tool_prefetch_v2 import (
    SAFE_READ_ONLY_EXACT,
    SAFE_SANDBOXED,
    UNKNOWN,
    canonicalize_command,
    classify_command_safety,
    command_class,
)
from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import write_csv


DEFAULT_TRACE_DIRS = ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]

FILE_CONTENT = "FILE_CONTENT"
DIRECTORY_LISTING = "DIRECTORY_LISTING"
GIT_STATUS = "GIT_STATUS"
GIT_DIFF = "GIT_DIFF"
GIT_LOG = "GIT_LOG"
GREP_INDEX = "GREP_INDEX"
PYTHON_INSPECTION_RESULT = "PYTHON_INSPECTION_RESULT"
TEST_RESULT_SANDBOXED = "TEST_RESULT_SANDBOXED"

READ_ONLY_ARTIFACT = "READ_ONLY_ARTIFACT"
SANDBOXED_ARTIFACT = "SANDBOXED_ARTIFACT"


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    artifact_type: str
    repo_family: str
    workspace_snapshot_hash: str
    path_scope: str
    query_terms: str
    command_class_source: str
    content_bytes: int
    generation_latency: float
    valid_until_step: int
    safety_level: str

    @property
    def key(self) -> str:
        return artifact_key(self.artifact_type, self.path_scope, self.query_terms)


def artifact_key(artifact_type: str, path_scope: str = "", query_terms: str = "") -> str:
    return f"{artifact_type}:{path_scope or '.'}:{query_terms or '*'}"


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _norm_path(path: str, cwd: str = ".") -> str:
    if not path or path in {"-", "--"}:
        return "."
    if "://" in path or path.startswith("$"):
        return path
    joined = path if path.startswith("/") else posixpath.join(cwd, path)
    out = posixpath.normpath(joined)
    if out == ".":
        return "."
    return out[2:] if out.startswith("./") else out


def _extract_cwd_and_command(command: str) -> tuple[str, str]:
    cmd = " ".join((command or "").replace("\\\n", " ").split())
    cwd = "."
    while True:
        match = re.match(r"^cd\s+([^;&|]+)\s*&&\s*(.+)$", cmd)
        if not match:
            return cwd, cmd
        cwd = _norm_path(match.group(1).strip(), cwd)
        cmd = match.group(2).strip()


def _split_chain(command: str) -> list[str]:
    _cwd, cmd = _extract_cwd_and_command(command)
    parts: list[str] = []
    buf: list[str] = []
    quote = ""
    escape = False
    i = 0
    while i < len(cmd):
        ch = cmd[i]
        if escape:
            buf.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            i += 1
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if cmd.startswith("&&", i):
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            i += 2
            continue
        if ch == ";":
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts


def _snapshot_hash(ev: Event) -> tuple[str, str, bool]:
    parts = {
        "patch_hash": ev.patch_hash or "",
        "patch_hash_prefix": ev.patch_hash_prefix or "",
        "modified_files_count": int(ev.modified_files_count or 0),
        "git_diff_stat_bytes": int(ev.git_diff_stat_bytes or 0),
        "branch_id": ev.branch_id,
        "step_id": ev.step_id,
    }
    source = "trace_patch_and_git_fields"
    available = any([parts["patch_hash"], parts["patch_hash_prefix"], parts["modified_files_count"], parts["git_diff_stat_bytes"]])
    if not available:
        source = "trace_branch_step_fallback"
    return stable_hash(json.dumps(parts, sort_keys=True))[:20], source, True


def _extract_literal_pattern(args: list[str]) -> tuple[str, list[str]]:
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("-"):
            if a in {"-e", "-f", "--regexp", "--glob", "-g"} and i + 1 < len(args):
                if a in {"-e", "--regexp"}:
                    positional.append(args[i + 1])
                i += 2
                continue
            i += 1
            continue
        positional.append(a)
        i += 1
    if not positional:
        return "", ["."]
    return positional[0], positional[1:] or ["."]


def command_artifacts(command: str | None, repo: str, snapshot_hash: str, step_id: int) -> list[Artifact]:
    cmd = command or ""
    safety, cls, _reason = classify_command_safety(cmd)
    if safety == UNKNOWN or cls in {"file_write", "shell_other"} and safety != SAFE_READ_ONLY_EXACT:
        return []
    cwd, stripped = _extract_cwd_and_command(cmd)
    parts = _split_chain(cmd)
    if len(parts) > 1:
        artifacts: list[Artifact] = []
        for part in parts:
            sub_cmd = f"cd {cwd} && {part}" if cwd != "." else part
            sub = command_artifacts(sub_cmd, repo, snapshot_hash, step_id)
            if not sub:
                return []
            artifacts.extend(sub)
        return artifacts
    try:
        argv = shlex.split(stripped)
    except Exception:
        return []
    if not argv:
        return []
    exe = Path(argv[0]).name
    args = argv[1:]
    safety_level = SANDBOXED_ARTIFACT if safety == SAFE_SANDBOXED else READ_ONLY_ARTIFACT
    base_latency = 0.2 if safety_level == READ_ONLY_ARTIFACT else 1.0
    def art(typ: str, path_scope: str = ".", query: str = "", latency: float = base_latency) -> Artifact:
        key = artifact_key(typ, path_scope, query)
        return Artifact(
            artifact_id=stable_hash((repo, snapshot_hash, key))[:20],
            artifact_type=typ,
            repo_family=repo,
            workspace_snapshot_hash=snapshot_hash,
            path_scope=path_scope,
            query_terms=query,
            command_class_source=cls,
            content_bytes=4096 if typ in {FILE_CONTENT, GREP_INDEX} else 1024,
            generation_latency=latency,
            valid_until_step=step_id + 1,
            safety_level=safety_level,
        )
    if exe in {"cat", "less"}:
        return [art(FILE_CONTENT, _norm_path(a, cwd), latency=0.18) for a in args if not a.startswith("-")]
    if exe in {"head", "tail"}:
        paths = [a for a in args if not a.startswith("-")]
        return [art(FILE_CONTENT, _norm_path(a, cwd), latency=0.18) for a in paths[-1:]]
    if exe == "sed" and "-i" not in args:
        paths = [a for a in args if not a.startswith("-") and not a.startswith("s/") and not a.startswith("/")]
        return [art(FILE_CONTENT, _norm_path(a, cwd), latency=0.2) for a in paths[-1:]]
    if exe in {"ls", "tree"}:
        paths = [a for a in args if not a.startswith("-")] or ["."]
        return [art(DIRECTORY_LISTING, _norm_path(p, cwd), latency=0.16) for p in paths]
    if exe == "find":
        root = next((a for a in args if not a.startswith("-")), ".")
        return [art(DIRECTORY_LISTING, _norm_path(root, cwd), latency=0.25)]
    if exe in {"grep", "rg"}:
        pattern, paths = _extract_literal_pattern(args)
        path_scope = _norm_path(paths[0], cwd) if paths else "."
        return [art(GREP_INDEX, path_scope, pattern, latency=0.35)]
    if exe == "git" and args:
        sub = args[0]
        if sub == "status":
            return [art(GIT_STATUS, ".", latency=0.2)]
        if sub == "diff":
            return [art(GIT_DIFF, ".", latency=0.25)]
        if sub in {"log", "show"}:
            return [art(GIT_LOG, ".", " ".join(args[:3]), latency=0.25)]
    if exe in {"pytest", "tox", "nox"} or "unittest" in args or (exe in {"python", "python3"} and "-m" in args and any(x in args for x in {"pytest", "unittest"})):
        return [art(TEST_RESULT_SANDBOXED, ".", canonicalize_command(cmd), latency=1.0)]
    if exe in {"python", "python3"} and safety == SAFE_READ_ONLY_EXACT:
        return [art(PYTHON_INSPECTION_RESULT, ".", canonicalize_command(cmd), latency=0.4)]
    return []


def can_answer(actual_command: str | None, artifact: Artifact, repo: str, snapshot_hash: str, step_id: int) -> bool:
    if artifact.repo_family != repo or artifact.workspace_snapshot_hash != snapshot_hash or artifact.valid_until_step < step_id:
        return False
    required = command_artifacts(actual_command, repo, snapshot_hash, step_id)
    if not required:
        return False
    return any(_artifact_covers(artifact, req) for req in required)


def command_answerable_by(actual_command: str | None, artifacts: list[Artifact], repo: str, snapshot_hash: str, step_id: int) -> bool:
    required = command_artifacts(actual_command, repo, snapshot_hash, step_id)
    if not required:
        return False
    for req in required:
        if not any(_artifact_covers(a, req) and a.workspace_snapshot_hash == snapshot_hash and a.valid_until_step >= step_id for a in artifacts):
            return False
    return True


def _artifact_covers(artifact: Artifact, required: Artifact) -> bool:
    if artifact.safety_level != required.safety_level:
        return False
    if artifact.artifact_type == required.artifact_type and artifact.path_scope == required.path_scope:
        if required.artifact_type in {GREP_INDEX, GIT_LOG, TEST_RESULT_SANDBOXED, PYTHON_INSPECTION_RESULT}:
            return artifact.query_terms == required.query_terms
        return True
    if artifact.artifact_type == FILE_CONTENT and required.artifact_type == GREP_INDEX:
        return required.path_scope == artifact.path_scope
    if artifact.artifact_type == DIRECTORY_LISTING and required.artifact_type == DIRECTORY_LISTING:
        return required.path_scope == artifact.path_scope or required.path_scope.startswith(artifact.path_scope.rstrip("/") + "/")
    return False


def reconstruct_observation(actual_command: str | None, artifact: Artifact) -> dict[str, Any]:
    return {
        "lossless": True,
        "actual_command": actual_command or "",
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "path_scope": artifact.path_scope,
        "query_terms": artifact.query_terms,
    }


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
            last_returncode = 0
            error_count = 0
            for idx, ev in enumerate(events):
                if ev.node_type != "tool":
                    continue
                snapshot, snapshot_source, available = _snapshot_hash(ev)
                safety, cls, reason = classify_command_safety(ev.command, ev.tool_type)
                artifacts = command_artifacts(ev.command, repo, snapshot, ev.step_id)
                rows.append(
                    {
                        "trace_source": trace.metadata.get("source", ""),
                        "repo_family": repo,
                        "branch_id": branch,
                        "step_index": idx,
                        "step_id": ev.step_id,
                        "event_id": ev.event_id,
                        "command": ev.command or "",
                        "canonical_tool_key": canonicalize_command(ev.command),
                        "command_class": cls,
                        "safety_level": safety,
                        "safety_reason": reason,
                        "artifact_keys": json.dumps([a.key for a in artifacts]),
                        "artifact_types": json.dumps([a.artifact_type for a in artifacts]),
                        "answerable_by_artifact": str(bool(artifacts)).lower(),
                        "prev_canonical_tool_key": prev_key,
                        "prev_command_class": prev_class,
                        "previous_returncode": last_returncode,
                        "observation_token_length": int(ev.observation_tokens or 0),
                        "error_count": error_count,
                        "recent_modified_files": int(ev.modified_files_count or 0),
                        "workspace_snapshot_hash": snapshot,
                        "snapshot_hash_source": snapshot_source,
                        "snapshot_hash_available": str(available).lower(),
                        "tool_latency": _tool_latency(ev),
                    }
                )
                last_returncode = int(ev.exit_code or 0)
                error_count += int(last_returncode != 0)
                prev_class = cls
                prev_key = canonicalize_command(ev.command)
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


class ArtifactPredictor:
    def __init__(self) -> None:
        self.by_repo_prev_key: Counter[tuple[str, str, str]] = Counter()
        self.by_repo_prev_class: Counter[tuple[str, str, str]] = Counter()
        self.by_prev_class: Counter[tuple[str, str]] = Counter()
        self.global_counts: Counter[str] = Counter()
        self.artifact_by_key: dict[str, Artifact] = {}
        self.latency_by_key: dict[str, float] = {}
        self.latency_by_type: dict[str, float] = {}
        self.exact_by_repo_prev_class: Counter[tuple[str, str, str]] = Counter()
        self.class_by_repo_prev_class: Counter[tuple[str, str, str]] = Counter()

    def fit(self, records: list[dict[str, Any]]) -> None:
        key_lat: dict[str, list[float]] = defaultdict(list)
        type_lat: dict[str, list[float]] = defaultdict(list)
        for row in records:
            repo = str(row["repo_family"])
            prev_key = str(row["prev_canonical_tool_key"])
            prev_class = str(row["prev_command_class"])
            snapshot = str(row["workspace_snapshot_hash"])
            artifacts = command_artifacts(str(row.get("command", "")), repo, snapshot, int(float(row.get("step_id", 0) or 0)))
            for art in artifacts:
                self.by_repo_prev_key[(repo, prev_key, art.key)] += 5
                self.by_repo_prev_class[(repo, prev_class, art.key)] += 3
                self.by_prev_class[(prev_class, art.key)] += 2
                self.global_counts[art.key] += 1
                self.artifact_by_key[art.key] = art
                key_lat[art.key].append(float(row.get("tool_latency", 0.0)))
                type_lat[art.artifact_type].append(float(row.get("tool_latency", 0.0)))
            self.exact_by_repo_prev_class[(repo, prev_class, str(row["canonical_tool_key"]))] += 1
            self.class_by_repo_prev_class[(repo, prev_class, str(row["command_class"]))] += 1
        self.latency_by_key = {k: sorted(v)[len(v) // 2] for k, v in key_lat.items() if v}
        self.latency_by_type = {k: sorted(v)[len(v) // 2] for k, v in type_lat.items() if v}

    def predict(self, row: dict[str, Any], k: int = 3) -> list[tuple[Artifact, float]]:
        repo = str(row["repo_family"])
        prev_key = str(row["prev_canonical_tool_key"])
        prev_class = str(row["prev_command_class"])
        scored: Counter[str] = Counter()
        for (r, p, key), n in self.by_repo_prev_key.items():
            if r == repo and p == prev_key:
                scored[key] += n
        for (r, p, key), n in self.by_repo_prev_class.items():
            if r == repo and p == prev_class:
                scored[key] += n
        for (p, key), n in self.by_prev_class.items():
            if p == prev_class:
                scored[key] += n
        scored.update(self.global_counts)
        total = sum(scored.values()) or 1
        snapshot = str(row["workspace_snapshot_hash"])
        step = int(float(row.get("step_id", 0) or 0))
        out: list[tuple[Artifact, float]] = []
        for key, score in scored.most_common(max(k * 4, 12)):
            proto = self.artifact_by_key.get(key)
            if not proto:
                continue
            art = Artifact(
                artifact_id=stable_hash((repo, snapshot, key))[:20],
                artifact_type=proto.artifact_type,
                repo_family=repo,
                workspace_snapshot_hash=snapshot,
                path_scope=proto.path_scope,
                query_terms=proto.query_terms,
                command_class_source=prev_class,
                content_bytes=proto.content_bytes,
                generation_latency=self.predicted_generation_latency(proto),
                valid_until_step=step + 1,
                safety_level=proto.safety_level,
            )
            out.append((art, score / total))
            if len(out) >= k:
                break
        return out

    def predict_exact_key(self, row: dict[str, Any]) -> str:
        repo = str(row["repo_family"])
        prev_class = str(row["prev_command_class"])
        choices = Counter({k: n for (r, p, k), n in self.exact_by_repo_prev_class.items() if r == repo and p == prev_class})
        return choices.most_common(1)[0][0] if choices else ""

    def predict_class(self, row: dict[str, Any]) -> str:
        repo = str(row["repo_family"])
        prev_class = str(row["prev_command_class"])
        choices = Counter({k: n for (r, p, k), n in self.class_by_repo_prev_class.items() if r == repo and p == prev_class})
        return choices.most_common(1)[0][0] if choices else ""

    def predicted_generation_latency(self, artifact: Artifact) -> float:
        defaults = {
            DIRECTORY_LISTING: 0.020,
            FILE_CONTENT: 0.025,
            GREP_INDEX: 0.035,
            GIT_STATUS: 0.025,
            GIT_DIFF: 0.035,
            GIT_LOG: 0.035,
            PYTHON_INSPECTION_RESULT: 0.080,
            TEST_RESULT_SANDBOXED: 0.800,
        }
        floor = defaults.get(artifact.artifact_type, artifact.generation_latency)
        if artifact.safety_level == SANDBOXED_ARTIFACT:
            return max(floor, self.latency_by_key.get(artifact.key, self.latency_by_type.get(artifact.artifact_type, floor)) * 0.65)
        return floor


def evaluate_artifact_predictor(
    records: list[dict[str, Any]],
    predictor: ArtifactPredictor,
    out_csv: str | Path = "data/results/artifact_predictor_pr4_v12.csv",
) -> dict[str, Any]:
    _train, val = _split_records(records)
    counts = Counter()
    examples: list[dict[str, Any]] = []
    for row in val:
        preds = [a for a, _p in predictor.predict(row, 3)]
        repo = str(row["repo_family"])
        snapshot = str(row["workspace_snapshot_hash"])
        step = int(float(row.get("step_id", 0) or 0))
        required = command_artifacts(str(row.get("command", "")), repo, snapshot, step)
        answerable = bool(required)
        top1 = bool(preds[:1]) and command_answerable_by(str(row.get("command", "")), preds[:1], repo, snapshot, step)
        top3 = bool(preds) and command_answerable_by(str(row.get("command", "")), preds, repo, snapshot, step)
        counts["total"] += 1
        counts["answerable"] += int(answerable)
        counts["top1"] += int(top1)
        counts["top3"] += int(top3)
        for typ in {r.artifact_type for r in required}:
            counts[f"{typ}_total"] += 1
            counts[f"{typ}_hit"] += int(top3)
        for req in required:
            counts["artifact_required_total"] += 1
            counts["read_only_total"] += int(req.safety_level == READ_ONLY_ARTIFACT)
            counts["sandbox_total"] += int(req.safety_level == SANDBOXED_ARTIFACT)
        if len(examples) < 40:
            examples.append(
                {
                    "row_type": "example",
                    "event_id": row["event_id"],
                    "command_class": row["command_class"],
                    "actual_artifacts": json.dumps([r.key for r in required]),
                    "predicted_artifacts": json.dumps([p.key for p in preds]),
                    "artifact_top1_hit": str(top1).lower(),
                    "artifact_top3_hit": str(top3).lower(),
                    "snapshot_hash_available": row["snapshot_hash_available"],
                    "snapshot_hash_source": row["snapshot_hash_source"],
                }
            )
    total = max(1, counts["total"])
    required_total = max(1, counts["artifact_required_total"])
    aggregate = {
        "row_type": "aggregate",
        "validation_rows": counts["total"],
        "artifact_top1_hit": counts["top1"] / total,
        "artifact_top3_hit": counts["top3"] / total,
        "artifact_coverage": counts["answerable"] / total,
        "file_artifact_hit": counts[f"{FILE_CONTENT}_hit"] / max(1, counts[f"{FILE_CONTENT}_total"]),
        "dir_artifact_hit": counts[f"{DIRECTORY_LISTING}_hit"] / max(1, counts[f"{DIRECTORY_LISTING}_total"]),
        "grep_artifact_hit": counts[f"{GREP_INDEX}_hit"] / max(1, counts[f"{GREP_INDEX}_total"]),
        "git_artifact_hit": (counts[f"{GIT_STATUS}_hit"] + counts[f"{GIT_DIFF}_hit"] + counts[f"{GIT_LOG}_hit"]) / max(1, counts[f"{GIT_STATUS}_total"] + counts[f"{GIT_DIFF}_total"] + counts[f"{GIT_LOG}_total"]),
        "sandbox_test_artifact_hit": counts[f"{TEST_RESULT_SANDBOXED}_hit"] / max(1, counts[f"{TEST_RESULT_SANDBOXED}_total"]),
        "read_only_artifact_coverage": counts["read_only_total"] / required_total,
        "sandbox_artifact_coverage": counts["sandbox_total"] / required_total,
        "artifact_equivalent_not_class_match": "true",
    }
    write_csv(out_csv, examples + [aggregate])
    return aggregate


def write_tool_safety_classification(
    records: list[dict[str, Any]],
    out_csv: str | Path = "data/results/tool_safety_classification_pr4_v12.csv",
) -> None:
    rows: list[dict[str, Any]] = []
    for row in records:
        rows.append(
            {
                "event_id": row.get("event_id", ""),
                "repo_family": row.get("repo_family", ""),
                "branch_id": row.get("branch_id", ""),
                "step_id": row.get("step_id", ""),
                "command": row.get("command", ""),
                "canonical_tool_key": row.get("canonical_tool_key", ""),
                "command_class": row.get("command_class", ""),
                "safety_level": row.get("safety_level", ""),
                "safety_reason": row.get("safety_reason", ""),
                "answerable_by_artifact": row.get("answerable_by_artifact", "false"),
                "artifact_keys": row.get("artifact_keys", "[]"),
                "artifact_types": row.get("artifact_types", "[]"),
                "workspace_snapshot_hash": row.get("workspace_snapshot_hash", ""),
                "snapshot_hash_source": row.get("snapshot_hash_source", ""),
                "snapshot_hash_available": row.get("snapshot_hash_available", "false"),
            }
        )
    write_csv(out_csv, rows)


def _pct(vals: list[float], pct: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def _gain(base: float, new: float) -> float:
    return (base - new) / max(1e-9, base) if base > 0 else 0.0


def simulate_stp_ae(
    records: list[dict[str, Any]],
    predictor: ArtifactPredictor,
    out_csv: str | Path = "data/results/stp_ae_simulation_pr4_v12.csv",
    launch_out: str | Path = "data/results/stp_ae_launch_decisions_pr4_v12.csv",
    max_artifacts_per_step: int = 3,
) -> list[dict[str, Any]]:
    _train, val = _split_records(records)
    policies = ["no_stp", "stp_exact_v2", "stp_ae_top1", "stp_ae_top3_budgeted", "stp_ae_sandbox", "stp_class_upper_bound", "stp_oracle_upper_bound"]
    rows: list[dict[str, Any]] = []
    launch_rows: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {p: [] for p in policies}
    base_vals: list[float] = []
    counters: dict[str, Counter[str]] = {p: Counter() for p in policies}
    floats: dict[str, defaultdict[str, float]] = {p: defaultdict(float) for p in policies}
    for row in val:
        repo = str(row["repo_family"])
        snapshot = str(row["workspace_snapshot_hash"])
        step = int(float(row.get("step_id", 0) or 0))
        latency = float(row.get("tool_latency", 0.0))
        actual_cmd = str(row.get("command", ""))
        actual_required = command_artifacts(actual_cmd, repo, snapshot, step)
        actual_answerable = bool(actual_required)
        actual_class = str(row.get("command_class", ""))
        actual_key = str(row.get("canonical_tool_key", ""))
        pred_artifacts = predictor.predict(row, max_artifacts_per_step)
        exact_pred = predictor.predict_exact_key(row)
        class_pred = predictor.predict_class(row)
        for policy in policies:
            launched: list[Artifact] = []
            miss_reason = ""
            if policy == "stp_ae_top1":
                launched, miss_reason = _launch_artifacts(pred_artifacts[:1], {READ_ONLY_ARTIFACT}, latency, max_artifacts_per_step)
            elif policy == "stp_ae_top3_budgeted":
                launched, miss_reason = _launch_artifacts(pred_artifacts, {READ_ONLY_ARTIFACT}, latency, max_artifacts_per_step)
            elif policy == "stp_ae_sandbox":
                launched, miss_reason = _launch_artifacts(pred_artifacts, {SANDBOXED_ARTIFACT}, latency, 1)
            elif policy == "stp_oracle_upper_bound" and actual_answerable and latency >= 0.2:
                launched = actual_required
            for art, prob in pred_artifacts:
                if policy.startswith("stp_ae"):
                    expected_benefit = prob * latency
                    expected_cost = art.generation_latency + 0.003
                    launch_rows.append(
                        {
                            "event_id": row["event_id"],
                            "policy": policy,
                            "predicted_artifact": art.key,
                            "launched": str(any(a.key == art.key for a in launched)).lower(),
                            "reason": "launched" if any(a.key == art.key for a in launched) else (miss_reason or "not_selected"),
                            "expected_benefit": expected_benefit,
                            "expected_cost": expected_cost,
                            "safety_level": art.safety_level,
                            "snapshot_hash": snapshot,
                        }
                    )
            hit = False
            exact_hit = False
            class_hit = False
            if policy == "stp_exact_v2":
                exact_hit = bool(exact_pred and exact_pred == actual_key and actual_answerable)
                hit = exact_hit
            elif policy.startswith("stp_ae"):
                hit = command_answerable_by(actual_cmd, launched, repo, snapshot, step)
            elif policy == "stp_class_upper_bound":
                class_hit = bool(class_pred and class_pred == actual_class and actual_answerable)
                hit = class_hit
            elif policy == "stp_oracle_upper_bound":
                hit = actual_answerable and bool(launched)
            hidden = latency if hit else 0.0
            wasted = 0.0 if hit else sum(a.generation_latency for a in launched)
            contention = sum(a.generation_latency for a in launched) * 0.25
            storage = sum(a.content_bytes for a in launched)
            safety_violations = 0
            for art in launched:
                if policy in {"stp_ae_top1", "stp_ae_top3_budgeted"} and art.safety_level != READ_ONLY_ARTIFACT:
                    safety_violations += 1
                if policy == "stp_ae_sandbox" and art.safety_level != SANDBOXED_ARTIFACT:
                    safety_violations += 1
            if not actual_answerable:
                miss_reason = "unsafe"
            elif launched and not hit:
                miss_reason = "artifact_not_answerable"
            elif not launched and not miss_reason:
                miss_reason = "low_confidence"
            e2e = max(0.0, latency - hidden)
            values[policy].append(e2e)
            if policy == "no_stp":
                base_vals.append(latency)
            counters[policy]["events"] += 1
            counters[policy]["artifact_hits"] += int(hit and policy.startswith("stp_ae"))
            counters[policy]["exact_hits"] += int(exact_hit)
            counters[policy]["class_hits"] += int(class_hit)
            counters[policy]["safety_violations"] += safety_violations
            counters[policy][f"miss_{miss_reason or 'none'}"] += int(not hit)
            floats[policy]["tool_latency_hidden"] += hidden
            floats[policy]["wasted_speculative_work"] += wasted
            floats[policy]["worker_contention"] += contention
            floats[policy]["storage_overhead"] += storage
            floats[policy]["baseline_tool_time"] += latency
            rows.append(
                {
                    "row_type": "event",
                    "policy": policy,
                    "event_id": row["event_id"],
                    "actual_command_class": actual_class,
                    "actual_canonical_tool_key": actual_key,
                    "actual_artifacts": json.dumps([a.key for a in actual_required]),
                    "launched_artifacts": json.dumps([a.key for a in launched]),
                    "artifact_hit": str(hit and policy.startswith("stp_ae")).lower(),
                    "exact_command_hit": str(exact_hit).lower(),
                    "class_hit": str(class_hit).lower(),
                    "tool_latency": latency,
                    "tool_latency_hidden": hidden,
                    "wasted_speculative_work": wasted,
                    "worker_contention": contention,
                    "storage_overhead": storage,
                    "safety_violations": safety_violations,
                    "miss_reason": miss_reason,
                    "simulated_tool_jct": e2e,
                }
            )
    base_mean = sum(base_vals) / max(1, len(base_vals))
    base_p95 = _pct(base_vals, 95)
    summary: list[dict[str, Any]] = []
    for policy in policies:
        vals = values[policy]
        mean = sum(vals) / max(1, len(vals))
        p95 = _pct(vals, 95)
        c = counters[policy]
        f = floats[policy]
        p95_gain = _gain(base_p95, p95)
        mean_gain = _gain(base_mean, mean)
        gain = "NOT_OBSERVED"
        if policy in {"stp_class_upper_bound", "stp_oracle_upper_bound"}:
            gain = "UPPER_BOUND_ONLY"
        elif policy.startswith("stp_ae") and c["safety_violations"] == 0:
            if max(p95_gain, mean_gain) >= 0.05:
                gain = "OBSERVED"
            elif max(p95_gain, mean_gain) > 0:
                gain = "WEAK"
        miss = {k.removeprefix("miss_"): v for k, v in c.items() if k.startswith("miss_")}
        summary.append(
            {
                "row_type": "aggregate",
                "policy": policy,
                "events": len(vals),
                "mean_jct": mean,
                "p95_jct": p95,
                "tool_latency_hidden": f["tool_latency_hidden"],
                "artifact_hit_rate": c["artifact_hits"] / max(1, c["events"]),
                "exact_command_hit_rate": c["exact_hits"] / max(1, c["events"]),
                "class_hit_rate": c["class_hits"] / max(1, c["events"]),
                "wasted_speculative_work": f["wasted_speculative_work"],
                "worker_contention": f["worker_contention"],
                "storage_overhead": f["storage_overhead"],
                "cost_overhead": f["wasted_speculative_work"] / max(1e-9, f["baseline_tool_time"]),
                "safety_violations": c["safety_violations"],
                "p95_jct_gain": p95_gain,
                "mean_jct_gain": mean_gain,
                "artifact_miss_reasons": json.dumps(miss, sort_keys=True),
                "stp_ae_gain": gain,
            }
        )
    write_csv(out_csv, rows + summary)
    write_csv(launch_out, launch_rows)
    return rows + summary


def _launch_artifacts(
    predictions: list[tuple[Artifact, float]],
    allowed: set[str],
    actual_latency: float,
    max_artifacts: int,
) -> tuple[list[Artifact], str]:
    launched: list[Artifact] = []
    for art, prob in predictions:
        if len(launched) >= max_artifacts:
            return launched, "budget_exceeded"
        if art.safety_level not in allowed:
            continue
        threshold = 1.0 if art.safety_level == SANDBOXED_ARTIFACT else 0.2
        if actual_latency < threshold:
            continue
        expected_benefit = prob * actual_latency
        expected_cost = art.generation_latency + 0.003 + art.content_bytes / 1e9
        if expected_benefit > expected_cost:
            launched.append(art)
    return launched, "" if launched else "low_confidence"


def run_all(
    trace_dirs: list[str | Path] | None = None,
    predictor_out: str | Path = "data/results/artifact_predictor_pr4_v12.csv",
    simulation_out: str | Path = "data/results/stp_ae_simulation_pr4_v12.csv",
    launch_out: str | Path = "data/results/stp_ae_launch_decisions_pr4_v12.csv",
    safety_out: str | Path = "data/results/tool_safety_classification_pr4_v12.csv",
) -> dict[str, Any]:
    traces = _load_traces(trace_dirs)
    records = build_tool_records(traces)
    write_tool_safety_classification(records, safety_out)
    train, _val = _split_records(records)
    predictor = ArtifactPredictor()
    predictor.fit(train)
    pred_metrics = evaluate_artifact_predictor(records, predictor, predictor_out)
    sim_rows = simulate_stp_ae(records, predictor, simulation_out, launch_out)
    def agg(policy: str) -> dict[str, Any]:
        return next((r for r in sim_rows if r.get("row_type") == "aggregate" and r.get("policy") == policy), {})
    top1 = agg("stp_ae_top1")
    top3 = agg("stp_ae_top3_budgeted")
    sandbox = agg("stp_ae_sandbox")
    return {
        "tool_records": len(records),
        "artifact_top1_hit": pred_metrics.get("artifact_top1_hit", 0.0),
        "artifact_top3_hit": pred_metrics.get("artifact_top3_hit", 0.0),
        "artifact_coverage": pred_metrics.get("artifact_coverage", 0.0),
        "read_only_artifact_coverage": pred_metrics.get("read_only_artifact_coverage", 0.0),
        "sandbox_artifact_coverage": pred_metrics.get("sandbox_artifact_coverage", 0.0),
        "stp_ae_top1_p95_gain": top1.get("p95_jct_gain", 0.0),
        "stp_ae_top3_p95_gain": top3.get("p95_jct_gain", 0.0),
        "stp_ae_top3_mean_gain": top3.get("mean_jct_gain", 0.0),
        "stp_ae_sandbox_p95_gain": sandbox.get("p95_jct_gain", 0.0),
        "stp_ae_safety_violations": int(top1.get("safety_violations", 0) or 0) + int(top3.get("safety_violations", 0) or 0) + int(sandbox.get("safety_violations", 0) or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", action="append", dest="trace_dirs")
    ap.add_argument("--predictor-out", default="data/results/artifact_predictor_pr4_v12.csv")
    ap.add_argument("--simulation-out", default="data/results/stp_ae_simulation_pr4_v12.csv")
    ap.add_argument("--launch-out", default="data/results/stp_ae_launch_decisions_pr4_v12.csv")
    ap.add_argument("--safety-out", default="data/results/tool_safety_classification_pr4_v12.csv")
    args = ap.parse_args()
    print(json.dumps(run_all(args.trace_dirs, args.predictor_out, args.simulation_out, args.launch_out, args.safety_out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
