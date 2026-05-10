from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _first_text(obj: Any, key: str) -> str:
    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
        for nested in obj.values():
            found = _first_text(nested, key)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _first_text(item, key)
            if found:
                return found
    return ""


def _rollout_id(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("rollout_"):
            return part.replace(".traj.json", "").replace(".json", "")
    return "rollout_0"


def summarize_patch_capture(traj_root: str | Path, out: str | Path) -> list[dict[str, Any]]:
    root = Path(traj_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".traj") or path.name.endswith(".traj.json") or path.suffix == ".json"):
            continue
        if any("_mini_extra_" in part for part in path.parts):
            continue
        obj = _load_json(path)
        info = obj.get("info") if isinstance(obj.get("info"), dict) else {}
        rel = path.relative_to(root)
        instance_id = str(info.get("instance_id") or _first_text(obj, "instance_id") or (rel.parts[0] if len(rel.parts) > 1 else path.stem))
        patch = str(info.get("agentweaver_patch") or "")
        rows.append(
            {
                "instance_id": instance_id,
                "rollout_id": str(info.get("rollout_id") or _rollout_id(path)),
                "trajectory": str(path),
                "workspace_detected": str(bool(info.get("agentweaver_patch_workspace_detected"))).lower(),
                "workspace": info.get("agentweaver_patch_workspace", ""),
                "git_diff_exit_code": info.get("agentweaver_patch_git_diff_exit_code", ""),
                "patch_bytes": info.get("agentweaver_patch_bytes", len(patch.encode("utf-8")) if patch else 0),
                "patch_empty": str(bool(info.get("agentweaver_patch_empty", not bool(patch)))).lower(),
                "patch_empty_reason": info.get("agentweaver_patch_empty_reason", ""),
                "patch_capture_error": info.get("agentweaver_patch_capture_error", ""),
            }
        )
    fields = [
        "instance_id",
        "rollout_id",
        "trajectory",
        "workspace_detected",
        "workspace",
        "git_diff_exit_code",
        "patch_bytes",
        "patch_empty",
        "patch_empty_reason",
        "patch_capture_error",
    ]
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-root", required=True)
    ap.add_argument("--out", default="data/results/patch_capture_debug_pr3_v4.csv")
    args = ap.parse_args()
    rows = summarize_patch_capture(args.traj_root, args.out)
    errors = sum(1 for r in rows if r.get("patch_capture_error"))
    empty = sum(1 for r in rows if r.get("patch_empty") == "true")
    print(json.dumps({"rows": len(rows), "patch_capture_errors": errors, "patch_empty": empty, "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
