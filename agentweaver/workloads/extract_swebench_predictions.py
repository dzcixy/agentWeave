from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def _first_text(obj: Any, keys: tuple[str, ...]) -> str:
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                nested = _first_text(value, keys)
                if nested:
                    return nested
        for value in obj.values():
            nested = _first_text(value, keys)
            if nested:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _first_text(item, keys)
            if nested:
                return nested
    return ""


def _steps(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        for key in ("trajectory", "steps", "history", "turns", "events"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
    if isinstance(obj, list):
        return obj
    return []


def _patch(obj: Any) -> str:
    patch = _first_text(obj, ("patch", "diff", "final_patch", "model_patch"))
    if patch:
        return patch
    for step in _steps(obj):
        patch = _first_text(step, ("patch", "diff", "final_patch", "model_patch"))
        if patch:
            return patch
    return ""


def _instance_id(path: Path, obj: Any, root: Path) -> str:
    value = _first_text(obj, ("instance_id",))
    if value:
        return value
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return path.stem.replace(".traj", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name-or-path", default="qwen-coder-7b")
    args = ap.parse_args()

    root = Path(args.traj_root)
    out = Path(args.out)
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".traj") or path.name.endswith(".traj.json") or path.suffix == ".json"):
            continue
        if "_mini_extra_" in path.parts:
            continue
        obj = _load_json(path)
        patch = _patch(obj)
        if not patch:
            continue
        rows.append(
            {
                "instance_id": _instance_id(path, obj, root),
                "model_name_or_path": args.model_name_or_path,
                "model_patch": patch,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"predictions": len(rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
