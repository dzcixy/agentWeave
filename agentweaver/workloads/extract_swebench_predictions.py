from __future__ import annotations

import argparse
import csv
import json
import re
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


def _candidate_texts(obj: Any, keys: tuple[str, ...], prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{prefix}.{key}" if prefix else key
            if key in keys and isinstance(value, str) and value.strip():
                rows.append((here, value))
            if isinstance(value, (dict, list)):
                rows.extend(_candidate_texts(value, keys, here))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            rows.extend(_candidate_texts(item, keys, f"{prefix}[{i}]"))
    return rows


def _steps(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        for key in ("trajectory", "steps", "history", "turns", "events"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
    if isinstance(obj, list):
        return obj
    return []


def _messages(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("messages"), list):
            rows.extend([m for m in obj["messages"] if isinstance(m, dict)])
        for value in obj.values():
            if isinstance(value, (dict, list)):
                rows.extend(_messages(value))
    elif isinstance(obj, list):
        for item in obj:
            rows.extend(_messages(item))
    if rows:
        return rows
    return []


def _is_unified_diff(text: str) -> bool:
    return bool(
        text
        and (
            "diff --git " in text
            or (re.search(r"(?m)^---\s+(?:a/|\S)", text) and re.search(r"(?m)^\+\+\+\s+(?:b/|\S)", text) and re.search(r"(?m)^@@", text))
        )
    )


def _extract_diff_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for m in re.finditer(r"```(?:diff|patch)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        block = m.group(1).strip()
        if _is_unified_diff(block):
            blocks.append(block)
    if _is_unified_diff(text):
        start = text.find("diff --git ")
        if start >= 0:
            blocks.append(text[start:].strip())
        else:
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("--- "):
                    candidate = "\n".join(lines[i:]).strip()
                    if _is_unified_diff(candidate):
                        blocks.append(candidate)
                        break
    return blocks


def _patch(obj: Any) -> tuple[str, str, str]:
    if isinstance(obj, dict):
        info = obj.get("info") if isinstance(obj.get("info"), dict) else {}
        submission = info.get("submission") if isinstance(info, dict) else ""
        if isinstance(submission, str) and submission.strip():
            return submission.strip(), "info.submission", "top_level_submission"
    for key in ("final_patch", "model_patch", "patch", "diff"):
        if isinstance(obj, dict) and isinstance(obj.get(key), str) and obj[key].strip():
            return obj[key].strip(), key, "top_level_patch_field"
    for path, text in _candidate_texts(obj, ("final_patch", "model_patch", "patch", "diff")):
        if text.strip():
            return text.strip(), path, "nested_patch_field"
    for msg in reversed(_messages(obj)):
        if msg.get("role") != "assistant":
            continue
        for block in _extract_diff_blocks(str(msg.get("content") or "")):
            return block, "messages.assistant.diff_block", "assistant_unified_diff"
        extra = msg.get("extra")
        if isinstance(extra, dict):
            response = extra.get("response")
            if isinstance(response, dict):
                for block in _extract_diff_blocks(json.dumps(response, ensure_ascii=False)):
                    return block, "messages.assistant.extra.response", "assistant_response_unified_diff"
    for step in _steps(obj):
        for path, text in _candidate_texts(step, ("patch_file", "patch_path", "diff_file", "diff_path")):
            p = Path(text)
            if p.exists() and p.is_file():
                content = p.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    return content.strip(), path, "tool_recorded_patch_file"
    return "", "", "no_patch_found"


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
    ap.add_argument("--report")
    args = ap.parse_args()

    root = Path(args.traj_root)
    out = Path(args.out)
    report = Path(args.report) if args.report else out.with_name(out.stem.replace("_predictions", "") + "_patch_extraction_report.csv")
    rows: list[dict[str, str]] = []
    report_rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".traj") or path.name.endswith(".traj.json") or path.suffix == ".json"):
            continue
        if any("_mini_extra_" in part for part in path.parts):
            continue
        obj = _load_json(path)
        instance_id = _instance_id(path, obj, root)
        patch, source, source_type = _patch(obj)
        if not patch:
            report_rows.append(
                {
                    "instance_id": instance_id,
                    "trajectory": str(path),
                    "patch_extracted": "false",
                    "patch_source": source,
                    "patch_source_type": source_type,
                    "patch_bytes": "0",
                    "reason": "no non-empty submission/final_patch/patch/diff/unified-diff block found",
                }
            )
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": args.model_name_or_path,
                "model_patch": patch,
            }
        )
        report_rows.append(
            {
                "instance_id": instance_id,
                "trajectory": str(path),
                "patch_extracted": "true",
                "patch_source": source,
                "patch_source_type": source_type,
                "patch_bytes": str(len(patch.encode("utf-8"))),
                "reason": "",
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="") as f:
        fields = ["instance_id", "trajectory", "patch_extracted", "patch_source", "patch_source_type", "patch_bytes", "reason"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)
    status = "PASS" if rows else "FAIL_NO_PATCH"
    print(json.dumps({"PATCH_EXTRACTION": status, "predictions": len(rows), "out": str(out), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
