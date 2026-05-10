from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


def _norm_result(value: Any) -> str | None:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    text = str(value).strip().lower()
    if text in {"pass", "passed", "resolved", "true", "1"}:
        return "pass"
    if text in {"fail", "failed", "unresolved", "false", "0"}:
        return "fail"
    return None


def _load_csv(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            instance_id = row.get("instance_id") or row.get("instance") or row.get("id")
            if not instance_id:
                continue
            for key in ("resolved", "success", "verifier_result", "status", "result"):
                if key in row:
                    result = _norm_result(row.get(key))
                    if result:
                        rows[instance_id] = result
                        break
    return rows


def _load_json_obj(obj: Any) -> dict[str, str]:
    rows: dict[str, str] = {}
    if isinstance(obj, dict):
        for key in ("resolved_ids", "resolved"):
            value = obj.get(key)
            if isinstance(value, list):
                rows.update({str(x): "pass" for x in value})
        for key in ("unresolved_ids", "unresolved", "failed_ids", "failed"):
            value = obj.get(key)
            if isinstance(value, list):
                rows.update({str(x): "fail" for x in value})
        if "instance_id" in obj:
            result = None
            for key in ("resolved", "success", "verifier_result", "status", "result"):
                if key in obj:
                    result = _norm_result(obj[key])
                    if result:
                        break
            if result:
                rows[str(obj["instance_id"])] = result
        for value in obj.values():
            rows.update(_load_json_obj(value))
    elif isinstance(obj, list):
        for item in obj:
            rows.update(_load_json_obj(item))
    return rows


def load_verifier_results(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    if p.suffix.lower() == ".csv":
        return _load_csv(p)
    if p.suffix.lower() == ".jsonl":
        rows: dict[str, str] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.update(_load_json_obj(json.loads(line)))
        return rows
    try:
        return _load_json_obj(json.loads(p.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return {}


def merge_verifier_results(trace_dir: str | Path, results_path: str | Path, out_dir: str | Path, summary_out: str | Path) -> list[dict[str, Any]]:
    verdicts = load_verifier_results(results_path)
    outp = ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []
    for trace in load_trace_dir(trace_dir):
        instance_id = str(trace.metadata.get("instance_id") or (trace.events[0].instance_id if trace.events else ""))
        verdict = verdicts.get(instance_id)
        events: list[Event] = []
        for ev in trace.events:
            row = asdict(ev)
            if ev.node_type == "verifier":
                if verdict == "pass":
                    row["verifier_result"] = "pass"
                    row["success"] = True
                elif verdict == "fail":
                    row["verifier_result"] = "fail"
                    row["success"] = False
                else:
                    row["verifier_result"] = ev.verifier_result or "unknown"
                    row["success"] = ev.success
            events.append(Event(**row))
        rollout_id = str(trace.metadata.get("rollout_id") or "rollout_0")
        out_path = outp / f"{instance_id}_{rollout_id}.jsonl"
        Trace({**trace.metadata, "official_verifier_merged": bool(verdict), "official_verifier_result": verdict or "unknown"}, events).to_jsonl(out_path)
        rows.append(
            {
                "instance_id": instance_id,
                "official_verifier_merged": str(bool(verdict)).lower(),
                "verifier_result": verdict or "unknown",
                "num_verifier_events": sum(1 for e in events if e.node_type == "verifier"),
            }
        )
    write_csv(summary_out, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--official-results", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()
    rows = merge_verifier_results(args.trace_dir, args.official_results, args.out_dir, args.summary_out)
    merged = sum(1 for r in rows if r["official_verifier_merged"] == "true")
    print(json.dumps({"merged_instances": merged, "out_dir": args.out_dir, "summary": args.summary_out}, indent=2))


if __name__ == "__main__":
    main()
