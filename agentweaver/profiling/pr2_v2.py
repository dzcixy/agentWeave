from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from agentweaver.utils.io import ensure_dir, write_csv


REQUIRED_REAL_POLICIES = [
    "naive_wafer",
    "static_branch_pinning",
    "wafer_fcfs",
    "acd_only",
    "acd_bes",
    "acd_nisp",
    "full_agentweaver",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_union_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, rows, fields)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _num(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _ok(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).startswith("ok")


def combine_profiles(paths: list[str | Path], out: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if p.exists():
            rows.extend(_read_csv(p))
    _write_union_csv(out, rows)
    return rows


def _case_key(row: dict[str, Any]) -> tuple[int, int, int]:
    shared = int(_float(row, "shared_prefix_tokens_actual", _float(row, "shared_prefix_tokens_target")))
    suffix = int(_float(row, "unique_suffix_tokens_actual", _float(row, "unique_suffix_tokens_target")))
    num = int(_float(row, "num_requests"))
    return shared, suffix, num


def prefix_reuse_effect(noprefix_csv: str | Path, prefix_csv: str | Path, out: str | Path) -> list[dict[str, Any]]:
    noprefix_rows = [r for r in _read_csv(noprefix_csv) if _ok(r)]
    prefix_rows = [r for r in _read_csv(prefix_csv) if _ok(r)]
    global_prefix_reliable, _, _ = _prefix_metrics_status(prefix_rows)
    by_np: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    by_pf: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in noprefix_rows:
        by_np[_case_key(row)].append(row)
    for row in prefix_rows:
        by_pf[_case_key(row)].append(row)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(by_np) & set(by_pf)):
        shared, suffix, num = key
        np_lat = [_float(r, "per_request_latency_mean", math.nan) for r in by_np[key]]
        pf_lat = [_float(r, "per_request_latency_mean", math.nan) for r in by_pf[key]]
        np_lat = [x for x in np_lat if not math.isnan(x) and x > 0]
        pf_lat = [x for x in pf_lat if not math.isnan(x) and x > 0]
        if not np_lat or not pf_lat:
            continue
        prefix_reliable = global_prefix_reliable and any(
            str(r.get("prefix_cache_metrics_reliable", "")).lower() == "true" for r in by_pf[key]
        )
        reliable_cache_hit_tokens = ""
        if prefix_reliable:
            vals = [_num(r.get("cached_prompt_tokens_delta"), math.nan) for r in by_pf[key]]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                reliable_cache_hit_tokens = sum(vals) / len(vals)
        client_repeated = max((_float(r, "client_repeated_prefix_tokens") for r in by_pf[key]), default=shared * max(0, num - 1))
        np_mean = sum(np_lat) / len(np_lat)
        pf_mean = sum(pf_lat) / len(pf_lat)
        reduction = (np_mean - pf_mean) / np_mean if np_mean > 0 else 0.0
        rows.append(
            {
                "shared_prefix_tokens": shared,
                "unique_suffix_tokens": suffix,
                "num_requests": num,
                "noprefix_latency_mean": np_mean,
                "prefix_latency_mean": pf_mean,
                "latency_reduction": reduction,
                "client_repeated_prefix_tokens": int(client_repeated),
                "reliable_cache_hit_tokens": reliable_cache_hit_tokens,
                "prefix_metrics_reliable": str(prefix_reliable).lower(),
                "prefix_cache_counters_used_as_evidence": str(prefix_reliable).lower(),
            }
        )
    write_csv(out, rows)
    return rows


def real_policy_comparison(all_policies_csv: str | Path, out: str | Path) -> list[dict[str, Any]]:
    rows = _read_csv(all_policies_csv)
    aggregate = {r["policy"]: r for r in rows if r.get("instance_id") == "AGGREGATE"}
    baseline = aggregate.get("naive_wafer", {})
    metrics = [
        "jct",
        "time_to_first_success",
        "prefill_tokens_avoided",
        "branch_wasted_tokens",
        "resume_prefill_tokens",
        "region_utilization",
    ]
    out_rows: list[dict[str, Any]] = []
    for policy in REQUIRED_REAL_POLICIES:
        row = aggregate.get(policy)
        if not row:
            out_rows.append({"policy": policy, "status": "missing"})
            continue
        out_row: dict[str, Any] = {"policy": policy, "status": "ok"}
        for metric in metrics:
            value = _float(row, metric)
            base = _float(baseline, metric, math.nan) if baseline else math.nan
            out_row[metric] = value
            if not math.isnan(base) and base != 0:
                out_row[f"{metric}_delta_vs_naive"] = value - base
                out_row[f"{metric}_ratio_vs_naive"] = value / base
            else:
                out_row[f"{metric}_delta_vs_naive"] = ""
                out_row[f"{metric}_ratio_vs_naive"] = ""
        out_rows.append(out_row)
    write_csv(out, out_rows)
    return out_rows


def _parse_fit_report(path: str | Path) -> dict[str, str]:
    p = Path(path)
    fields: dict[str, str] = {}
    if not p.exists():
        return fields
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        line = line.strip().lstrip("-").strip()
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def _parse_key_value_report(path: str | Path) -> dict[str, str]:
    p = Path(path)
    fields: dict[str, str] = {}
    if not p.exists():
        return fields
    for line in p.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            fields[key.strip()] = value.strip()
    return fields


def _pr1_gate() -> str:
    for path in (Path("data/results/pr1_final_report.md"), Path("data/results/pr1_report.md")):
        if path.exists() and "PR1_GATE = PASS" in path.read_text(encoding="utf-8"):
            return "PASS"
    return "FAIL"


def _model_family(model_path: str, model_name: str) -> str:
    text = f"{model_path} {model_name}".lower()
    if "coder" in text:
        return "coder"
    if "qwen2.5-7b" in text or "qwen2.5" in text:
        return "general"
    return "unknown"


def _ttft_coverage(length_rows: list[dict[str, str]]) -> float:
    ok = [r for r in length_rows if _ok(r)]
    with_ttft = [r for r in ok if _float(r, "ttft", math.nan) > 0]
    return len(with_ttft) / max(1, len(ok))


def _output_length_status(length_rows: list[dict[str, str]]) -> tuple[str, str]:
    ok = [r for r in length_rows if _ok(r)]
    if not ok:
        return "FAIL", "no successful length rows"
    notes: list[str] = []
    status = "PASS"
    uncontrolled = any(str(r.get("status", "")).startswith("ok_uncontrolled") for r in ok)
    if uncontrolled:
        status = "WARNING"
        notes.append("endpoint rejected output-control extension and profiling used uncontrolled fallback")
    for target in (32, 128, 512):
        rows = [r for r in ok if int(_float(r, "output_tokens_target", -1)) == target]
        good = [r for r in rows if _float(r, "output_tokens_actual") >= 0.8 * target]
        rate = len(good) / max(1, len(rows))
        notes.append(f"target_{target}_actual_ge_0.8_target_rate={rate:.3f}")
        if rows and rate < 0.90:
            status = "FAIL"
    rows_1024 = [r for r in ok if int(_float(r, "output_tokens_target", -1)) == 1024]
    if rows_1024:
        rate = sum(1 for r in rows_1024 if _float(r, "output_tokens_actual") >= 0.8 * 1024) / len(rows_1024)
        notes.append(f"target_1024_actual_ge_0.8_target_rate={rate:.3f}")
        if rate < 0.90 and status == "PASS":
            status = "WARNING"
    return status, "; ".join(notes)


def _profile_isolation_status(rows: list[dict[str, str]]) -> tuple[str, str]:
    checked = [row for row in rows if row.get("mode") in {"length", "concurrency"}]
    if not checked:
        return "FAIL", "no no-prefix length/concurrency rows available"
    contaminated: list[str] = []
    for row in checked:
        for key in ("cached_prompt_tokens_delta", "prefix_cache_hits_delta"):
            val = _num(row.get(key), math.nan)
            if not math.isnan(val) and val > 0:
                contaminated.append(f"{row.get('mode')}:{row.get('request_id', row.get('repeat'))}:{key}={val}")
    if contaminated:
        return "FAIL", "; ".join(contaminated[:10])
    return "PASS", "no reliable no-prefix cache deltas observed"


def _prefix_metrics_status(prefix_rows: list[dict[str, str]]) -> tuple[bool, str, str]:
    reliable = any(str(r.get("prefix_cache_metrics_reliable", "")).lower() == "true" for r in prefix_rows)
    if not reliable:
        return False, "WARNING", "strict parser did not find reliable prefix cache counters"
    offenders: list[str] = []
    max_per_request = 0.0
    for row in prefix_rows:
        shared = int(_float(row, "shared_prefix_tokens_actual", _float(row, "shared_prefix_tokens_target")))
        if shared != 0:
            continue
        hits = _num(row.get("prefix_cache_hits_delta"), 0.0)
        cached = _num(row.get("cached_prompt_tokens_delta"), 0.0)
        if hits > 0 or cached > 0:
            prompt = max(_num(row.get("prompt_tokens_delta"), math.nan), _num(row.get("prompt_tokens"), math.nan), 1.0)
            num_requests = max(_num(row.get("num_requests"), 1.0), 1.0)
            ratio = max(hits, cached) / max(prompt, 1.0)
            per_request = max(hits, cached) / num_requests
            max_per_request = max(max_per_request, per_request)
            offenders.append(
                f"shared=0 hits={hits} cached={cached} ratio={ratio:.6f} per_request={per_request:.3f}"
            )
    if offenders:
        note = (
            "; ".join(offenders[:10])
            + "; treating vLLM prefix cache counters as unreliable for evidence because "
            "zero client-shared-prefix prompts still report small block/template-level cache deltas; "
            "prefix reuse benefit is evaluated by no-prefix vs prefix latency comparison"
        )
        if max_per_request <= 64.0:
            return False, "WARNING", note
        return False, "FAIL", note
    return reliable, "PASS", "shared_prefix_tokens=0 has no reliable cache-hit delta"


def _prefix_latency_benefit(effect_rows: list[dict[str, str]]) -> tuple[str, str]:
    positive_cases = [
        _float(r, "latency_reduction", math.nan)
        for r in effect_rows
        if _float(r, "shared_prefix_tokens", 0) > 0 and not math.isnan(_float(r, "latency_reduction", math.nan))
    ]
    if not positive_cases:
        return "FAIL", "no comparable no-prefix vs prefix cases"
    good = [x for x in positive_cases if x > 0]
    if len(good) == len(positive_cases):
        return "PASS", f"{len(good)}/{len(positive_cases)} shared-prefix cases faster on prefix server"
    if good:
        return "WARNING", f"{len(good)}/{len(positive_cases)} shared-prefix cases faster on prefix server"
    return "WARNING", "prefix-enabled server was not faster for shared-prefix cases"


def _real_all_policies_status(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        return "FAIL", "missing real agent-like policy CSV"
    by_instance: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_instance[row.get("instance_id", "")].add(row.get("policy", ""))
    required = set(REQUIRED_REAL_POLICIES)
    missing: list[str] = []
    for instance, policies in sorted(by_instance.items()):
        if instance and not required.issubset(policies):
            missing.append(f"{instance}:{sorted(required - policies)}")
    if "AGGREGATE" not in by_instance:
        missing.append("AGGREGATE row missing")
    if missing:
        return "FAIL", "; ".join(missing[:10])
    return "PASS", f"{len(by_instance) - 1} instances plus aggregate include all policies"


def _full_agentweaver_benefit_note(rows: list[dict[str, str]]) -> str:
    aggregate = {r.get("policy", ""): r for r in rows if r.get("instance_id") == "AGGREGATE"}
    naive = aggregate.get("naive_wafer")
    full = aggregate.get("full_agentweaver")
    if not naive or not full:
        return "missing naive_wafer or full_agentweaver aggregate"
    naive_jct = _float(naive, "jct", math.nan)
    full_jct = _float(full, "jct", math.nan)
    if math.isnan(naive_jct) or math.isnan(full_jct) or naive_jct <= 0:
        return "insufficient aggregate JCT data"
    benefit = (naive_jct - full_jct) / naive_jct
    if benefit > 0:
        return f"full_agentweaver_jct_reduction_vs_naive = {benefit:.6f}"
    return (
        f"full_agentweaver_jct_reduction_vs_naive = {benefit:.6f}; "
        "no positive JCT benefit observed, report mechanism counters instead of claiming a speedup"
    )


def generate_report(args: argparse.Namespace) -> dict[str, str]:
    existing = _parse_key_value_report(args.out)
    length_rows = _read_csv(args.length_csv)
    concurrency_rows = _read_csv(args.concurrency_csv)
    prefix_rows = _read_csv(args.prefix_csv)
    raw_rows = length_rows + concurrency_rows + prefix_rows
    effect_rows = _read_csv(args.prefix_effect_csv)
    real_rows = _read_csv(args.real_all_policies_csv)
    fit = _parse_fit_report(args.fit_report_md)
    model: dict[str, Any] = {}
    if Path(args.model_json).exists():
        model = json.loads(Path(args.model_json).read_text(encoding="utf-8"))

    length_ok = sum(1 for r in length_rows if _ok(r))
    conc_ok = sum(1 for r in concurrency_rows if _ok(r))
    prefix_ok = sum(1 for r in prefix_rows if _ok(r))
    max_conc = max([int(_float(r, "concurrency")) for r in concurrency_rows if _ok(r)] or [0])
    ttft = _ttft_coverage(length_rows)
    output_status, output_note = _output_length_status(length_rows)
    isolation, isolation_note = _profile_isolation_status(length_rows + concurrency_rows)
    if args.single_server_debug:
        isolation = "FAIL_SINGLE_SERVER_DEBUG"
        isolation_note = "single-server debug mode cannot isolate no-prefix and prefix-cache effects"
    prefix_reliable, prefix_sanity, prefix_sanity_note = _prefix_metrics_status(prefix_rows)
    prefix_benefit, prefix_benefit_note = _prefix_latency_benefit(effect_rows)
    real_status, real_note = _real_all_policies_status(real_rows)

    median = fit.get("median_absolute_percentage_error", "")
    p95 = fit.get("p95_absolute_percentage_error", "")
    median_f = _num(median, math.nan)
    p95_f = _num(p95, math.nan)
    latency_quality = "PASS" if not math.isnan(median_f) and not math.isnan(p95_f) and median_f < 0.15 and p95_f < 0.25 and ttft >= 0.95 else "WARNING"
    if not length_ok:
        latency_quality = "FAIL"
    h100_profile = "PASS" if length_ok > 0 and conc_ok > 0 and prefix_ok > 0 and not args.single_server_debug else "FAIL"
    if args.single_server_debug and length_ok and conc_ok and prefix_ok:
        h100_profile = "WARNING"
    model_path = args.model_path or existing.get("MODEL_PATH", "")
    tokenizer_path = args.tokenizer_path or existing.get("TOKENIZER_PATH", "")
    noprefix_server = args.noprefix_server or existing.get("NO_PREFIX_SERVER", "")
    prefix_server = args.prefix_server or existing.get("PREFIX_SERVER", "")
    model_name = args.model or existing.get("MODEL_NAME", "")
    model_family = _model_family(model_path, model_name)
    blockers = []
    if h100_profile != "PASS":
        blockers.append("complete isolated no-prefix and prefix H100 profile")
    if latency_quality != "PASS":
        blockers.append("latency model does not satisfy median<15% and p95<25% with TTFT coverage>=95%")
    if isolation != "PASS":
        blockers.append("profile isolation failed or single-server debug was used")
    if real_status != "PASS":
        blockers.append("real agent-like replay does not include all policies")
    if model_family != "coder":
        blockers.append("not final paper Coder model")
    if not blockers:
        blockers.append("SWE-agent/SWE-bench intentionally not started in PR2-v2")

    fields: dict[str, str] = {
        "PR1_GATE": _pr1_gate(),
        "OLD_PR2_V1_STATUS": "INVALID_FOR_PAPER",
        "OLD_PR2_V1_INVALID_REASON": (
            "PR2-v1 used high-error latency fit, non-streaming TTFT gaps, target output tokens, "
            "fuzzy metrics, possible prefix-cache contamination, and only full_agentweaver real replay."
        ),
        "H100_PROFILE": h100_profile,
        "LATENCY_MODEL_QUALITY": latency_quality,
        "PROFILE_ISOLATION": isolation,
        "TTFT_COVERAGE": f"{ttft:.6f}",
        "OUTPUT_LENGTH_CONTROL": output_status,
        "PREFIX_CACHE_METRICS_RELIABLE": str(prefix_reliable).lower(),
        "PREFIX_METRICS_SANITY": prefix_sanity,
        "PREFIX_REUSE_LATENCY_BENEFIT": prefix_benefit,
        "REAL_AGENTLIKE_ALL_POLICIES": real_status,
        "MODEL_PATH": model_path,
        "MODEL_FAMILY": model_family,
        "TOKENIZER_PATH": tokenizer_path,
        "NO_PREFIX_SERVER": noprefix_server,
        "PREFIX_SERVER": prefix_server,
        "LENGTH_SWEEP_SUCCESS_CASES": str(length_ok),
        "CONCURRENCY_SWEEP_SUCCESS_CASES": str(conc_ok),
        "PREFIX_SWEEP_SUCCESS_CASES": str(prefix_ok),
        "MAX_SUCCESSFUL_CONCURRENCY": str(max_conc),
        "MEDIAN_LATENCY_MODEL_ERROR": median,
        "P95_LATENCY_MODEL_ERROR": p95,
        "LATENCY_MODEL_MODE": str(model.get("mode", fit.get("LATENCY_MODEL_MODE", ""))),
        "REMAINING_BLOCKERS_FOR_SWE_AGENT": "; ".join(blockers),
    }
    notes = {
        "PROFILE_ISOLATION_NOTE": isolation_note,
        "OUTPUT_LENGTH_CONTROL_NOTE": output_note,
        "PREFIX_METRICS_SANITY_NOTE": prefix_sanity_note,
        "PREFIX_REUSE_LATENCY_BENEFIT_NOTE": prefix_benefit_note,
        "REAL_AGENTLIKE_ALL_POLICIES_NOTE": real_note,
        "REAL_AGENTLIKE_FULL_AGENTWEAVER_BENEFIT_NOTE": _full_agentweaver_benefit_note(real_rows),
        "PROFILE_ROWS_TOTAL": str(len(raw_rows)),
    }
    path = Path(args.out)
    ensure_dir(path.parent)
    lines = ["# PR2-v2 Report", ""]
    lines.extend(f"{key} = {value}" for key, value in fields.items())
    lines.append("")
    lines.append("## Notes")
    lines.extend(f"{key} = {value}" for key, value in notes.items())
    if model_family != "coder":
        lines.append("MODEL_WARNING = not final paper model")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_combine = sub.add_parser("combine-profiles")
    p_combine.add_argument("--out", required=True)
    p_combine.add_argument("paths", nargs="+")

    p_prefix = sub.add_parser("prefix-effect")
    p_prefix.add_argument("--noprefix-csv", required=True)
    p_prefix.add_argument("--prefix-csv", required=True)
    p_prefix.add_argument("--out", required=True)

    p_real = sub.add_parser("real-comparison")
    p_real.add_argument("--all-policies-csv", required=True)
    p_real.add_argument("--out", required=True)

    p_report = sub.add_parser("report")
    p_report.add_argument("--out", default="data/results/pr2_v2_report.md")
    p_report.add_argument("--length-csv", default="data/profiles/h100_profile_length_pr2_v2.csv")
    p_report.add_argument("--concurrency-csv", default="data/profiles/h100_profile_concurrency_pr2_v2.csv")
    p_report.add_argument("--prefix-csv", default="data/profiles/h100_profile_prefix_pr2_v2.csv")
    p_report.add_argument("--prefix-effect-csv", default="data/results/prefix_reuse_effect_pr2_v2.csv")
    p_report.add_argument("--fit-report-md", default="data/results/h100_latency_fit_report_pr2_v2.md")
    p_report.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    p_report.add_argument("--real-all-policies-csv", default="data/results/real_agentlike_replay_all_policies_pr2_v2.csv")
    p_report.add_argument("--model", default="qwen-coder-7b")
    p_report.add_argument("--model-path", default="")
    p_report.add_argument("--tokenizer-path", default="")
    p_report.add_argument("--noprefix-server", default="")
    p_report.add_argument("--prefix-server", default="")
    p_report.add_argument("--single-server-debug", action="store_true")

    args = ap.parse_args()
    if args.cmd == "combine-profiles":
        rows = combine_profiles(args.paths, args.out)
        print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))
    elif args.cmd == "prefix-effect":
        rows = prefix_reuse_effect(args.noprefix_csv, args.prefix_csv, args.out)
        print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))
    elif args.cmd == "real-comparison":
        rows = real_policy_comparison(args.all_policies_csv, args.out)
        print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))
    elif args.cmd == "report":
        fields = generate_report(args)
        print(json.dumps(fields, indent=2))


if __name__ == "__main__":
    main()
