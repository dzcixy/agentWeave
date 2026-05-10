from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.utils.io import ensure_dir, write_csv


PLACEHOLDER = LatencyModel()


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def _ok(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).startswith("ok")


def _read_rows(raw: str | Path) -> list[dict[str, Any]]:
    with Path(raw).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _length_training(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    out = []
    for r in rows:
        if r.get("mode") != "length" or not _ok(r):
            continue
        inp = _float(r, "input_tokens_actual", _float(r, "client_prompt_tokens"))
        out_target = _float(r, "output_tokens_target")
        out_actual = _float(r, "output_tokens_actual", _float(r, "completion_tokens_client", out_target))
        e2e = _float(r, "e2e_latency")
        if inp <= 0 or e2e <= 0:
            continue
        ttft = _float(r, "ttft", math.nan)
        if math.isnan(ttft) or ttft <= 0 or ttft > e2e:
            prefill = e2e * inp / max(1.0, inp + out_actual)
            decode = e2e - prefill
        else:
            prefill = ttft
            decode = max(0.0, e2e - ttft)
        out.append({"input": inp, "output": max(1.0, out_actual or out_target), "e2e": e2e, "prefill": prefill, "decode": decode})
    return out


def _concurrency_factors(rows: list[dict[str, Any]], base_model: LatencyModel) -> dict[str, float]:
    by_c: dict[int, list[float]] = {}
    for r in rows:
        if r.get("mode") != "concurrency" or not _ok(r):
            continue
        c = int(_float(r, "concurrency", 1))
        inp = _float(r, "input_tokens_actual", _float(r, "input_tokens_target"))
        out = _float(r, "output_tokens_target")
        measured = _float(r, "per_request_latency_mean")
        pred = base_model.predict_prefill(inp) + base_model.predict_decode(inp, out)
        if measured > 0 and pred > 0:
            by_c.setdefault(c, []).append(measured / pred)
    return {str(k): float(np.median(v)) for k, v in by_c.items() if v}


def _errors(rows: list[dict[str, float]], model: LatencyModel) -> list[dict[str, float]]:
    errs = []
    for r in rows:
        pred = model.predict_llm(int(r["input"]), int(r["output"]))
        actual = r["e2e"]
        if actual > 0:
            errs.append(
                {
                    "input_tokens": r["input"],
                    "output_tokens": r["output"],
                    "actual": actual,
                    "predicted": pred,
                    "ape": abs(pred - actual) / actual,
                }
            )
    return errs


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    idx = min(len(xs) - 1, max(0, int(round((p / 100) * (len(xs) - 1)))))
    return xs[idx]


def _plot_fit(train: list[dict[str, float]], model: LatencyModel, plot_dir: Path, prefix_rows: list[dict[str, Any]], conc_rows: list[dict[str, Any]]) -> None:
    ensure_dir(plot_dir)
    if train:
        xs = np.array([r["input"] for r in train])
        ys = np.array([r["prefill"] for r in train])
        plt.figure()
        plt.scatter(xs, ys, s=12, label="measured")
        grid = np.linspace(xs.min(), xs.max(), 100)
        plt.plot(grid, [model.predict_prefill(x) for x in grid], label="fit")
        plt.xlabel("input tokens")
        plt.ylabel("prefill/TTFT proxy (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "profile_fit_prefill.pdf")
        plt.close()

        ys = np.array([r["decode"] for r in train])
        plt.figure()
        plt.scatter(xs, ys, s=12, label="measured")
        plt.plot(grid, [model.predict_decode(x, 512) for x in grid], label="fit@512 output")
        plt.xlabel("context tokens")
        plt.ylabel("decode proxy (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "profile_fit_decode.pdf")
        plt.close()

    plt.figure()
    if prefix_rows:
        x = [_float(r, "shared_prefix_tokens_actual", _float(r, "shared_prefix_tokens_target")) for r in prefix_rows if _ok(r)]
        y = [_float(r, "per_request_latency_mean") for r in prefix_rows if _ok(r)]
        plt.scatter(x, y, s=12)
    plt.xlabel("shared prefix tokens")
    plt.ylabel("per-request latency (s)")
    plt.tight_layout()
    plt.savefig(plot_dir / "profile_fit_prefix_reuse.pdf")
    plt.close()

    plt.figure()
    if conc_rows:
        x = [_float(r, "concurrency") for r in conc_rows if _ok(r)]
        y = [_float(r, "per_request_latency_mean") for r in conc_rows if _ok(r)]
        plt.scatter(x, y, s=12)
    plt.xlabel("concurrency")
    plt.ylabel("per-request latency (s)")
    plt.tight_layout()
    plt.savefig(plot_dir / "profile_fit_concurrency.pdf")
    plt.close()


def fit(
    raw: str | Path,
    out: str | Path,
    plot_dir: str | Path = "data/plots",
    report_csv: str | Path = "data/results/h100_latency_fit_report.csv",
    report_md: str | Path = "data/results/h100_latency_fit_report.md",
) -> LatencyModel:
    rows = _read_rows(raw)
    train = _length_training(rows)
    failed = [r for r in rows if not _ok(r)]
    if len(train) < 4:
        raise RuntimeError(f"not enough successful length samples to fit H100 latency model: {len(train)}")
    n = np.array([r["input"] for r in train])
    y = np.array([r["prefill"] for r in train])
    X = np.vstack([n, n * n, np.ones_like(n)]).T
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
    out_tok = np.array([r["output"] for r in train])
    ctx = n
    yd = np.array([r["decode"] / max(1.0, r["output"]) for r in train])
    Xd = np.vstack([np.ones_like(ctx), ctx]).T
    d, e = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    lm = LatencyModel(
        prefill_a=float(max(0.0, a)),
        prefill_b=float(max(0.0, b)),
        prefill_c=float(max(0.0, c)),
        decode_d=float(max(0.0, d)),
        decode_e=float(max(0.0, e)),
        measured=True,
        source=str(raw),
    )
    lm.queue_factors = _concurrency_factors(rows, lm)
    errs = _errors(train, lm)
    apes = [e["ape"] for e in errs]
    median = _pct(apes, 50)
    p95 = _pct(apes, 95)
    maxe = max(apes) if apes else float("nan")
    if median < 0.15 and p95 < 0.25:
        quality = "PASS"
    else:
        quality = "WARNING"
    lm.quality = quality
    ensure_dir(Path(out).parent)
    lm.to_json(out)
    ensure_dir(Path(report_csv).parent)
    write_csv(report_csv, errs)
    prefix_rows = [r for r in rows if r.get("mode") == "prefix"]
    conc_rows = [r for r in rows if r.get("mode") == "concurrency"]
    _plot_fit(train, lm, ensure_dir(plot_dir), prefix_rows, conc_rows)
    buckets: dict[str, list[float]] = {}
    for e in errs:
        bucket = f"input<={int(2 ** math.ceil(math.log2(max(1, e['input_tokens']))))}"
        buckets.setdefault(bucket, []).append(e["ape"])
    bucket_lines = "\n".join(f"- {k}: median APE={_pct(v,50):.4f}, n={len(v)}" for k, v in sorted(buckets.items()))
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(
        f"""# H100 Latency Fit Report

- training_samples = {len(train)}
- failed_samples_excluded = {len(failed)}
- median_absolute_percentage_error = {median:.6f}
- p95_absolute_percentage_error = {p95:.6f}
- max_error = {maxe:.6f}
- interpolation_fallback_used = false
- LATENCY_MODEL_QUALITY = {quality}

## Error By Input Length Bucket

{bucket_lines}
""",
        encoding="utf-8",
    )
    return lm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/profiles/h100_profile_raw.csv")
    ap.add_argument("--out", default="data/profiles/h100_latency_model.json")
    ap.add_argument("--plot-dir", default="data/plots")
    ap.add_argument("--report-csv", default="data/results/h100_latency_fit_report.csv")
    ap.add_argument("--report-md", default="data/results/h100_latency_fit_report.md")
    args = ap.parse_args()
    lm = fit(args.raw, args.out, args.plot_dir, args.report_csv, args.report_md)
    print(json.dumps(lm.__dict__, indent=2))


if __name__ == "__main__":
    main()
