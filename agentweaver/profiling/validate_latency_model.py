from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.profiling.fit_latency_model import (
    _float,
    _ok,
    _pct,
    _read_rows,
    _summary,
    fit_latency_model_from_training,
    length_training_from_rows,
    prediction_errors,
)
from agentweaver.utils.io import ensure_dir, write_csv


def _aggregate_apes(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    vals = [float(r["ape"]) for r in rows if not math.isnan(float(r["ape"]))]
    return _pct(vals, 50), _pct(vals, 95), max(vals) if vals else float("nan")


def _summary_row(
    validation_type: str,
    bucket: str,
    train: list[dict[str, float]],
    test_errs: list[dict[str, float | str]],
    seed: int | str = "",
    mode: str = "",
) -> dict[str, Any]:
    median, p95, maxe = _summary(test_errs)
    return {
        "validation_type": validation_type,
        "bucket": bucket,
        "seed": seed,
        "n_train": len(train),
        "n_test": len(test_errs),
        "model_mode": mode,
        "median_ape": median,
        "p95_ape": p95,
        "max_ape": maxe,
    }


def _fit_and_eval(
    train: list[dict[str, float]],
    test: list[dict[str, float]],
    source: str,
    all_rows: list[dict[str, Any]],
) -> tuple[str, list[dict[str, float | str]]]:
    model, _ = fit_latency_model_from_training(train, source, all_rows)
    return model.mode, prediction_errors(test, model)


def _random_splits(
    train_rows: list[dict[str, float]],
    all_rows: list[dict[str, Any]],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, float | str]]]:
    summaries: list[dict[str, Any]] = []
    all_errs: list[dict[str, float | str]] = []
    n = len(train_rows)
    test_n = max(1, int(round(0.2 * n)))
    for seed in seeds:
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        test_idx = set(idx[:test_n])
        fold_train = [r for i, r in enumerate(train_rows) if i not in test_idx]
        fold_test = [r for i, r in enumerate(train_rows) if i in test_idx]
        mode, errs = _fit_and_eval(fold_train, fold_test, f"random_split_seed_{seed}", all_rows)
        for err in errs:
            err["validation_type"] = "random_split"
            err["bucket"] = "all"
            err["seed"] = seed
        summaries.append(_summary_row("random_split", "all", fold_train, errs, seed, mode))
        all_errs.extend(errs)
    median, p95, maxe = _aggregate_apes(all_errs)
    summaries.append(
        {
            "validation_type": "random_split",
            "bucket": "ALL",
            "seed": "aggregate",
            "n_train": "",
            "n_test": len(all_errs),
            "model_mode": "mixed",
            "median_ape": median,
            "p95_ape": p95,
            "max_ape": maxe,
        }
    )
    return summaries, all_errs


def _leave_one_bucket(
    train_rows: list[dict[str, float]],
    all_rows: list[dict[str, Any]],
    key: str,
    validation_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, float | str]]]:
    by_bucket: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in train_rows:
        bucket_value = int(row[key])
        by_bucket[bucket_value].append(row)
    summaries: list[dict[str, Any]] = []
    all_errs: list[dict[str, float | str]] = []
    for bucket, test in sorted(by_bucket.items()):
        fold_train = [r for r in train_rows if int(r[key]) != bucket]
        if len(fold_train) < 4:
            continue
        mode, errs = _fit_and_eval(fold_train, test, f"{validation_type}_{bucket}", all_rows)
        for err in errs:
            err["validation_type"] = validation_type
            err["bucket"] = str(bucket)
            err["seed"] = ""
        summaries.append(_summary_row(validation_type, str(bucket), fold_train, errs, "", mode))
        all_errs.extend(errs)
    median, p95, maxe = _aggregate_apes(all_errs)
    summaries.append(
        {
            "validation_type": validation_type,
            "bucket": "ALL",
            "seed": "aggregate",
            "n_train": "",
            "n_test": len(all_errs),
            "model_mode": "mixed",
            "median_ape": median,
            "p95_ape": p95,
            "max_ape": maxe,
        }
    )
    return summaries, all_errs


def _concurrency_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in raw_rows:
        if row.get("mode") != "concurrency" or not _ok(row):
            continue
        concurrency = int(_float(row, "concurrency", 1))
        inp = _float(row, "input_tokens_actual", _float(row, "input_tokens_target"))
        out = _float(row, "output_tokens_actual_mean", _float(row, "output_tokens_target"))
        measured = _float(row, "per_request_latency_mean")
        if concurrency <= 0 or inp <= 0 or out <= 0 or measured <= 0:
            continue
        rows.append(
            {
                "concurrency": float(concurrency),
                "input": inp,
                "output": out,
                "measured": measured,
            }
        )
    return rows


def _interpolate_factor(concurrency: int, factors: dict[int, float]) -> float:
    if concurrency in factors:
        return factors[concurrency]
    if not factors:
        return 1.0
    xs = sorted(factors)
    if concurrency <= xs[0]:
        return factors[xs[0]]
    if concurrency >= xs[-1]:
        return factors[xs[-1]]
    for lo, hi in zip(xs, xs[1:]):
        if lo <= concurrency <= hi:
            frac = (concurrency - lo) / max(1, hi - lo)
            return factors[lo] + frac * (factors[hi] - factors[lo])
    return factors[min(xs, key=lambda x: abs(x - concurrency))]


def _concurrency_holdout(
    raw_rows: list[dict[str, Any]],
    train_rows: list[dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conc_rows = _concurrency_rows(raw_rows)
    if not conc_rows:
        return [], []
    base_model, _ = fit_latency_model_from_training(train_rows, "concurrency_holdout_base", raw_rows)
    by_conc: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in conc_rows:
        by_conc[int(row["concurrency"])].append(row)
    summaries: list[dict[str, Any]] = []
    all_errs: list[dict[str, Any]] = []
    for held, test in sorted(by_conc.items()):
        ratio_by_c: dict[int, list[float]] = defaultdict(list)
        for c, rows in by_conc.items():
            if c == held:
                continue
            for row in rows:
                base = base_model.predict_prefill(row["input"]) + base_model.predict_decode(row["input"], row["output"])
                if base > 0:
                    ratio_by_c[c].append(row["measured"] / base)
        factors = {c: sorted(vals)[len(vals) // 2] for c, vals in ratio_by_c.items() if vals}
        errs: list[dict[str, Any]] = []
        for row in test:
            base = base_model.predict_prefill(row["input"]) + base_model.predict_decode(row["input"], row["output"])
            pred = base * _interpolate_factor(held, factors)
            ape = abs(pred - row["measured"]) / row["measured"]
            err = {
                "validation_type": "concurrency_holdout",
                "bucket": str(held),
                "seed": "",
                "input_tokens": row["input"],
                "output_tokens_actual": row["output"],
                "actual_e2e": row["measured"],
                "predicted_e2e": pred,
                "ape": ape,
                "model_mode": base_model.mode,
            }
            errs.append(err)
            all_errs.append(err)
        summaries.append(_summary_row("concurrency_holdout", str(held), train_rows, errs, "", base_model.mode))
    median, p95, maxe = _aggregate_apes(all_errs)
    summaries.append(
        {
            "validation_type": "concurrency_holdout",
            "bucket": "ALL",
            "seed": "aggregate",
            "n_train": "",
            "n_test": len(all_errs),
            "model_mode": base_model.mode,
            "median_ape": median,
            "p95_ape": p95,
            "max_ape": maxe,
        }
    )
    return summaries, all_errs


def _agg(summary_rows: list[dict[str, Any]], validation_type: str) -> dict[str, Any]:
    for row in summary_rows:
        if row.get("validation_type") == validation_type and row.get("bucket") == "ALL":
            return row
    return {}


def _worst_bucket(summary_rows: list[dict[str, Any]]) -> str:
    candidates = [r for r in summary_rows if r.get("bucket") != "ALL" and r.get("n_test")]
    if not candidates:
        return ""
    worst = max(candidates, key=lambda r: float(r.get("p95_ape") or 0.0))
    return f"{worst.get('validation_type')}:{worst.get('bucket')}:p95={float(worst.get('p95_ape')):.6f}"


def _plot(summary_rows: list[dict[str, Any]], out: str | Path) -> None:
    aggregate = [r for r in summary_rows if r.get("bucket") == "ALL" and r.get("validation_type") != "concurrency_holdout"]
    if not aggregate:
        return
    labels = [str(r["validation_type"]).replace("_", "\n") for r in aggregate]
    med = [float(r["median_ape"]) * 100 for r in aggregate]
    p95 = [float(r["p95_ape"]) * 100 for r in aggregate]
    x = list(range(len(labels)))
    plt.figure(figsize=(7.2, 3.8))
    plt.bar([i - 0.18 for i in x], med, width=0.36, label="median APE")
    plt.bar([i + 0.18 for i in x], p95, width=0.36, label="p95 APE")
    plt.axhline(15, color="tab:green", linestyle="--", linewidth=1, label="median PASS ref")
    plt.axhline(25, color="tab:red", linestyle=":", linewidth=1, label="p95 PASS ref")
    plt.ylabel("APE (%)")
    plt.xticks(x, labels)
    plt.legend(fontsize=8)
    plt.tight_layout()
    ensure_dir(Path(out).parent)
    plt.savefig(out)
    plt.close()


def validate(
    raw: str | Path,
    report_csv: str | Path,
    report_md: str | Path,
    plot_out: str | Path,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    raw_rows = _read_rows(raw)
    train_rows = length_training_from_rows(raw_rows)
    if len(train_rows) < 8:
        raise RuntimeError(f"not enough length rows for holdout validation: {len(train_rows)}")
    seeds = seeds or [0, 1, 2, 3, 4]

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    rows, errs = _random_splits(train_rows, raw_rows, seeds)
    summary_rows.extend(rows)
    detail_rows.extend(errs)

    rows, errs = _leave_one_bucket(train_rows, raw_rows, "input", "leave_one_input_length")
    summary_rows.extend(rows)
    detail_rows.extend(errs)

    rows, errs = _leave_one_bucket(train_rows, raw_rows, "target_output", "leave_one_output_length")
    summary_rows.extend(rows)
    detail_rows.extend(errs)

    rows, errs = _concurrency_holdout(raw_rows, train_rows)
    summary_rows.extend(rows)
    detail_rows.extend(errs)

    rnd = _agg(summary_rows, "random_split")
    lin = _agg(summary_rows, "leave_one_input_length")
    lout = _agg(summary_rows, "leave_one_output_length")
    conc = _agg(summary_rows, "concurrency_holdout")
    def f(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, "nan"))
        except Exception:
            return float("nan")

    required_present = bool(rnd and lin and lout)
    quality = "FAIL"
    if required_present:
        passes = (
            f(rnd, "median_ape") < 0.15
            and f(rnd, "p95_ape") < 0.25
            and f(lin, "median_ape") < 0.20
            and f(lout, "median_ape") < 0.20
        )
        quality = "PASS" if passes else "WARNING"

    ensure_dir(Path(report_csv).parent)
    write_csv(report_csv, summary_rows)
    _plot(summary_rows, plot_out)
    worst = _worst_bucket(summary_rows)
    report = {
        "random_split_median_ape": f(rnd, "median_ape"),
        "random_split_p95_ape": f(rnd, "p95_ape"),
        "leave_input_median_ape": f(lin, "median_ape"),
        "leave_input_p95_ape": f(lin, "p95_ape"),
        "leave_output_median_ape": f(lout, "median_ape"),
        "leave_output_p95_ape": f(lout, "p95_ape"),
        "concurrency_holdout_median_ape": f(conc, "median_ape"),
        "concurrency_holdout_p95_ape": f(conc, "p95_ape"),
        "worst_bucket": worst,
        "HOLDOUT_LATENCY_MODEL_QUALITY": quality,
        "detail_rows": len(detail_rows),
    }
    lines = ["# H100 Latency Holdout Validation PR2-v2.1", ""]
    for key, value in report.items():
        if isinstance(value, float):
            lines.append(f"{key} = {value:.6f}")
        else:
            lines.append(f"{key} = {value}")
    if quality != "PASS":
        lines.append("recommendation = use interpolation fallback or collect more boundary profile points before paper claims")
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/profiles/h100_profile_raw_pr2_v2.csv")
    ap.add_argument("--report-csv", default="data/results/h100_latency_holdout_report_pr2_v2_1.csv")
    ap.add_argument("--report-md", default="data/results/h100_latency_holdout_report_pr2_v2_1.md")
    ap.add_argument("--plot-out", default="data/plots/profile_holdout_error_pr2_v2_1.pdf")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    report = validate(args.raw, args.report_csv, args.report_md, args.plot_out, seeds)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
