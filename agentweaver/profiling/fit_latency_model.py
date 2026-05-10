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
    out: list[dict[str, float]] = []
    for r in rows:
        if r.get("mode") != "length" or not _ok(r):
            continue
        inp = _float(r, "input_tokens_actual", _float(r, "client_prompt_tokens"))
        out_actual = _float(r, "output_tokens_actual", _float(r, "completion_tokens_client"))
        e2e = _float(r, "e2e_latency")
        ttft = _float(r, "ttft", math.nan)
        if inp <= 0 or out_actual <= 0 or e2e <= 0:
            continue
        if math.isnan(ttft) or ttft <= 0 or ttft > e2e:
            continue
        decode = max(0.0, e2e - ttft)
        out.append(
            {
                "input": inp,
                "output": out_actual,
                "target_output": _float(r, "output_tokens_target", out_actual),
                "e2e": e2e,
                "prefill": ttft,
                "decode": decode,
                "tpot": decode / max(1.0, out_actual),
            }
        )
    return out


def _concurrency_factors(rows: list[dict[str, Any]], base_model: LatencyModel) -> dict[str, float]:
    by_c: dict[int, list[float]] = {}
    for r in rows:
        if r.get("mode") != "concurrency" or not _ok(r):
            continue
        c = int(_float(r, "concurrency", 1))
        inp = _float(r, "input_tokens_actual", _float(r, "input_tokens_target"))
        out = _float(r, "output_tokens_actual_mean", _float(r, "output_tokens_target"))
        measured = _float(r, "per_request_latency_mean")
        pred = base_model.predict_prefill(inp) + base_model.predict_decode(inp, out)
        if measured > 0 and pred > 0:
            by_c.setdefault(c, []).append(measured / pred)
    return {str(k): float(np.median(v)) for k, v in by_c.items() if v}


def _fit_parametric(train: list[dict[str, float]], source: str) -> LatencyModel:
    n = np.array([r["input"] for r in train], dtype=float)
    prefill = np.array([r["prefill"] for r in train], dtype=float)
    x_prefill = np.vstack([n, n * n, np.ones_like(n)]).T
    a, b, c = np.linalg.lstsq(x_prefill, prefill, rcond=None)[0]

    ctx = np.array([r["input"] for r in train], dtype=float)
    tpot = np.array([r["tpot"] for r in train], dtype=float)
    x_decode = np.vstack([np.ones_like(ctx), ctx]).T
    d, e = np.linalg.lstsq(x_decode, tpot, rcond=None)[0]
    return LatencyModel(
        mode="parametric",
        prefill_a=float(max(0.0, a)),
        prefill_b=float(max(0.0, b)),
        prefill_c=float(max(0.0, c)),
        decode_d=float(max(0.0, d)),
        decode_e=float(max(0.0, e)),
        measured=True,
        source=source,
    )


def _make_interpolation_model(train: list[dict[str, float]], parametric: LatencyModel, source: str) -> LatencyModel:
    lm = LatencyModel(
        mode="interpolation",
        prefill_a=parametric.prefill_a,
        prefill_b=parametric.prefill_b,
        prefill_c=parametric.prefill_c,
        decode_d=parametric.decode_d,
        decode_e=parametric.decode_e,
        interpolation_points=[
            {
                "input": float(r["input"]),
                "output": float(r["output"]),
                "prefill": float(r["prefill"]),
                "decode": float(r["decode"]),
                "tpot": float(r["tpot"]),
                "e2e": float(r["e2e"]),
            }
            for r in train
        ],
        measured=True,
        source=source,
    )
    return lm


def _errors(rows: list[dict[str, float]], model: LatencyModel) -> list[dict[str, float | str]]:
    errs: list[dict[str, float | str]] = []
    for r in rows:
        pred_prefill = model.predict_prefill(r["input"])
        pred_decode = model.predict_decode(r["input"], r["output"])
        pred = pred_prefill + pred_decode
        actual = r["e2e"]
        if actual > 0:
            errs.append(
                {
                    "input_tokens": r["input"],
                    "output_tokens_actual": r["output"],
                    "output_tokens_target": r["target_output"],
                    "actual_e2e": actual,
                    "actual_ttft_prefill_proxy": r["prefill"],
                    "actual_decode": r["decode"],
                    "predicted_prefill": pred_prefill,
                    "predicted_decode": pred_decode,
                    "predicted_e2e": pred,
                    "ape": abs(pred - actual) / actual,
                    "model_mode": model.mode,
                }
            )
    return errs


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    idx = min(len(xs) - 1, max(0, int(round((p / 100) * (len(xs) - 1)))))
    return xs[idx]


def _plot_name(base: str, suffix: str) -> str:
    return f"{base}_{suffix}.pdf" if suffix else f"{base}.pdf"


def _plot_fit(
    train: list[dict[str, float]],
    model: LatencyModel,
    plot_dir: Path,
    prefix_rows: list[dict[str, Any]],
    conc_rows: list[dict[str, Any]],
    suffix: str = "",
) -> None:
    ensure_dir(plot_dir)
    if train:
        xs = np.array([r["input"] for r in train])
        ys = np.array([r["prefill"] for r in train])
        plt.figure()
        plt.scatter(xs, ys, s=12, label="streaming TTFT")
        grid = np.linspace(xs.min(), xs.max(), 100)
        plt.plot(grid, [model.predict_prefill(x) for x in grid], label=f"{model.mode} model")
        plt.xlabel("input tokens")
        plt.ylabel("prefill proxy: TTFT (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / _plot_name("profile_fit_prefill", suffix))
        plt.close()

        out_tokens = np.array([r["output"] for r in train])
        decode = np.array([r["decode"] for r in train])
        plt.figure()
        plt.scatter(out_tokens, decode, s=12, label="measured decode")
        order = np.argsort(out_tokens)
        median_ctx = float(np.median(xs))
        sorted_out = out_tokens[order]
        plt.plot(sorted_out, [model.predict_decode(median_ctx, x) for x in sorted_out], label=f"{model.mode}@median ctx")
        plt.xlabel("actual output tokens")
        plt.ylabel("decode time: e2e - TTFT (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / _plot_name("profile_fit_decode", suffix))
        plt.close()

    plt.figure()
    if prefix_rows:
        x = [_float(r, "shared_prefix_tokens_actual", _float(r, "shared_prefix_tokens_target")) for r in prefix_rows if _ok(r)]
        y = [_float(r, "per_request_latency_mean") for r in prefix_rows if _ok(r)]
        plt.scatter(x, y, s=12)
    plt.xlabel("shared prefix tokens")
    plt.ylabel("per-request latency (s)")
    plt.tight_layout()
    plt.savefig(plot_dir / _plot_name("profile_fit_prefix_reuse", suffix))
    plt.close()

    plt.figure()
    if conc_rows:
        x = [_float(r, "concurrency") for r in conc_rows if _ok(r)]
        y = [_float(r, "per_request_latency_mean") for r in conc_rows if _ok(r)]
        plt.scatter(x, y, s=12)
    plt.xlabel("concurrency")
    plt.ylabel("per-request latency (s)")
    plt.tight_layout()
    plt.savefig(plot_dir / _plot_name("profile_fit_concurrency", suffix))
    plt.close()


def _summary(errs: list[dict[str, float | str]]) -> tuple[float, float, float]:
    apes = [float(e["ape"]) for e in errs]
    return _pct(apes, 50), _pct(apes, 95), max(apes) if apes else float("nan")


def length_training_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Return length-sweep training samples that have streaming TTFT."""
    return _length_training(rows)


def fit_latency_model_from_training(
    train: list[dict[str, float]],
    source: str,
    all_rows: list[dict[str, Any]] | None = None,
    force_mode: str = "auto",
) -> tuple[LatencyModel, dict[str, float | str | bool]]:
    """Fit a latency model from already-cleaned training rows.

    force_mode may be "auto", "parametric", or "interpolation". Auto keeps the
    PR2-v2 rule: use interpolation when the parametric in-sample p95 exceeds 25%.
    """
    if len(train) < 4:
        raise RuntimeError(f"not enough training rows for latency model: {len(train)}")
    if force_mode not in {"auto", "parametric", "interpolation"}:
        raise ValueError(f"unknown force_mode {force_mode}")
    parametric = _fit_parametric(train, source)
    parametric.queue_factors = _concurrency_factors(all_rows or [], parametric)
    parametric_errs = _errors(train, parametric)
    param_median, param_p95, param_max = _summary(parametric_errs)

    interpolation_used = force_mode == "interpolation"
    if force_mode == "auto" and not (param_median < 0.15 and param_p95 < 0.25):
        interpolation_used = True

    lm = parametric
    if interpolation_used:
        lm = _make_interpolation_model(train, parametric, source)
        lm.queue_factors = _concurrency_factors(all_rows or [], lm)

    final_errs = _errors(train, lm)
    median, p95, maxe = _summary(final_errs)
    return lm, {
        "parametric_median_absolute_percentage_error": param_median,
        "parametric_p95_absolute_percentage_error": param_p95,
        "parametric_max_error": param_max,
        "interpolation_fallback_used": interpolation_used,
        "median_absolute_percentage_error": median,
        "p95_absolute_percentage_error": p95,
        "max_error": maxe,
        "latency_model_mode": lm.mode,
    }


def prediction_errors(rows: list[dict[str, float]], model: LatencyModel) -> list[dict[str, float | str]]:
    return _errors(rows, model)


def fit(
    raw: str | Path,
    out: str | Path,
    plot_dir: str | Path = "data/plots",
    report_csv: str | Path = "data/results/h100_latency_fit_report.csv",
    report_md: str | Path = "data/results/h100_latency_fit_report.md",
    plot_suffix: str = "",
) -> LatencyModel:
    rows = _read_rows(raw)
    train = _length_training(rows)
    failed = [r for r in rows if r.get("mode") == "length" and not _ok(r)]
    ok_length = [r for r in rows if r.get("mode") == "length" and _ok(r)]
    ttft_coverage = len(train) / max(1, len(ok_length))
    if len(train) < 4:
        raise RuntimeError(f"not enough successful streaming-TTFT length samples to fit H100 latency model: {len(train)}")

    lm, fit_stats = fit_latency_model_from_training(train, str(raw), rows)
    param_median = float(fit_stats["parametric_median_absolute_percentage_error"])
    param_p95 = float(fit_stats["parametric_p95_absolute_percentage_error"])
    param_max = float(fit_stats["parametric_max_error"])
    interpolation_used = bool(fit_stats["interpolation_fallback_used"])

    final_errs = prediction_errors(train, lm)
    median, p95, maxe = _summary(final_errs)
    quality = "PASS" if median < 0.15 and p95 < 0.25 and ttft_coverage >= 0.95 else "WARNING"
    lm.quality = quality
    ensure_dir(Path(out).parent)
    lm.to_json(out)
    ensure_dir(Path(report_csv).parent)
    write_csv(report_csv, final_errs)
    prefix_rows = [r for r in rows if r.get("mode") == "prefix"]
    conc_rows = [r for r in rows if r.get("mode") == "concurrency"]
    _plot_fit(train, lm, ensure_dir(plot_dir), prefix_rows, conc_rows, plot_suffix)
    buckets: dict[str, list[float]] = {}
    for e in final_errs:
        bucket = f"input<={int(2 ** math.ceil(math.log2(max(1, float(e['input_tokens'])))))}"
        buckets.setdefault(bucket, []).append(float(e["ape"]))
    bucket_lines = "\n".join(f"- {k}: median APE={_pct(v,50):.4f}, n={len(v)}" for k, v in sorted(buckets.items()))
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(
        f"""# H100 Latency Fit Report PR2-v2

- training_samples = {len(train)}
- failed_length_samples_excluded = {len(failed)}
- ttft_coverage = {ttft_coverage:.6f}
- prefill_proxy = streaming TTFT; no e2e proportional split is used.
- decode_proxy = e2e_latency - TTFT.
- output_token_source = actual completion tokens from server usage or client tokenizer.
- parametric_median_absolute_percentage_error = {param_median:.6f}
- parametric_p95_absolute_percentage_error = {param_p95:.6f}
- parametric_max_error = {param_max:.6f}
- interpolation_fallback_used = {str(interpolation_used).lower()}
- LATENCY_MODEL_MODE = {lm.mode}
- median_absolute_percentage_error = {median:.6f}
- p95_absolute_percentage_error = {p95:.6f}
- max_error = {maxe:.6f}
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
    ap.add_argument("--plot-suffix", default="")
    ap.add_argument("--report-csv", default="data/results/h100_latency_fit_report.csv")
    ap.add_argument("--report-md", default="data/results/h100_latency_fit_report.md")
    args = ap.parse_args()
    lm = fit(args.raw, args.out, args.plot_dir, args.report_csv, args.report_md, args.plot_suffix)
    print(json.dumps(lm.__dict__, indent=2))


if __name__ == "__main__":
    main()
