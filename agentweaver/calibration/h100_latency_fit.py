from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.utils.io import ensure_dir, write_csv


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _least_squares(x: list[list[float]], y: list[float]) -> list[float]:
    try:
        import numpy as np

        coef, *_ = np.linalg.lstsq(np.asarray(x, dtype=float), np.asarray(y, dtype=float), rcond=None)
        return [float(v) for v in coef]
    except Exception:
        # Small normal-equation fallback for three coefficients.
        n = len(x[0]) if x else 0
        a = [[sum(row[i] * row[j] for row in x) for j in range(n)] for i in range(n)]
        b = [sum(row[i] * val for row, val in zip(x, y)) for i in range(n)]
        for i in range(n):
            pivot = a[i][i] or 1e-12
            for j in range(i, n):
                a[i][j] /= pivot
            b[i] /= pivot
            for k in range(n):
                if k == i:
                    continue
                factor = a[k][i]
                for j in range(i, n):
                    a[k][j] -= factor * a[i][j]
                b[k] -= factor * b[i]
        return b


def fit_latency(
    raw_csv: str | Path = "data/calibration/h100_vllm_latency_raw_pr4_v13.csv",
    out_json: str | Path = "data/calibration/h100_vllm_latency_fit_pr4_v13.json",
    report_md: str | Path = "data/results/h100_calibration_report_pr4_v13.md",
    compare_csv: str | Path = "data/results/h100_latency_model_comparison_pr4_v13.csv",
) -> dict[str, Any]:
    raw = [r for r in _read_csv(raw_csv) if r.get("status") == "OK"]
    if not raw:
        if not Path(raw_csv).exists():
            write_csv(
                raw_csv,
                [
                    {
                        "request_id": "NOT_RUN",
                        "model": "",
                        "prompt_tokens": 0,
                        "output_tokens": 0,
                        "batch_size": 0,
                        "ttft": "",
                        "total_latency": "",
                        "prefill_latency_est": "",
                        "decode_latency_est": "",
                        "tokens_per_sec": "",
                        "gpu_count": 0,
                        "gpu_name": "",
                        "status": "NOT_RUN",
                    }
                ],
            )
        fields = {
            "H100_CALIBRATION": "NOT_RUN",
            "H100_CALIBRATION_STATUS": "NOT_RUN",
            "RAW_ROWS": 0,
            "FITTED_MODEL_AVAILABLE": "false",
        }
        Path(report_md).parent.mkdir(parents=True, exist_ok=True)
        Path(report_md).write_text(
            "# H100 Calibration PR4-v13\n\n"
            + "\n".join(f"{k} = {v}" for k, v in fields.items())
            + "\n\nNo vLLM profiling rows were available; no fitted model or fake data was generated.\n",
            encoding="utf-8",
        )
        ensure_dir(Path(out_json).parent)
        Path(out_json).write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_csv(compare_csv, [])
        return fields

    pre_x, pre_y, dec_x, dec_y = [], [], [], []
    for row in raw:
        inp = _f(row, "prompt_tokens")
        out = max(1.0, _f(row, "output_tokens"))
        batch = _f(row, "batch_size")
        pre_x.append([1.0, inp, inp * inp])
        pre_y.append(_f(row, "prefill_latency_est"))
        dec_x.append([1.0, inp, batch])
        dec_y.append(_f(row, "decode_latency_est") / out)
    pre = _least_squares(pre_x, pre_y)
    dec = _least_squares(dec_x, dec_y)
    fit = {
        "H100_CALIBRATION": "OK",
        "H100_CALIBRATION_STATUS": "OK",
        "raw_rows": len(raw),
        "prefill_latency": {"a0": pre[0], "a1": pre[1], "a2": pre[2]},
        "decode_latency_per_token": {"b0": dec[0], "b1": dec[1], "b2": dec[2]},
    }
    ensure_dir(Path(out_json).parent)
    Path(out_json).write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lm = LatencyModel()
    compare: list[dict[str, Any]] = []
    for inp in [512, 1024, 2048, 4096, 8192, 16384]:
        old_prefill = lm.predict_prefill(inp)
        new_prefill = pre[0] + pre[1] * inp + pre[2] * inp * inp
        compare.append(
            {
                "input_tokens": inp,
                "analytic_prefill_latency": old_prefill,
                "h100_fit_prefill_latency": new_prefill,
                "relative_delta": (new_prefill - old_prefill) / max(1e-9, old_prefill),
            }
        )
    write_csv(compare_csv, compare)
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(
        "# H100 Calibration PR4-v13\n\n"
        + "\n".join(
            [
                "H100_CALIBRATION = OK",
                "H100_CALIBRATION_STATUS = OK",
                f"RAW_ROWS = {len(raw)}",
                f"PREFILL_COEFFS = {json.dumps(fit['prefill_latency'], sort_keys=True)}",
                f"DECODE_COEFFS = {json.dumps(fit['decode_latency_per_token'], sort_keys=True)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/calibration/h100_vllm_latency_raw_pr4_v13.csv")
    ap.add_argument("--out-json", default="data/calibration/h100_vllm_latency_fit_pr4_v13.json")
    ap.add_argument("--report", default="data/results/h100_calibration_report_pr4_v13.md")
    ap.add_argument("--compare", default="data/results/h100_latency_model_comparison_pr4_v13.csv")
    args = ap.parse_args()
    print(json.dumps(fit_latency(args.raw, args.out_json, args.report, args.compare), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
