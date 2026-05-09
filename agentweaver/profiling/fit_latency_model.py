from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.utils.io import ensure_dir


def fit(raw: str | Path, out: str | Path, plot_dir: str | Path = "data/plots") -> LatencyModel:
    rows = []
    with Path(raw).open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok").startswith("ok"):
                rows.append(r)
    if len(rows) < 3:
        lm = LatencyModel()
    else:
        n = np.array([float(r["input_tokens"]) for r in rows])
        y = np.array([float(r["estimated_prefill"]) / max(1.0, float(r["batch_size"])) for r in rows])
        X = np.vstack([n, n * n, np.ones_like(n)]).T
        a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
        m = np.array([float(r["output_tokens"]) for r in rows])
        ctx = n
        yd = np.array([float(r["estimated_decode"]) / np.maximum(1.0, m)])
        Xd = np.vstack([np.ones_like(ctx), ctx]).T
        d, e = np.linalg.lstsq(Xd, yd, rcond=None)[0]
        lm = LatencyModel(float(max(0, a)), float(max(0, b)), float(max(0, c)), float(max(0, d)), float(max(0, e)))
    ensure_dir(Path(out).parent)
    lm.to_json(out)
    pd = ensure_dir(plot_dir)
    if rows:
        xs = np.array([float(r["input_tokens"]) for r in rows])
        ys = np.array([float(r["estimated_prefill"]) for r in rows])
        plt.figure()
        plt.scatter(xs, ys, s=8, label="measured")
        grid = np.linspace(xs.min(), xs.max(), 100)
        plt.plot(grid, [lm.predict_prefill(x) for x in grid], label="fit")
        plt.xlabel("input tokens")
        plt.ylabel("prefill latency (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pd / "profile_fit_prefill.pdf")
        plt.close()
        ys2 = np.array([float(r["estimated_decode"]) for r in rows])
        plt.figure()
        plt.scatter(xs, ys2, s=8, label="measured")
        plt.plot(grid, [lm.predict_decode(x, 512) for x in grid], label="fit@512 tok")
        plt.xlabel("context tokens")
        plt.ylabel("decode latency (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pd / "profile_fit_decode.pdf")
        plt.close()
    return lm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/profiles/h100_profile_raw.csv")
    ap.add_argument("--out", default="data/profiles/h100_latency_model.json")
    ap.add_argument("--plot-dir", default="data/plots")
    args = ap.parse_args()
    print(json.dumps(fit(args.raw, args.out, args.plot_dir).__dict__, indent=2))


if __name__ == "__main__":
    main()
