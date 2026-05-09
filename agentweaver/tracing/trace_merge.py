from __future__ import annotations

import argparse

from agentweaver.tracing.trace_schema import load_trace_dir, merge_traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merge_traces(load_trace_dir(args.trace_dir)).to_jsonl(args.out)


if __name__ == "__main__":
    main()
