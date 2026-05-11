#!/usr/bin/env bash
set -euo pipefail

SIZE="${SIZE:-medium}"
python -m agentweaver.analysis.pr4_algo_v8 run-all --size "${SIZE}"

