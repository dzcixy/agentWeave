#!/usr/bin/env bash
set -euo pipefail

TRIALS="${TRIALS:-100}"

python -m agentweaver.analysis.pr4_algo_v5 run-all --taps-trials "${TRIALS}"
