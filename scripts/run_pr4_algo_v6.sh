#!/usr/bin/env bash
set -euo pipefail

TRIALS="${TRIALS:-100}"
python -m agentweaver.analysis.pr4_algo_v6 run-all --trials "${TRIALS}"
