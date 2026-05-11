#!/usr/bin/env bash
set -euo pipefail

TAPS_TRIALS=50
PABB_TRIALS=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --taps-trials)
      TAPS_TRIALS="$2"
      shift 2
      ;;
    --pabb-trials)
      PABB_TRIALS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PYTHONDONTWRITEBYTECODE=1 python -m agentweaver.analysis.pr4_algo_v4 run-all \
  --taps-trials "${TAPS_TRIALS}" \
  --pabb-trials "${PABB_TRIALS}"
