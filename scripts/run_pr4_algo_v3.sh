#!/usr/bin/env bash
set -euo pipefail

MAX_CONFIGS=64
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-configs)
      MAX_CONFIGS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PYTHONDONTWRITEBYTECODE=1 python -m agentweaver.analysis.pr4_algo_v3 run-all --max-configs "${MAX_CONFIGS}"
