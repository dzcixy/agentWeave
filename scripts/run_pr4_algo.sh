#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
  else
    PYTHON=(python)
  fi
else
  PYTHON=("$PYTHON_BIN")
fi

"${PYTHON[@]}" -m agentweaver.analysis.pr4_algo run-all
