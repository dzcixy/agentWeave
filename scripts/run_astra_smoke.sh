#!/usr/bin/env bash
set -euo pipefail

REPORT="data/results/astra_run_report.md"
LOG_DIR="data/logs/astra"
LOG_PATH="${LOG_DIR}/astra_smoke.log"

mkdir -p "$(dirname "${REPORT}")" "${LOG_DIR}"

python -m agentweaver.astra.run_astra_smoke

ASTRA_AVAILABLE=false
ASTRA_SMOKE_RUN=false
ASTRA_OUTPUT_CYCLES=""
ASTRA_BIN="${ASTRA_BIN:-}"

if [[ -n "${ASTRA_SIM_PATH:-}" ]]; then
  if [[ -n "${ASTRA_BIN}" && -x "${ASTRA_BIN}" ]]; then
    ASTRA_AVAILABLE=true
  elif [[ -x "${ASTRA_SIM_PATH}" ]]; then
    ASTRA_BIN="${ASTRA_SIM_PATH}"
    ASTRA_AVAILABLE=true
  elif [[ -d "${ASTRA_SIM_PATH}" ]]; then
    CANDIDATE="$(find "${ASTRA_SIM_PATH}" -type f -perm -111 \( -name 'astra*' -o -name '*Astra*' \) 2>/dev/null | head -n 1 || true)"
    if [[ -n "${CANDIDATE}" ]]; then
      ASTRA_BIN="${CANDIDATE}"
      ASTRA_AVAILABLE=true
    fi
  fi
fi

if [[ "${ASTRA_AVAILABLE}" == "true" && -n "${ASTRA_RUN_CMD:-}" ]]; then
  set +e
  bash -lc "${ASTRA_RUN_CMD}" >"${LOG_PATH}" 2>&1
  STATUS=$?
  set -e
  if [[ ${STATUS} -eq 0 ]]; then
    ASTRA_SMOKE_RUN=true
    ASTRA_OUTPUT_CYCLES="$(grep -Eio 'cycles[^0-9]*[0-9]+' "${LOG_PATH}" | tail -n 1 | grep -Eo '[0-9]+' || true)"
  fi
fi

{
  echo "# ASTRA Run Report"
  echo
  echo "ASTRA_SIM_AVAILABLE = ${ASTRA_AVAILABLE}"
  echo "ASTRA_SMOKE_RUN = ${ASTRA_SMOKE_RUN}"
  echo "ASTRA_OUTPUT_LOG_PATH = ${LOG_PATH}"
  echo "ASTRA_OUTPUT_CYCLES = ${ASTRA_OUTPUT_CYCLES}"
  echo "ASTRA_EXPORT_FORMAT = intermediate_json"
  echo
  if [[ "${ASTRA_AVAILABLE}" != "true" ]]; then
    echo "ASTRA_SIM_PATH was not set to a runnable ASTRA-sim binary or checkout."
    echo "Set ASTRA_SIM_PATH and optionally ASTRA_BIN after building ASTRA-sim."
  elif [[ -z "${ASTRA_RUN_CMD:-}" ]]; then
    echo "ASTRA binary was found at ${ASTRA_BIN}, but ASTRA_RUN_CMD was not provided."
    echo "No cycles are reported because ASTRA-sim CLI conventions vary by checkout."
  else
    echo "ASTRA_RUN_CMD was executed; inspect the log for raw simulator output."
  fi
} >"${REPORT}"

cat "${REPORT}"
