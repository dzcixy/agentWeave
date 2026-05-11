#!/usr/bin/env bash
set -euo pipefail

TRACE_ROOT="${1:-data/astra_traces}"
if [[ -z "${ASTRA_SIM_PATH:-}" ]]; then
  echo "ASTRA_SIM_PATH is not set; not running ASTRA." >&2
  exit 2
fi
if [[ ! -x "${ASTRA_SIM_PATH}" ]]; then
  echo "ASTRA_SIM_PATH does not point to an executable: ${ASTRA_SIM_PATH}" >&2
  exit 2
fi
if [[ ! -d "${TRACE_ROOT}" ]]; then
  echo "Trace root not found: ${TRACE_ROOT}" >&2
  exit 2
fi

mapfile -t traces < <(find "${TRACE_ROOT}" -name '*.et' -o -name '*.et.json' | sort)
if [[ "${#traces[@]}" -eq 0 ]]; then
  echo "No Chakra traces found under ${TRACE_ROOT}" >&2
  exit 2
fi

echo "Found ${#traces[@]} trace files. This script prints runnable ASTRA commands; it does not infer cycles."
for trace in "${traces[@]}"; do
  echo "${ASTRA_SIM_PATH} --workload-configuration=${trace}"
done
