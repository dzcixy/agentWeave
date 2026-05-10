#!/usr/bin/env bash
set -euo pipefail

SERVER=http://localhost:8001/v1
MODEL=qwen-coder-7b
TOKENIZER_PATH=/data2/model_zoo/Qwen2.5-Coder-7B-Instruct
NUM_INSTANCES=1
ROLLOUTS=1
MAX_STEPS=20
RUN_ID=mini_swe_pr3
OUT_DIR=data/traces/mini_swe_pr3
TRAJ_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --num-instances|--instances) NUM_INSTANCES="$2"; shift 2 ;;
    --rollouts) ROLLOUTS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; OUT_DIR="data/traces/$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --traj-dir) TRAJ_DIR="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

if [ -n "$TRAJ_DIR" ]; then
  mkdir -p "$OUT_DIR"
  found=0
  while IFS= read -r traj; do
    found=1
    name=$(basename "$traj")
    name=${name%.json}
    name=${name%.traj}
    "${PYTHON[@]}" -m agentweaver.tracing.mini_swe_trace_adapter \
      --traj "$traj" \
      --out "$OUT_DIR/${name}.jsonl"
  done < <(find "$TRAJ_DIR" -type f \( -name '*.traj' -o -name '*.traj.json' -o -name '*.json' \) | sort)
  if [ "$found" = "0" ]; then
    echo "no mini-SWE-agent trajectory files found under $TRAJ_DIR" >&2
    exit 1
  fi
  echo "converted mini-SWE-agent trajectories to $OUT_DIR"
  exit 0
fi

if ! command -v mini-swe-agent >/dev/null 2>&1 && ! "${PYTHON[@]}" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mini_swe_agent") else 1)
PY
then
  cat >&2 <<'EOF'
mini-SWE-agent is not installed in this environment.

Install mini-SWE-agent in the active environment following its upstream docs, then either:
  1. run mini-SWE-agent to produce real .traj/.traj.json files and pass --traj-dir <dir>, or
  2. set MINI_SWE_AGENT_CMD to your local runner command and adapt this skeleton.

This script does not generate fake traces.
EOF
  exit 1
fi

if ! curl -fsS "$SERVER/models" >/dev/null 2>&1; then
  echo "local OpenAI-compatible vLLM server is not reachable at $SERVER" >&2
  echo "start vLLM first, then rerun this script with --server http://localhost:<port>/v1" >&2
  exit 1
fi

if [ ! -e "$TOKENIZER_PATH" ]; then
  echo "tokenizer path does not exist locally: $TOKENIZER_PATH" >&2
  echo "pass --tokenizer-path /path/to/Qwen2.5-Coder-7B-Instruct" >&2
  exit 1
fi

cat >&2 <<EOF
mini-SWE-agent appears to be installed, but this repository does not assume a stable upstream CLI.
Run mini-SWE-agent with your local command against:
  server: $SERVER
  model: $MODEL
  tokenizer: $TOKENIZER_PATH
  num_instances: $NUM_INSTANCES
  rollouts: $ROLLOUTS
  max_steps: $MAX_STEPS

Then convert the real trajectories with:
  bash scripts/run_mini_swe_trace_pr3.sh --traj-dir <traj_dir> --out-dir $OUT_DIR --server $SERVER --tokenizer-path $TOKENIZER_PATH

No fake trace was generated.
EOF
exit 1
