#!/usr/bin/env bash
set -euo pipefail

SERVER=http://localhost:8001/v1
MODEL=qwen-coder-7b
TOKENIZER_PATH=/data2/model_zoo/Qwen2.5-Coder-7B-Instruct
NUM_INSTANCES=1
ROLLOUTS=4
MAX_STEPS=20
RUN_ID=mini_swe_multibranch_pr3
OUT_DIR=data/traces/mini_swe_multibranch_pr3
TRAJ_ROOT=""

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
    --traj-root|--traj-dir) TRAJ_ROOT="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ "$ROLLOUTS" -lt 2 ]; then
  echo "multi-branch collection requires --rollouts >= 2" >&2
  exit 1
fi

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

if [ -n "$TRAJ_ROOT" ]; then
  mkdir -p "$OUT_DIR"
  found=0
  while IFS= read -r traj; do
    found=1
    rel=${traj#"$TRAJ_ROOT"/}
    instance=$(dirname "$rel" | tr '/' '_')
    base=$(basename "$traj")
    base=${base%.json}
    base=${base%.traj}
    out_name="${instance}_${base}.jsonl"
    "${PYTHON[@]}" -m agentweaver.tracing.mini_swe_trace_adapter \
      --traj "$traj" \
      --out "$OUT_DIR/$out_name"
  done < <(find "$TRAJ_ROOT" -type f \( -name '*.traj' -o -name '*.traj.json' -o -name '*.json' \) | sort)
  if [ "$found" = "0" ]; then
    echo "no mini-SWE-agent trajectory files found under $TRAJ_ROOT" >&2
    exit 1
  fi
  echo "converted multi-branch mini-SWE-agent trajectories to $OUT_DIR"
  exit 0
fi

if ! command -v mini-swe-agent >/dev/null 2>&1 && ! "${PYTHON[@]}" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mini_swe_agent") else 1)
PY
then
  cat >&2 <<'EOF'
mini-SWE-agent is not installed in this environment.

Install mini-SWE-agent following its upstream docs, run real multi-rollout trajectories
against your local vLLM server, then pass --traj-root <traj_root> to convert them.

This script does not generate fake traces.
EOF
  exit 1
fi

cat >&2 <<EOF
This PR3 skeleton does not run full SWE-bench and does not synthesize trajectories.

Run mini-SWE-agent externally with multiple rollouts per instance against:
  server: $SERVER
  model: $MODEL
  tokenizer: $TOKENIZER_PATH
  num_instances: $NUM_INSTANCES
  rollouts: $ROLLOUTS
  max_steps: $MAX_STEPS

Store real trajectories in a directory shaped like:
  <traj_root>/<instance_id>/rollout_0.traj.json
  <traj_root>/<instance_id>/rollout_1.traj.json

Then convert them with:
  bash scripts/run_mini_swe_multibranch_pr3.sh --traj-root <traj_root> --out-dir $OUT_DIR --server $SERVER --tokenizer-path $TOKENIZER_PATH --rollouts $ROLLOUTS --max-steps $MAX_STEPS
EOF
exit 1
