#!/usr/bin/env bash
set -euo pipefail

SERVER=${SERVER:-http://localhost:8001/v1}
MODEL=${MODEL:-qwen-coder-7b}
MAX_TOKENS=${MAX_TOKENS:-32768}
RESPONSE_MAX_TOKENS=${RESPONSE_MAX_TOKENS:-1024}
MAX_STEPS=${MAX_STEPS:-10}
TEMPERATURE=${TEMPERATURE:-0.2}
RESULTS_DIR=${RESULTS_DIR:-data/results}
EXTERNAL_DIR=${EXTERNAL_DIR:-external}
REGISTRY=configs/miniswe_litellm_registry.json
CONFIG=configs/miniswe_swebench_vllm.yaml
HELP_OUT="$RESULTS_DIR/mini_extra_swebench_help.txt"
HELP_SINGLE_OUT="$RESULTS_DIR/mini_extra_swebench_single_help.txt"
REPORT="$RESULTS_DIR/pr3_setup_report.md"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --response-max-tokens) RESPONSE_MAX_TOKENS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; HELP_OUT="$2/mini_extra_swebench_help.txt"; HELP_SINGLE_OUT="$2/mini_extra_swebench_single_help.txt"; REPORT="$2/pr3_setup_report.md"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RESULTS_DIR" configs "$EXTERNAL_DIR"

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  PYTHON=(python)
fi

PIP_BASE=()
if command -v uv >/dev/null 2>&1; then
  PIP_BASE=(uv pip install)
elif command -v pip >/dev/null 2>&1; then
  PIP_BASE=(pip install)
else
  echo "neither uv nor pip is available" >&2
  exit 1
fi

module_exists() {
  local module=$1
  "${PYTHON[@]}" - "$module" <<'PY' >/dev/null 2>&1
import importlib.util
import sys
raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

mini_extra_cmd() {
  if command -v mini-extra >/dev/null 2>&1; then
    printf '%s\n' "mini-extra"
  elif [ -x ".venv/bin/mini-extra" ]; then
    printf '%s\n' ".venv/bin/mini-extra"
  elif command -v uv >/dev/null 2>&1; then
    printf '%s\n' "uv run mini-extra"
  else
    return 1
  fi
}

mini_installed=false
swebench_installed=false
litellm_installed=false
mini_extra_available=false
mini_extra_help_captured=false
registry_created=false
config_created=false
install_notes=()

if mini_extra_cmd >/dev/null 2>&1 || module_exists minisweagent || module_exists mini_swe_agent; then
  mini_installed=true
else
  if "${PIP_BASE[@]}" mini-swe-agent; then
    mini_installed=true
  else
    install_notes+=("pip package install for mini-swe-agent failed; trying upstream clone")
    if [ ! -d "$EXTERNAL_DIR/mini-swe-agent/.git" ]; then
      git clone https://github.com/SWE-agent/mini-swe-agent "$EXTERNAL_DIR/mini-swe-agent"
    fi
    if "${PIP_BASE[@]}" -e "$EXTERNAL_DIR/mini-swe-agent"; then
      mini_installed=true
    else
      mini_installed=false
      install_notes+=("failed to install mini-SWE-agent from package and editable clone")
    fi
  fi
fi

if module_exists swebench; then
  swebench_installed=true
else
  if "${PIP_BASE[@]}" datasets swebench; then
    swebench_installed=true
  else
    swebench_installed=false
    install_notes+=("failed to install datasets/swebench; check Python version, Docker extras, and package resolver output")
  fi
fi

if module_exists litellm; then
  litellm_installed=true
else
  if "${PIP_BASE[@]}" litellm; then
    litellm_installed=true
  else
    litellm_installed=false
    install_notes+=("failed to install litellm")
  fi
fi

cat > "$REGISTRY" <<EOF
{
  "$MODEL": {
    "max_tokens": $MAX_TOKENS,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat"
  },
  "hosted_vllm/$MODEL": {
    "max_tokens": $MAX_TOKENS,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat"
  }
}
EOF
registry_created=true

cat > "$CONFIG" <<EOF
# PR3 local mini-SWE-agent/SWE-bench override.
# Runner scripts merge this after the built-in swebench config:
#   mini-extra swebench-single -c swebench_backticks -c configs/miniswe_swebench_vllm.yaml ...
agent:
  step_limit: $MAX_STEPS
  cost_limit: 0
environment:
  environment_class: docker
  pull_timeout: 1200
  container_timeout: 2h
  timeout: 60
model:
  model_name: hosted_vllm/$MODEL
  model_class: litellm_textbased
  litellm_model_registry: configs/miniswe_litellm_registry.json
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: $SERVER
    api_key: dummy
    temperature: $TEMPERATURE
    max_tokens: $RESPONSE_MAX_TOKENS
    drop_params: true
EOF
config_created=true

if MINI_EXTRA=$(mini_extra_cmd); then
  # shellcheck disable=SC2206
  MINI_EXTRA_ARR=( $MINI_EXTRA )
  if "${MINI_EXTRA_ARR[@]}" swebench --help > "$HELP_OUT" 2>&1; then
    mini_extra_available=true
    mini_extra_help_captured=true
  else
    mini_extra_available=false
    mini_extra_help_captured=false
  fi
  "${MINI_EXTRA_ARR[@]}" swebench-single --help > "$HELP_SINGLE_OUT" 2>&1 || true
else
  echo "mini-extra not found" > "$HELP_OUT"
  echo "mini-extra not found" > "$HELP_SINGLE_OUT"
fi

if [ "$mini_installed" = true ] && mini_extra_cmd >/dev/null 2>&1; then
  mini_extra_available=true
fi

ready=false
if [ "$mini_installed" = true ] && [ "$mini_extra_available" = true ] && [ "$swebench_installed" = true ] && [ "$litellm_installed" = true ] && [ "$registry_created" = true ] && [ "$config_created" = true ] && [ "$mini_extra_help_captured" = true ]; then
  ready=true
fi

{
  echo "# PR3 mini-SWE-agent/SWE-bench Setup Report"
  echo
  echo "MINI_SWE_AGENT_INSTALLED = $mini_installed"
  echo "MINI_EXTRA_AVAILABLE = $mini_extra_available"
  echo "SWE_BENCH_INSTALLED = $swebench_installed"
  echo "LITELLM_INSTALLED = $litellm_installed"
  echo "LITELLM_REGISTRY_CREATED = $registry_created"
  echo "MINISWE_CONFIG_CREATED = $config_created"
  echo "MINI_EXTRA_HELP_CAPTURED = $mini_extra_help_captured"
  echo "READY_TO_RUN_MINISWE = $ready"
  echo
  echo "## Files"
  echo "- registry: $REGISTRY"
  echo "- config: $CONFIG"
  echo "- mini-extra swebench help: $HELP_OUT"
  echo "- mini-extra swebench-single help: $HELP_SINGLE_OUT"
  if [ "${#install_notes[@]}" -gt 0 ]; then
    echo
    echo "## Notes"
    for note in "${install_notes[@]}"; do
      echo "- $note"
    done
  fi
} > "$REPORT"

cat "$REPORT"

if [ "$ready" != true ]; then
  exit 1
fi
