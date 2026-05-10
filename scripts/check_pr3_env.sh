#!/usr/bin/env bash
set -euo pipefail

SERVER=http://localhost:8001/v1
METRICS_URL=http://localhost:8001/metrics
TOKENIZER_PATH=/data2/model_zoo/Qwen2.5-Coder-7B-Instruct
DATASET=princeton-nlp/SWE-bench_Lite
SPLIT=test
OUT_JSON=data/results/pr3_env_report.json
OUT_MD=data/results/pr3_env_report.md

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --metrics-url) METRICS_URL="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --out-json) OUT_JSON="$2"; shift 2 ;;
    --out-md) OUT_MD="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$OUT_MD")"

export PR3_SERVER="$SERVER"
export PR3_METRICS_URL="$METRICS_URL"
export PR3_TOKENIZER_PATH="$TOKENIZER_PATH"
export PR3_DATASET="$DATASET"
export PR3_SPLIT="$SPLIT"
export PR3_OUT_JSON="$OUT_JSON"
export PR3_OUT_MD="$OUT_MD"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

python - <<'PY'
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(cmd: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        text = (cp.stdout or "") + (cp.stderr or "")
        return cp.returncode == 0, text.strip()[-4000:]
    except Exception as exc:
        return False, str(exc)


def http_ok(url: str, timeout: int = 5) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agentweaver-pr3-env"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(512).decode("utf-8", errors="replace")
            return 200 <= resp.status < 300, body
    except Exception as exc:
        return False, str(exc)


def mem_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kb = float(line.split()[1])
                return round(kb / 1024 / 1024, 2)
    except Exception:
        pass
    return 0.0


def disk_free_gb(path: str = ".") -> float:
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024**3), 2)


venv_python = str(Path(".venv/bin/python")) if Path(".venv/bin/python").exists() else sys.executable


def import_check(module: str) -> bool:
    ok, _ = run([venv_python, "-c", f"import {module}"], timeout=20)
    return ok


def module_exists_any(modules: list[str]) -> bool:
    code = (
        "import importlib.util, sys; "
        f"mods={modules!r}; "
        "sys.exit(0 if any(importlib.util.find_spec(m) for m in mods) else 1)"
    )
    ok, _ = run([venv_python, "-c", code], timeout=20)
    return ok


server = os.environ["PR3_SERVER"].rstrip("/")
metrics_url = os.environ["PR3_METRICS_URL"]
tokenizer_path = os.environ["PR3_TOKENIZER_PATH"]
dataset = os.environ["PR3_DATASET"]
split = os.environ["PR3_SPLIT"]
out_json = Path(os.environ["PR3_OUT_JSON"])
out_md = Path(os.environ["PR3_OUT_MD"])

python_available = True
uv_available = shutil.which("uv") is not None

docker_cmd = shutil.which("docker")
docker_available = False
docker_runnable = False
docker_info_ok = False
if docker_cmd:
    docker_info_ok, _ = run(["docker", "info"], timeout=20)
    docker_ps_ok, _ = run(["docker", "ps"], timeout=20)
    docker_available = docker_info_ok
    docker_runnable = docker_ps_ok

gpu_visible = False
cuda_report = ""
if shutil.which("nvidia-smi"):
    gpu_ok, cuda_report = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=15)
    gpu_visible = gpu_ok
else:
    cuda_report = "nvidia-smi not found"

models_url = server + "/models"
vllm_models_ok, vllm_models_text = http_ok(models_url)
metrics_ok, metrics_text = http_ok(metrics_url)
vllm_server_available = vllm_models_ok

candidate_paths = [
    tokenizer_path,
    "/data2/model_zoo/Qwen2.5-Coder-7B-Instruct",
    "/data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct",
]
model_path_selected = ""
for p in candidate_paths:
    if p and Path(p).exists():
        model_path_selected = p
        break
if not model_path_selected:
    model_path_selected = "Qwen/Qwen2.5-Coder-7B-Instruct"

mini_extra_path = shutil.which("mini-extra")
if not mini_extra_path and Path(".venv/bin/mini-extra").exists():
    mini_extra_path = ".venv/bin/mini-extra"
mini_extra_ok, mini_extra_help = run([mini_extra_path, "--help"], timeout=20) if mini_extra_path else (False, "mini-extra not found")
mini_import_ok = module_exists_any(["minisweagent", "mini_swe_agent"])
mini_installed = bool(mini_extra_ok or mini_import_ok)

swebench_installed = import_check("swebench")
datasets_installed = import_check("datasets")
dataset_access = "FAIL"
dataset_text = "datasets module unavailable"
if datasets_installed:
    code = (
        "from datasets import load_dataset; "
        f"ds=load_dataset({dataset!r}, split={split!r}+'[:1]'); "
        "print(len(ds)); "
        "print(ds[0].get('instance_id',''))"
    )
    ds_ok, dataset_text = run([venv_python, "-c", code], timeout=180)
    dataset_access = "PASS" if ds_ok else "FAIL"

pytest_cmd = ["pytest", "-q"]
if not shutil.which("pytest") and uv_available:
    pytest_cmd = ["uv", "run", "pytest", "-q"]
pytest_pass, pytest_text = run(pytest_cmd, timeout=600)

if not vllm_server_available:
    pr3_env = "FAIL"
elif not pytest_pass:
    pr3_env = "FAIL"
elif not mini_installed or not swebench_installed or dataset_access != "PASS" or not docker_runnable:
    pr3_env = "WARNING"
else:
    pr3_env = "PASS"

report = {
    "PR3_ENV": pr3_env,
    "PYTHON_AVAILABLE": python_available,
    "PYTHON_VERSION": platform.python_version(),
    "UV_AVAILABLE": uv_available,
    "DOCKER_AVAILABLE": docker_available,
    "DOCKER_RUNNABLE": docker_runnable,
    "DISK_FREE_GB": disk_free_gb("."),
    "MEMORY_GB": mem_gb(),
    "CPU_CORES": os.cpu_count() or 0,
    "GPU_CUDA_VISIBLE": gpu_visible,
    "CUDA_REPORT": cuda_report,
    "VLLM_SERVER_AVAILABLE": vllm_server_available,
    "VLLM_MODELS_ENDPOINT": "PASS" if vllm_models_ok else "FAIL",
    "VLLM_METRICS_AVAILABLE": metrics_ok,
    "MODEL_PATH_SELECTED": model_path_selected,
    "MINI_SWE_AGENT_INSTALLED": mini_installed,
    "MINI_EXTRA_AVAILABLE": mini_extra_ok,
    "SWE_BENCH_INSTALLED": swebench_installed,
    "DATASETS_INSTALLED": datasets_installed,
    "DATASET_ACCESS": dataset_access,
    "PYTEST_PASS": pytest_pass,
    "SERVER": server,
    "METRICS_URL": metrics_url,
    "TIMESTAMP_UNIX": int(time.time()),
    "DETAILS": {
        "vllm_models": vllm_models_text,
        "vllm_metrics": metrics_text[:500],
        "mini_extra": mini_extra_help[:1000],
        "dataset": dataset_text[:1000],
        "pytest": pytest_text[-2000:],
    },
}

out_json.parent.mkdir(parents=True, exist_ok=True)
out_md.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

fields = [
    "PR3_ENV",
    "DOCKER_AVAILABLE",
    "DOCKER_RUNNABLE",
    "DISK_FREE_GB",
    "MEMORY_GB",
    "CPU_CORES",
    "VLLM_SERVER_AVAILABLE",
    "VLLM_MODELS_ENDPOINT",
    "VLLM_METRICS_AVAILABLE",
    "MODEL_PATH_SELECTED",
    "MINI_SWE_AGENT_INSTALLED",
    "SWE_BENCH_INSTALLED",
    "DATASET_ACCESS",
    "PYTEST_PASS",
]
lines = ["# PR3 Environment Report", ""]
lines.extend(f"{k} = {str(report[k]).lower() if isinstance(report[k], bool) else report[k]}" for k in fields)
lines.extend(
    [
        "",
        "## Notes",
        f"- SERVER = {server}",
        f"- METRICS_URL = {metrics_url}",
        f"- UV_AVAILABLE = {str(uv_available).lower()}",
        f"- GPU_CUDA_VISIBLE = {str(gpu_visible).lower()}",
        "- Docker is required only for optional official SWE-bench harness evaluation.",
        "- Missing mini-SWE-agent or SWE-bench dependencies should be handled by scripts/setup_pr3_miniswe_swebench.sh.",
    ]
)
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({k: report[k] for k in fields}, indent=2))
PY
