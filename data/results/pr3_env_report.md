# PR3 Environment Report

PR3_ENV = PASS
DOCKER_AVAILABLE = true
DOCKER_RUNNABLE = true
DISK_FREE_GB = 193.05
MEMORY_GB = 1007.52
CPU_CORES = 128
VLLM_SERVER_AVAILABLE = true
VLLM_MODELS_ENDPOINT = PASS
VLLM_METRICS_AVAILABLE = true
MODEL_PATH_SELECTED = /data2/model_zoo/Qwen2.5-Coder-7B-Instruct
MINI_SWE_AGENT_INSTALLED = true
SWE_BENCH_INSTALLED = true
DATASET_ACCESS = PASS
PYTEST_PASS = true

## Notes
- SERVER = http://localhost:8001/v1
- METRICS_URL = http://localhost:8001/metrics
- UV_AVAILABLE = true
- GPU_CUDA_VISIBLE = true
- Docker is required only for optional official SWE-bench harness evaluation.
- Missing mini-SWE-agent or SWE-bench dependencies should be handled by scripts/setup_pr3_miniswe_swebench.sh.
