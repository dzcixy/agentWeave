from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def _read_instances(path: str | Path, limit: int) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"instance list does not exist: {p}")
    ids = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]
    return ids[:limit]


def _run(cmd: list[str], *, log_path: Path, env: dict[str, str], timeout: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, text=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        return proc.returncode


def _capture(cmd: list[str], timeout: int = 20) -> tuple[bool, str]:
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return cp.returncode == 0, (cp.stdout or "") + (cp.stderr or "")
    except Exception as exc:
        return False, str(exc)


def _has_flag(help_text: str, *names: str) -> str | None:
    for name in names:
        if name in help_text:
            return name
    return None


def _traj_score(path: Path) -> tuple[int, float]:
    score = 0
    name = path.name.lower()
    if ".traj" in name or "trajectory" in name:
        score += 10
    if path.suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and any(k in obj for k in ("trajectory", "steps", "history", "turns", "messages", "events")):
                score += 20
        except Exception:
            pass
    return score, path.stat().st_mtime


def _find_trajectory(run_out: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("*.traj", "*.traj.json", "*.json"):
        candidates.extend(p for p in run_out.rglob(pattern) if p.is_file())
    candidates = [p for p in candidates if not p.name.endswith(".log") and "report" not in p.name.lower()]
    if not candidates:
        return None
    candidates.sort(key=_traj_score, reverse=True)
    best = candidates[0]
    if _traj_score(best)[0] <= 0 and best.suffix == ".json":
        return None
    return best


def _copy_trajectory(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _template_command(template: str, values: dict[str, str]) -> list[str]:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return shlex.split(rendered)


def _auto_command(
    *,
    help_text: str,
    subcommand: list[str],
    instance_id: str,
    run_out: Path,
    server: str,
    model: str,
    max_steps: int,
    seed: int,
    temperature: float,
    config: str,
) -> list[str]:
    cmd = list(subcommand)
    config_flag = _has_flag(help_text, "--config", "--config-path", "--config_path")
    if config_flag:
        cmd.extend(
            [
                config_flag,
                "swebench_backticks",
                config_flag,
                config,
                config_flag,
                f"agent.step_limit={max_steps}",
                config_flag,
                "agent.cost_limit=0",
                config_flag,
                f"model.model_kwargs.temperature={temperature}",
            ]
        )
    subset_flag = _has_flag(help_text, "--subset")
    if subset_flag:
        cmd.extend([subset_flag, "lite"])
    dataset_flag = _has_flag(help_text, "--dataset-name", "--dataset_name", "--dataset")
    if dataset_flag:
        cmd.extend([dataset_flag, "princeton-nlp/SWE-bench_Lite"])
    split_flag = _has_flag(help_text, "--split")
    if split_flag:
        cmd.extend([split_flag, "test"])
    instance_flag = _has_flag(help_text, "--instance-id", "--instance_id", "--instance")
    if not instance_flag:
        raise RuntimeError("mini-extra help does not expose an instance id flag; set MINI_SWE_AGENT_RUN_CMD")
    cmd.extend([instance_flag, instance_id])
    model_flag = _has_flag(help_text, "--model-name", "--model_name", "--model")
    if model_flag:
        cmd.extend([model_flag, f"hosted_vllm/{model}"])
    api_base_flag = _has_flag(help_text, "--api-base", "--api_base", "--base-url", "--base_url")
    if api_base_flag:
        cmd.extend([api_base_flag, server])
    out_flag = _has_flag(help_text, "--output-dir", "--output_dir", "--traj-dir", "--traj_dir", "--output")
    if out_flag:
        output_value = run_out / "trajectory.traj.json" if "Output trajectory file" in help_text else run_out
        cmd.extend([out_flag, str(output_value)])
    steps_flag = _has_flag(help_text, "--max-steps", "--max_steps", "--max-iterations", "--max_iterations", "--max-iters", "--max_iters")
    if steps_flag:
        cmd.extend([steps_flag, str(max_steps)])
    seed_flag = _has_flag(help_text, "--seed")
    if seed_flag:
        cmd.extend([seed_flag, str(seed)])
    temp_flag = _has_flag(help_text, "--temperature")
    if temp_flag:
        cmd.extend([temp_flag, str(temperature)])
    env_flag = _has_flag(help_text, "--environment-class")
    if env_flag:
        cmd.extend([env_flag, "docker"])
    if "--yolo" in help_text:
        cmd.append("--yolo")
    if "--cost-limit" in help_text:
        cmd.extend(["--cost-limit", "0"])
    if "--exit-immediately" in help_text:
        cmd.append("--exit-immediately")
    return cmd


def _choose_subcommand(results_dir: Path) -> tuple[list[str], str, str]:
    mini_extra = shutil.which("mini-extra")
    if not mini_extra and Path(".venv/bin/mini-extra").exists():
        mini_extra = ".venv/bin/mini-extra"
    if not mini_extra:
        raise RuntimeError("mini-extra is not installed; run scripts/setup_pr3_miniswe_swebench.sh")
    single_ok, single_help = _capture([mini_extra, "swebench-single", "--help"])
    (results_dir / "mini_extra_swebench_single_help.txt").write_text(single_help, encoding="utf-8")
    if single_ok:
        return [mini_extra, "swebench-single"], single_help, "swebench-single"
    bench_ok, bench_help = _capture([mini_extra, "swebench", "--help"])
    (results_dir / "mini_extra_swebench_help.txt").write_text(bench_help, encoding="utf-8")
    if bench_ok:
        return [mini_extra, "swebench"], bench_help, "swebench"
    raise RuntimeError("neither `mini-extra swebench-single --help` nor `mini-extra swebench --help` ran successfully")


def _write_failures(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "instance_id", "rollout_id", "status", "log_path", "message"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_real(args: argparse.Namespace) -> int:
    instances = _read_instances(args.instance_list, args.num_instances)
    raw_root = Path(args.raw_root)
    logs_root = Path(args.logs_dir) / args.run_id
    results = Path(args.results_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": env.get("OPENAI_API_KEY", "dummy"),
            "OPENAI_BASE_URL": args.server,
            "OPENAI_API_BASE": args.server,
            "LITELLM_API_BASE": args.server,
            "NO_PROXY": env.get("NO_PROXY", "localhost,127.0.0.1"),
            "no_proxy": env.get("no_proxy", "localhost,127.0.0.1"),
        }
    )

    template = os.environ.get("MINI_SWE_AGENT_RUN_CMD", "").strip()
    if template:
        subcommand = []
        help_text = ""
        cli_name = "env-template"
    else:
        subcommand, help_text, cli_name = _choose_subcommand(results)

    failures: list[dict[str, str]] = []
    successes = 0
    for instance_id in instances:
        for rollout in range(args.rollouts):
            rollout_id = f"rollout_{rollout}"
            run_out = raw_root / instance_id / f"_mini_extra_{rollout_id}"
            run_out.mkdir(parents=True, exist_ok=True)
            log_path = logs_root / instance_id / f"{rollout_id}.log"
            seed = args.seed_base + rollout
            temperature = args.temperature + (0.05 * rollout if args.rollouts > 1 else 0.0)
            values = {
                "instance_id": instance_id,
                "rollout_id": rollout_id,
                "run_out": str(run_out),
                "raw_root": str(raw_root),
                "server": args.server,
                "model": args.model,
                "hosted_model": f"hosted_vllm/{args.model}",
                "max_steps": str(args.max_steps),
                "seed": str(seed),
                "temperature": str(temperature),
                "config": args.config,
            }
            try:
                if template:
                    cmd = _template_command(template, values)
                else:
                    cmd = _auto_command(
                        help_text=help_text,
                        subcommand=subcommand,
                        instance_id=instance_id,
                        run_out=run_out,
                        server=args.server,
                        model=args.model,
                        max_steps=args.max_steps,
                        seed=seed,
                        temperature=temperature,
                        config=args.config,
                    )
            except Exception as exc:
                failures.append(
                    {
                        "run_id": args.run_id,
                        "instance_id": instance_id,
                        "rollout_id": rollout_id,
                        "status": "not_run",
                        "log_path": str(log_path),
                        "message": str(exc),
                    }
                )
                continue

            rc = _run(cmd, log_path=log_path, env=env, timeout=args.timeout_seconds)
            traj = _find_trajectory(run_out)
            target = raw_root / instance_id / f"{rollout_id}.traj.json"
            if traj:
                _copy_trajectory(traj, target)
                successes += 1
                failures.append(
                    {
                        "run_id": args.run_id,
                        "instance_id": instance_id,
                        "rollout_id": rollout_id,
                        "status": "success" if rc == 0 else "trajectory_with_error",
                        "log_path": str(log_path),
                        "message": f"returncode={rc}; cli={cli_name}; trajectory={traj}",
                    }
                )
            else:
                failures.append(
                    {
                        "run_id": args.run_id,
                        "instance_id": instance_id,
                        "rollout_id": rollout_id,
                        "status": "failed",
                        "log_path": str(log_path),
                        "message": f"returncode={rc}; trajectory_found={bool(traj)}",
                    }
                )

    failures_path = results / f"{args.run_id}_run_failures.csv"
    _write_failures(failures_path, failures)
    summary = {
        "run_id": args.run_id,
        "instances_requested": len(instances),
        "rollouts": args.rollouts,
        "successful_trajectories": successes,
        "failures_csv": str(failures_path),
        "raw_root": str(raw_root),
    }
    (results / f"{args.run_id}_runner_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if successes > 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-list", required=True)
    ap.add_argument("--num-instances", type=int, required=True)
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--server", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="qwen-coder-7b")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed-base", type=int, default=1337)
    ap.add_argument("--timeout-seconds", type=int, default=3600)
    ap.add_argument("--config", default="configs/miniswe_swebench_vllm.yaml")
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--logs-dir", default="data/logs")
    args = ap.parse_args()
    raise SystemExit(run_real(args))


if __name__ == "__main__":
    main()
