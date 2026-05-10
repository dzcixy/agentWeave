from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any


PREFERRED_REPOS = (
    "django/django",
    "sympy/sympy",
    "scikit-learn/scikit-learn",
    "matplotlib/matplotlib",
    "pytest-dev/pytest",
    "pallets/flask",
)

AVOID_REPO_HINTS = (
    "tensorflow",
    "pytorch",
    "ray-project",
    "apache",
)


def _repo(row: dict[str, Any]) -> str:
    return str(row.get("repo") or row.get("repo_name") or "")


def _instance_id(row: dict[str, Any]) -> str:
    return str(row.get("instance_id") or "")


def _local_swebench_image_ids() -> set[str]:
    try:
        cp = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], text=True, capture_output=True, timeout=20)
    except Exception:
        return set()
    if cp.returncode != 0:
        return set()
    out: set[str] = set()
    for line in cp.stdout.splitlines():
        marker = "sweb.eval.x86_64."
        if marker not in line or not line.endswith(":latest"):
            continue
        iid = line.split(marker, 1)[1].rsplit(":latest", 1)[0].replace("_1776_", "__")
        out.add(iid)
    return out


def _score(row: dict[str, Any], local_images: set[str] | None = None) -> tuple[int, str]:
    repo = _repo(row)
    iid = _instance_id(row)
    local_rank = 0 if local_images and iid in local_images else 1
    if any(h in repo.lower() for h in AVOID_REPO_HINTS):
        repo_rank = 100
    else:
        try:
            repo_rank = PREFERRED_REPOS.index(repo)
        except ValueError:
            repo_rank = 50
    text_size = len(str(row.get("problem_statement") or "")) + len(str(row.get("hints_text") or ""))
    return local_rank * 1_000_000_000 + repo_rank * 1_000_000 + text_size, iid


def _load_dataset(name: str, split: str):
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("datasets is not installed; run scripts/setup_pr3_miniswe_swebench.sh") from exc
    return load_dataset(name, split=split)


def select_instances(
    dataset: str,
    split: str,
    num_instances: int,
    explicit_ids: list[str] | None = None,
    prefer_local_docker_images: bool = True,
) -> tuple[list[str], int, list[dict[str, Any]]]:
    ds = _load_dataset(dataset, split)
    rows = [dict(r) for r in ds]
    by_id = {_instance_id(r): r for r in rows if _instance_id(r)}
    if explicit_ids:
        missing = [iid for iid in explicit_ids if iid not in by_id]
        if missing:
            raise RuntimeError(f"explicit SWE-bench instance ids not found: {', '.join(missing)}")
        selected = explicit_ids[:num_instances]
    else:
        local_images = _local_swebench_image_ids() if prefer_local_docker_images else set()
        selected = [_instance_id(r) for r in sorted(rows, key=lambda r: _score(r, local_images)) if _instance_id(r)][:num_instances]
    return selected, len(rows), [by_id[iid] for iid in selected if iid in by_id]


def write_report(
    path: Path,
    *,
    dataset: str,
    split: str,
    num_available: int,
    selected_ids: list[str],
    ok: bool,
    error: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SWE-bench Instance Selection Report",
        "",
        f"DATASET_ACCESS = {'PASS' if ok else 'FAIL'}",
        f"DATASET_NAME = {dataset}",
        f"SPLIT = {split}",
        f"NUM_AVAILABLE = {num_available}",
        f"NUM_SELECTED = {len(selected_ids)}",
        "SELECTED_INSTANCE_IDS = " + ",".join(selected_ids),
    ]
    if error:
        lines.extend(["", "## Error", error])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-instances", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--explicit-instance-id", action="append", default=[])
    ap.add_argument("--explicit-instance-file")
    ap.add_argument("--prefer-local-docker-images", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--report", default="data/results/swebench_instance_selection_report.md")
    args = ap.parse_args()

    explicit = list(args.explicit_instance_id)
    if args.explicit_instance_file:
        explicit.extend(
            line.strip()
            for line in Path(args.explicit_instance_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    out = Path(args.out)
    report = Path(args.report)
    try:
        selected, num_available, rows = select_instances(
            args.dataset,
            args.split,
            args.num_instances,
            explicit_ids=explicit or None,
            prefer_local_docker_images=args.prefer_local_docker_images,
        )
        if len(selected) < args.num_instances:
            raise RuntimeError(f"only selected {len(selected)} instances from {num_available} available rows")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(selected) + "\n", encoding="utf-8")
        write_report(
            report,
            dataset=args.dataset,
            split=args.split,
            num_available=num_available,
            selected_ids=selected,
            ok=True,
        )
        out_report = out.with_suffix(".selection_report.md")
        write_report(
            out_report,
            dataset=args.dataset,
            split=args.split,
            num_available=num_available,
            selected_ids=selected,
            ok=True,
        )
        repos = sorted({_repo(r) for r in rows if _repo(r)})
        print(
            {
                "DATASET_ACCESS": "PASS",
                "NUM_AVAILABLE": num_available,
                "NUM_SELECTED": len(selected),
                "SELECTED_INSTANCE_IDS": selected,
                "SELECTED_REPOS": repos,
            }
        )
    except Exception as exc:
        write_report(
            report,
            dataset=args.dataset,
            split=args.split,
            num_available=0,
            selected_ids=[],
            ok=False,
            error=str(exc),
        )
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
