#!/usr/bin/env python3
"""Run PlaScan 3D reconstruction over prepared photogrammetry benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "testData" / "photogrammetry_benchmarks"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "benchmark_runs" / "photogrammetry_benchmarks"


@dataclass(frozen=True)
class ReadyDataset:
    dataset_id: str
    list_file: Path
    image_count: int


def count_list_items(list_file: Path) -> int:
    count = 0
    for raw_line in list_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            count += 1
    return count


def discover_ready_datasets(root: Path, selected: set[str] | None = None) -> list[ReadyDataset]:
    if not root.exists():
        return []

    datasets: list[ReadyDataset] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset_id = dataset_dir.name
        if selected and dataset_id not in selected:
            continue

        candidates = [
            dataset_dir / "prepared" / "plascan" / "image_camera.lis",
            dataset_dir / "image_camera.lis",
        ]
        list_file = next((path for path in candidates if path.exists()), None)
        if list_file is None:
            continue

        datasets.append(ReadyDataset(
            dataset_id=dataset_id,
            list_file=list_file.resolve(),
            image_count=count_list_items(list_file),
        ))
    return datasets


def build_reconstruction_command(
    cli_path: Path,
    list_file: Path,
    output_dir: Path,
    stage: str,
    device: str,
    quality: int,
    threads: int,
    timeout: int | None = None,
) -> list[str]:
    del timeout
    command = [
        str(cli_path),
        str(list_file),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--quality",
        str(quality),
        "--threads",
        str(threads),
        "--force",
    ]
    if stage == "sfm":
        command.append("--stop-after-sfm")
    elif stage == "mvs":
        command.append("--skip-mesh")
    return command


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_dataset(command: list[str], output_dir: Path, timeout: int | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout if timeout and timeout > 0 else None,
            check=False,
        )
        elapsed = time.monotonic() - started
        (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        return {
            "returncode": completed.returncode,
            "status": "ok" if completed.returncode == 0 else "failed",
            "elapsed_seconds": elapsed,
            "stdout_log": str(output_dir / "stdout.log"),
            "stderr_log": str(output_dir / "stderr.log"),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        (output_dir / "stdout.log").write_text(exc.stdout or "", encoding="utf-8")
        (output_dir / "stderr.log").write_text(exc.stderr or "", encoding="utf-8")
        return {
            "returncode": 124,
            "status": "timeout",
            "elapsed_seconds": elapsed,
            "stdout_log": str(output_dir / "stdout.log"),
            "stderr_log": str(output_dir / "stderr.log"),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PlaScan photogrammetry benchmark reconstruction batches")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="benchmark root directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="run output root")
    parser.add_argument("--cli", type=Path, default=REPO_ROOT / "build" / "bin" / "three_d_reconstruction_cli")
    parser.add_argument("--stage", choices=["sfm", "mvs", "full"], default="sfm")
    parser.add_argument("--dataset", action="append", default=[], help="dataset id to run; repeatable")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--quality", type=int, default=3)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=0, help="per-dataset timeout in seconds; 0 disables it")
    parser.add_argument("--limit", type=int, default=0, help="run at most this many discovered datasets")
    parser.add_argument("--dry-run", action="store_true", help="write planned commands without running the CLI")
    parser.add_argument("--summary", type=Path, help="summary JSON path; default is output-dir/summary.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_root = args.output_dir.resolve()
    summary_path = args.summary.resolve() if args.summary else output_root / "summary.json"
    selected = set(args.dataset) if args.dataset else None
    datasets = discover_ready_datasets(root, selected)
    if args.limit > 0:
        datasets = datasets[:args.limit]

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if args.dry_run else "running",
        "root": str(root),
        "output_dir": str(output_root),
        "stage": args.stage,
        "datasets": [],
    }

    if not args.dry_run and not args.cli.exists():
        print(f"three_d_reconstruction_cli not found: {args.cli}", file=sys.stderr)
        print("请先编译: cmake --build build --target three_d_reconstruction_cli", file=sys.stderr)
        summary["status"] = "failed"
        summary["reason"] = f"CLI not found: {args.cli}"
        write_json(summary_path, summary)
        return 127

    timeout = args.timeout if args.timeout > 0 else None
    for dataset in datasets:
        run_output_dir = output_root / dataset.dataset_id / args.stage
        command = build_reconstruction_command(
            cli_path=args.cli.resolve(),
            list_file=dataset.list_file,
            output_dir=run_output_dir,
            stage=args.stage,
            device=args.device,
            quality=args.quality,
            threads=args.threads,
            timeout=timeout,
        )
        entry = {
            "dataset_id": dataset.dataset_id,
            "list_file": str(dataset.list_file),
            "image_count": dataset.image_count,
            "output_dir": str(run_output_dir),
            "command": command,
        }
        if args.dry_run:
            entry["status"] = "planned"
        else:
            entry.update(run_dataset(command, run_output_dir, timeout))
        summary["datasets"].append(entry)

    if args.dry_run:
        summary["status"] = "planned"
    elif not datasets:
        summary["status"] = "failed"
        summary["reason"] = "no prepared benchmark datasets found"
    elif all(item.get("status") == "ok" for item in summary["datasets"]):
        summary["status"] = "ok"
    else:
        summary["status"] = "partial"

    write_json(summary_path, summary)
    print(f"summary={summary_path}")
    return 0 if summary["status"] in {"ok", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
