#!/usr/bin/env python3
"""Download and export SAM2.1 TorchScript models for PlaScan.

This script assumes the PlaScan Python runtime already contains torch and sam2.
It does not install Python packages. The GUI calls it as a long-running process
and reads line-oriented progress from stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Sam21InstallVariant:
    token: str
    checkpoint_name: str
    url: str


BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

VARIANTS: dict[str, Sam21InstallVariant] = {
    "tiny": Sam21InstallVariant(
        token="tiny",
        checkpoint_name="sam2.1_hiera_tiny.pt",
        url=f"{BASE_URL}/sam2.1_hiera_tiny.pt",
    ),
    "small": Sam21InstallVariant(
        token="small",
        checkpoint_name="sam2.1_hiera_small.pt",
        url=f"{BASE_URL}/sam2.1_hiera_small.pt",
    ),
    "base_plus": Sam21InstallVariant(
        token="base_plus",
        checkpoint_name="sam2.1_hiera_base_plus.pt",
        url=f"{BASE_URL}/sam2.1_hiera_base_plus.pt",
    ),
    "large": Sam21InstallVariant(
        token="large",
        checkpoint_name="sam2.1_hiera_large.pt",
        url=f"{BASE_URL}/sam2.1_hiera_large.pt",
    ),
}


def clean_path(path: Path) -> str:
    return path.expanduser().resolve().as_posix()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_python_executable() -> Path:
    raw = os.environ.get("PLASCAN_PYTHON_EXECUTABLE") or os.environ.get("PLASCAN_PYTHON") or sys.executable
    return Path(raw).expanduser()


def emit(message: str) -> None:
    print(message, flush=True)


def build_export_command(
    *,
    python_exe: Path,
    source_dir: Path,
    model_dir: Path,
    variant: Sam21InstallVariant,
    devices: str,
    input_size: int,
) -> list[str]:
    export_script = source_dir / "scripts" / "models" / "export_sam21_torchscript.py"
    checkpoint = model_dir / variant.checkpoint_name
    return [
        clean_path(python_exe),
        clean_path(export_script),
        "--variant",
        variant.token,
        "--checkpoint",
        clean_path(checkpoint),
        "--devices",
        devices,
        "--input-size",
        str(input_size),
        "--output-dir",
        clean_path(model_dir),
    ]


def download_checkpoint(url: str, destination: Path, *, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        emit(f"PROGRESS checkpoint exists: {destination}")
        return

    tmp_path = destination.with_suffix(destination.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()

    emit(f"PROGRESS download started: {url}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as output:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else 0
        downloaded = 0
        last_percent = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = int(downloaded * 100 / total)
                if percent != last_percent and (percent % 5 == 0 or percent == 100):
                    emit(f"PROGRESS download {percent}% {downloaded}/{total}")
                    last_percent = percent
            else:
                emit(f"PROGRESS download {downloaded} bytes")

    tmp_path.replace(destination)
    emit(f"PROGRESS download finished: {destination}")


def run_export(command: list[str]) -> int:
    emit("PROGRESS export started")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        emit(f"EXPORT {line.rstrip()}")
    return process.wait()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="tiny")
    parser.add_argument("--source-dir", type=Path, default=repo_root())
    parser.add_argument("--model-dir", type=Path, default=repo_root() / "resources" / "models")
    parser.add_argument("--python-executable", type=Path, default=default_python_executable())
    parser.add_argument("--devices", default="auto", help="auto, cpu, cuda, or cpu,cuda")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    variant = VARIANTS[args.variant]
    source_dir = args.source_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    python_exe = args.python_executable.expanduser().resolve()

    export_script = source_dir / "scripts" / "models" / "export_sam21_torchscript.py"
    if not export_script.exists():
        raise FileNotFoundError(f"SAM2.1 export script not found: {export_script}")
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")

    checkpoint = model_dir / variant.checkpoint_name
    command = build_export_command(
        python_exe=python_exe,
        source_dir=source_dir,
        model_dir=model_dir,
        variant=variant,
        devices=args.devices,
        input_size=args.input_size,
    )

    emit(f"PROGRESS variant {variant.token}")
    emit(f"PROGRESS model_dir {model_dir}")
    emit(f"PROGRESS python {python_exe}")
    if args.dry_run:
        emit(f"DRYRUN download {variant.url} -> {checkpoint}")
        emit("DRYRUN export " + " ".join(command))
        return 0

    download_checkpoint(variant.url, checkpoint, force=args.force_download)
    exit_code = run_export(command)
    if exit_code != 0:
        raise RuntimeError(f"SAM2.1 TorchScript export failed with exit code {exit_code}")
    emit("PROGRESS install finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
