#!/usr/bin/env python3
"""Create PlaScan's fixed isolated Python development runtime.

The recommended development and packaging runtime lives at:

    .venv

This script intentionally uses the standard-library venv module instead of
conda so the same workflow can be used on Windows and Linux.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from env_common import default_output_dir, host_platform, repo_root as detect_repo_root, run, write_env_files


BASE_PACKAGES = [
    "numpy",
    "opencv-python",
    "pymupdf",
    "kornia",
    "git+https://github.com/cvg/LightGlue.git",
]

TORCH_PACKAGES = [
    "torch",
    "torchvision",
]


@dataclass(frozen=True)
class DependencyPlan:
    torch_index_url: str
    torch_packages: list[str]
    base_packages: list[str]
    extra_packages: list[str]


def default_runtime_dir(repo_root: Path) -> Path:
    return repo_root / ".venv"


def runtime_python_path(runtime_dir: Path, platform_name: str | None = None) -> Path:
    platform_value = platform_name or host_platform()
    if platform_value == "windows":
        return runtime_dir / "Scripts" / "python.exe"
    return runtime_dir / "bin" / "python"


def torch_index_url(device: str, cuda_wheel: str) -> str:
    if device == "cpu":
        return "https://download.pytorch.org/whl/cpu"
    return f"https://download.pytorch.org/whl/{cuda_wheel}"


def dependency_plan(
    *,
    device: str,
    cuda_wheel: str,
    extra_packages: list[str] | None = None,
    torch_index_url_override: str = "",
) -> DependencyPlan:
    return DependencyPlan(
        torch_index_url=torch_index_url_override or torch_index_url(device, cuda_wheel),
        torch_packages=list(TORCH_PACKAGES),
        base_packages=list(BASE_PACKAGES),
        extra_packages=list(extra_packages or []),
    )


def environment_values(
    *,
    repo_root: Path,
    runtime_dir: Path,
    python_exe: Path,
    device: str,
    cuda_wheel: str,
    cuda_root: str,
) -> dict[str, str]:
    values = {
        "PLASCAN_PYTHON_RUNTIME_DIR": str(runtime_dir),
        "PLASCAN_PYTHON_EXECUTABLE": str(python_exe),
        "PLASCAN_PYTHON": str(python_exe),
        "PLASCAN_MODEL_DIR": str(repo_root / "resources" / "models"),
        "PLASCAN_SCRIPT_DIR": str(repo_root / "scripts"),
        "PLASCAN_PYTHON_DEVICE": device,
        "PLASCAN_PYTHON_CUDA_WHEEL": cuda_wheel,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUDAToolkit_ROOT": cuda_root,
        "CUDA_TOOLKIT_ROOT_DIR": cuda_root,
    }
    return values


def write_runtime_env_files(
    *,
    output_dir: Path,
    repo_root: Path,
    runtime_dir: Path,
    python_exe: Path,
    device: str,
    cuda_wheel: str,
    cuda_root: str,
) -> Path:
    return write_env_files(
        output_dir,
        environment_values(
            repo_root=repo_root,
            runtime_dir=runtime_dir,
            python_exe=python_exe,
            device=device,
            cuda_wheel=cuda_wheel,
            cuda_root=cuda_root,
        ),
    )


def create_runtime(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_dir).expanduser().resolve() if args.source_dir else detect_repo_root()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve() if args.runtime_dir else default_runtime_dir(source_dir)
    python_exe = runtime_python_path(runtime_dir)

    if not args.skip_create:
        bootstrap_python = args.python or sys.executable
        run([bootstrap_python, "-m", "venv", str(runtime_dir)], dry_run=args.dry_run)

    if not args.skip_install:
        plan = dependency_plan(
            device=args.device,
            cuda_wheel=args.cuda_wheel,
            extra_packages=args.extra_package,
            torch_index_url_override=args.torch_index_url or "",
        )
        run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], dry_run=args.dry_run)
        run(
            [str(python_exe), "-m", "pip", "install", *plan.torch_packages, "--index-url", plan.torch_index_url],
            dry_run=args.dry_run,
        )
        packages = plan.base_packages + plan.extra_packages
        if packages:
            run([str(python_exe), "-m", "pip", "install", *packages], dry_run=args.dry_run)

    return write_runtime_env_files(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        repo_root=source_dir,
        runtime_dir=runtime_dir,
        python_exe=python_exe,
        device=args.device,
        cuda_wheel=args.cuda_wheel,
        cuda_root=args.cuda_root or "",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", help="PlaScan source tree root. Defaults to the current script repository.")
    parser.add_argument("--runtime-dir", help="Runtime directory. Defaults to .venv under the source tree.")
    parser.add_argument("--python", help="Bootstrap Python executable. Defaults to the current Python.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-wheel", default="cu130", help="PyTorch CUDA wheel tag, for example cu128 or cu130.")
    parser.add_argument("--torch-index-url", help="Override the PyTorch pip index URL.")
    parser.add_argument("--cuda-root", help="Existing CUDA Toolkit root to write into env files.")
    parser.add_argument("--extra-package", action="append", default=[], help="Additional pip package to install.")
    parser.add_argument("--skip-create", action="store_true", help="Use an existing runtime instead of creating it.")
    parser.add_argument("--skip-install", action="store_true", help="Only create/detect the runtime and write env files.")
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    env_file = create_runtime(parse_args())
    print(f"Environment JSON: {env_file}")


if __name__ == "__main__":
    main()
