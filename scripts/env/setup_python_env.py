#!/usr/bin/env python3
"""Create or update the Python runtime used by PlaScan helper scripts."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from env_common import (
    capture,
    default_output_dir,
    fail,
    host_platform,
    python_executable_from_prefix,
    run,
    write_env_files,
)


BASE_PACKAGES = [
    "numpy",
    "scipy",
    "opencv-python",
    "kornia",
    "git+https://github.com/cvg/LightGlue.git",
]

OPTIONAL_PACKAGES: list[str] = []


def conda_executable() -> str:
    exe = shutil.which("conda") or shutil.which("mamba")
    if not exe:
        fail("conda/mamba not found. Use --manager venv or install Miniconda/Mambaforge.")
    return exe


def conda_python_from_name(conda: str, name: str, dry_run: bool) -> Path:
    code = "import sys; print(sys.executable)"
    out = capture([conda, "run", "-n", name, "python", "-c", code], dry_run=dry_run)
    if dry_run:
        suffix = "python.exe" if host_platform() == "windows" else "bin/python"
        return Path(f"<conda-env:{name}>") / suffix
    return Path(out)


def torch_index_url(device: str, cuda_wheel: str) -> str:
    if device == "cpu":
        return "https://download.pytorch.org/whl/cpu"
    return f"https://download.pytorch.org/whl/{cuda_wheel}"


def create_or_update_env(args: argparse.Namespace) -> Path:
    manager = args.manager
    if manager == "auto":
        manager = "conda" if shutil.which("conda") or shutil.which("mamba") else "venv"

    if manager == "conda":
        conda = conda_executable()
        if args.prefix:
            prefix = Path(args.prefix).expanduser().resolve()
            if not args.skip_create:
                run([conda, "create", "-y", "-p", str(prefix), f"python={args.python}", "pip"], dry_run=args.dry_run)
            python = python_executable_from_prefix(prefix)
        else:
            if not args.skip_create:
                run([conda, "create", "-y", "-n", args.name, f"python={args.python}", "pip"], dry_run=args.dry_run)
            python = conda_python_from_name(conda, args.name, args.dry_run)
    else:
        prefix = Path(args.prefix or (default_output_dir() / "python")).expanduser().resolve()
        if not args.skip_create:
            run([sys.executable, "-m", "venv", str(prefix)], dry_run=args.dry_run)
        python = python_executable_from_prefix(prefix)

    if not args.skip_install:
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"], dry_run=args.dry_run)
        index_url = args.torch_index_url or torch_index_url(args.device, args.cuda_wheel)
        run(
            [str(python), "-m", "pip", "install", "torch", "torchvision", "--index-url", index_url],
            dry_run=args.dry_run,
        )
        packages = list(BASE_PACKAGES)
        if args.with_optional:
            packages.extend(OPTIONAL_PACKAGES)
        packages.extend(args.extra_package)
        if packages:
            run([str(python), "-m", "pip", "install", *packages], dry_run=args.dry_run)

    values = {
        "PLASCAN_PYTHON_EXECUTABLE": str(python),
        "PLASCAN_PYTHON": str(python),
        "CUDAToolkit_ROOT": args.cuda_root or "",
        "CUDA_TOOLKIT_ROOT_DIR": args.cuda_root or "",
    }
    return write_env_files(Path(args.output_dir).expanduser().resolve(), values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager", choices=["auto", "conda", "venv"], default="auto")
    parser.add_argument("--name", default="plascan", help="conda environment name when --prefix is not used")
    parser.add_argument("--prefix", help="conda/venv prefix path")
    parser.add_argument("--python", default="3.12", help="Python version for new conda envs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-wheel", default="cu128", help="PyTorch CUDA wheel tag, e.g. cu121/cu124/cu128")
    parser.add_argument("--torch-index-url", help="Override PyTorch pip index URL")
    parser.add_argument("--cuda-root", help="Existing CUDA Toolkit root to write into env files")
    parser.add_argument("--with-optional", action="store_true", help="Install optional experimental Python packages")
    parser.add_argument("--extra-package", action="append", default=[], help="Additional pip package to install")
    parser.add_argument("--skip-create", action="store_true", help="Use an existing env instead of creating it")
    parser.add_argument("--skip-install", action="store_true", help="Only detect/write environment files")
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    env_file = create_or_update_env(parse_args())
    print(f"Environment JSON: {env_file}")


if __name__ == "__main__":
    main()
