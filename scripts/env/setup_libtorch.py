#!/usr/bin/env python3
"""Download or register a LibTorch distribution for PlaScan CMake builds."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from env_common import default_output_dir, fail, host_platform, write_env_files


def default_libtorch_url(version: str, device: str, cuda_wheel: str) -> str:
    if device == "cpu":
        channel = "cpu"
        suffix = "cpu"
    else:
        channel = cuda_wheel
        suffix = cuda_wheel

    if host_platform() == "windows":
        archive = f"libtorch-win-shared-with-deps-{version}%2B{suffix}.zip"
    else:
        archive = f"libtorch-cxx11-abi-shared-with-deps-{version}%2B{suffix}.zip"
    return f"https://download.pytorch.org/libtorch/{channel}/{archive}"


def extract_zip(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    libtorch_root = destination / "libtorch"
    if not (libtorch_root / "share" / "cmake" / "Torch" / "TorchConfig.cmake").exists():
        fail(f"LibTorch archive did not produce a valid libtorch directory under {destination}")
    return libtorch_root


def download(url: str, archive: Path, dry_run: bool) -> None:
    print(f"+ download {url}")
    if dry_run:
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--version", default="2.7.1")
    parser.add_argument("--cuda-wheel", default="cu128", help="LibTorch CUDA archive tag, e.g. cu121/cu124/cu128")
    parser.add_argument("--url", help="Override LibTorch zip URL")
    parser.add_argument("--archive", help="Use an existing local LibTorch zip archive")
    parser.add_argument("--libtorch-root", help="Register an existing extracted libtorch directory")
    parser.add_argument("--install-dir", default=str(default_output_dir() / "libtorch"))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--cuda-root", help="Existing CUDA Toolkit root to write into env files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.libtorch_root:
        libtorch_root = Path(args.libtorch_root).expanduser().resolve()
        if not (libtorch_root / "share" / "cmake" / "Torch" / "TorchConfig.cmake").exists():
            fail(f"TorchConfig.cmake not found under {libtorch_root}")
    else:
        install_dir = Path(args.install_dir).expanduser().resolve()
        archive = Path(args.archive).expanduser().resolve() if args.archive else install_dir / "downloads" / "libtorch.zip"
        if not args.archive:
            download(args.url or default_libtorch_url(args.version, args.device, args.cuda_wheel), archive, args.dry_run)
        if args.dry_run:
            libtorch_root = install_dir / "libtorch"
        else:
            if install_dir.exists() and not args.archive:
                shutil.rmtree(install_dir / "libtorch", ignore_errors=True)
            libtorch_root = extract_zip(archive, install_dir)

    torch_dir = libtorch_root / "share" / "cmake" / "Torch"
    values = {
        "PLASCAN_TORCH_DIR": str(torch_dir),
        "Torch_DIR": str(torch_dir),
        "CUDAToolkit_ROOT": args.cuda_root or "",
        "CUDA_TOOLKIT_ROOT_DIR": args.cuda_root or "",
    }
    env_file = write_env_files(Path(args.output_dir).expanduser().resolve(), values)
    print(f"Environment JSON: {env_file}")


if __name__ == "__main__":
    main()
