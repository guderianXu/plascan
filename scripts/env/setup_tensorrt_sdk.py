#!/usr/bin/env python3
"""Acquire and validate the pinned TensorRT C++ SDK used by PlaScan."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


TENSORRT_VERSION = "10.15.1.29"
TENSORRT_ARCHIVE_NAME = "TensorRT-10.15.1.29.Windows.amd64.cuda-13.1.zip"
TENSORRT_DOWNLOAD_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/"
    f"10.15.1/zip/{TENSORRT_ARCHIVE_NAME}"
)
TENSORRT_LICENSE_URL = "https://docs.nvidia.com/deeplearning/tensorrt/latest/license.html"
TENSORRT_ARCHIVE_SIZE = 1_934_486_030
# Filled from the pinned official NVIDIA archive. Keep this value mandatory so a
# changed CDN object cannot silently enter a reproducible PlaScan build.
TENSORRT_ARCHIVE_SHA256 = "83304c1f9ab86534f083bc4864691b38dee34135fd1aa0391644531c7c4e1e1e"


def _is_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "on", "true", "yes"}


def _library_exists(root: Path, patterns: tuple[str, ...]) -> bool:
    for directory in (root / "lib", root / "lib" / "x64"):
        if directory.is_dir() and any(directory.glob(pattern) for pattern in patterns):
            return True
    return False


def _runtime_exists(root: Path, patterns: tuple[str, ...]) -> bool:
    for directory in (root / "bin", root / "lib", root / "lib" / "x64"):
        if directory.is_dir() and any(directory.glob(pattern) for pattern in patterns):
            return True
    return False


def is_valid_sdk(root: Path) -> bool:
    """Return whether *root* contains the complete C++ build/runtime SDK."""
    return all(
        (
            (root / "include" / "NvInferRuntime.h").is_file(),
            (root / "include" / "NvOnnxParser.h").is_file(),
            _library_exists(root, ("nvinfer_10.lib", "nvinfer.lib")),
            _library_exists(root, ("nvonnxparser_10.lib", "nvonnxparser.lib")),
            _runtime_exists(root, ("nvinfer_10.dll", "nvinfer.dll")),
            _runtime_exists(root, ("nvonnxparser_10.dll", "nvonnxparser.dll")),
            _runtime_exists(root, ("nvinfer_plugin_10.dll", "nvinfer_plugin.dll")),
            _runtime_exists(root, ("nvinfer_builder_resource_*.dll",)),
        )
    )


def find_sdk_root(candidate: Path) -> Path | None:
    candidate = candidate.expanduser().resolve()
    if is_valid_sdk(candidate):
        return candidate
    if candidate.is_dir():
        for child in sorted(candidate.iterdir()):
            if child.is_dir() and is_valid_sdk(child):
                return child.resolve()
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path) -> None:
    actual_size = path.stat().st_size
    if actual_size != TENSORRT_ARCHIVE_SIZE:
        raise RuntimeError(
            f"TensorRT archive size mismatch: {path} has {actual_size} bytes, "
            f"expected {TENSORRT_ARCHIVE_SIZE}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256.lower() != TENSORRT_ARCHIVE_SHA256.lower():
        raise RuntimeError(
            f"TensorRT archive SHA-256 mismatch: {path} has {actual_sha256}, "
            f"expected {TENSORRT_ARCHIVE_SHA256}"
        )


def _download_with_curl(url: str, destination: Path) -> None:
    curl = shutil.which("curl")
    if not curl:
        with urllib.request.urlopen(url) as response, destination.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        return
    subprocess.run(
        [
            curl,
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def download_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / TENSORRT_ARCHIVE_NAME
    if archive.is_file():
        validate_archive(archive)
        return archive

    partial = archive.with_suffix(f"{archive.suffix}.part")
    print(f"Downloading TensorRT {TENSORRT_VERSION} to {archive}")
    _download_with_curl(TENSORRT_DOWNLOAD_URL, partial)
    validate_archive(partial)
    partial.replace(archive)
    return archive


def install_archive(archive: Path, install_root: Path) -> Path:
    existing = find_sdk_root(install_root)
    if existing:
        return existing

    install_parent = install_root.parent.resolve()
    staging = install_parent / f".{install_root.name}.extracting"
    install_parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        print(f"Extracting TensorRT {TENSORRT_VERSION} to {install_root}")
        with zipfile.ZipFile(archive) as package:
            package.extractall(staging)
        extracted = find_sdk_root(staging)
        if not extracted:
            raise RuntimeError(f"TensorRT archive does not contain a complete C++ SDK: {archive}")
        if install_root.exists():
            shutil.rmtree(install_root)
        if extracted == staging:
            staging.replace(install_root)
        else:
            shutil.move(str(extracted), str(install_root))
            shutil.rmtree(staging)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    validated = find_sdk_root(install_root)
    if not validated:
        raise RuntimeError(f"Installed TensorRT SDK failed validation: {install_root}")
    return validated


def prepare_tensorrt_sdk(
    repo_root: Path,
    environment: dict[str, str],
    *,
    requested_root: str = "",
    requested_archive: str = "",
    auto_install: bool = True,
    accept_license: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """Find or install TensorRT; return None so callers can fall back safely."""
    candidates = [
        requested_root,
        environment.get("TENSORRT_ROOT", ""),
        environment.get("TensorRT_ROOT", ""),
        str(repo_root / "build" / "env" / "sdk" / "tensorrt" / TENSORRT_VERSION),
    ]
    for value in candidates:
        if value:
            found = find_sdk_root(Path(value))
            if found:
                print(f"Using TensorRT SDK: {found}")
                return found

    if not auto_install or sys.platform != "win32":
        return None

    archive = Path(requested_archive).expanduser().resolve() if requested_archive else None
    if archive and not archive.is_file():
        print(f"warning: requested TensorRT archive does not exist: {archive}", file=sys.stderr)
        return None

    if not archive and not accept_license:
        print(
            "warning: TensorRT SDK is not installed. Automatic download requires one-time "
            "license acceptance via --accept-tensorrt-license or "
            "PLASCAN_ACCEPT_TENSORRT_LICENSE=ON; continuing with the CUDA/CPU fallback. "
            f"License: {TENSORRT_LICENSE_URL}",
            file=sys.stderr,
        )
        return None

    install_root = repo_root / "build" / "env" / "sdk" / "tensorrt" / TENSORRT_VERSION
    if dry_run:
        print(f"Would install TensorRT {TENSORRT_VERSION} into {install_root}")
        return None

    try:
        if archive:
            validate_archive(archive)
        else:
            cache_dir = (
                repo_root / "build" / "env" / "downloads" / "tensorrt" / TENSORRT_VERSION
            )
            archive = download_archive(cache_dir)
        return install_archive(archive, install_root)
    except (OSError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(
            f"warning: TensorRT automatic installation failed: {error}; "
            "continuing with the CUDA/CPU fallback.",
            file=sys.stderr,
        )
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--root", default="", help="Use an existing TensorRT SDK root")
    parser.add_argument("--archive", default="", help="Install from an existing official ZIP")
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = dict(os.environ)
    root = prepare_tensorrt_sdk(
        Path(args.repo_root).resolve(),
        environment,
        requested_root=args.root,
        requested_archive=args.archive,
        accept_license=args.accept_license or _is_enabled(
            environment.get("PLASCAN_ACCEPT_TENSORRT_LICENSE")
        ),
        dry_run=args.dry_run,
    )
    if root:
        print(root)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
