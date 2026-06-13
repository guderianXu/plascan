#!/usr/bin/env python3
"""Prepare vcpkg for PlaScan manifest builds."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from env_common import default_output_dir, fail, host_platform, repo_root, run, write_env_files


def default_triplet() -> str:
    return "x64-windows" if host_platform() == "windows" else "x64-linux-dynamic"


def default_vcpkg_root() -> Path:
    env_root = os.environ.get("VCPKG_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return default_output_dir() / "vcpkg"


def vcpkg_executable(root: Path) -> Path:
    return root / ("vcpkg.exe" if host_platform() == "windows" else "vcpkg")


def bootstrap_script(root: Path) -> Path:
    return root / ("bootstrap-vcpkg.bat" if host_platform() == "windows" else "bootstrap-vcpkg.sh")


def clone_vcpkg(root: Path, repo: str, branch: str, dry_run: bool) -> None:
    if root.exists():
        return
    cmd = ["git", "clone"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([repo, str(root)])
    run(cmd, dry_run=dry_run)


def bootstrap_vcpkg(root: Path, dry_run: bool) -> None:
    exe = vcpkg_executable(root)
    if exe.exists():
        return
    script = bootstrap_script(root)
    if not dry_run and not script.exists():
        fail(f"vcpkg bootstrap script not found: {script}")
    if host_platform() == "windows":
        run([str(script)], dry_run=dry_run, cwd=root)
    else:
        run(["bash", str(script)], dry_run=dry_run, cwd=root)


def install_manifest(args: argparse.Namespace, root: Path, exe: Path) -> None:
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    if not (manifest_root / "vcpkg.json").exists():
        fail(f"vcpkg.json not found under manifest root: {manifest_root}")

    cmd = [
        str(exe),
        "install",
        "--triplet",
        args.triplet,
        f"--x-manifest-root={manifest_root}",
    ]
    if args.host_triplet:
        cmd.extend(["--host-triplet", args.host_triplet])
    for source in args.binarysource:
        cmd.append(f"--binarysource={source}")
    cmd.extend(args.vcpkg_arg)
    run(cmd, dry_run=args.dry_run, cwd=root)


def write_vcpkg_environment(args: argparse.Namespace, root: Path) -> Path:
    toolchain = root / "scripts" / "buildsystems" / "vcpkg.cmake"
    values = {
        "VCPKG_ROOT": str(root),
        "VCPKG_TARGET_TRIPLET": args.triplet,
        "CMAKE_TOOLCHAIN_FILE": str(toolchain),
    }
    return write_env_files(Path(args.output_dir).expanduser().resolve(), values, stem="plascan-vcpkg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(default_vcpkg_root()), help="vcpkg root directory")
    parser.add_argument("--repo", default="https://github.com/microsoft/vcpkg.git")
    parser.add_argument("--branch", default="", help="Optional vcpkg branch/tag to clone")
    parser.add_argument("--triplet", default=default_triplet())
    parser.add_argument("--host-triplet", default="")
    parser.add_argument("--manifest-root", default=str(repo_root()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--clone", action="store_true", help="Clone vcpkg when --root does not exist")
    parser.add_argument("--install", action="store_true", help="Run vcpkg install for the PlaScan manifest")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--binarysource", action="append", default=[], help="Forwarded vcpkg binary cache source")
    parser.add_argument("--vcpkg-arg", action="append", default=[], help="Additional argument forwarded to vcpkg install")
    parser.add_argument("--configure", action="store_true", help="Run configure_with_env.py after writing config")
    parser.add_argument("--build-type", choices=["debug", "release"], default="release")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        if not args.clone:
            fail(f"vcpkg root does not exist: {root}. Re-run with --clone or set VCPKG_ROOT.")
        clone_vcpkg(root, args.repo, args.branch, args.dry_run)

    if not args.skip_bootstrap:
        bootstrap_vcpkg(root, args.dry_run)

    exe = vcpkg_executable(root)
    if not args.dry_run and not exe.exists():
        fail(f"vcpkg executable not found after bootstrap: {exe}")

    if args.install:
        install_manifest(args, root, exe)

    env_file = write_vcpkg_environment(args, root)
    print(f"vcpkg environment JSON: {env_file}")

    if args.configure:
        configure_script = Path(__file__).resolve().parent / "configure_with_env.py"
        python = shutil.which("python") or shutil.which("python3")
        if not python:
            fail("python executable not found for configure step")
        run(
            [
                python,
                str(configure_script),
                "--vcpkg-file",
                str(env_file),
                "--build-type",
                args.build_type,
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
