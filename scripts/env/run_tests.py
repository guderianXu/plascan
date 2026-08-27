#!/usr/bin/env python3
"""Run CTest with a portable default parallelism using all logical CPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


def default_parallel_jobs(logical_cpus: int | None = None) -> int:
    """Return all available logical CPUs, with at least one worker."""
    cpu_count = os.cpu_count() if logical_cpus is None else logical_cpus
    if cpu_count is None or cpu_count < 1:
        cpu_count = 1
    return cpu_count


def has_parallel_override(ctest_args: Sequence[str]) -> bool:
    """Return whether the forwarded CTest arguments already select parallelism."""
    for argument in ctest_args:
        if argument in {"-j", "--parallel"}:
            return True
        if argument.startswith("--parallel="):
            return True
        if argument.startswith("-j") and len(argument) > 2:
            return True
    return False


def build_ctest_command(
    ctest_args: Sequence[str],
    *,
    jobs: int | None = None,
    environment: Mapping[str, str] | None = None,
    logical_cpus: int | None = None,
) -> list[str]:
    """Build a CTest command while preserving explicit parallel overrides."""
    forwarded_args = list(ctest_args)
    if jobs is not None and jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if jobs is not None and has_parallel_override(forwarded_args):
        raise ValueError("use either --jobs or a forwarded CTest --parallel/-j option, not both")

    command = ["ctest"]
    configured_environment = os.environ if environment is None else environment
    has_environment_override = bool(configured_environment.get("CTEST_PARALLEL_LEVEL", "").strip())
    if not has_parallel_override(forwarded_args) and (jobs is not None or not has_environment_override):
        parallel_jobs = jobs if jobs is not None else default_parallel_jobs(logical_cpus)
        command.extend(["--parallel", str(parallel_jobs)])
    command.extend(forwarded_args)
    return command


def ctest_test_directory(
    ctest_args: Sequence[str],
    *,
    working_directory: Path | None = None,
) -> Path | None:
    """Return the resolved directory passed to CTest through --test-dir."""
    for index, argument in enumerate(ctest_args):
        if argument == "--test-dir" and index + 1 < len(ctest_args):
            value = ctest_args[index + 1]
            break
        if argument.startswith("--test-dir="):
            value = argument.partition("=")[2]
            break
    else:
        return None

    if not value:
        return None
    base_directory = Path.cwd() if working_directory is None else working_directory
    test_directory = Path(value)
    if not test_directory.is_absolute():
        test_directory = base_directory / test_directory
    return test_directory.resolve()


def build_test_environment(
    ctest_args: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    working_directory: Path | None = None,
) -> dict[str, str]:
    """Bind Windows build-tree DLLs and Qt plugins for CTest discovery and runs."""
    result = dict(os.environ if environment is None else environment)
    effective_platform = os.name if platform_name is None else platform_name
    if effective_platform != "nt":
        return result

    test_directory = ctest_test_directory(
        ctest_args, working_directory=working_directory)
    if test_directory is None:
        return result

    runtime_directories = [test_directory / "bin", test_directory / "tests"]
    existing_runtime_directories = [
        str(directory) for directory in runtime_directories if directory.is_dir()
    ]
    if existing_runtime_directories:
        current_path = result.get("PATH", "")
        path_entries = existing_runtime_directories
        if current_path:
            path_entries.append(current_path)
        result["PATH"] = os.pathsep.join(path_entries)

    qt_plugin_root = test_directory / "bin"
    if (qt_plugin_root / "platforms").is_dir():
        current_plugin_path = result.get("QT_PLUGIN_PATH", "")
        plugin_paths = [str(qt_plugin_root)]
        if current_plugin_path:
            plugin_paths.append(current_plugin_path)
        result["QT_PLUGIN_PATH"] = os.pathsep.join(plugin_paths)

    return result


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        help="Override the default test worker count; must be at least 1.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the CTest command without running it.")
    return parser.parse_known_args()


def main() -> int:
    args, ctest_args = parse_args()
    try:
        command = build_ctest_command(ctest_args, jobs=args.jobs)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print("+ " + subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0
    test_environment = build_test_environment(ctest_args)
    return subprocess.run(command, check=False, env=test_environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
