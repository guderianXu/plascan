#!/usr/bin/env python3
"""Run CTest with a portable default parallelism of half the logical CPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Mapping, Sequence


def default_parallel_jobs(logical_cpus: int | None = None) -> int:
    """Return half the available logical CPUs, with at least one worker."""
    cpu_count = os.cpu_count() if logical_cpus is None else logical_cpus
    if cpu_count is None or cpu_count < 1:
        cpu_count = 1
    return max(1, cpu_count // 2)


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
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
