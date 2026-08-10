#!/usr/bin/env python3
"""Resolve PlaScan executables across single- and multi-config builds."""

from __future__ import annotations

import os
from pathlib import Path


_CONFIGURATION_DIRECTORIES = ("Release", "RelWithDebInfo", "Debug")


def _executable_names(name: str) -> tuple[str, ...]:
    if name.lower().endswith(".exe"):
        return (name,)
    if os.name == "nt":
        return (f"{name}.exe", name)
    return (name, f"{name}.exe")


def executable_candidates(build_dir: Path, name: str) -> tuple[Path, ...]:
    """Return deterministic candidate paths for an executable in a build tree."""

    root = Path(build_dir)
    directories = [root, root / "bin"]
    directories.extend(root / "bin" / config for config in _CONFIGURATION_DIRECTORIES)
    directories.extend(root / config for config in _CONFIGURATION_DIRECTORIES)

    candidates: list[Path] = []
    for directory in directories:
        for executable_name in _executable_names(name):
            candidate = directory / executable_name
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def resolve_build_executable(build_dir: Path, name: str) -> Path:
    """Resolve an executable from a CMake build tree.

    When no candidate exists, return the conventional ``bin`` candidate so the
    caller can report a stable and actionable missing-path error.
    """

    candidates = executable_candidates(build_dir, name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    fallback_name = _executable_names(name)[0]
    return Path(build_dir) / "bin" / fallback_name


def resolve_explicit_executable(path: Path) -> Path:
    """Resolve an explicit path, including CMake multi-config subdirectories."""

    requested = Path(path)
    executable_names = _executable_names(requested.name)
    candidates = [requested]
    candidates.extend(requested.with_name(name) for name in executable_names)
    candidates.extend(
        requested.parent / config / name
        for config in _CONFIGURATION_DIRECTORIES
        for name in executable_names
    )

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    return requested
