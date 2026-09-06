#!/usr/bin/env python3
"""Named browser-debug fixtures and their safety defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any


FIXTURE_PROJECTS = {
    "south_building": Path(
        "testData/photogrammetry_benchmarks/test_colmap_south_building/"
        "test_colmap_south_building.plascan"
    ),
}


def fixture_catalog(root: Path) -> dict[str, dict[str, Any]]:
    catalog = {}
    for name, relative_path in FIXTURE_PROJECTS.items():
        project = (root / relative_path).resolve()
        catalog[name] = {
            "project": str(project),
            "available": project.is_file(),
            "read_only_default": True,
            "mode": "copied metadata/images with sparse derived-resource placeholders",
            "scenario": str(root / f"scripts/dev/browser_gui_scenarios/{name}_open.json"),
        }
    return catalog


def fixture_project(root: Path, name: str) -> Path:
    try:
        project = (root / FIXTURE_PROJECTS[name]).resolve()
    except KeyError as error:
        raise ValueError(f"unknown browser fixture: {name}") from error
    if not project.is_file():
        raise ValueError(f"browser fixture project does not exist: {project}")
    return project
