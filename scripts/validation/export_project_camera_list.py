#!/usr/bin/env python3
"""Export the active image cameras from a PlaScan archive for CLI validation."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export project_files.json image cameras as TSAI files and an "
            "image_camera.lis file without modifying the project archive."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--aerial-report",
        type=Path,
        help=(
            "Optional aerial_triangulation_cli_report.json. When provided, "
            "pending_camera_updates replace the cameras stored in the project "
            "without modifying the archive."
        ),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def scalar(camera: dict[str, Any], name: str) -> float:
    value = camera.get(name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"camera field {name!r} is missing or non-numeric")
    return float(value)


def vector(camera: dict[str, Any], name: str, size: int) -> list[float]:
    value = camera.get(name)
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"camera field {name!r} must contain {size} values")
    return [float(item) for item in value]


def tsai_text(camera: dict[str, Any]) -> str:
    center = vector(camera, "C", 3)
    rotation = vector(camera, "R", 9)
    depth_direction = -1 if bool(camera.get("depth_axis_flipped", False)) else 1
    lines = [
        f"fu = {scalar(camera, 'fu'):.17g}",
        f"fv = {scalar(camera, 'fv'):.17g}",
        f"cu = {scalar(camera, 'cu'):.17g}",
        f"cv = {scalar(camera, 'cv'):.17g}",
        "c = " + " ".join(f"{value:.17g}" for value in center),
        "r = " + " ".join(f"{value:.17g}" for value in rotation),
        f"k1 = {float(camera.get('k1', 0.0)):.17g}",
        f"k2 = {float(camera.get('k2', 0.0)):.17g}",
        f"k3 = {float(camera.get('k3', 0.0)):.17g}",
        f"p1 = {float(camera.get('p1', 0.0)):.17g}",
        f"p2 = {float(camera.get('p2', 0.0)):.17g}",
        f"pitch = {float(camera.get('pitch', 1.0)):.17g}",
        f"u_direction = {int(camera.get('u_direction', 1))}",
        f"v_direction = {int(camera.get('v_direction', 1))}",
        f"w_direction = {depth_direction}",
        "",
    ]
    return "\n".join(lines)


def normalized_path_key(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def report_camera_updates(report_path: Path | None) -> dict[str, dict[str, Any]]:
    if report_path is None:
        return {}
    resolved_report = report_path.resolve()
    if not resolved_report.is_file():
        raise FileNotFoundError(
            f"aerial triangulation report not found: {resolved_report}"
        )
    report = json.loads(resolved_report.read_text(encoding="utf-8"))
    reconstruction_result = report.get("reconstruction_result", {})
    updates = reconstruction_result.get("pending_camera_updates", {})
    if not isinstance(updates, dict) or not updates:
        raise ValueError(
            "aerial triangulation report has no pending_camera_updates: "
            f"{resolved_report}"
        )
    normalized_updates: dict[str, dict[str, Any]] = {}
    for image_path, camera in updates.items():
        if not isinstance(camera, dict):
            raise ValueError(
                f"pending camera update for {image_path!r} is not an object"
            )
        normalized_updates[normalized_path_key(image_path)] = camera
    return normalized_updates


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    if not project.is_file():
        raise FileNotFoundError(f"PlaScan project archive not found: {project}")

    with zipfile.ZipFile(project) as archive:
        try:
            project_files = json.loads(
                archive.read("project_files.json").decode("utf-8")
            )
        except KeyError as error:
            raise ValueError(
                f"project archive has no project_files.json: {project}"
            ) from error

    images = project_files.get("images", [])
    if not isinstance(images, list) or not images:
        raise ValueError(f"project archive has no active images: {project}")
    report_updates = report_camera_updates(args.aerial_report)

    output_directory = args.output_directory.resolve()
    camera_directory = output_directory / "cameras"
    camera_directory.mkdir(parents=True, exist_ok=True)
    list_lines: list[str] = []
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"image record {index} is not an object")
        image_path = Path(str(image.get("path", "")))
        camera = image.get("camera")
        if not image_path.is_file():
            raise FileNotFoundError(f"image record {index} is missing: {image_path}")
        if not isinstance(camera, dict):
            raise ValueError(f"image record {index} has no camera object")
        if report_updates:
            camera = report_updates.get(normalized_path_key(image_path))
            if camera is None:
                raise ValueError(
                    "aerial triangulation report has no camera update for "
                    f"image record {index}: {image_path}"
                )
        camera_path = camera_directory / f"{index:04d}_{image_path.stem}.tsai"
        camera_path.write_text(tsai_text(camera), encoding="utf-8")
        list_lines.append(f'"{image_path}" "{camera_path}"')

    list_path = output_directory / "image_camera.lis"
    list_path.write_text("\n".join(list_lines) + "\n", encoding="utf-8")
    source = "aerial report" if report_updates else "project archive"
    print(f"Exported {len(images)} cameras from {source}: {list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
