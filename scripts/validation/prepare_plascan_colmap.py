#!/usr/bin/env python3
"""Export a PlaScan MVS manifest and sparse SfM report as a COLMAP text model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prepare_middlebury_colmap import rotation_to_quaternion


@dataclass(frozen=True)
class ManifestCamera:
    name: str
    image_path: Path
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PlaScan's current cameras and sparse tracks for an external MVS baseline."
    )
    parser.add_argument("--mvs-manifest", required=True, type=Path)
    parser.add_argument("--sparse-points", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-count", type=int, default=10)
    return parser.parse_args()


def read_cameras(path: Path) -> list[ManifestCamera]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cameras: dict[str, ManifestCamera] = {}
    for frame in payload.get("frames", []):
        image_path = Path(frame.get("ref_image", ""))
        model = frame.get("camera_model", {})
        name = image_path.name
        rotation_values = model.get("rotation_world_to_camera")
        translation_values = model.get("translation_world_to_camera")
        if not name or not isinstance(rotation_values, list) or len(rotation_values) != 9:
            continue
        if not isinstance(translation_values, list) or len(translation_values) != 3:
            continue
        camera = ManifestCamera(
            name=name,
            image_path=image_path,
            width=int(frame.get("grid_width", 0)),
            height=int(frame.get("grid_height", 0)),
            fx=float(model["fx"]),
            fy=float(model["fy"]),
            cx=float(model["cx"]),
            cy=float(model["cy"]),
            rotation=np.asarray(rotation_values, dtype=np.float64).reshape(3, 3),
            translation=np.asarray(translation_values, dtype=np.float64),
        )
        if camera.width <= 0 or camera.height <= 0:
            raise ValueError(f"Invalid image dimensions for {name} in {path}")
        if not camera.image_path.is_file():
            raise FileNotFoundError(f"Manifest image not found: {camera.image_path}")
        cameras[name] = camera
    if len(cameras) < 2:
        raise ValueError(f"Fewer than two valid cameras found in {path}")
    return [cameras[name] for name in sorted(cameras)]


def ring_source_names(cameras: list[ManifestCamera], reference_index: int, count: int) -> list[str]:
    image_count = len(cameras)
    candidates: list[tuple[int, str]] = []
    for source_index, camera in enumerate(cameras):
        if source_index == reference_index:
            continue
        direct = abs(source_index - reference_index)
        candidates.append((min(direct, image_count - direct), camera.name))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in candidates[:count]]


def build_tracks(
    sparse_payload: dict[str, Any],
    image_ids: dict[str, int],
) -> tuple[dict[str, list[tuple[float, float, int]]], list[dict[str, Any]]]:
    image_observations: dict[str, list[tuple[float, float, int]]] = {
        name: [] for name in image_ids
    }
    exported_points: list[dict[str, Any]] = []
    for source_point in sparse_payload.get("points", []):
        point_id = len(exported_points) + 1
        track: list[tuple[int, int]] = []
        seen_images: set[str] = set()
        for observation in source_point.get("observations", []):
            image_name = Path(observation.get("image_name", "")).name
            xy = observation.get("xy")
            if image_name not in image_ids or image_name in seen_images:
                continue
            if not isinstance(xy, list) or len(xy) != 2:
                continue
            point2d_index = len(image_observations[image_name])
            image_observations[image_name].append((float(xy[0]), float(xy[1]), point_id))
            track.append((image_ids[image_name], point2d_index))
            seen_images.add(image_name)
        if len(track) < 2:
            for image_id, point2d_index in track:
                image_name = next(name for name, value in image_ids.items() if value == image_id)
                image_observations[image_name][point2d_index] = (
                    image_observations[image_name][point2d_index][0],
                    image_observations[image_name][point2d_index][1],
                    -1,
                )
            continue
        xyz = source_point.get("point_xyz")
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError(f"Sparse point {point_id} has no valid point_xyz")
        exported_points.append(
            {
                "id": point_id,
                "xyz": [float(value) for value in xyz],
                "error": float(source_point.get("rms_reproj_px", 0.0)),
                "track": track,
            }
        )
    return image_observations, exported_points


def write_model(
    cameras: list[ManifestCamera],
    sparse_payload: dict[str, Any],
    output_dir: Path,
    source_count: int,
) -> dict[str, Any]:
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    image_ids = {camera.name: index for index, camera in enumerate(cameras, start=1)}
    camera_ids = dict(image_ids)
    observations, points = build_tracks(sparse_payload, image_ids)

    camera_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera in cameras:
        camera_lines.append(
            f"{camera_ids[camera.name]} PINHOLE {camera.width} {camera.height} "
            f"{camera.fx:.17g} {camera.fy:.17g} {camera.cx:.17g} {camera.cy:.17g}"
        )
    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")

    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(cameras)}",
    ]
    for camera in cameras:
        quaternion = rotation_to_quaternion(camera.rotation)
        pose = " ".join(
            f"{value:.17g}" for value in (*quaternion, *camera.translation)
        )
        image_lines.append(
            f"{image_ids[camera.name]} {pose} {camera_ids[camera.name]} {camera.name}"
        )
        image_lines.append(
            " ".join(
                f"{x:.17g} {y:.17g} {point_id}"
                for x, y, point_id in observations[camera.name]
            )
        )
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

    point_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(points)}",
    ]
    for point in points:
        track = " ".join(f"{image_id} {index}" for image_id, index in point["track"])
        xyz = " ".join(f"{value:.17g}" for value in point["xyz"])
        point_lines.append(
            f"{point['id']} {xyz} 128 128 128 {point['error']:.17g} {track}"
        )
    (sparse_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")

    config_lines: list[str] = []
    actual_source_count = min(max(source_count, 1), len(cameras) - 1)
    for reference_index, camera in enumerate(cameras):
        config_lines.append(camera.name)
        config_lines.append(
            ",".join(ring_source_names(cameras, reference_index, actual_source_count))
        )
    (output_dir / "patch-match.cfg").write_text(
        "\n".join(config_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "camera_count": len(cameras),
        "point_count": len(points),
        "observation_count": sum(len(values) for values in observations.values()),
        "image_root": str(cameras[0].image_path.parent),
        "coordinate_contract": "PlaScan current SfM world coordinates copied without pose estimation",
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.mvs_manifest.resolve()
    sparse_path = args.sparse_points.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MVS manifest not found: {manifest_path}")
    if not sparse_path.is_file():
        raise FileNotFoundError(f"Sparse point report not found: {sparse_path}")
    cameras = read_cameras(manifest_path)
    sparse_payload = json.loads(sparse_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    summary = write_model(cameras, sparse_payload, output_dir, args.source_count)
    summary.update(
        {
            "mvs_manifest": str(manifest_path),
            "sparse_points": str(sparse_path),
        }
    )
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {summary['camera_count']} cameras, {summary['point_count']} points, "
        f"and {summary['observation_count']} observations in {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
