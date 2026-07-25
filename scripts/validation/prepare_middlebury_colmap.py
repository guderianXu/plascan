#!/usr/bin/env python3
"""Prepare a fixed-camera COLMAP workspace from Middlebury *_par.txt data."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraRecord:
    image_name: str
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Middlebury K[R|t] cameras to a fixed COLMAP text model."
    )
    parser.add_argument("--par-file", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--bbox-min",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional object bounding-box minimum, used to derive PatchMatch depth limits.",
    )
    parser.add_argument(
        "--bbox-max",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional object bounding-box maximum, used to derive PatchMatch depth limits.",
    )
    parser.add_argument(
        "--source-count",
        type=int,
        default=10,
        help="Number of ring-neighbour source views per reference image (default: 10).",
    )
    return parser.parse_args()


def parse_middlebury_cameras(par_file: Path, image_dir: Path) -> list[CameraRecord]:
    if not par_file.is_file():
        raise FileNotFoundError(f"Middlebury camera file not found: {par_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    lines = [
        line.strip()
        for line in par_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"Middlebury camera file is empty: {par_file}")

    try:
        expected_count = int(lines[0])
        camera_lines = lines[1:]
    except ValueError:
        expected_count = len(lines)
        camera_lines = lines
    if len(camera_lines) != expected_count:
        raise ValueError(
            f"Camera count mismatch in {par_file}: header={expected_count}, records={len(camera_lines)}"
        )

    records: list[CameraRecord] = []
    for line_number, line in enumerate(camera_lines, start=2):
        fields = line.split()
        if len(fields) != 22:
            raise ValueError(
                f"Expected image name plus 21 values at {par_file}:{line_number}, got {len(fields)} fields"
            )
        image_name = fields[0]
        values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        intrinsic = values[:9].reshape(3, 3)
        rotation = values[9:18].reshape(3, 3)
        translation = values[18:21]
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Camera references a missing image: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        if not np.allclose(intrinsic[[0, 1, 2], [1, 0, 0]], 0.0, atol=1.0e-10):
            raise ValueError(f"COLMAP PINHOLE cannot preserve skewed intrinsic matrix for {image_name}")
        # Middlebury text rounds the calibrated matrices; the observed residual is
        # about 1.6e-6, which the normalized quaternion safely absorbs.
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=5.0e-6):
            raise ValueError(f"Camera rotation is not orthonormal for {image_name}")
        records.append(
            CameraRecord(
                image_name=image_name,
                intrinsic=intrinsic,
                rotation=rotation,
                translation=translation,
                width=width,
                height=height,
            )
        )
    return records


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return COLMAP's Hamilton quaternion (qw, qx, qy, qz)."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    normalized = np.asarray(quaternion, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    if normalized[0] < 0.0:
        normalized *= -1.0
    return tuple(float(value) for value in normalized)


def camera_key(record: CameraRecord) -> tuple[float | int, ...]:
    intrinsic = record.intrinsic
    return (
        record.width,
        record.height,
        round(float(intrinsic[0, 0]), 12),
        round(float(intrinsic[1, 1]), 12),
        round(float(intrinsic[0, 2]), 12),
        round(float(intrinsic[1, 2]), 12),
    )


def write_colmap_model(records: list[CameraRecord], output_dir: Path) -> None:
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    camera_ids: dict[tuple[float | int, ...], int] = {}
    for record in records:
        camera_ids.setdefault(camera_key(record), len(camera_ids) + 1)

    camera_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(camera_ids)}",
    ]
    for key, camera_id in sorted(camera_ids.items(), key=lambda item: item[1]):
        width, height, fx, fy, cx, cy = key
        camera_lines.append(f"{camera_id} PINHOLE {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}")
    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")

    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(records)}, mean observations per image: 0",
    ]
    for image_id, record in enumerate(records, start=1):
        quaternion = rotation_to_quaternion(record.rotation)
        values = (*quaternion, *record.translation)
        pose = " ".join(f"{value:.17g}" for value in values)
        image_lines.extend(
            [
                f"{image_id} {pose} {camera_ids[camera_key(record)]} {record.image_name}",
                "",
            ]
        )
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )


def ring_source_names(records: list[CameraRecord], reference_index: int, source_count: int) -> list[str]:
    candidates: list[tuple[int, str]] = []
    image_count = len(records)
    for source_index, record in enumerate(records):
        if source_index == reference_index:
            continue
        direct_distance = abs(source_index - reference_index)
        ring_distance = min(direct_distance, image_count - direct_distance)
        candidates.append((ring_distance, record.image_name))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in candidates[:source_count]]


def write_patch_match_config(records: list[CameraRecord], output_dir: Path, source_count: int) -> None:
    if source_count < 1:
        raise ValueError("--source-count must be positive")
    config_lines: list[str] = []
    for reference_index, record in enumerate(records):
        config_lines.append(record.image_name)
        config_lines.append(",".join(ring_source_names(records, reference_index, source_count)))
    (output_dir / "patch-match.cfg").write_text("\n".join(config_lines) + "\n", encoding="utf-8")


def bbox_depth_limits(
    records: list[CameraRecord],
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> dict[str, object]:
    minimum = np.asarray(bbox_min, dtype=np.float64)
    maximum = np.asarray(bbox_max, dtype=np.float64)
    if np.any(maximum <= minimum):
        raise ValueError("Every --bbox-max component must be greater than --bbox-min")
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    per_image: dict[str, list[float]] = {}
    all_depths: list[float] = []
    for record in records:
        depths = (record.rotation @ corners.T + record.translation[:, None])[2]
        positive_depths = depths[depths > 0.0]
        if positive_depths.size == 0:
            raise ValueError(f"Bounding box is behind camera {record.image_name}")
        depth_min = float(np.min(positive_depths))
        depth_max = float(np.max(positive_depths))
        per_image[record.image_name] = [depth_min, depth_max]
        all_depths.extend([depth_min, depth_max])
    span = max(all_depths) - min(all_depths)
    margin = max(0.02 * span, 1.0e-4)
    return {
        "global_depth_min": max(min(all_depths) - margin, 1.0e-6),
        "global_depth_max": max(all_depths) + margin,
        "per_image": per_image,
    }


def main() -> int:
    args = parse_args()
    if (args.bbox_min is None) != (args.bbox_max is None):
        raise ValueError("--bbox-min and --bbox-max must be specified together")
    records = parse_middlebury_cameras(args.par_file.resolve(), args.image_dir.resolve())
    output_dir = args.output_dir.resolve()
    write_colmap_model(records, output_dir)
    write_patch_match_config(records, output_dir, min(args.source_count, len(records) - 1))
    summary: dict[str, object] = {
        "camera_count": len(records),
        "image_dir": str(args.image_dir.resolve()),
        "par_file": str(args.par_file.resolve()),
        "coordinate_contract": "Middlebury Xc = R_wc * Xw + t_wc; copied without pose estimation",
    }
    if args.bbox_min is not None and args.bbox_max is not None:
        summary["depth_limits"] = bbox_depth_limits(records, tuple(args.bbox_min), tuple(args.bbox_max))
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(records)} fixed cameras in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
