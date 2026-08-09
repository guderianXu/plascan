#!/usr/bin/env python3
"""Prepare Tanks and Temples Ignatius for PlaScan LiDAR-constrained BA."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .estimate_lidar_normals import (
        deterministic_subsample,
        estimate_normals,
        read_ply_xyz,
        voxel_downsample,
        write_plane_ply,
    )
    from .prepare_photogrammetry_benchmarks import CameraRecord, write_tsai
except ImportError:
    from estimate_lidar_normals import (  # type: ignore[no-redef]
        deterministic_subsample,
        estimate_normals,
        read_ply_xyz,
        voxel_downsample,
        write_plane_ply,
    )
    from prepare_photogrammetry_benchmarks import CameraRecord, write_tsai  # type: ignore[no-redef]


DEFAULT_OUTPUT_SUBDIR = Path("prepared") / "plascan_lidar_ba"
OUTPUT_MARKER_NAME = ".plascan_ignatius_lidar_ba_output"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class AlignedPoseSet:
    matrices: tuple[np.ndarray, ...]
    scale: float
    alignment_rotation: np.ndarray
    alignment_translation: np.ndarray


def read_matrix4(path: Path) -> np.ndarray:
    rows = [
        [float(token) for token in line.split()]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix in {path}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"matrix contains non-finite values: {path}")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
        raise ValueError(f"matrix has an invalid homogeneous row: {path}")
    return matrix


def parse_tanks_pose_log(path: Path) -> list[np.ndarray]:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) % 5 != 0:
        raise ValueError(f"pose log must contain five lines per camera: {path}")

    indexed_poses: list[tuple[int, np.ndarray]] = []
    for offset in range(0, len(lines), 5):
        header = lines[offset].split()
        if len(header) != 3:
            raise ValueError(f"invalid pose header at line {offset + 1}: {lines[offset]}")
        camera_index = int(header[0])
        rows = [[float(token) for token in lines[offset + row].split()] for row in range(1, 5)]
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"invalid pose matrix for camera {camera_index}: {matrix.shape}")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
            raise ValueError(f"camera {camera_index} has an invalid homogeneous pose row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3):
            raise ValueError(f"camera {camera_index} pose rotation is not orthonormal")
        indexed_poses.append((camera_index, matrix))

    indexed_poses.sort(key=lambda item: item[0])
    expected_indices = list(range(len(indexed_poses)))
    actual_indices = [item[0] for item in indexed_poses]
    if actual_indices != expected_indices:
        raise ValueError(
            f"pose indices must be contiguous from zero, got {actual_indices[:5]}..."
        )
    return [matrix for _, matrix in indexed_poses]


def align_camera_poses(
    camera_to_world_poses: list[np.ndarray],
    colmap_to_laser: np.ndarray,
) -> AlignedPoseSet:
    linear = colmap_to_laser[:3, :3]
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("COLMAP-to-laser transform must have a positive finite scale")
    scale = float(np.cbrt(determinant))
    alignment_rotation = linear / scale
    if not np.allclose(alignment_rotation.T @ alignment_rotation, np.eye(3), atol=2.0e-5):
        raise ValueError("COLMAP-to-laser linear block is not a similarity transform")
    if not np.isclose(np.linalg.det(alignment_rotation), 1.0, atol=2.0e-5):
        raise ValueError("COLMAP-to-laser transform contains a reflection")

    translation = colmap_to_laser[:3, 3]
    aligned: list[np.ndarray] = []
    for pose in camera_to_world_poses:
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = alignment_rotation @ pose[:3, :3]
        output[:3, 3] = linear @ pose[:3, 3] + translation
        aligned.append(output)
    return AlignedPoseSet(tuple(aligned), scale, alignment_rotation, translation.copy())


def parse_scanner_positions(path: Path) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_number} requires '<ply> X Y Z'")
        position = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        if not np.all(np.isfinite(position)):
            raise ValueError(f"{path}:{line_number} contains a non-finite scanner position")
        positions[parts[0].casefold()] = position
    if not positions:
        raise ValueError(f"scanner position file is empty: {path}")
    return positions


def portable_path_token(path: Path, base_dir: Path) -> str:
    """Return a POSIX relative token, or an absolute token across Windows drives."""
    try:
        token = os.path.relpath(path, base_dir)
    except ValueError:
        token = str(path.resolve())
    return token.replace("\\", "/")


def prepare_cameras(
    images_dir: Path,
    pose_log: Path,
    transform_path: Path,
    output_dir: Path,
    *,
    image_width: int,
    image_height: int,
    focal_ratio: float,
) -> dict[str, object]:
    images = sorted(
        path for path in Path(images_dir).iterdir() if path.suffix.casefold() in IMAGE_SUFFIXES
    )
    poses = parse_tanks_pose_log(pose_log)
    if len(images) != len(poses):
        raise ValueError(f"image/pose count mismatch: {len(images)} images, {len(poses)} poses")
    for index, image_path in enumerate(images, start=1):
        expected_stem = f"{index:06d}"
        if image_path.stem != expected_stem:
            raise ValueError(
                f"pose {index - 1} must map to {expected_stem}, got {image_path.name}"
            )

    aligned = align_camera_poses(poses, read_matrix4(transform_path))
    cameras_dir = output_dir / "cameras"
    cameras_dir.mkdir(parents=True, exist_ok=True)
    fx = focal_ratio * image_width
    fy = fx
    cx = image_width / 2.0
    cy = image_height / 2.0
    list_lines: list[str] = []
    image_lines: list[str] = []
    for image_path, pose in zip(images, aligned.matrices):
        camera_path = cameras_dir / f"{image_path.stem}.tsai"
        camera = CameraRecord(
            image_name=image_path.name,
            K=[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            rotation_camera_to_world=pose[:3, :3].tolist(),
            center=pose[:3, 3].tolist(),
        )
        write_tsai(camera_path, camera)
        image_token = shlex.quote(portable_path_token(image_path, output_dir))
        camera_token = shlex.quote(portable_path_token(camera_path, output_dir))
        list_lines.append(f"{image_token} {camera_token}")
        image_lines.append(image_token)

    (output_dir / "image_camera.lis").write_text("\n".join(list_lines) + "\n", encoding="utf-8")
    (output_dir / "images.lis").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    centers = np.asarray([pose[:3, 3] for pose in aligned.matrices])
    return {
        "image_count": len(images),
        "pose_convention": "camera_to_world",
        "output_coordinate_frame": "Ignatius laser ground truth",
        "similarity_scale": aligned.scale,
        "focal_x_pixels": fx,
        "focal_y_pixels": fy,
        "principal_x_pixels": cx,
        "principal_y_pixels": cy,
        "camera_center_bounds_min": centers.min(axis=0).tolist(),
        "camera_center_bounds_max": centers.max(axis=0).tolist(),
    }


def prepare_laser_cloud(
    scans_dir: Path,
    scanner_positions_path: Path,
    output_path: Path,
    *,
    voxel_size: float,
    k_neighbors: int,
    max_points_per_scan: int,
    max_output_points: int,
    chunk_size: int,
    workers: int,
) -> dict[str, object]:
    positions = parse_scanner_positions(scanner_positions_path)
    scan_paths = sorted(Path(scans_dir).glob("*.ply"))
    if not scan_paths:
        raise FileNotFoundError(f"no individual scan PLY files found under {scans_dir}")

    merged_points: list[np.ndarray] = []
    merged_normals: list[np.ndarray] = []
    merged_curvature: list[np.ndarray] = []
    scan_summaries: list[dict[str, object]] = []
    for scan_path in scan_paths:
        position = positions.get(scan_path.name.casefold())
        if position is None:
            raise ValueError(f"scanner position is missing for {scan_path.name}")
        points, _ = read_ply_xyz(scan_path)
        input_count = points.shape[0]
        points, = voxel_downsample(points, voxel_size)
        voxel_count = points.shape[0]
        points, = deterministic_subsample(points, max_points_per_scan)
        normals, curvature = estimate_normals(
            points,
            k_neighbors,
            viewpoint=position,
            chunk_size=chunk_size,
            workers=workers,
        )
        merged_points.append(points)
        merged_normals.append(normals)
        merged_curvature.append(curvature)
        scan_summaries.append(
            {
                "file": str(scan_path.resolve()),
                "input_vertex_count": int(input_count),
                "voxel_vertex_count": int(voxel_count),
                "normal_vertex_count": int(points.shape[0]),
                "scanner_position": position.tolist(),
                "curvature_mean": float(np.mean(curvature)),
            }
        )

    points = np.concatenate(merged_points)
    normals = np.concatenate(merged_normals)
    curvature = np.concatenate(merged_curvature)
    merged_count = points.shape[0]
    points, normals, curvature = voxel_downsample(
        points, voxel_size, normals, curvature
    )
    uniform_count = points.shape[0]
    points, normals, curvature = deterministic_subsample(
        points, max_output_points, normals, curvature
    )
    write_plane_ply(
        output_path,
        points,
        normals,
        curvature,
        generator="prepare_tanks_ignatius_lidar_ba.py",
    )
    return {
        "source_scan_count": len(scan_paths),
        "merged_normal_vertex_count": int(merged_count),
        "uniform_vertex_count": int(uniform_count),
        "output_vertex_count": int(points.shape[0]),
        "voxel_size_m": float(voxel_size),
        "k_neighbors": int(k_neighbors),
        "output_path": str(output_path.resolve()),
        "normal_length_min": float(np.min(np.linalg.norm(normals, axis=1))),
        "normal_length_max": float(np.max(np.linalg.norm(normals, axis=1))),
        "curvature_mean": float(np.mean(curvature)),
        "scans": scan_summaries,
    }


def reset_output_directory(output_dir: Path, dataset_root: Path, overwrite: bool) -> None:
    resolved_output = output_dir.resolve()
    resolved_dataset = dataset_root.resolve()
    default_output = (resolved_dataset / DEFAULT_OUTPUT_SUBDIR).resolve()
    marker_path = resolved_output / OUTPUT_MARKER_NAME
    protected_trees = [
        (resolved_dataset / "archives").resolve(),
        (resolved_dataset / "extracted").resolve(),
    ]
    if resolved_output.parent == resolved_output or resolved_output == resolved_dataset:
        raise ValueError(f"refusing unsafe output directory: {resolved_output}")
    if resolved_dataset.is_relative_to(resolved_output):
        raise ValueError(f"output directory contains the source dataset: {resolved_output}")
    if any(
        resolved_output.is_relative_to(protected) or protected.is_relative_to(resolved_output)
        for protected in protected_trees
    ):
        raise ValueError(f"output overlaps protected source data: {resolved_output}")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        marker_path.write_text("PlaScan Ignatius LiDAR BA output\n", encoding="utf-8")
        return
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite or choose another directory: {output_dir}")
    if any(output_dir.iterdir()):
        if resolved_output != default_output and not marker_path.is_file():
            raise ValueError(
                "refusing to overwrite an unmarked custom output directory: "
                f"{resolved_output}"
            )
        shutil.rmtree(resolved_output)
        resolved_output.mkdir(parents=True)
    marker_path.write_text("PlaScan Ignatius LiDAR BA output\n", encoding="utf-8")


def prepare_ignatius(
    dataset_root: Path,
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
    skip_laser: bool = False,
    image_width: int = 1920,
    image_height: int = 1080,
    focal_ratio: float = 0.7,
    voxel_size: float = 0.005,
    k_neighbors: int = 24,
    max_points_per_scan: int = 100000,
    max_output_points: int = 500000,
    chunk_size: int = 10000,
    workers: int = -1,
) -> dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    output_dir = (Path(output_dir) if output_dir else dataset_root / DEFAULT_OUTPUT_SUBDIR).resolve()
    reset_output_directory(output_dir, dataset_root, overwrite)

    camera_summary = prepare_cameras(
        dataset_root / "extracted" / "Ignatius",
        dataset_root / "archives" / "Ignatius_COLMAP_SfM.log",
        dataset_root / "archives" / "Ignatius_trans.txt",
        output_dir,
        image_width=image_width,
        image_height=image_height,
        focal_ratio=focal_ratio,
    )
    laser_summary: dict[str, object] | None = None
    if not skip_laser:
        scans_dir = dataset_root / "extracted" / "individual_scans" / "Ignatius"
        laser_summary = prepare_laser_cloud(
            scans_dir,
            scans_dir / "scanner_pos.txt",
            output_dir / "lidar" / "Ignatius_lidar_planes.ply",
            voxel_size=voxel_size,
            k_neighbors=k_neighbors,
            max_points_per_scan=max_points_per_scan,
            max_output_points=max_output_points,
            chunk_size=chunk_size,
            workers=workers,
        )

    summary: dict[str, object] = {
        "dataset": "Tanks and Temples Ignatius",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "camera_preparation": camera_summary,
        "laser_preparation": laser_summary,
        "recommended_bundle_adjust": {
            "laser_association_max_distance_m": 0.05,
            "laser_voxel_size_m": 0.0,
            "laser_max_curvature": 0.2,
            "laser_max_samples": max_output_points,
            "laser_sigma_m": 0.0025,
            "laser_weight_initial": 0.0,
            "laser_effective_weight": 160000.0,
            "laser_huber_delta_m": 0.05,
            "laser_missing_normals_as_height_planes": False,
            "refine_camera_pose": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-laser", action="store_true", help="prepare aligned cameras only")
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--focal-ratio", type=float, default=0.7)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--k-neighbors", type=int, default=24)
    parser.add_argument("--max-points-per-scan", type=int, default=100000)
    parser.add_argument("--max-output-points", type=int, default=500000)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = prepare_ignatius(
            args.dataset_root,
            args.output_dir,
            overwrite=args.overwrite,
            skip_laser=args.skip_laser,
            image_width=args.image_width,
            image_height=args.image_height,
            focal_ratio=args.focal_ratio,
            voxel_size=args.voxel_size,
            k_neighbors=args.k_neighbors,
            max_points_per_scan=args.max_points_per_scan,
            max_output_points=args.max_output_points,
            chunk_size=args.chunk_size,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"Ignatius preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
