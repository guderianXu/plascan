#!/usr/bin/env python3
"""Compare PlaScan depth artifacts with a reference mesh in the same coordinates."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import trimesh


FAST_DEPTH_HEADER = struct.Struct("<16siii4xQ")
FAST_DEPTH_MAGIC = b"PLASDEPTHMAT01\x00\x00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a registered reference mesh into MVS cameras and measure "
            "per-frame depth coverage, signed bias, and absolute residuals."
        )
    )
    parser.add_argument("--mvs-manifest", type=Path, required=True)
    parser.add_argument("--reference-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-samples", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--minimum-reference-samples-per-pixel",
        type=int,
        default=2,
    )
    return parser.parse_args()


def read_float_grid(path: Path, width: int, height: int) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(FAST_DEPTH_HEADER.size)
        magic, rows, columns, cv_type, data_bytes = FAST_DEPTH_HEADER.unpack(
            header
        )
        if magic != FAST_DEPTH_MAGIC or cv_type != 5:
            raise ValueError(f"Unsupported PlaScan depth matrix: {path}")
        if rows != height or columns != width:
            raise ValueError(
                f"{path}: manifest grid is {width}x{height}, "
                f"artifact grid is {columns}x{rows}"
            )
        expected_bytes = width * height * np.dtype(np.float32).itemsize
        if data_bytes != expected_bytes:
            raise ValueError(
                f"{path}: expected {expected_bytes} payload bytes, "
                f"found {data_bytes}"
            )
        values = np.fromfile(
            stream,
            dtype="<f4",
            count=width * height,
        )
    return values.reshape(height, width)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "p10": float(np.percentile(values, 10.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "mean": float(np.mean(values)),
    }


def project_reference_depth(
    points: np.ndarray,
    camera: dict,
    width: int,
    height: int,
    minimum_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(
        camera["rotation_world_to_camera"], dtype=np.float64
    ).reshape(3, 3)
    translation = np.asarray(
        camera["translation_world_to_camera"], dtype=np.float64
    )
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2]
    visible = np.isfinite(z) & (z > 0.0)
    x = camera_points[:, 0]
    y = camera_points[:, 1]
    u = camera["fx"] * x / np.maximum(z, 1.0e-12) + camera["cx"]
    v = camera["fy"] * y / np.maximum(z, 1.0e-12) + camera["cy"]
    columns = np.rint(u).astype(np.int64)
    rows = np.rint(v).astype(np.int64)
    visible &= (
        (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    columns = columns[visible]
    rows = rows[visible]
    depth = z[visible]
    flat_indices = rows * width + columns

    z_buffer = np.full(width * height, np.inf, dtype=np.float64)
    sample_counts = np.zeros(width * height, dtype=np.int32)
    np.minimum.at(z_buffer, flat_indices, depth)
    np.add.at(sample_counts, flat_indices, 1)
    z_buffer[sample_counts < max(1, minimum_samples)] = np.nan
    return z_buffer.reshape(height, width), sample_counts.reshape(height, width)


def main() -> int:
    args = parse_args()
    manifest_path = args.mvs_manifest.resolve()
    mesh_path = args.reference_mesh.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MVS manifest not found: {manifest_path}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Reference mesh not found: {mesh_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in MVS manifest: {manifest_path}")

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Reference mesh is empty: {mesh_path}")
    np.random.seed(args.seed)
    points, _ = trimesh.sample.sample_surface(
        mesh, max(10_000, args.surface_samples)
    )
    points = np.asarray(points, dtype=np.float64)

    frame_results: list[dict] = []
    for frame in frames:
        width = int(frame["grid_width"])
        height = int(frame["grid_height"])
        depth_path = Path(frame["raw_depth_path"])
        if not depth_path.is_file():
            raise FileNotFoundError(
                f"Depth artifact missing for frame {frame.get('ref_index')}: "
                f"{depth_path}"
            )
        depth = read_float_grid(depth_path, width, height)
        reference_depth, sample_counts = project_reference_depth(
            points,
            frame["camera_model"],
            width,
            height,
            args.minimum_reference_samples_per_pixel,
        )

        mask_path = Path(frame.get("valid_mask_path", ""))
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height, width):
                raise ValueError(f"Invalid mask artifact: {mask_path}")
            object_mask = mask > 0
        else:
            object_mask = np.ones((height, width), dtype=bool)

        reference_valid = (
            object_mask
            & np.isfinite(reference_depth)
            & (reference_depth > 0.0)
        )
        measured_valid = np.isfinite(depth) & (depth > 0.0)
        compared = reference_valid & measured_valid
        signed = depth[compared].astype(np.float64) - reference_depth[compared]
        absolute = np.abs(signed)
        normalized = absolute / np.maximum(reference_depth[compared], 1.0e-12)
        reference_count = int(np.count_nonzero(reference_valid))
        compared_count = int(np.count_nonzero(compared))
        frame_results.append(
            {
                "ref_index": int(frame["ref_index"]),
                "ref_image": frame.get("ref_image", ""),
                "acceptance": frame.get("acceptance", ""),
                "reference_pixel_count": reference_count,
                "compared_pixel_count": compared_count,
                "measured_reference_coverage": (
                    compared_count / reference_count
                    if reference_count > 0
                    else 0.0
                ),
                "reference_samples_per_pixel": percentile_summary(
                    sample_counts[reference_valid].astype(np.float64)
                ),
                "signed_depth_residual": percentile_summary(signed),
                "absolute_depth_residual": percentile_summary(absolute),
                "relative_absolute_depth_residual": percentile_summary(
                    normalized
                ),
            }
        )

    relative_p50 = [
        frame["relative_absolute_depth_residual"].get("p50", np.nan)
        for frame in frame_results
    ]
    relative_p90 = [
        frame["relative_absolute_depth_residual"].get("p90", np.nan)
        for frame in frame_results
    ]
    coverage = [
        frame["measured_reference_coverage"] for frame in frame_results
    ]
    report = {
        "mvs_manifest": str(manifest_path),
        "reference_mesh": str(mesh_path),
        "surface_samples": int(points.shape[0]),
        "frame_count": len(frame_results),
        "summary": {
            "coverage": percentile_summary(np.asarray(coverage)),
            "relative_absolute_depth_p50_across_frames": percentile_summary(
                np.asarray(relative_p50)
            ),
            "relative_absolute_depth_p90_across_frames": percentile_summary(
                np.asarray(relative_p90)
            ),
        },
        "frames": frame_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "Compared "
        f"{len(frame_results)} frames: "
        f"coverage median={np.nanmedian(coverage):.4f}, "
        f"relative residual median={np.nanmedian(relative_p50):.6f}, "
        f"P90 median={np.nanmedian(relative_p90):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
