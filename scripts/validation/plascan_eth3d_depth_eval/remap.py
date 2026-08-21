"""Map raw ETH3D depth into the official undistorted PINHOLE domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import (
    project_pinhole,
    unproject_thin_prism_fisheye,
    validate_camera_roundtrip,
)
from .camera_io import (
    ColmapCamera,
    colmap_pixels_to_raster_coordinates,
    raster_indices_to_colmap_pixels,
)


@dataclass(frozen=True)
class RemapResult:
    depth: np.ndarray
    diagnostics: dict[str, object]


def remap_eth3d_depth_to_undistorted(
    raw_depth: np.ndarray,
    raw_camera: ColmapCamera,
    target_camera: ColmapCamera,
    *,
    maximum_relative_depth_spread: float = 0.02,
    chunk_rows: int = 128,
    maximum_reprojection_error: float = 1.0e-4,
) -> RemapResult:
    """Forward-project each measured raw pixel into the prediction raster.

    Every raw array index is first shifted by ``+0.5`` into COLMAP pixel
    coordinates, inverted through the exact THIN_PRISM_FISHEYE model, and
    projected through the official undistorted PINHOLE model. The target
    COLMAP coordinate is shifted by ``-0.5`` before deterministic nearest-pixel
    rasterization. The original depth scalar stays attached to that camera ray.
    """

    if raw_camera.model != "THIN_PRISM_FISHEYE":
        raise ValueError("raw_camera must use THIN_PRISM_FISHEYE")
    if target_camera.model != "PINHOLE":
        raise ValueError("target_camera must use PINHOLE")
    raw_depth = np.asarray(raw_depth)
    if raw_depth.shape != (raw_camera.height, raw_camera.width):
        raise ValueError(
            f"Raw depth shape is {raw_depth.shape}, expected "
            f"{(raw_camera.height, raw_camera.width)}"
        )
    if (
        not np.isfinite(maximum_relative_depth_spread)
        or maximum_relative_depth_spread < 0.0
    ):
        raise ValueError(
            "maximum_relative_depth_spread must be finite and non-negative"
        )
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    roundtrip = validate_camera_roundtrip(
        raw_camera,
        target_camera,
        maximum_reprojection_error=maximum_reprojection_error,
    )
    raw_valid, raw_boundary_safe = depth_boundary_safe_mask(
        raw_depth,
        maximum_relative_depth_spread=maximum_relative_depth_spread,
    )
    target_shape = (target_camera.height, target_camera.width)
    output = np.full(target_shape, np.nan, dtype=np.float32)
    best_distance = np.full(target_shape, np.inf, dtype=np.float64)
    projected_candidate_count = 0
    outside_target_count = 0
    maximum_raw_reprojection_error = 0.0
    maximum_inverse_iterations = 0

    for row_begin in range(0, raw_camera.height, chunk_rows):
        row_end = min(row_begin + chunk_rows, raw_camera.height)
        local_rows, columns = np.nonzero(raw_boundary_safe[row_begin:row_end])
        if columns.size == 0:
            continue
        rows = local_rows + row_begin
        raw_pixels = raster_indices_to_colmap_pixels(columns, rows)
        inverse = unproject_thin_prism_fisheye(
            raw_camera,
            raw_pixels,
            maximum_reprojection_error=maximum_reprojection_error,
            require_all=True,
        )
        maximum_raw_reprojection_error = max(
            maximum_raw_reprojection_error,
            float(np.max(inverse.reprojection_error_pixels)),
        )
        maximum_inverse_iterations = max(
            maximum_inverse_iterations,
            int(np.max(inverse.iterations)),
        )
        target_projection = project_pinhole(target_camera, inverse.rays)
        target_raster = colmap_pixels_to_raster_coordinates(
            target_projection.pixels
        )
        target_x = target_raster[:, 0]
        target_y = target_raster[:, 1]
        inside = target_projection.valid & (
            (target_x >= -0.5)
            & (target_x < target_camera.width - 0.5)
            & (target_y >= -0.5)
            & (target_y < target_camera.height - 0.5)
        )
        outside_target_count += int(np.count_nonzero(~inside))
        if not np.any(inside):
            continue

        target_x = target_x[inside]
        target_y = target_y[inside]
        target_columns = np.floor(target_x + 0.5).astype(np.int64)
        target_rows = np.floor(target_y + 0.5).astype(np.int64)
        target_indices = target_rows * target_camera.width + target_columns
        distances = np.hypot(
            target_x - target_columns,
            target_y - target_rows,
        )
        depths = raw_depth[rows[inside], columns[inside]].astype(
            np.float32, copy=False
        )
        source_indices = rows[inside] * raw_camera.width + columns[inside]
        projected_candidate_count += int(target_indices.size)

        order = np.lexsort((source_indices, distances, target_indices))
        sorted_target_indices = target_indices[order]
        first_for_target = np.r_[
            True,
            sorted_target_indices[1:] != sorted_target_indices[:-1],
        ]
        selected = order[first_for_target]
        selected_target_indices = target_indices[selected]
        selected_distances = distances[selected]
        selected_depths = depths[selected]
        flat_best_distance = best_distance.reshape(-1)
        improves = selected_distances < flat_best_distance[selected_target_indices]
        selected_target_indices = selected_target_indices[improves]
        flat_best_distance[selected_target_indices] = selected_distances[improves]
        output.reshape(-1)[selected_target_indices] = selected_depths[improves]

    output_valid = np.isfinite(output) & (output > 0.0)
    valid_count = int(np.count_nonzero(output_valid))
    chosen_distance = best_distance[output_valid].astype(np.float64)
    raw_valid_count = int(np.count_nonzero(raw_valid))
    raw_boundary_safe_count = int(np.count_nonzero(raw_boundary_safe))
    diagnostics = {
        "raw_pixel_count": int(raw_depth.size),
        "raw_valid_pixel_count": raw_valid_count,
        "depth_boundary_safe_raw_pixel_count": raw_boundary_safe_count,
        "depth_boundary_rejected_raw_pixel_count": int(
            np.count_nonzero(raw_valid & ~raw_boundary_safe)
        ),
        "projected_candidate_count": projected_candidate_count,
        "outside_target_sample_count": outside_target_count,
        "target_pixel_collision_count": projected_candidate_count - valid_count,
        "target_pixel_count": target_camera.width * target_camera.height,
        "valid_remapped_pixel_count": valid_count,
        "valid_remapped_fraction": float(
            valid_count / (target_camera.width * target_camera.height)
        ),
        "pixel_convention": {
            "raster_index_to_colmap_pixel": "(column + 0.5, row + 0.5)",
            "colmap_pixel_to_raster_coordinate": "(x - 0.5, y - 0.5)",
        },
        "rasterization": "nearest_target_pixel_minimum_center_distance",
        "target_center_distance_pixels": _distribution(chosen_distance),
        "maximum_relative_depth_spread": maximum_relative_depth_spread,
        "inverse_projection": {
            "sample_count": raw_boundary_safe_count,
            "maximum_iterations_used": maximum_inverse_iterations,
            "maximum_raw_reprojection_error_pixels": (
                maximum_raw_reprojection_error
            ),
            "required_reprojection_error_pixels": maximum_reprojection_error,
        },
        "camera_roundtrip": roundtrip,
    }
    return RemapResult(depth=output, diagnostics=diagnostics)


def depth_boundary_safe_mask(
    depth: np.ndarray,
    *,
    maximum_relative_depth_spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject measurements adjacent to a conflicting valid depth sample."""

    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    if (
        not np.isfinite(maximum_relative_depth_spread)
        or maximum_relative_depth_spread < 0.0
    ):
        raise ValueError(
            "maximum_relative_depth_spread must be finite and non-negative"
        )
    valid = np.isfinite(depth) & (depth > 0.0)
    safe = valid.copy()
    for row_slice, neighbor_row_slice, column_slice, neighbor_column_slice in (
        (slice(1, None), slice(None, -1), slice(None), slice(None)),
        (slice(None, -1), slice(1, None), slice(None), slice(None)),
        (slice(None), slice(None), slice(1, None), slice(None, -1)),
        (slice(None), slice(None), slice(None, -1), slice(1, None)),
    ):
        current_valid = valid[row_slice, column_slice]
        neighbor_valid = valid[neighbor_row_slice, neighbor_column_slice]
        common = current_valid & neighbor_valid
        current_depth = depth[row_slice, column_slice]
        neighbor_depth = depth[neighbor_row_slice, neighbor_column_slice]
        conflicting = np.zeros(common.shape, dtype=bool)
        difference = np.abs(
            current_depth[common].astype(np.float64)
            - neighbor_depth[common].astype(np.float64)
        )
        denominator = np.maximum(
            np.minimum(
                current_depth[common].astype(np.float64),
                neighbor_depth[common].astype(np.float64),
            ),
            np.finfo(np.float64).eps,
        )
        conflicting[common] = (
            difference / denominator > maximum_relative_depth_spread
        )
        safe_view = safe[row_slice, column_slice]
        safe_view[conflicting] = False
    return valid, safe


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "sample_count": 0,
            "p50": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "sample_count": int(values.size),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }
