"""Per-frame depth and evidence metrics with bounded working memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .accumulators import (
    DistributionAccumulator,
    SpreadAggregate,
    SupportAggregate,
)
from .fast_matrix import CV_16UC1, CV_32FC1, read_fast_matrix


CHUNK_PIXEL_COUNT = 1_000_000


def resize_nearest(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Match OpenCV INTER_NEAREST raster-to-grid sampling without cv2."""
    if values.shape == shape:
        return values
    rows = np.minimum(
        (np.arange(shape[0], dtype=np.int64) * values.shape[0]) // shape[0],
        values.shape[0] - 1,
    )
    columns = np.minimum(
        (np.arange(shape[1], dtype=np.int64) * values.shape[1]) // shape[1],
        values.shape[1] - 1,
    )
    return values[np.ix_(rows, columns)]


def depth_metrics(depth: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0)
    pixel_count = int(depth.size)
    valid_count = int(np.count_nonzero(valid))
    return {
        "pixel_count": pixel_count,
        "valid_pixel_count": valid_count,
        "coverage": valid_count / pixel_count,
        "zero_depth_count": int(np.count_nonzero(finite & (depth == 0.0))),
        # Keep negative and non-finite counts disjoint. Negative infinity is
        # represented by nonfinite_depth_count, not counted twice.
        "negative_depth_count": int(np.count_nonzero(finite & (depth < 0.0))),
        "nonfinite_depth_count": int(pixel_count - np.count_nonzero(finite)),
    }, valid


def chunk_rows(columns: int) -> int:
    return max(1, CHUNK_PIXEL_COUNT // max(1, columns))


def relative_depth_metrics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    common_valid: np.ndarray,
    sample_limit: int,
    aggregate: DistributionAccumulator,
    seed: int,
) -> dict[str, Any]:
    local = DistributionAccumulator(sample_limit, seed)
    rows_per_chunk = chunk_rows(baseline.shape[1])
    for begin in range(0, baseline.shape[0], rows_per_chunk):
        end = min(baseline.shape[0], begin + rows_per_chunk)
        mask = common_valid[begin:end]
        if not np.any(mask):
            continue
        # Promote before subtraction/division. Two finite positive float32
        # depths can otherwise overflow their float32 difference or ratio and
        # silently disappear when the distribution rejects non-finite values.
        baseline_values = np.asarray(
            np.asarray(baseline[begin:end])[mask], dtype=np.float64
        )
        candidate_values = np.asarray(
            np.asarray(candidate[begin:end])[mask], dtype=np.float64
        )
        difference = np.abs(candidate_values - baseline_values) / np.maximum(
            baseline_values,
            np.finfo(np.float32).eps,
        )
        local.add(difference)
        aggregate.add(difference)
    return local.summary()


def empty_support_metrics(
    valid_depth: np.ndarray, sample_limit: int, seed: int
) -> dict[str, Any]:
    return {
        "available": False,
        "artifact_path": None,
        "scope": "valid_depth_pixels",
        "scope_valid_pixel_count": int(np.count_nonzero(valid_depth)),
        "distribution": DistributionAccumulator(sample_limit, seed).summary(),
        "ratio_ge_2": None,
        "ratio_ge_3": None,
    }


def support_metrics(
    path: Path | None,
    depth_shape: tuple[int, int],
    valid_depth: np.ndarray,
    sample_limit: int,
    aggregate: SupportAggregate,
    seed: int,
    resample_source_to_depth_shape: bool = False,
) -> dict[str, Any]:
    if path is None:
        return empty_support_metrics(valid_depth, sample_limit, seed)
    support = read_fast_matrix(path, CV_16UC1)
    source_shape = support.shape
    if support.shape != depth_shape and resample_source_to_depth_shape:
        support = resize_nearest(support, depth_shape)
    if support.shape != depth_shape:
        raise ValueError(
            f"Geometry support shape mismatch: support={support.shape}, "
            f"depth={depth_shape}, path={path}"
        )

    local = DistributionAccumulator(sample_limit, seed)
    scoped_count = 0
    count_ge_2 = 0
    count_ge_3 = 0
    rows_per_chunk = chunk_rows(depth_shape[1])
    for begin in range(0, depth_shape[0], rows_per_chunk):
        end = min(depth_shape[0], begin + rows_per_chunk)
        mask = valid_depth[begin:end]
        if not np.any(mask):
            continue
        values = np.asarray(support[begin:end])[mask]
        scoped_count += int(values.size)
        count_ge_2 += int(np.count_nonzero(values >= 2))
        count_ge_3 += int(np.count_nonzero(values >= 3))
        local.add(values)
        aggregate.distribution.add(values)
    aggregate.available_frame_count += 1
    aggregate.scope_valid_pixel_count += scoped_count
    aggregate.count_ge_2 += count_ge_2
    aggregate.count_ge_3 += count_ge_3
    return {
        "available": True,
        "artifact_path": str(path),
        "source_shape": {"rows": source_shape[0], "columns": source_shape[1]},
        "nearest_resampled_to_depth_grid": source_shape != depth_shape,
        "scope": "valid_depth_pixels",
        "scope_valid_pixel_count": scoped_count,
        "distribution": local.summary(),
        "ratio_ge_2": count_ge_2 / scoped_count if scoped_count else None,
        "ratio_ge_3": count_ge_3 / scoped_count if scoped_count else None,
    }


def empty_spread_metrics(
    valid_depth: np.ndarray, sample_limit: int, seed: int
) -> dict[str, Any]:
    return {
        "available": False,
        "artifact_path": None,
        "scope": "valid_depth_pixels",
        "scope_valid_pixel_count": int(np.count_nonzero(valid_depth)),
        "nonfinite_count": 0,
        "negative_count": 0,
        "distribution": DistributionAccumulator(sample_limit, seed).summary(),
    }


def spread_metrics(
    path: Path | None,
    depth_shape: tuple[int, int],
    valid_depth: np.ndarray,
    sample_limit: int,
    aggregate: SpreadAggregate,
    seed: int,
    resample_source_to_depth_shape: bool = False,
) -> dict[str, Any]:
    if path is None:
        return empty_spread_metrics(valid_depth, sample_limit, seed)
    spread = read_fast_matrix(path, CV_32FC1)
    source_shape = spread.shape
    if spread.shape != depth_shape and resample_source_to_depth_shape:
        spread = resize_nearest(spread, depth_shape)
    if spread.shape != depth_shape:
        raise ValueError(
            f"Inverse-depth-spread shape mismatch: spread={spread.shape}, "
            f"depth={depth_shape}, path={path}"
        )

    local = DistributionAccumulator(sample_limit, seed)
    scoped_count = 0
    nonfinite_count = 0
    negative_count = 0
    rows_per_chunk = chunk_rows(depth_shape[1])
    for begin in range(0, depth_shape[0], rows_per_chunk):
        end = min(depth_shape[0], begin + rows_per_chunk)
        mask = valid_depth[begin:end]
        if not np.any(mask):
            continue
        values = np.asarray(spread[begin:end])[mask]
        scoped_count += int(values.size)
        finite = np.isfinite(values)
        nonfinite_count += int(values.size - np.count_nonzero(finite))
        negative_count += int(np.count_nonzero(finite & (values < 0.0)))
        usable = values[finite & (values >= 0.0)]
        local.add(usable)
        aggregate.distribution.add(usable)
    aggregate.available_frame_count += 1
    aggregate.scope_valid_pixel_count += scoped_count
    aggregate.nonfinite_count += nonfinite_count
    aggregate.negative_count += negative_count
    return {
        "available": True,
        "artifact_path": str(path),
        "source_shape": {"rows": source_shape[0], "columns": source_shape[1]},
        "nearest_resampled_to_depth_grid": source_shape != depth_shape,
        "scope": "valid_depth_pixels",
        "scope_valid_pixel_count": scoped_count,
        "nonfinite_count": nonfinite_count,
        "negative_count": negative_count,
        "distribution": local.summary(),
    }
