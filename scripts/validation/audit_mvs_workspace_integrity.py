#!/usr/bin/env python3
"""Audit MVS manifest bookkeeping and persisted support masks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


_COMPLETED_STATUSES = {"complete", "completed", "success", "succeeded"}
_GEOMETRY_SOURCE_ORDINAL_REVISION = 37
_FAST_MATRIX_MAGIC = b"PLASDEPTHMAT01\x00\x00"
_FAST_MATRIX_HEADER = struct.Struct("<16siii4xQ")
_CV_16UC1 = 2


@dataclass(frozen=True)
class MaskMetrics:
    width: int
    height: int
    channels: int
    dtype: str
    pixel_count: int
    nonzero_pixel_count: int
    nonzero_coverage: float
    minimum_value: float
    maximum_value: float
    all_white: bool
    all_black: bool


MaskReader = Callable[[Path], MaskMetrics]


@dataclass(frozen=True)
class SourceMaskMetrics:
    width: int
    height: int
    cv_type: int
    data_bytes: int
    file_size: int
    value_count: int
    nonzero_value_count: int
    maximum_value: int
    observed_bit_mask: int


SourceMaskReader = Callable[[Path], SourceMaskMetrics]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    if not math.isfinite(converted) or not converted.is_integer():
        return None
    return int(converted)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "minimum": None,
            "p50": None,
            "p90": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _read_mask_metrics(path: Path) -> MaskMetrics:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "support-mask decoding requires OpenCV and NumPy; initialize the "
            "repository Python runtime before running this audit"
        ) from error

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError("OpenCV could not decode the image")
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
        pixel_nonzero = image != 0
    elif image.ndim == 3:
        height, width, channels = image.shape
        pixel_nonzero = np.any(image != 0, axis=2)
    else:
        raise ValueError(f"unsupported mask array rank: {image.ndim}")

    pixel_count = int(width * height)
    nonzero_pixel_count = int(np.count_nonzero(pixel_nonzero))
    minimum_value = float(np.min(image))
    maximum_value = float(np.max(image))
    if np.issubdtype(image.dtype, np.integer):
        white_value = float(np.iinfo(image.dtype).max)
    elif np.issubdtype(image.dtype, np.bool_):
        white_value = 1.0
    else:
        white_value = 1.0
    return MaskMetrics(
        width=int(width),
        height=int(height),
        channels=int(channels),
        dtype=str(image.dtype),
        pixel_count=pixel_count,
        nonzero_pixel_count=nonzero_pixel_count,
        nonzero_coverage=(
            nonzero_pixel_count / pixel_count if pixel_count else 0.0
        ),
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        all_white=bool(
            pixel_count > 0
            and minimum_value == white_value
            and maximum_value == white_value
        ),
        all_black=bool(pixel_count > 0 and maximum_value == 0.0),
    )


def _read_geometry_source_mask_metrics(path: Path) -> SourceMaskMetrics:
    """Read and validate a persisted CV_16UC1 geometry-source mask."""
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "geometry-source-mask decoding requires NumPy; initialize the "
            "repository Python runtime before running this audit"
        ) from error

    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(_FAST_MATRIX_HEADER.size)
    if len(header) != _FAST_MATRIX_HEADER.size:
        raise ValueError(
            f"incomplete PlaScan matrix header: read {len(header)} bytes, "
            f"expected {_FAST_MATRIX_HEADER.size}"
        )

    magic, rows, columns, cv_type, data_bytes = _FAST_MATRIX_HEADER.unpack(header)
    if magic != _FAST_MATRIX_MAGIC:
        raise ValueError("unexpected PlaScan matrix magic")
    if rows <= 0 or columns <= 0:
        raise ValueError(f"invalid PlaScan matrix dimensions: {rows}x{columns}")
    if cv_type != _CV_16UC1:
        raise ValueError(
            f"unexpected PlaScan matrix type {cv_type}; expected CV_16UC1 "
            f"({_CV_16UC1})"
        )

    expected_data_bytes = rows * columns * 2
    if data_bytes != expected_data_bytes:
        raise ValueError(
            "PlaScan matrix payload size mismatch: "
            f"header={data_bytes}, expected={expected_data_bytes}"
        )
    expected_file_size = _FAST_MATRIX_HEADER.size + expected_data_bytes
    file_size = path.stat().st_size
    if file_size != expected_file_size:
        raise ValueError(
            "PlaScan matrix file size mismatch: "
            f"actual={file_size}, expected={expected_file_size}"
        )

    values = np.memmap(
        path,
        dtype=np.dtype("<u2"),
        mode="r",
        offset=_FAST_MATRIX_HEADER.size,
        shape=(rows * columns,),
    )
    nonzero_value_count = 0
    maximum_value = 0
    observed_bit_mask = 0
    chunk_size = 1 << 20
    for start in range(0, values.size, chunk_size):
        chunk = values[start : start + chunk_size]
        nonzero_value_count += int(np.count_nonzero(chunk))
        maximum_value = max(maximum_value, int(np.max(chunk)))
        observed_bit_mask |= int(np.bitwise_or.reduce(chunk))

    return SourceMaskMetrics(
        width=columns,
        height=rows,
        cv_type=cv_type,
        data_bytes=data_bytes,
        file_size=file_size,
        value_count=rows * columns,
        nonzero_value_count=nonzero_value_count,
        maximum_value=maximum_value,
        observed_bit_mask=observed_bit_mask,
    )


def _frames(document: dict[str, Any]) -> list[dict[str, Any]]:
    records = document.get("frames", document.get("depth_artifacts"))
    if not isinstance(records, list):
        raise ValueError("manifest must contain a frames array")
    frames: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"frame record {position} must be an object")
        frames.append(record)
    return frames


def _acceptance(frame: dict[str, Any]) -> str:
    value = frame.get("acceptance")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    decision = frame.get("quality_decision")
    if isinstance(decision, dict):
        value = decision.get("acceptance")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "unknown"


def _artifact_path(configured: str, manifest_path: Path) -> Path:
    path = Path(configured.replace("\\", "/"))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _sparse_residual_summary(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    available_true_count = 0
    available_false_count = 0
    missing_count = 0
    invalid_available_count = 0
    projected_samples: list[float] = []
    valid_samples: list[float] = []
    valid_ratios: list[float] = []
    median_errors: list[float] = []

    for frame in frames:
        decision = frame.get("quality_decision")
        residual = (
            decision.get("sparse_absolute_depth_residual")
            if isinstance(decision, dict)
            else None
        )
        if not isinstance(residual, dict):
            missing_count += 1
            continue
        available = residual.get("available")
        if available is True:
            available_true_count += 1
        elif available is False:
            available_false_count += 1
        else:
            invalid_available_count += 1
            continue

        if available is not True:
            continue
        projected = _finite_number(residual.get("projected_sample_count"))
        valid = _finite_number(residual.get("valid_sample_count"))
        ratio = _finite_number(residual.get("valid_sample_ratio"))
        median_error = _finite_number(residual.get("median_absolute_log_error"))
        if projected is not None:
            projected_samples.append(projected)
        if valid is not None:
            valid_samples.append(valid)
        if ratio is not None:
            valid_ratios.append(ratio)
        elif projected is not None and projected > 0.0 and valid is not None:
            valid_ratios.append(valid / projected)
        if median_error is not None:
            median_errors.append(median_error)

    return {
        "available_count": available_true_count,
        "unavailable_count": available_false_count,
        "missing_count": missing_count,
        "invalid_available_count": invalid_available_count,
        "projected_sample_count_distribution": _distribution(projected_samples),
        "valid_sample_count_distribution": _distribution(valid_samples),
        "valid_sample_ratio_distribution": _distribution(valid_ratios),
        "median_absolute_log_error_distribution": _distribution(median_errors),
    }


def _geometry_source_table_issues(
    configured: Any, ref_index: int | None
) -> tuple[list[int], list[str]]:
    if not isinstance(configured, list):
        return [], ["missing_or_non_array_table"]

    issues: list[str] = []
    if not 1 <= len(configured) <= 16:
        issues.append("table_length_out_of_range")

    indices: list[int] = []
    for value in configured:
        index = _integer(value)
        if index is None or index < 0:
            issues.append("invalid_source_index")
            continue
        indices.append(index)
    if len(indices) != len(configured):
        issues.append("table_contains_non_integer_or_negative_index")
    if len(set(indices)) != len(indices):
        issues.append("duplicate_source_index")
    if ref_index is not None and ref_index in indices:
        issues.append("table_contains_reference_index")
    return indices, list(dict.fromkeys(issues))


def _effective_algorithm_revision(
    frame: dict[str, Any], root_revision: int | None
) -> int | None:
    frame_revision = _integer(frame.get("algorithm_revision"))
    return frame_revision if frame_revision is not None else root_revision


def audit_manifest(
    document: dict[str, Any],
    manifest_path: Path,
    *,
    coverage_tolerance: float = 1.0e-6,
    mask_reader: MaskReader = _read_mask_metrics,
    source_mask_reader: SourceMaskReader = _read_geometry_source_mask_metrics,
) -> dict[str, Any]:
    if coverage_tolerance < 0.0 or not math.isfinite(coverage_tolerance):
        raise ValueError("coverage_tolerance must be a finite non-negative number")

    manifest_path = manifest_path.resolve()
    frames = _frames(document)
    root_revision = _integer(document.get("algorithm_revision"))
    statuses: Counter[str] = Counter()
    acceptances: Counter[str] = Counter()
    revisions: Counter[str] = Counter()
    duplicate_ref_indices = 0
    seen_ref_indices: set[int] = set()
    fusion_eligible_count = 0
    fusion_ineligible_count = 0
    fusion_eligibility_unknown_count = 0

    for frame in frames:
        status = str(frame.get("status", "unknown")).strip().lower() or "unknown"
        statuses[status] += 1
        acceptances[_acceptance(frame)] += 1
        revision = _integer(frame.get("algorithm_revision"))
        revisions[str(revision) if revision is not None else "unknown"] += 1
        ref_index = _integer(frame.get("ref_index"))
        if ref_index is not None:
            duplicate_ref_indices += int(ref_index in seen_ref_indices)
            seen_ref_indices.add(ref_index)
        eligible = frame.get("fusion_eligible")
        if eligible is True:
            fusion_eligible_count += 1
        elif eligible is False:
            fusion_ineligible_count += 1
        else:
            fusion_eligibility_unknown_count += 1

    complete_count = sum(
        count for status, count in statuses.items() if status in _COMPLETED_STATUSES
    )
    support_details: list[dict[str, Any]] = []
    support_counts: Counter[str] = Counter()
    support_coverages: list[float] = []
    coverage_differences: list[float] = []
    errors: list[str] = []
    warnings: list[str] = []

    for position, frame in enumerate(frames):
        status = str(frame.get("status", "unknown")).strip().lower() or "unknown"
        completed = status in _COMPLETED_STATUSES
        configured = frame.get("support_mask_path")
        configured_path = configured.strip() if isinstance(configured, str) else ""
        if completed:
            support_counts["expected"] += 1
        if not configured_path:
            if completed:
                support_counts["missing_path"] += 1
                errors.append(f"frame {frame.get('ref_index', position)}: support_mask_path is empty")
            continue

        support_counts["configured"] += 1
        path = _artifact_path(configured_path, manifest_path)
        detail: dict[str, Any] = {
            "ref_index": frame.get("ref_index"),
            "path": str(path),
            "completed": completed,
            "exists": path.is_file(),
            "issues": [],
        }
        if not path.is_file():
            support_counts["missing_file"] += 1
            detail["issues"].append("missing_file")
            errors.append(f"frame {frame.get('ref_index', position)}: mask does not exist: {path}")
            support_details.append(detail)
            continue

        support_counts["existing"] += 1
        try:
            metrics = mask_reader(path)
        except (OSError, RuntimeError, ValueError) as error:
            support_counts["unreadable"] += 1
            detail["issues"].append("unreadable")
            detail["decode_error"] = str(error)
            errors.append(
                f"frame {frame.get('ref_index', position)}: cannot decode mask {path}: {error}"
            )
            support_details.append(detail)
            continue

        support_counts["readable"] += 1
        detail.update(asdict(metrics))
        support_coverages.append(metrics.nonzero_coverage)
        if metrics.all_white:
            support_counts["all_white"] += 1
            detail["issues"].append("all_white")
        if metrics.all_black:
            support_counts["all_black"] += 1
            detail["issues"].append("all_black")
            if completed:
                errors.append(
                    f"frame {frame.get('ref_index', position)}: completed frame has an all-black support mask"
                )

        expected_width = _integer(frame.get("grid_width"))
        expected_height = _integer(frame.get("grid_height"))
        detail["manifest_width"] = expected_width
        detail["manifest_height"] = expected_height
        if (
            expected_width is not None
            and expected_width > 0
            and expected_height is not None
            and expected_height > 0
        ):
            if (metrics.width, metrics.height) != (expected_width, expected_height):
                support_counts["dimension_mismatch"] += 1
                detail["issues"].append("dimension_mismatch")
                errors.append(
                    f"frame {frame.get('ref_index', position)}: mask size "
                    f"{metrics.width}x{metrics.height} != manifest "
                    f"{expected_width}x{expected_height}"
                )
        else:
            support_counts["missing_manifest_dimensions"] += 1

        manifest_coverage = _finite_number(frame.get("mask_coverage"))
        detail["manifest_mask_coverage"] = manifest_coverage
        if manifest_coverage is None:
            support_counts["missing_manifest_coverage"] += 1
        else:
            difference = abs(metrics.nonzero_coverage - manifest_coverage)
            coverage_differences.append(difference)
            detail["coverage_absolute_difference"] = difference
            if difference > coverage_tolerance:
                support_counts["coverage_mismatch"] += 1
                detail["issues"].append("coverage_mismatch")
                errors.append(
                    f"frame {frame.get('ref_index', position)}: support coverage "
                    f"{metrics.nonzero_coverage:.9f} != manifest "
                    f"{manifest_coverage:.9f} (difference={difference:.9f})"
                )
        support_details.append(detail)

    source_details: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    source_table_issue_counts: Counter[str] = Counter()
    source_nonzero_counts: list[float] = []
    source_maximum_values: list[float] = []
    aggregate_observed_bit_mask = 0
    aggregate_out_of_range_bit_mask = 0

    for position, frame in enumerate(frames):
        status = str(frame.get("status", "unknown")).strip().lower() or "unknown"
        completed = status in _COMPLETED_STATUSES
        revision = _effective_algorithm_revision(frame, root_revision)
        if (
            not completed
            or revision is None
            or revision < _GEOMETRY_SOURCE_ORDINAL_REVISION
        ):
            continue

        source_counts["revision_completed"] += 1
        acceptance = _acceptance(frame)
        required = acceptance in {"accepted", "validation_only"}
        if required:
            source_counts["required"] += 1

        ref_index = _integer(frame.get("ref_index"))
        frame_label = frame.get("ref_index", position)
        configured = frame.get("raw_geometry_source_mask_path")
        configured_path = configured.strip() if isinstance(configured, str) else ""
        validate_table = required or bool(configured_path)
        source_indices: list[int] = []
        table_issues: list[str] = []
        if validate_table:
            source_indices, table_issues = _geometry_source_table_issues(
                frame.get("geometry_source_indices"), ref_index
            )
            if table_issues:
                source_counts["invalid_table"] += 1
                source_table_issue_counts.update(table_issues)
                errors.append(
                    f"frame {frame_label}: geometry_source_indices is invalid: "
                    f"{', '.join(table_issues)}"
                )
            else:
                source_counts["valid_table"] += 1

        detail: dict[str, Any] = {
            "ref_index": frame.get("ref_index"),
            "algorithm_revision": revision,
            "acceptance": acceptance,
            "required": required,
            "path": None,
            "exists": False,
            "geometry_source_indices": source_indices,
            "geometry_source_count": len(source_indices),
            "table_issues": table_issues,
            "issues": list(table_issues),
        }
        if not configured_path:
            if required:
                source_counts["missing_required_path"] += 1
                detail["issues"].append("missing_required_path")
                errors.append(
                    f"frame {frame_label}: raw_geometry_source_mask_path is empty"
                )
            source_details.append(detail)
            continue

        source_counts["configured"] += 1
        path = _artifact_path(configured_path, manifest_path)
        detail["path"] = str(path)
        detail["exists"] = path.is_file()
        if not path.is_file():
            source_counts["missing_file"] += 1
            detail["issues"].append("missing_file")
            errors.append(
                f"frame {frame_label}: geometry source mask does not exist: {path}"
            )
            source_details.append(detail)
            continue

        source_counts["existing"] += 1
        try:
            metrics = source_mask_reader(path)
        except (OSError, RuntimeError, ValueError) as error:
            source_counts["unreadable"] += 1
            detail["issues"].append("unreadable")
            detail["decode_error"] = str(error)
            errors.append(
                f"frame {frame_label}: cannot decode geometry source mask "
                f"{path}: {error}"
            )
            source_details.append(detail)
            continue

        source_counts["readable"] += 1
        detail.update(asdict(metrics))
        detail["observed_bit_mask_hex"] = f"0x{metrics.observed_bit_mask:04x}"
        source_nonzero_counts.append(float(metrics.nonzero_value_count))
        source_maximum_values.append(float(metrics.maximum_value))
        aggregate_observed_bit_mask |= metrics.observed_bit_mask

        expected_width = _integer(frame.get("grid_width"))
        expected_height = _integer(frame.get("grid_height"))
        detail["manifest_width"] = expected_width
        detail["manifest_height"] = expected_height
        if (
            expected_width is not None
            and expected_width > 0
            and expected_height is not None
            and expected_height > 0
            and (metrics.width, metrics.height)
            != (expected_width, expected_height)
        ):
            source_counts["dimension_mismatch"] += 1
            detail["issues"].append("dimension_mismatch")
            errors.append(
                f"frame {frame_label}: geometry source mask size "
                f"{metrics.width}x{metrics.height} != manifest "
                f"{expected_width}x{expected_height}"
            )

        if not table_issues:
            allowed_bit_mask = (1 << len(source_indices)) - 1
            out_of_range_bit_mask = (
                metrics.observed_bit_mask & (~allowed_bit_mask & 0xFFFF)
            )
            detail["allowed_bit_mask"] = allowed_bit_mask
            detail["allowed_bit_mask_hex"] = f"0x{allowed_bit_mask:04x}"
            detail["out_of_range_bit_mask"] = out_of_range_bit_mask
            detail["out_of_range_bit_mask_hex"] = (
                f"0x{out_of_range_bit_mask:04x}"
            )
            if out_of_range_bit_mask:
                source_counts["out_of_range_bit_mask"] += 1
                aggregate_out_of_range_bit_mask |= out_of_range_bit_mask
                detail["issues"].append("out_of_range_bit_mask")
                errors.append(
                    f"frame {frame_label}: geometry source mask uses bits "
                    f"0x{out_of_range_bit_mask:04x} outside "
                    f"geometry_source_indices length {len(source_indices)}"
                )
        source_details.append(detail)

    if support_counts["all_white"]:
        warnings.append(
            f"{support_counts['all_white']} support masks are entirely white; "
            "verify that full-frame support is intended"
        )
    if support_counts["missing_manifest_dimensions"]:
        warnings.append(
            f"{support_counts['missing_manifest_dimensions']} masks lack positive "
            "grid_width/grid_height metadata"
        )
    if support_counts["missing_manifest_coverage"]:
        warnings.append(
            f"{support_counts['missing_manifest_coverage']} masks lack numeric "
            "mask_coverage metadata"
        )

    counts = {
        "frame": len(frames),
        "complete": complete_count,
        "accepted": acceptances["accepted"],
        "validation": acceptances["validation_only"],
        "validation_only": acceptances["validation_only"],
        "rejected": acceptances["rejected"],
        "fusion_eligible": fusion_eligible_count,
        "fusion_ineligible": fusion_ineligible_count,
        "fusion_eligibility_unknown": fusion_eligibility_unknown_count,
    }
    return {
        "schema": "plascan.mvs.workspace_integrity_audit.v2",
        "input": str(manifest_path),
        "coverage_tolerance": coverage_tolerance,
        "integrity_passed": not errors,
        "counts": counts,
        "status_distribution": dict(sorted(statuses.items())),
        "acceptance_distribution": dict(sorted(acceptances.items())),
        "duplicate_ref_index_count": duplicate_ref_indices,
        "algorithm_revision": {
            "manifest": root_revision,
            "frame_distribution": dict(sorted(revisions.items())),
        },
        "support_masks": {
            "expected_count": support_counts["expected"],
            "configured_count": support_counts["configured"],
            "existing_count": support_counts["existing"],
            "readable_count": support_counts["readable"],
            "missing_path_count": support_counts["missing_path"],
            "missing_file_count": support_counts["missing_file"],
            "unreadable_count": support_counts["unreadable"],
            "dimension_mismatch_count": support_counts["dimension_mismatch"],
            "all_white_count": support_counts["all_white"],
            "all_black_count": support_counts["all_black"],
            "coverage_mismatch_count": support_counts["coverage_mismatch"],
            "missing_manifest_dimensions_count": support_counts[
                "missing_manifest_dimensions"
            ],
            "missing_manifest_coverage_count": support_counts[
                "missing_manifest_coverage"
            ],
            "nonzero_coverage_distribution": _distribution(support_coverages),
            "coverage_absolute_difference_distribution": _distribution(
                coverage_differences
            ),
            "details": support_details,
        },
        "geometry_source_masks": {
            "contract_revision": _GEOMETRY_SOURCE_ORDINAL_REVISION,
            "revision_completed_count": source_counts["revision_completed"],
            "required_count": source_counts["required"],
            "configured_count": source_counts["configured"],
            "existing_count": source_counts["existing"],
            "readable_count": source_counts["readable"],
            "missing_required_path_count": source_counts[
                "missing_required_path"
            ],
            "missing_file_count": source_counts["missing_file"],
            "unreadable_count": source_counts["unreadable"],
            "valid_table_count": source_counts["valid_table"],
            "invalid_table_count": source_counts["invalid_table"],
            "table_issue_distribution": dict(
                sorted(source_table_issue_counts.items())
            ),
            "dimension_mismatch_count": source_counts["dimension_mismatch"],
            "out_of_range_bit_mask_count": source_counts[
                "out_of_range_bit_mask"
            ],
            "aggregate_observed_bit_mask": aggregate_observed_bit_mask,
            "aggregate_observed_bit_mask_hex": (
                f"0x{aggregate_observed_bit_mask:04x}"
            ),
            "aggregate_out_of_range_bit_mask": (
                aggregate_out_of_range_bit_mask
            ),
            "aggregate_out_of_range_bit_mask_hex": (
                f"0x{aggregate_out_of_range_bit_mask:04x}"
            ),
            "nonzero_value_count_distribution": _distribution(
                source_nonzero_counts
            ),
            "maximum_value_distribution": _distribution(
                source_maximum_values
            ),
            "details": source_details,
        },
        "sparse_absolute_depth_residual": _sparse_residual_summary(frames),
        "errors": errors,
        "warnings": warnings,
    }


def _print_summary(result: dict[str, Any], output: Path | None) -> None:
    counts = result["counts"]
    masks = result["support_masks"]
    source_masks = result["geometry_source_masks"]
    sparse = result["sparse_absolute_depth_residual"]
    state = "PASS" if result["integrity_passed"] else "FAIL"
    print(f"MVS workspace integrity: {state}")
    print(
        "frames: "
        f"total={counts['frame']} complete={counts['complete']} "
        f"accepted={counts['accepted']} validation={counts['validation']} "
        f"rejected={counts['rejected']} fusion_eligible={counts['fusion_eligible']}"
    )
    print(
        "support masks: "
        f"existing={masks['existing_count']}/{masks['expected_count']} "
        f"readable={masks['readable_count']} white={masks['all_white_count']} "
        f"black={masks['all_black_count']} size_mismatch="
        f"{masks['dimension_mismatch_count']} coverage_mismatch="
        f"{masks['coverage_mismatch_count']}"
    )
    print(
        "geometry source masks: "
        f"existing={source_masks['existing_count']}/"
        f"{source_masks['configured_count']} "
        f"required={source_masks['required_count']} "
        f"readable={source_masks['readable_count']} "
        f"invalid_table={source_masks['invalid_table_count']} "
        f"out_of_range_bits={source_masks['out_of_range_bit_mask_count']}"
    )
    median_stats = sparse["median_absolute_log_error_distribution"]
    print(
        "sparse residual: "
        f"available={sparse['available_count']} unavailable="
        f"{sparse['unavailable_count']} missing={sparse['missing_count']} "
        f"median_error_p50={median_stats['p50']} p90={median_stats['p90']}"
    )
    print(
        f"issues: errors={len(result['errors'])} warnings={len(result['warnings'])}"
    )
    for error in result["errors"][:3]:
        print(f"  error: {error}")
    if len(result["errors"]) > 3:
        print(f"  ... {len(result['errors']) - 3} more errors in the JSON report")
    if output is not None:
        print(f"json: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit MVS frame decisions, algorithm revisions, support masks, "
            "geometry source ordinals, and sparse absolute-depth residual "
            "diagnostics."
        )
    )
    parser.add_argument("manifest", type=Path, help="path to mvs_manifest.json")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=1.0e-6,
        help="maximum absolute mask-coverage difference (default: 1e-6)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("manifest root must be a JSON object")
        result = audit_manifest(
            document,
            args.manifest,
            coverage_tolerance=args.coverage_tolerance,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"MVS workspace integrity audit failed: {error}", file=sys.stderr)
        return 2

    _print_summary(result, args.output)
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
