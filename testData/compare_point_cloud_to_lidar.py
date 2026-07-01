#!/usr/bin/env python3
"""Compare an image-derived point cloud against a LiDAR/reference point cloud."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Point3 = tuple[float, float, float]
TEXT_POINT_EXTENSIONS = {".csv", ".txt", ".xyz"}
NEAREST_NEIGHBOR_METHODS = {"auto", "brute", "kd-tree"}
AUTO_KD_TREE_MIN_REFERENCE_POINTS = 64
PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


@dataclass(frozen=True)
class KdNode:
    point: Point3
    axis: int
    left: "KdNode | None"
    right: "KdNode | None"


@dataclass(frozen=True)
class PlyHeader:
    format_name: str
    vertex_count: int
    vertex_properties: list[tuple[str, str]]
    data_offset: int


def sampled_indices(count: int, max_points: int | None) -> list[int]:
    if count <= 0:
        return []
    if max_points is None or max_points <= 0 or max_points >= count:
        return list(range(count))
    if max_points == 1:
        return [0]
    indices = [round(i * (count - 1) / (max_points - 1)) for i in range(max_points)]
    result: list[int] = []
    seen: set[int] = set()
    for index in indices:
        bounded = max(0, min(count - 1, int(index)))
        if bounded not in seen:
            seen.add(bounded)
            result.append(bounded)
    return result


def read_ply_header(path: Path) -> PlyHeader:
    path = Path(path)
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    format_name: str | None = None
    in_vertex_element = False

    with path.open("rb") as handle:
        first = handle.readline()
        if first.strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"{path}: missing PLY end_header")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: PLY header must be ASCII") from exc
            if line == "end_header":
                return PlyHeader(
                    format_name=format_name or "",
                    vertex_count=vertex_count if vertex_count is not None else -1,
                    vertex_properties=vertex_properties,
                    data_offset=handle.tell(),
                )
            if line.startswith("format "):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"{path}: invalid PLY format line: {line}")
                format_name = parts[1]
                continue
            if line.startswith("element "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"{path}: invalid PLY element line: {line}")
                in_vertex_element = parts[1] == "vertex"
                if in_vertex_element:
                    try:
                        vertex_count = int(parts[2])
                    except ValueError as exc:
                        raise ValueError(f"{path}: invalid vertex count: {parts[2]}") from exc
                    if vertex_count < 0:
                        raise ValueError(f"{path}: vertex count must be non-negative")
                continue
            if in_vertex_element and line.startswith("property "):
                parts = line.split()
                if len(parts) < 3 or parts[1] == "list":
                    raise ValueError(f"{path}: unsupported vertex property line: {line}")
                vertex_properties.append((parts[1], parts[-1]))


def validate_ply_xyz_properties(path: Path, header: PlyHeader) -> tuple[int, int, int]:
    if header.vertex_count < 0:
        raise ValueError(f"{path}: missing PLY vertex element")
    if not header.format_name:
        raise ValueError(f"{path}: missing PLY format")
    property_names = [name for _, name in header.vertex_properties]
    for coordinate in ("x", "y", "z"):
        if coordinate not in property_names:
            raise ValueError(f"{path}: missing PLY vertex property {coordinate}")
    return property_names.index("x"), property_names.index("y"), property_names.index("z")


def read_ascii_ply_points(path: Path, max_points: int | None = None) -> list[Point3]:
    path = Path(path)
    header = read_ply_header(path)
    if header.format_name != "ascii":
        raise ValueError(f"{path}: only ASCII PLY format ascii 1.0 is supported by read_ascii_ply_points")
    x_index, y_index, z_index = validate_ply_xyz_properties(path, header)

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path} is not a PLY file")

    header_end = -1
    for index, raw_line in enumerate(lines):
        if raw_line.strip() == "end_header":
            header_end = index
            break

    if header_end < 0:
        raise ValueError(f"{path}: missing PLY end_header")
    points: list[Point3] = []
    first_vertex_line = header_end + 1
    if len(lines) < first_vertex_line + header.vertex_count:
        raise ValueError(f"{path}: expected {header.vertex_count} vertex rows")

    for row_index in sampled_indices(header.vertex_count, max_points):
        line = lines[first_vertex_line + row_index].strip()
        parts = line.split()
        if len(parts) < len(header.vertex_properties):
            raise ValueError(f"{path}: vertex row {row_index + 1} has too few fields")
        try:
            point = (float(parts[x_index]), float(parts[y_index]), float(parts[z_index]))
        except ValueError as exc:
            raise ValueError(f"{path}: invalid numeric value in vertex row {row_index + 1}") from exc
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"{path}: non-finite coordinate in vertex row {row_index + 1}")
        points.append(point)

    if not points:
        raise ValueError(f"{path}: no valid PLY vertex points")
    return points


def read_binary_little_endian_ply_points(path: Path, max_points: int | None = None) -> list[Point3]:
    path = Path(path)
    header = read_ply_header(path)
    if header.format_name != "binary_little_endian":
        raise ValueError(f"{path}: unsupported PLY format {header.format_name}")
    x_index, y_index, z_index = validate_ply_xyz_properties(path, header)

    struct_codes: list[str] = []
    for property_type, property_name in header.vertex_properties:
        code = PLY_SCALAR_FORMATS.get(property_type)
        if code is None:
            raise ValueError(f"{path}: unsupported PLY vertex property type {property_type} for {property_name}")
        struct_codes.append(code)

    row_struct = struct.Struct("<" + "".join(struct_codes))
    points: list[Point3] = []
    with path.open("rb") as handle:
        for row_index in sampled_indices(header.vertex_count, max_points):
            handle.seek(header.data_offset + row_index * row_struct.size)
            row = handle.read(row_struct.size)
            if len(row) != row_struct.size:
                raise ValueError(f"{path}: expected complete binary vertex row {row_index + 1}")
            values = row_struct.unpack(row)
            point = (float(values[x_index]), float(values[y_index]), float(values[z_index]))
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"{path}: non-finite coordinate in vertex row {row_index + 1}")
            points.append(point)

    if not points:
        raise ValueError(f"{path}: no valid PLY vertex points")
    return points


def read_ply_points(path: Path, max_points: int | None = None) -> list[Point3]:
    path = Path(path)
    header = read_ply_header(path)
    if header.format_name == "ascii":
        return read_ascii_ply_points(path, max_points=max_points)
    if header.format_name == "binary_little_endian":
        return read_binary_little_endian_ply_points(path, max_points=max_points)
    raise ValueError(f"{path}: unsupported PLY format {header.format_name}")


def read_text_points(path: Path) -> list[Point3]:
    path = Path(path)
    points: list[Point3] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace(",", " ").split()
            if len(tokens) < 3:
                raise ValueError(f"{path}: text point row {line_number} has fewer than 3 fields")
            try:
                point = (float(tokens[0]), float(tokens[1]), float(tokens[2]))
            except ValueError:
                if not points:
                    continue
                raise ValueError(f"{path}: invalid numeric value in text point row {line_number}")
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"{path}: non-finite coordinate in text point row {line_number}")
            points.append(point)

    if not points:
        raise ValueError(f"{path}: no valid text point rows")
    return points


def read_points(path: Path, max_points: int | None = None) -> list[Point3]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return read_ply_points(path, max_points=max_points)
    if suffix in TEXT_POINT_EXTENSIONS:
        points = read_text_points(path)
        return [points[index] for index in sampled_indices(len(points), max_points)]
    raise ValueError(f"{path}: unsupported point cloud extension {suffix}")


def squared_distance(left: Point3, right: Point3) -> float:
    return (
        (left[0] - right[0]) ** 2
        + (left[1] - right[1]) ** 2
        + (left[2] - right[2]) ** 2
    )


def brute_force_nearest_neighbor_distances(
    source_points: list[Point3],
    reference_points: list[Point3],
) -> list[float]:
    return [
        math.sqrt(min(squared_distance(source_point, ref_point) for ref_point in reference_points))
        for source_point in source_points
    ]


def brute_force_nearest_neighbor_deltas(
    source_points: list[Point3],
    reference_points: list[Point3],
) -> list[Point3]:
    deltas: list[Point3] = []
    for source_point in source_points:
        nearest = min(reference_points, key=lambda ref_point: squared_distance(source_point, ref_point))
        deltas.append((
            source_point[0] - nearest[0],
            source_point[1] - nearest[1],
            source_point[2] - nearest[2],
        ))
    return deltas


def build_kd_tree(points: list[Point3], depth: int = 0) -> KdNode | None:
    if not points:
        return None
    axis = depth % 3
    ordered = sorted(points, key=lambda point: (point[axis], point[0], point[1], point[2]))
    mid = len(ordered) // 2
    return KdNode(
        point=ordered[mid],
        axis=axis,
        left=build_kd_tree(ordered[:mid], depth + 1),
        right=build_kd_tree(ordered[mid + 1:], depth + 1),
    )


def kd_tree_nearest_squared(node: KdNode | None, query: Point3, best_sq: float = math.inf) -> float:
    if node is None:
        return best_sq

    point_sq = squared_distance(query, node.point)
    if point_sq < best_sq:
        best_sq = point_sq

    axis = node.axis
    delta = query[axis] - node.point[axis]
    near_branch = node.left if delta < 0.0 else node.right
    far_branch = node.right if delta < 0.0 else node.left

    best_sq = kd_tree_nearest_squared(near_branch, query, best_sq)
    if delta * delta <= best_sq:
        best_sq = kd_tree_nearest_squared(far_branch, query, best_sq)
    return best_sq


def kd_tree_nearest_point(
    node: KdNode | None,
    query: Point3,
    best_point: Point3 | None = None,
    best_sq: float = math.inf,
) -> tuple[Point3 | None, float]:
    if node is None:
        return best_point, best_sq

    point_sq = squared_distance(query, node.point)
    if point_sq < best_sq:
        best_sq = point_sq
        best_point = node.point

    axis = node.axis
    delta = query[axis] - node.point[axis]
    near_branch = node.left if delta < 0.0 else node.right
    far_branch = node.right if delta < 0.0 else node.left

    best_point, best_sq = kd_tree_nearest_point(near_branch, query, best_point, best_sq)
    if delta * delta <= best_sq:
        best_point, best_sq = kd_tree_nearest_point(far_branch, query, best_point, best_sq)
    return best_point, best_sq


def kd_tree_nearest_neighbor_distances(
    source_points: list[Point3],
    reference_points: list[Point3],
) -> list[float]:
    tree = build_kd_tree(reference_points)
    return [math.sqrt(kd_tree_nearest_squared(tree, source_point)) for source_point in source_points]


def kd_tree_nearest_neighbor_deltas(
    source_points: list[Point3],
    reference_points: list[Point3],
) -> list[Point3]:
    tree = build_kd_tree(reference_points)
    deltas: list[Point3] = []
    for source_point in source_points:
        nearest, _ = kd_tree_nearest_point(tree, source_point)
        if nearest is None:
            raise ValueError("reference point cloud has no points")
        deltas.append((
            source_point[0] - nearest[0],
            source_point[1] - nearest[1],
            source_point[2] - nearest[2],
        ))
    return deltas


def resolve_nearest_neighbor_method(method: str, reference_point_count: int) -> str:
    if method not in NEAREST_NEIGHBOR_METHODS:
        raise ValueError(f"unsupported nearest-neighbor method: {method}")
    if method == "auto":
        return "kd-tree" if reference_point_count >= AUTO_KD_TREE_MIN_REFERENCE_POINTS else "brute"
    return method


def nearest_neighbor_deltas(
    source_points: list[Point3],
    reference_points: list[Point3],
    method: str = "auto",
) -> list[Point3]:
    if not source_points:
        raise ValueError("source point cloud has no points")
    if not reference_points:
        raise ValueError("reference point cloud has no points")

    resolved_method = resolve_nearest_neighbor_method(method, len(reference_points))
    if resolved_method == "kd-tree":
        return kd_tree_nearest_neighbor_deltas(source_points, reference_points)
    return brute_force_nearest_neighbor_deltas(source_points, reference_points)


def nearest_neighbor_distances(
    source_points: list[Point3],
    reference_points: list[Point3],
    method: str = "auto",
) -> list[float]:
    if not source_points:
        raise ValueError("source point cloud has no points")
    if not reference_points:
        raise ValueError("reference point cloud has no points")

    resolved_method = resolve_nearest_neighbor_method(method, len(reference_points))
    if resolved_method == "kd-tree":
        return kd_tree_nearest_neighbor_distances(source_points, reference_points)
    return brute_force_nearest_neighbor_distances(source_points, reference_points)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) * 0.5


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile for empty values")
    ordered = sorted(values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    index = max(0, min(len(ordered) - 1, index))
    return ordered[index]


def distance_metrics(distances: list[float]) -> dict[str, float]:
    if not distances:
        raise ValueError("cannot compute metrics for empty distances")
    mean = sum(distances) / len(distances)
    rmse = math.sqrt(sum(value * value for value in distances) / len(distances))
    return {
        "mean": mean,
        "rmse": rmse,
        "median": median(distances),
        "p95": nearest_rank_percentile(distances, 95.0),
        "max": max(distances),
    }


def vertical_error_metrics(deltas: list[Point3]) -> dict[str, float]:
    if not deltas:
        raise ValueError("cannot compute vertical error metrics for empty deltas")
    signed = [delta[2] for delta in deltas]
    absolute = [abs(value) for value in signed]
    mean_signed = sum(signed) / len(signed)
    mean_abs = sum(absolute) / len(absolute)
    rmse = math.sqrt(sum(value * value for value in signed) / len(signed))
    return {
        "mean_signed": mean_signed,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "p50_abs": median(absolute),
        "p95_abs": nearest_rank_percentile(absolute, 95.0),
        "max_abs": max(absolute),
    }


def local_roughness_metrics(
    points: list[Point3],
    grid_cells: int = 120,
    min_cell_points: int = 5,
) -> dict[str, Any]:
    if not points:
        raise ValueError("cannot compute local roughness for empty points")
    if grid_cells <= 0:
        raise ValueError("roughness grid cells must be positive")
    if min_cell_points <= 0:
        raise ValueError("roughness minimum cell points must be positive")

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span_x = max(max_x - min_x, sys.float_info.epsilon)
    span_y = max(max_y - min_y, sys.float_info.epsilon)

    cells: dict[tuple[int, int], list[float]] = {}
    for x, y, z in points:
        cell_x = int((x - min_x) / span_x * grid_cells)
        cell_y = int((y - min_y) / span_y * grid_cells)
        cell_x = max(0, min(grid_cells - 1, cell_x))
        cell_y = max(0, min(grid_cells - 1, cell_y))
        cells.setdefault((cell_x, cell_y), []).append(z)

    z_ranges: list[float] = []
    z_stds: list[float] = []
    cell_counts: list[int] = []
    for values in cells.values():
        if len(values) < min_cell_points:
            continue
        cell_counts.append(len(values))
        z_min = min(values)
        z_max = max(values)
        z_ranges.append(z_max - z_min)
        mean_z = sum(values) / len(values)
        z_stds.append(math.sqrt(sum((value - mean_z) ** 2 for value in values) / len(values)))

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return nearest_rank_percentile(values, percentile) if values else None

    return {
        "grid_cells": grid_cells,
        "valid_cells": len(z_ranges),
        "min_count": min_cell_points,
        "cell_count_p50": median([float(count) for count in cell_counts]) if cell_counts else None,
        "z_range_in_cell": {
            "median": median(z_ranges) if z_ranges else None,
            "p84": percentile_or_none(z_ranges, 84.0),
            "p95": percentile_or_none(z_ranges, 95.0),
            "max": max(z_ranges) if z_ranges else None,
        },
        "z_std_in_cell": {
            "median": median(z_stds) if z_stds else None,
            "p84": percentile_or_none(z_stds, 84.0),
            "p95": percentile_or_none(z_stds, 95.0),
            "max": max(z_stds) if z_stds else None,
        },
    }


def reference_coverage_metrics(
    reference_points: list[Point3],
    source_points: list[Point3],
    coverage_radius_m: float | None,
    nearest_neighbor_method: str = "auto",
) -> dict[str, Any]:
    resolved_method = resolve_nearest_neighbor_method(nearest_neighbor_method, len(source_points))
    if coverage_radius_m is None:
        return {
            "radius_m": None,
            "covered_points": None,
            "total_points": len(reference_points),
            "covered_percent": None,
            "nearest_neighbor_method": resolved_method,
        }
    if coverage_radius_m < 0.0 or not math.isfinite(coverage_radius_m):
        raise ValueError("coverage radius must be a finite non-negative value")

    distances = nearest_neighbor_distances(reference_points, source_points, method=resolved_method)
    covered_points = sum(1 for distance in distances if distance <= coverage_radius_m)
    covered_percent = covered_points * 100.0 / len(reference_points)
    return {
        "radius_m": coverage_radius_m,
        "covered_points": covered_points,
        "total_points": len(reference_points),
        "covered_percent": covered_percent,
        "nearest_neighbor_method": resolved_method,
    }


def build_quality_gate(
    metrics: dict[str, float],
    vertical_metrics: dict[str, float],
    reference_coverage: dict[str, Any],
    source_local_roughness: dict[str, Any],
    max_rmse_m: float | None,
    max_p95_m: float | None,
    max_vertical_rmse_m: float | None,
    max_vertical_p95_m: float | None,
    min_reference_coverage_percent: float | None,
    max_local_z_range_p95_m: float | None,
) -> dict[str, Any]:
    failure_codes: list[str] = []
    failure_reasons: list[str] = []
    if max_rmse_m is not None and metrics["rmse"] > max_rmse_m:
        failure_codes.append("rmse_above_threshold")
        failure_reasons.append("Nearest-neighbor RMSE is above the accepted threshold.")
    if max_p95_m is not None and metrics["p95"] > max_p95_m:
        failure_codes.append("p95_above_threshold")
        failure_reasons.append("Nearest-neighbor p95 distance is above the accepted threshold.")
    if max_vertical_rmse_m is not None and vertical_metrics["rmse"] > max_vertical_rmse_m:
        failure_codes.append("vertical_rmse_above_threshold")
        failure_reasons.append("Signed vertical RMSE is above the accepted threshold.")
    if max_vertical_p95_m is not None and vertical_metrics["p95_abs"] > max_vertical_p95_m:
        failure_codes.append("vertical_p95_above_threshold")
        failure_reasons.append("Absolute vertical p95 error is above the accepted threshold.")
    covered_percent = reference_coverage["covered_percent"]
    if (
        min_reference_coverage_percent is not None
        and covered_percent is not None
        and covered_percent < min_reference_coverage_percent
    ):
        failure_codes.append("reference_coverage_below_threshold")
        failure_reasons.append("Reference LiDAR coverage is below the accepted threshold.")
    if min_reference_coverage_percent is not None and covered_percent is None:
        failure_codes.append("missing_reference_coverage_radius")
        failure_reasons.append("A reference coverage quality gate needs a coverage radius.")
    local_z_range_p95 = source_local_roughness["z_range_in_cell"]["p95"]
    if max_local_z_range_p95_m is not None and local_z_range_p95 is None:
        failure_codes.append("missing_local_roughness")
        failure_reasons.append("A local roughness quality gate needs enough points in local grid cells.")
    if (
        max_local_z_range_p95_m is not None
        and local_z_range_p95 is not None
        and local_z_range_p95 > max_local_z_range_p95_m
    ):
        failure_codes.append("local_z_range_p95_above_threshold")
        failure_reasons.append("Local terrain thickness p95 is above the accepted threshold.")
    return {
        "passed": not failure_codes,
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "max_rmse_m": max_rmse_m,
            "max_p95_m": max_p95_m,
            "max_vertical_rmse_m": max_vertical_rmse_m,
            "max_vertical_p95_m": max_vertical_p95_m,
            "coverage_radius_m": reference_coverage["radius_m"],
            "min_reference_coverage_percent": min_reference_coverage_percent,
            "max_local_z_range_p95_m": max_local_z_range_p95_m,
        },
    }


def compare_point_clouds(
    source_path: Path,
    reference_path: Path,
    max_rmse_m: float | None = None,
    max_p95_m: float | None = None,
    max_vertical_rmse_m: float | None = None,
    max_vertical_p95_m: float | None = None,
    coverage_radius_m: float | None = None,
    min_reference_coverage_percent: float | None = None,
    max_local_z_range_p95_m: float | None = None,
    roughness_grid_cells: int = 120,
    roughness_min_cell_points: int = 5,
    nearest_neighbor_method: str = "auto",
    max_source_points: int | None = None,
    max_reference_points: int | None = None,
) -> dict[str, Any]:
    source_points = read_points(Path(source_path), max_points=max_source_points)
    reference_points = read_points(Path(reference_path), max_points=max_reference_points)
    resolved_method = resolve_nearest_neighbor_method(nearest_neighbor_method, len(reference_points))
    deltas = nearest_neighbor_deltas(source_points, reference_points, method=resolved_method)
    distances = [math.sqrt(dx * dx + dy * dy + dz * dz) for dx, dy, dz in deltas]
    metrics = distance_metrics(distances)
    vertical_metrics = vertical_error_metrics(deltas)
    reference_coverage = reference_coverage_metrics(
        reference_points,
        source_points,
        coverage_radius_m,
        nearest_neighbor_method=nearest_neighbor_method,
    )
    source_local_roughness = local_roughness_metrics(
        source_points,
        grid_cells=roughness_grid_cells,
        min_cell_points=roughness_min_cell_points,
    )
    reference_local_roughness = local_roughness_metrics(
        reference_points,
        grid_cells=roughness_grid_cells,
        min_cell_points=roughness_min_cell_points,
    )
    return {
        "source": str(Path(source_path)),
        "reference": str(Path(reference_path)),
        "nearest_neighbor_method": resolved_method,
        "source_points": len(source_points),
        "reference_points": len(reference_points),
        "sampling": {
            "max_source_points": max_source_points,
            "max_reference_points": max_reference_points,
        },
        "distance_m": metrics,
        "vertical_error_m": vertical_metrics,
        "reference_coverage": reference_coverage,
        "source_local_roughness": source_local_roughness,
        "reference_local_roughness": reference_local_roughness,
        "quality_gate": build_quality_gate(
            metrics,
            vertical_metrics,
            reference_coverage,
            source_local_roughness,
            max_rmse_m,
            max_p95_m,
            max_vertical_rmse_m,
            max_vertical_p95_m,
            min_reference_coverage_percent,
            max_local_z_range_p95_m,
        ),
    }


def percent_reduction(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0.0:
        return None
    return max(0.0, (before - after) * 100.0 / before)


def build_improvement_quality_gate(
    candidate_gate: dict[str, Any],
    improvement: dict[str, Any],
    min_local_z_range_p95_improvement_percent: float | None,
) -> dict[str, Any]:
    failure_codes = list(candidate_gate["failure_codes"])
    failure_reasons = list(candidate_gate["failure_reasons"])
    reduction = improvement["local_z_range_p95_reduction_percent"]
    if min_local_z_range_p95_improvement_percent is not None and reduction is None:
        failure_codes.append("missing_local_z_range_p95_improvement")
        failure_reasons.append("Local terrain thickness improvement could not be measured.")
    if (
        min_local_z_range_p95_improvement_percent is not None
        and reduction is not None
        and reduction < min_local_z_range_p95_improvement_percent
    ):
        failure_codes.append("local_z_range_p95_improvement_below_threshold")
        failure_reasons.append("Refined cloud did not reduce local terrain thickness enough.")

    thresholds = dict(candidate_gate["thresholds"])
    thresholds["min_local_z_range_p95_improvement_percent"] = min_local_z_range_p95_improvement_percent
    return {
        "passed": not failure_codes,
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "failure_reasons": failure_reasons,
        "thresholds": thresholds,
    }


def compare_point_cloud_improvement(
    baseline_source_path: Path,
    candidate_source_path: Path,
    reference_path: Path,
    max_rmse_m: float | None = None,
    max_p95_m: float | None = None,
    max_vertical_rmse_m: float | None = None,
    max_vertical_p95_m: float | None = None,
    coverage_radius_m: float | None = None,
    min_reference_coverage_percent: float | None = None,
    max_local_z_range_p95_m: float | None = None,
    min_local_z_range_p95_improvement_percent: float | None = None,
    roughness_grid_cells: int = 120,
    roughness_min_cell_points: int = 5,
    nearest_neighbor_method: str = "auto",
    max_source_points: int | None = None,
    max_reference_points: int | None = None,
) -> dict[str, Any]:
    baseline = compare_point_clouds(
        baseline_source_path,
        reference_path,
        max_rmse_m=max_rmse_m,
        max_p95_m=max_p95_m,
        max_vertical_rmse_m=max_vertical_rmse_m,
        max_vertical_p95_m=max_vertical_p95_m,
        coverage_radius_m=coverage_radius_m,
        min_reference_coverage_percent=min_reference_coverage_percent,
        max_local_z_range_p95_m=max_local_z_range_p95_m,
        roughness_grid_cells=roughness_grid_cells,
        roughness_min_cell_points=roughness_min_cell_points,
        nearest_neighbor_method=nearest_neighbor_method,
        max_source_points=max_source_points,
        max_reference_points=max_reference_points,
    )
    candidate = compare_point_clouds(
        candidate_source_path,
        reference_path,
        max_rmse_m=max_rmse_m,
        max_p95_m=max_p95_m,
        max_vertical_rmse_m=max_vertical_rmse_m,
        max_vertical_p95_m=max_vertical_p95_m,
        coverage_radius_m=coverage_radius_m,
        min_reference_coverage_percent=min_reference_coverage_percent,
        max_local_z_range_p95_m=max_local_z_range_p95_m,
        roughness_grid_cells=roughness_grid_cells,
        roughness_min_cell_points=roughness_min_cell_points,
        nearest_neighbor_method=nearest_neighbor_method,
        max_source_points=max_source_points,
        max_reference_points=max_reference_points,
    )

    baseline_z_p95 = baseline["source_local_roughness"]["z_range_in_cell"]["p95"]
    candidate_z_p95 = candidate["source_local_roughness"]["z_range_in_cell"]["p95"]
    improvement = {
        "local_z_range_p95_before_m": baseline_z_p95,
        "local_z_range_p95_after_m": candidate_z_p95,
        "local_z_range_p95_reduction_percent": percent_reduction(baseline_z_p95, candidate_z_p95),
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "reference": str(Path(reference_path)),
        "improvement": improvement,
        "quality_gate": build_improvement_quality_gate(
            candidate["quality_gate"],
            improvement,
            min_local_z_range_p95_improvement_percent,
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an image-derived ASCII PLY cloud against a LiDAR/reference ASCII PLY cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True, help="source/reconstruction ASCII PLY")
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="optional raw/baseline source cloud for raw-vs-refined improvement reports",
    )
    parser.add_argument("--reference", type=Path, required=True, help="reference/LiDAR ASCII PLY")
    parser.add_argument("--output-json", type=Path, required=True, help="comparison JSON output")
    parser.add_argument("--max-rmse-m", type=float, help="optional RMSE acceptance threshold in meters")
    parser.add_argument("--max-p95-m", type=float, help="optional p95 acceptance threshold in meters")
    parser.add_argument(
        "--max-vertical-rmse-m",
        type=float,
        help="optional signed vertical RMSE acceptance threshold in meters",
    )
    parser.add_argument(
        "--max-vertical-p95-m",
        type=float,
        help="optional absolute vertical p95 acceptance threshold in meters",
    )
    parser.add_argument(
        "--nearest-neighbor-method",
        choices=sorted(NEAREST_NEIGHBOR_METHODS),
        default="auto",
        help="nearest-neighbor search method",
    )
    parser.add_argument(
        "--max-source-points",
        type=int,
        help="optional deterministic sample size for the source/reconstruction cloud",
    )
    parser.add_argument(
        "--max-reference-points",
        type=int,
        help="optional deterministic sample size for the reference cloud",
    )
    parser.add_argument(
        "--coverage-radius-m",
        type=float,
        help="optional radius for measuring how many reference points are covered by the source cloud",
    )
    parser.add_argument(
        "--min-reference-coverage-percent",
        type=float,
        help="optional minimum percent of reference points covered within --coverage-radius-m",
    )
    parser.add_argument(
        "--roughness-grid-cells",
        type=int,
        default=120,
        help="number of XY cells per axis for local terrain thickness statistics",
    )
    parser.add_argument(
        "--roughness-min-cell-points",
        type=int,
        default=5,
        help="minimum points in a cell before local terrain thickness is measured",
    )
    parser.add_argument(
        "--max-local-z-range-p95-m",
        type=float,
        help="optional p95 local terrain thickness acceptance threshold in meters",
    )
    parser.add_argument(
        "--min-local-z-range-p95-improvement-percent",
        type=float,
        help="optional minimum local terrain thickness p95 reduction when --baseline-source is set",
    )
    parser.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help="return a non-zero exit code when the point-cloud quality gate fails",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_reference_coverage_percent is not None and args.coverage_radius_m is None:
        print(
            "failed to compare point clouds: coverage radius is required for reference coverage gate",
            file=sys.stderr,
        )
        return 1
    try:
        if args.baseline_source:
            comparison = compare_point_cloud_improvement(
                args.baseline_source,
                args.source,
                args.reference,
                max_rmse_m=args.max_rmse_m,
                max_p95_m=args.max_p95_m,
                max_vertical_rmse_m=args.max_vertical_rmse_m,
                max_vertical_p95_m=args.max_vertical_p95_m,
                coverage_radius_m=args.coverage_radius_m,
                min_reference_coverage_percent=args.min_reference_coverage_percent,
                max_local_z_range_p95_m=args.max_local_z_range_p95_m,
                min_local_z_range_p95_improvement_percent=args
                    .min_local_z_range_p95_improvement_percent,
                roughness_grid_cells=args.roughness_grid_cells,
                roughness_min_cell_points=args.roughness_min_cell_points,
                nearest_neighbor_method=args.nearest_neighbor_method,
                max_source_points=args.max_source_points,
                max_reference_points=args.max_reference_points,
            )
        else:
            comparison = compare_point_clouds(
                args.source,
                args.reference,
                max_rmse_m=args.max_rmse_m,
                max_p95_m=args.max_p95_m,
                max_vertical_rmse_m=args.max_vertical_rmse_m,
                max_vertical_p95_m=args.max_vertical_p95_m,
                coverage_radius_m=args.coverage_radius_m,
                min_reference_coverage_percent=args.min_reference_coverage_percent,
                max_local_z_range_p95_m=args.max_local_z_range_p95_m,
                roughness_grid_cells=args.roughness_grid_cells,
                roughness_min_cell_points=args.roughness_min_cell_points,
                nearest_neighbor_method=args.nearest_neighbor_method,
                max_source_points=args.max_source_points,
                max_reference_points=args.max_reference_points,
            )
    except (OSError, ValueError) as exc:
        print(f"failed to compare point clouds: {exc}", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {args.output_json}")
    if args.fail_on_quality_gate and not comparison["quality_gate"]["passed"]:
        print("quality gate failed: point cloud should not be accepted automatically", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
