#!/usr/bin/env python3
"""Compare an image-derived point cloud against a LiDAR/reference point cloud."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Point3 = tuple[float, float, float]
TEXT_POINT_EXTENSIONS = {".csv", ".txt", ".xyz"}
NEAREST_NEIGHBOR_METHODS = {"auto", "brute", "kd-tree"}
AUTO_KD_TREE_MIN_REFERENCE_POINTS = 64


@dataclass(frozen=True)
class KdNode:
    point: Point3
    axis: int
    left: "KdNode | None"
    right: "KdNode | None"


def read_ascii_ply_points(path: Path) -> list[Point3]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path} is not a PLY file")

    vertex_count: int | None = None
    vertex_properties: list[str] = []
    in_vertex_element = False
    header_end = -1
    for index, raw_line in enumerate(lines[1:], start=1):
        line = raw_line.strip()
        if line == "end_header":
            header_end = index
            break
        if line.startswith("format "):
            if line != "format ascii 1.0":
                raise ValueError(f"{path}: only ASCII PLY format ascii 1.0 is supported")
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
            vertex_properties.append(parts[-1])

    if header_end < 0:
        raise ValueError(f"{path}: missing PLY end_header")
    if vertex_count is None:
        raise ValueError(f"{path}: missing PLY vertex element")
    for coordinate in ("x", "y", "z"):
        if coordinate not in vertex_properties:
            raise ValueError(f"{path}: missing PLY vertex property {coordinate}")

    x_index = vertex_properties.index("x")
    y_index = vertex_properties.index("y")
    z_index = vertex_properties.index("z")
    points: list[Point3] = []
    first_vertex_line = header_end + 1
    if len(lines) < first_vertex_line + vertex_count:
        raise ValueError(f"{path}: expected {vertex_count} vertex rows")

    for row_index in range(vertex_count):
        line = lines[first_vertex_line + row_index].strip()
        parts = line.split()
        if len(parts) < len(vertex_properties):
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


def read_points(path: Path) -> list[Point3]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return read_ascii_ply_points(path)
    if suffix in TEXT_POINT_EXTENSIONS:
        return read_text_points(path)
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


def kd_tree_nearest_neighbor_distances(
    source_points: list[Point3],
    reference_points: list[Point3],
) -> list[float]:
    tree = build_kd_tree(reference_points)
    return [math.sqrt(kd_tree_nearest_squared(tree, source_point)) for source_point in source_points]


def resolve_nearest_neighbor_method(method: str, reference_point_count: int) -> str:
    if method not in NEAREST_NEIGHBOR_METHODS:
        raise ValueError(f"unsupported nearest-neighbor method: {method}")
    if method == "auto":
        return "kd-tree" if reference_point_count >= AUTO_KD_TREE_MIN_REFERENCE_POINTS else "brute"
    return method


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
    reference_coverage: dict[str, Any],
    max_rmse_m: float | None,
    max_p95_m: float | None,
    min_reference_coverage_percent: float | None,
) -> dict[str, Any]:
    failure_codes: list[str] = []
    failure_reasons: list[str] = []
    if max_rmse_m is not None and metrics["rmse"] > max_rmse_m:
        failure_codes.append("rmse_above_threshold")
        failure_reasons.append("Nearest-neighbor RMSE is above the accepted threshold.")
    if max_p95_m is not None and metrics["p95"] > max_p95_m:
        failure_codes.append("p95_above_threshold")
        failure_reasons.append("Nearest-neighbor p95 distance is above the accepted threshold.")
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
    return {
        "passed": not failure_codes,
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "max_rmse_m": max_rmse_m,
            "max_p95_m": max_p95_m,
            "coverage_radius_m": reference_coverage["radius_m"],
            "min_reference_coverage_percent": min_reference_coverage_percent,
        },
    }


def compare_point_clouds(
    source_path: Path,
    reference_path: Path,
    max_rmse_m: float | None = None,
    max_p95_m: float | None = None,
    coverage_radius_m: float | None = None,
    min_reference_coverage_percent: float | None = None,
    nearest_neighbor_method: str = "auto",
) -> dict[str, Any]:
    source_points = read_points(Path(source_path))
    reference_points = read_points(Path(reference_path))
    resolved_method = resolve_nearest_neighbor_method(nearest_neighbor_method, len(reference_points))
    distances = nearest_neighbor_distances(source_points, reference_points, method=resolved_method)
    metrics = distance_metrics(distances)
    reference_coverage = reference_coverage_metrics(
        reference_points,
        source_points,
        coverage_radius_m,
        nearest_neighbor_method=nearest_neighbor_method,
    )
    return {
        "source": str(Path(source_path)),
        "reference": str(Path(reference_path)),
        "nearest_neighbor_method": resolved_method,
        "source_points": len(source_points),
        "reference_points": len(reference_points),
        "distance_m": metrics,
        "reference_coverage": reference_coverage,
        "quality_gate": build_quality_gate(
            metrics,
            reference_coverage,
            max_rmse_m,
            max_p95_m,
            min_reference_coverage_percent,
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an image-derived ASCII PLY cloud against a LiDAR/reference ASCII PLY cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True, help="source/reconstruction ASCII PLY")
    parser.add_argument("--reference", type=Path, required=True, help="reference/LiDAR ASCII PLY")
    parser.add_argument("--output-json", type=Path, required=True, help="comparison JSON output")
    parser.add_argument("--max-rmse-m", type=float, help="optional RMSE acceptance threshold in meters")
    parser.add_argument("--max-p95-m", type=float, help="optional p95 acceptance threshold in meters")
    parser.add_argument(
        "--nearest-neighbor-method",
        choices=sorted(NEAREST_NEIGHBOR_METHODS),
        default="auto",
        help="nearest-neighbor search method",
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
        comparison = compare_point_clouds(
            args.source,
            args.reference,
            max_rmse_m=args.max_rmse_m,
            max_p95_m=args.max_p95_m,
            coverage_radius_m=args.coverage_radius_m,
            min_reference_coverage_percent=args.min_reference_coverage_percent,
            nearest_neighbor_method=args.nearest_neighbor_method,
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
