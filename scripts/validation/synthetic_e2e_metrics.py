#!/usr/bin/env python3
"""Truth-based metrics and reports for PlaScan synthetic E2E runs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
import trimesh
from scipy.spatial import cKDTree


REPORT_SCHEMA = "plascan.synthetic_e2e_report.v1"

THRESHOLDS: dict[str, dict[str, dict[str, float]]] = {
    "plascan_synthetic_3d_object": {
        "coarse": {
            "sfm_reprojection_px": 1.5,
            "sfm_min_points": 40,
            "dense_rmse_m": 0.45,
            "dense_p95_m": 0.80,
            "dense_min_coverage": 0.20,
            "mesh_chamfer_m": 0.40,
            "mesh_p95_m": 0.75,
            "radial_dem_rmse_m": 0.45,
            "radial_dem_min_coverage": 0.65,
        },
        "medium": {
            "sfm_reprojection_px": 1.2,
            "sfm_min_points": 80,
            "dense_rmse_m": 0.32,
            "dense_p95_m": 0.60,
            "dense_min_coverage": 0.30,
            "mesh_chamfer_m": 0.30,
            "mesh_p95_m": 0.55,
            "radial_dem_rmse_m": 0.30,
            "radial_dem_min_coverage": 0.75,
        },
        "fine": {
            "sfm_reprojection_px": 1.0,
            "sfm_min_points": 120,
            "dense_rmse_m": 0.22,
            "dense_p95_m": 0.42,
            "dense_min_coverage": 0.42,
            "mesh_chamfer_m": 0.22,
            "mesh_p95_m": 0.40,
            "radial_dem_rmse_m": 0.20,
            "radial_dem_min_coverage": 0.85,
        },
    },
    "plascan_synthetic_terrain": {
        "coarse": {
            "sfm_reprojection_px": 1.5,
            "sfm_min_points": 250,
            "dense_rmse_m": 0.40,
            "dense_p95_m": 0.75,
            "dense_min_coverage": 0.35,
            "mesh_chamfer_m": 0.38,
            "mesh_p95_m": 0.70,
            "dem_rmse_m": 0.35,
            "dem_nmad_m": 0.30,
            "dem_min_coverage": 0.70,
        },
        "medium": {
            "sfm_reprojection_px": 1.2,
            "sfm_min_points": 700,
            "dense_rmse_m": 0.28,
            "dense_p95_m": 0.52,
            "dense_min_coverage": 0.50,
            "mesh_chamfer_m": 0.28,
            "mesh_p95_m": 0.50,
            "dem_rmse_m": 0.24,
            "dem_nmad_m": 0.20,
            "dem_min_coverage": 0.82,
        },
        "fine": {
            "sfm_reprojection_px": 1.0,
            "sfm_min_points": 1200,
            "dense_rmse_m": 0.20,
            "dense_p95_m": 0.38,
            "dense_min_coverage": 0.65,
            "mesh_chamfer_m": 0.20,
            "mesh_p95_m": 0.36,
            "dem_rmse_m": 0.16,
            "dem_nmad_m": 0.13,
            "dem_min_coverage": 0.90,
        },
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact(report_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = report_path.parent / path
    return path.resolve()


def nested_value(payload: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for keys in paths:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value not in (None, ""):
            return value
    return None


def geometry(path: Path) -> tuple[np.ndarray, trimesh.Trimesh | None]:
    loaded = trimesh.load(path, process=False)
    meshes: list[trimesh.Trimesh] = []
    if isinstance(loaded, trimesh.Scene):
        meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        vertices = np.concatenate([np.asarray(item.vertices) for item in meshes], axis=0)
        mesh = trimesh.util.concatenate(meshes) if meshes else None
    elif isinstance(loaded, trimesh.Trimesh):
        vertices = np.asarray(loaded.vertices)
        mesh = loaded if len(loaded.faces) else None
    elif isinstance(loaded, trimesh.points.PointCloud):
        vertices = np.asarray(loaded.vertices)
        mesh = None
    else:
        raise ValueError(f"unsupported geometry payload: {type(loaded).__name__}")
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"invalid geometry vertices: {path}")
    return vertices, mesh


def deterministic_subset(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), maximum, replace=False)]


def surface_samples(mesh: trimesh.Trimesh | None, vertices: np.ndarray, count: int, seed: int) -> np.ndarray:
    if mesh is None or not len(mesh.faces):
        return deterministic_subset(vertices, count, seed)
    np.random.seed(seed)
    return np.asarray(trimesh.sample.sample_surface(mesh, count)[0], dtype=np.float64)


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def nmad(errors: np.ndarray) -> float:
    median = float(np.median(errors))
    return float(1.4826 * np.median(np.abs(errors - median)))


def check(
    checks: list[dict[str, Any]],
    check_id: str,
    label: str,
    value: float,
    operator: str,
    threshold: float,
    unit: str = "",
) -> None:
    passed = value <= threshold if operator == "<=" else value >= threshold
    checks.append(
        {
            "id": check_id,
            "label": label,
            "value": float(value),
            "operator": operator,
            "threshold": float(threshold),
            "unit": unit,
            "passed": bool(passed),
        }
    )


def bilinear_grid(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = grid.shape
    valid = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    map_x = np.clip(x, 0.0, width - 1).astype(np.float32).reshape(-1, 1)
    map_y = np.clip(y, 0.0, height - 1).astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        grid.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    return sampled.astype(np.float64), valid


def object_errors(points: np.ndarray, radius_grid: np.ndarray) -> tuple[np.ndarray, float]:
    radii = np.linalg.norm(points, axis=1)
    nonzero = radii > 1.0e-9
    unit = points[nonzero] / radii[nonzero, None]
    longitude = np.arctan2(unit[:, 1], unit[:, 0])
    latitude = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
    map_x = (longitude / (2.0 * math.pi) + 0.5) * radius_grid.shape[1]
    map_x = np.mod(map_x, radius_grid.shape[1])
    map_y = (0.5 - latitude / math.pi) * (radius_grid.shape[0] - 1)
    truth, _ = bilinear_grid(radius_grid, map_x, map_y)
    errors = np.abs(radii[nonzero] - truth)
    bins_y = np.clip((map_y / radius_grid.shape[0] * 36).astype(int), 0, 35)
    bins_x = np.mod((map_x / radius_grid.shape[1] * 72).astype(int), 72)
    coverage = len(np.unique(bins_y * 72 + bins_x)) / float(36 * 72)
    return errors, coverage


def terrain_errors(
    points: np.ndarray,
    dem: np.ndarray,
    terrain_width_m: float,
    terrain_height_m: float,
) -> tuple[np.ndarray, float, float]:
    x_index = (points[:, 0] / terrain_width_m + 0.5) * (dem.shape[1] - 1)
    y_index = (0.5 - points[:, 1] / terrain_height_m) * (dem.shape[0] - 1)
    truth, valid = bilinear_grid(dem, x_index, y_index)
    errors = points[valid, 2] - truth[valid]
    bins_x = np.clip((x_index[valid] / max(dem.shape[1] - 1, 1) * 63).astype(int), 0, 63)
    bins_y = np.clip((y_index[valid] / max(dem.shape[0] - 1, 1) * 63).astype(int), 0, 63)
    coverage = len(np.unique(bins_y * 64 + bins_x)) / float(64 * 64)
    return errors, coverage, float(np.mean(valid))


def chamfer_metrics(
    candidate_path: Path,
    truth_path: Path,
    sample_count: int,
) -> dict[str, Any]:
    candidate_vertices, candidate_mesh = geometry(candidate_path)
    truth_vertices, truth_mesh = geometry(truth_path)
    candidate_points = surface_samples(candidate_mesh, candidate_vertices, sample_count, 20260914)
    truth_points = surface_samples(truth_mesh, truth_vertices, sample_count, 20260915)
    candidate_to_truth = cKDTree(truth_points).query(candidate_points, workers=-1)[0]
    truth_to_candidate = cKDTree(candidate_points).query(truth_points, workers=-1)[0]
    return {
        "candidate_to_truth": distribution(candidate_to_truth),
        "truth_to_candidate": distribution(truth_to_candidate),
        "chamfer_l1_mean": float(
            0.5 * (np.mean(candidate_to_truth) + np.mean(truth_to_candidate))
        ),
        "symmetric_p95": float(
            0.5 * (np.quantile(candidate_to_truth, 0.95) + np.quantile(truth_to_candidate, 0.95))
        ),
        "sample_count_per_direction": int(sample_count),
    }


def terrain_chamfer_metrics(
    candidate_path: Path,
    truth_path: Path,
    sample_count: int,
) -> dict[str, Any]:
    candidate_vertices, candidate_mesh = geometry(candidate_path)
    truth_vertices, truth_mesh = geometry(truth_path)
    candidate_points = surface_samples(
        candidate_mesh,
        candidate_vertices,
        sample_count,
        20260914,
    )
    truth_points_full = surface_samples(
        truth_mesh,
        truth_vertices,
        max(sample_count * 3, sample_count),
        20260915,
    )
    bounds_min = np.min(candidate_points[:, :2], axis=0)
    bounds_max = np.max(candidate_points[:, :2], axis=0)
    in_candidate_domain = np.all(
        (truth_points_full[:, :2] >= bounds_min)
        & (truth_points_full[:, :2] <= bounds_max),
        axis=1,
    )
    truth_points = deterministic_subset(
        truth_points_full[in_candidate_domain],
        sample_count,
        20260916,
    )
    if not len(truth_points):
        raise ValueError("candidate terrain mesh does not overlap the synthetic truth")
    candidate_to_truth = cKDTree(truth_points_full).query(
        candidate_points,
        workers=-1,
    )[0]
    truth_to_candidate = cKDTree(candidate_points).query(
        truth_points,
        workers=-1,
    )[0]
    return {
        "candidate_to_truth": distribution(candidate_to_truth),
        "truth_to_candidate": distribution(truth_to_candidate),
        "chamfer_l1_mean": float(
            0.5 * (np.mean(candidate_to_truth) + np.mean(truth_to_candidate))
        ),
        "symmetric_p95": float(
            0.5
            * (
                np.quantile(candidate_to_truth, 0.95)
                + np.quantile(truth_to_candidate, 0.95)
            )
        ),
        "candidate_xy_bounds_m": [
            float(bounds_min[0]),
            float(bounds_min[1]),
            float(bounds_max[0]),
            float(bounds_max[1]),
        ],
        "truth_domain_sample_fraction": float(np.mean(in_candidate_domain)),
        "candidate_sample_count": int(len(candidate_points)),
        "truth_sample_count": int(len(truth_points)),
    }


def raster_band(
    path: Path,
    preferred_band: int = 1,
    validity_band: int | None = None,
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    with rasterio.open(path) as source:
        band = preferred_band if preferred_band <= source.count else 1
        data = source.read(band).astype(np.float64)
        mask = source.read_masks(band) > 0
        if validity_band is not None and validity_band <= source.count:
            mask &= source.read(validity_band) > 0
        if source.nodata is not None and np.isfinite(source.nodata):
            mask &= data != source.nodata
        mask &= np.isfinite(data)
        return data, mask, source.transform


def compare_radial_raster(candidate_path: Path, truth: np.ndarray) -> dict[str, Any]:
    candidate, valid, _ = raster_band(candidate_path)
    truth_resized = cv2.resize(
        truth.astype(np.float32),
        (candidate.shape[1], candidate.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float64)
    errors = candidate[valid] - truth_resized[valid]
    return {
        "coverage": float(np.mean(valid)),
        "error": distribution(np.abs(errors)),
        "bias_m": float(np.mean(errors)),
        "nmad_m": nmad(errors),
        "shape": list(candidate.shape),
    }


def compare_planar_dem(
    candidate_path: Path,
    truth: np.ndarray,
    terrain_width_m: float,
    terrain_height_m: float,
) -> dict[str, Any]:
    # PlaScan terrain DEMs use XYZA bands; third-party/reference DEMs are
    # commonly single-band. Read Z plus validity when present, otherwise band 1.
    candidate, valid, transform = raster_band(
        candidate_path,
        preferred_band=3,
        validity_band=4,
    )
    rows, columns = np.nonzero(valid)
    x = transform.c + (columns + 0.5) * transform.a + (rows + 0.5) * transform.b
    y = transform.f + (columns + 0.5) * transform.d + (rows + 0.5) * transform.e
    x_index = (x / terrain_width_m + 0.5) * (truth.shape[1] - 1)
    y_index = (0.5 - y / terrain_height_m) * (truth.shape[0] - 1)
    truth_values, in_bounds = bilinear_grid(truth, x_index, y_index)
    values = candidate[rows, columns]
    errors = values[in_bounds] - truth_values[in_bounds]
    if not len(errors):
        raise ValueError("candidate DEM does not overlap the synthetic truth extent")
    truth_domain_fraction = float(np.mean(in_bounds))
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    truth_area_coverage = min(
        1.0,
        float(np.count_nonzero(in_bounds) * pixel_area)
        / (terrain_width_m * terrain_height_m),
    )
    return {
        "coverage": truth_area_coverage,
        "valid_pixel_fraction": float(np.mean(valid)),
        "overlap_fraction_of_valid": truth_domain_fraction,
        "truth_area_coverage": truth_area_coverage,
        "error": distribution(np.abs(errors)),
        "bias_m": float(np.mean(errors)),
        "nmad_m": nmad(errors),
        "shape": list(candidate.shape),
    }


def evaluate_sfm(
    manifest: dict[str, Any],
    pipeline: dict[str, Any],
    thresholds: dict[str, float],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    sfm = dict(pipeline.get("sfm", {}))
    view_count = int(manifest["summary"]["view_count"])
    registered = int(sfm.get("registered_images", 0))
    registration = registered / max(view_count, 1)
    points = int(sfm.get("points", 0))
    reprojection = float(sfm.get("mean_reprojection_error", math.inf))
    check(checks, "sfm_registration", "SfM 影像注册率", registration, ">=", 1.0)
    check(checks, "sfm_points", "SfM 稀疏点数", points, ">=", thresholds["sfm_min_points"], "points")
    check(
        checks,
        "sfm_reprojection",
        "SfM 平均重投影误差",
        reprojection,
        "<=",
        thresholds["sfm_reprojection_px"],
        "px",
    )
    return {
        "registered_images": registered,
        "expected_images": view_count,
        "registration_fraction": registration,
        "points": points,
        "mean_reprojection_error_px": reprojection,
    }


def pipeline_artifacts(report_path: Path, pipeline: dict[str, Any]) -> dict[str, Path | None]:
    return {
        "sparse_cloud": resolve_artifact(report_path, nested_value(pipeline, ("sfm", "sparse_cloud"))),
        "dense_cloud": resolve_artifact(
            report_path,
            nested_value(pipeline, ("dense", "refined_point_cloud"), ("dense", "point_cloud")),
        ),
        "mesh": resolve_artifact(
            report_path,
            nested_value(
                pipeline,
                ("model", "final_model_path"),
                ("model", "model_ply"),
                ("model", "mesh_ply"),
            ),
        ),
        "dem": resolve_artifact(report_path, nested_value(pipeline, ("terrain", "dem", "dem_tif"))),
        "dom": resolve_artifact(
            report_path,
            nested_value(
                pipeline,
                ("terrain", "dom", "output_path"),
                ("terrain", "dom", "dom_tif"),
                ("terrain", "dom", "dom_png"),
            ),
        ),
    }


def require_artifact(
    artifacts: dict[str, Path | None],
    key: str,
    checks: list[dict[str, Any]],
) -> Path | None:
    path = artifacts.get(key)
    exists = path is not None and path.is_file()
    check(checks, f"artifact_{key}", f"产物存在：{key}", float(exists), ">=", 1.0)
    return path if exists else None


def evaluate_geometry(
    kind: str,
    manifest: dict[str, Any],
    dataset_dir: Path,
    artifacts: dict[str, Path | None],
    thresholds: dict[str, float],
    checks: list[dict[str, Any]],
    sample_count: int,
) -> dict[str, Any]:
    dense_path = require_artifact(artifacts, "dense_cloud", checks)
    mesh_path = require_artifact(artifacts, "mesh", checks)
    result: dict[str, Any] = {}
    if kind == "plascan_synthetic_3d_object":
        truth_grid = np.load(dataset_dir / "ground_truth" / "radial_radius.npy", allow_pickle=False)
        truth_mesh = dataset_dir / "ground_truth" / "object.ply"
        if dense_path:
            dense_points = deterministic_subset(geometry(dense_path)[0], sample_count, 11)
            errors, coverage = object_errors(dense_points, truth_grid)
            result["dense"] = {
                "points_evaluated": len(dense_points),
                "radial_absolute_error": distribution(errors),
                "spherical_bin_coverage": coverage,
            }
            check(checks, "dense_rmse", "稠密点云径向 RMSE", result["dense"]["radial_absolute_error"]["rmse"], "<=", thresholds["dense_rmse_m"], "m")
            check(checks, "dense_p95", "稠密点云径向 P95", result["dense"]["radial_absolute_error"]["p95"], "<=", thresholds["dense_p95_m"], "m")
            check(checks, "dense_coverage", "稠密点云球面覆盖率", coverage, ">=", thresholds["dense_min_coverage"])
    else:
        truth_grid = np.load(dataset_dir / "ground_truth" / "dem.npy", allow_pickle=False)
        truth_mesh = dataset_dir / "ground_truth" / "terrain.ply"
        width = float(manifest["terrain"]["width_m"])
        height = float(manifest["terrain"]["height_m"])
        if dense_path:
            dense_points = deterministic_subset(geometry(dense_path)[0], sample_count, 11)
            errors, coverage, in_bounds = terrain_errors(dense_points, truth_grid, width, height)
            absolute = np.abs(errors)
            result["dense"] = {
                "points_evaluated": len(dense_points),
                "height_absolute_error": distribution(absolute),
                "height_bias_m": float(np.mean(errors)),
                "height_nmad_m": nmad(errors),
                "grid_coverage": coverage,
                "in_truth_bounds_fraction": in_bounds,
            }
            check(checks, "dense_rmse", "稠密点云高程 RMSE", result["dense"]["height_absolute_error"]["rmse"], "<=", thresholds["dense_rmse_m"], "m")
            check(checks, "dense_p95", "稠密点云高程 P95", result["dense"]["height_absolute_error"]["p95"], "<=", thresholds["dense_p95_m"], "m")
            check(checks, "dense_coverage", "稠密点云地形覆盖率", coverage, ">=", thresholds["dense_min_coverage"])
    if mesh_path:
        mesh_metrics = (
            chamfer_metrics(mesh_path, truth_mesh, sample_count)
            if kind == "plascan_synthetic_3d_object"
            else terrain_chamfer_metrics(mesh_path, truth_mesh, sample_count)
        )
        result["mesh"] = mesh_metrics
        check(checks, "mesh_chamfer", "网格对称 Chamfer-L1", mesh_metrics["chamfer_l1_mean"], "<=", thresholds["mesh_chamfer_m"], "m")
        check(checks, "mesh_p95", "网格对称 P95", mesh_metrics["symmetric_p95"], "<=", thresholds["mesh_p95_m"], "m")
    return result


def evaluate_products(
    kind: str,
    manifest: dict[str, Any],
    dataset_dir: Path,
    report_path: Path,
    small_body_report_path: Path | None,
    pipeline: dict[str, Any],
    small_body: dict[str, Any] | None,
    thresholds: dict[str, float],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if kind == "plascan_synthetic_3d_object":
        payload = small_body or {}
        radial_path = resolve_artifact(
            small_body_report_path or report_path,
            nested_value(payload, ("elevation_dem_tif",), ("artifacts", "elevation_dem_tif")),
        )
        exists = radial_path is not None and radial_path.is_file()
        check(checks, "artifact_radial_dem", "产物存在：radial DEM", float(exists), ">=", 1.0)
        if exists and radial_path:
            truth = np.load(dataset_dir / "ground_truth" / "radial_elevation.npy", allow_pickle=False)
            metrics = compare_radial_raster(radial_path, truth)
            result["radial_dem"] = metrics
            check(checks, "radial_dem_rmse", "径向 DEM RMSE", metrics["error"]["rmse"], "<=", thresholds["radial_dem_rmse_m"], "m")
            check(checks, "radial_dem_coverage", "径向 DEM 覆盖率", metrics["coverage"], ">=", thresholds["radial_dem_min_coverage"])
    else:
        artifacts = pipeline_artifacts(report_path, pipeline)
        dem_path = require_artifact(artifacts, "dem", checks)
        require_artifact(artifacts, "dom", checks)
        if dem_path:
            truth = np.load(dataset_dir / "ground_truth" / "dem.npy", allow_pickle=False)
            metrics = compare_planar_dem(
                dem_path,
                truth,
                float(manifest["terrain"]["width_m"]),
                float(manifest["terrain"]["height_m"]),
            )
            result["dem"] = metrics
            check(checks, "dem_rmse", "DEM 高程 RMSE", metrics["error"]["rmse"], "<=", thresholds["dem_rmse_m"], "m")
            check(checks, "dem_nmad", "DEM 高程 NMAD", metrics["nmad_m"], "<=", thresholds["dem_nmad_m"], "m")
            check(checks, "dem_coverage", "DEM 有效覆盖率", metrics["coverage"], ">=", thresholds["dem_min_coverage"])
    return result


def evaluate_run(
    dataset_dir: Path,
    pipeline_report_path: Path,
    small_body_report_path: Path | None = None,
    sample_count: int = 20_000,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    pipeline_report_path = pipeline_report_path.resolve()
    manifest = load_json(dataset_dir / "manifest.json")
    pipeline = load_json(pipeline_report_path)
    small_body = load_json(small_body_report_path.resolve()) if small_body_report_path else None
    kind = str(manifest.get("dataset_kind", ""))
    quality = str(manifest.get("quality_level", "medium"))
    if kind not in THRESHOLDS:
        raise ValueError(f"unsupported synthetic dataset kind: {kind}")
    if quality not in THRESHOLDS[kind]:
        raise ValueError(f"unsupported quality level for E2E evaluation: {quality}")
    thresholds = THRESHOLDS[kind][quality]
    checks: list[dict[str, Any]] = []
    pipeline_ok = pipeline.get("status") == "ok"
    check(checks, "pipeline_status", "PlaScan 流程状态", float(pipeline_ok), ">=", 1.0)
    sfm = evaluate_sfm(manifest, pipeline, thresholds, checks)
    artifacts = pipeline_artifacts(pipeline_report_path, pipeline)
    geometry_metrics = evaluate_geometry(
        kind,
        manifest,
        dataset_dir,
        artifacts,
        thresholds,
        checks,
        sample_count,
    )
    product_metrics = evaluate_products(
        kind,
        manifest,
        dataset_dir,
        pipeline_report_path,
        small_body_report_path,
        pipeline,
        small_body,
        thresholds,
        checks,
    )
    passed = all(item["passed"] for item in checks)
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass" if passed else "fail",
        "dataset": {
            "directory": str(dataset_dir),
            "kind": kind,
            "quality_level": quality,
            "seed": manifest.get("seed"),
            "manifest": str(dataset_dir / "manifest.json"),
        },
        "pipeline": {
            "report": str(pipeline_report_path),
            "status": pipeline.get("status"),
            "timings": pipeline.get("timings", {}),
        },
        "thresholds": thresholds,
        "checks": checks,
        "metrics": {"sfm": sfm, "geometry": geometry_metrics, "products": product_metrics},
        "artifacts": {key: str(value) if value else None for key, value in artifacts.items()},
    }


def write_html_report(report: dict[str, Any], path: Path) -> None:
    status_class = "pass" if report["status"] == "pass" else "fail"
    check_rows = "\n".join(
        "<tr><td>{}</td><td>{:.6g} {}</td><td>{} {:.6g} {}</td><td class=\"{}\">{}</td></tr>".format(
            html.escape(item["label"]),
            item["value"],
            html.escape(item["unit"]),
            html.escape(item["operator"]),
            item["threshold"],
            html.escape(item["unit"]),
            "pass" if item["passed"] else "fail",
            "通过" if item["passed"] else "失败",
        )
        for item in report["checks"]
    )
    artifact_rows = "\n".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(value or '未生成')}</td></tr>"
        for key, value in report["artifacts"].items()
    )
    details = html.escape(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>PlaScan 合成数据全流程报告</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#20242a}}
h1,h2{{color:#17324d}} table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}
th,td{{border:1px solid #d6dde5;padding:8px;text-align:left}} th{{background:#eef3f8}}
.pass{{color:#08752b;font-weight:700}} .fail{{color:#b42318;font-weight:700}}
pre{{background:#f5f7f9;padding:14px;overflow:auto}} .summary{{padding:14px;border-left:5px solid #64748b;background:#f8fafc}}
</style></head><body>
<h1>PlaScan 合成数据全流程测试报告</h1>
<div class="summary"><p>结果：<span class="{status_class}">{html.escape(report['status'].upper())}</span></p>
<p>数据类型：{html.escape(report['dataset']['kind'])}；精细度：{html.escape(report['dataset']['quality_level'])}</p>
<p>数据目录：{html.escape(report['dataset']['directory'])}</p></div>
<h2>验收检查</h2><table><thead><tr><th>指标</th><th>实测</th><th>门限</th><th>结论</th></tr></thead><tbody>{check_rows}</tbody></table>
<h2>产物</h2><table><tbody>{artifact_rows}</tbody></table>
<h2>完整指标</h2><pre>{details}</pre>
</body></html>"""
    path.write_text(document, encoding="utf-8", newline="\n")
