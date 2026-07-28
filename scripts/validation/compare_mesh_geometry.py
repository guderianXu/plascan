#!/usr/bin/env python3
"""Compare two registered triangle meshes with topology and surface metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure topology, triangle quality, distance, and normal agreement between meshes."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--proximity-chunk-size",
        type=int,
        default=250,
        help="Maximum query points per closest-surface batch; lower this for highly fragmented meshes.",
    )
    parser.add_argument(
        "--aligned-candidate-output",
        type=Path,
        help="Optional path for exporting the camera-aligned candidate mesh used by the comparison.",
    )
    parser.add_argument(
        "--reference-middlebury-par",
        type=Path,
        help="Optional Middlebury *_par.txt whose camera centers define reference coordinates.",
    )
    parser.add_argument(
        "--candidate-mvs-manifest",
        type=Path,
        help="Optional PlaScan mvs_manifest.json used with --reference-middlebury-par for camera-center Sim(3).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.00025, 0.0005, 0.001],
        help="Absolute surface-distance thresholds in model units.",
    )
    return parser.parse_args()


def read_middlebury_camera_centers(path: Path) -> dict[str, np.ndarray]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if lines and len(lines[0].split()) == 1:
        lines = lines[1:]
    centers: dict[str, np.ndarray] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 22:
            raise ValueError(f"Invalid Middlebury camera record in {path}: {line}")
        values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        rotation = values[9:18].reshape(3, 3)
        translation = values[18:21]
        centers[Path(fields[0]).name] = -rotation.T @ translation
    return centers


def read_middlebury_camera_rotations(path: Path) -> dict[str, np.ndarray]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if lines and len(lines[0].split()) == 1:
        lines = lines[1:]
    rotations: dict[str, np.ndarray] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 22:
            values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
            rotations[Path(fields[0]).name] = values[9:18].reshape(3, 3)
    return rotations


def read_manifest_camera_centers(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    centers: dict[str, np.ndarray] = {}
    for frame in payload.get("frames", []):
        image_name = Path(frame.get("ref_image", "")).name
        camera_center = frame.get("camera_model", {}).get("camera_center")
        if image_name and isinstance(camera_center, list) and len(camera_center) == 3:
            centers[image_name] = np.asarray(camera_center, dtype=np.float64)
    if not centers:
        raise ValueError(f"No camera centers found in PlaScan manifest: {path}")
    return centers


def read_manifest_camera_rotations(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rotations: dict[str, np.ndarray] = {}
    for frame in payload.get("frames", []):
        image_name = Path(frame.get("ref_image", "")).name
        values = frame.get("camera_model", {}).get("rotation_world_to_camera")
        if image_name and isinstance(values, list) and len(values) == 9:
            rotations[image_name] = np.asarray(values, dtype=np.float64).reshape(3, 3)
    return rotations


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate target = scale * rotation * source + translation via Umeyama."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Camera correspondence arrays must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("At least three camera correspondences are required for Sim(3)")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_transposed = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transposed) < 0.0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_transposed
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    transformed = (scale * (rotation @ source.T)).T + translation
    residuals = np.linalg.norm(transformed - target, axis=1)
    diagnostics = {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "camera_rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "camera_max_error": float(np.max(residuals)),
        "camera_count": len(source),
    }
    return transform, diagnostics


def align_candidate_from_cameras(
    candidate: trimesh.Trimesh,
    middlebury_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    reference_centers = read_middlebury_camera_centers(middlebury_path)
    candidate_centers = read_manifest_camera_centers(manifest_path)
    common_names = sorted(set(reference_centers) & set(candidate_centers))
    if len(common_names) < 3:
        raise ValueError(
            f"Only {len(common_names)} matching camera names between {middlebury_path} and {manifest_path}"
        )
    source = np.stack([candidate_centers[name] for name in common_names])
    target = np.stack([reference_centers[name] for name in common_names])
    transform, diagnostics = estimate_similarity(source, target)
    candidate.apply_transform(transform)
    reference_rotations = read_middlebury_camera_rotations(middlebury_path)
    candidate_rotations = read_manifest_camera_rotations(manifest_path)
    world_rotation = np.asarray(diagnostics["rotation"], dtype=np.float64)
    rotation_errors: list[float] = []
    for name in common_names:
        if name not in reference_rotations or name not in candidate_rotations:
            continue
        predicted_reference_rotation = candidate_rotations[name] @ world_rotation.T
        relative = reference_rotations[name] @ predicted_reference_rotation.T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        rotation_errors.append(float(np.rad2deg(np.arccos(cosine))))
    if rotation_errors:
        diagnostics["camera_rotation_error_degrees"] = quantiles(
            np.asarray(rotation_errors, dtype=np.float64)
        )
    diagnostics["camera_names"] = common_names
    diagnostics["candidate_to_reference_transform"] = transform.tolist()
    return diagnostics


def load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found: {path}")
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No triangle geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported mesh payload in {path}: {type(loaded).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh contains no triangles: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Mesh contains non-finite vertices: {path}")
    return mesh


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def mesh_statistics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    triangles = mesh.triangles
    edge_lengths = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    longest_edges = np.max(edge_lengths, axis=1)
    aspect_ratios = np.divide(
        longest_edges * longest_edges,
        np.maximum(double_areas, np.finfo(np.float64).eps),
    )
    unique_edge_counts = np.bincount(mesh.edges_unique_inverse)
    boundary_edge_count = int(np.count_nonzero(unique_edge_counts == 1))
    non_manifold_edge_count = int(np.count_nonzero(unique_edge_counts > 2))
    components = mesh.split(only_watertight=False)
    component_face_counts = sorted((int(len(component.faces)) for component in components), reverse=True)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "surface_area": float(mesh.area),
        "bbox_min": [float(value) for value in mesh.bounds[0]],
        "bbox_max": [float(value) for value in mesh.bounds[1]],
        "bbox_diagonal": float(np.linalg.norm(mesh.extents)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "connected_components": len(component_face_counts),
        "largest_component_face_fraction": float(component_face_counts[0] / len(mesh.faces)),
        "boundary_edges": boundary_edge_count,
        "non_manifold_edges": non_manifold_edge_count,
        "triangle_area": quantiles(double_areas * 0.5),
        "triangle_aspect_ratio": quantiles(aspect_ratios),
        "triangle_aspect_over_10_fraction": float(np.mean(aspect_ratios > 10.0)),
        "triangle_aspect_over_20_fraction": float(np.mean(aspect_ratios > 20.0)),
    }


def closest_surface_metrics(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    sample_count: int,
    seed: int,
    thresholds: list[float],
    chunk_size: int,
) -> dict[str, Any]:
    np.random.seed(seed)
    points, source_face_ids = trimesh.sample.sample_surface(source, sample_count)
    distances: list[np.ndarray] = []
    target_face_ids: list[np.ndarray] = []
    for offset in range(0, sample_count, chunk_size):
        chunk = points[offset : offset + chunk_size]
        _, chunk_distances, chunk_face_ids = trimesh.proximity.closest_point(target, chunk)
        distances.append(chunk_distances)
        target_face_ids.append(chunk_face_ids)
    distance_array = np.concatenate(distances)
    target_faces = np.concatenate(target_face_ids)
    source_normals = source.face_normals[source_face_ids]
    target_normals = target.face_normals[target_faces]
    normal_dots = np.abs(np.einsum("ij,ij->i", source_normals, target_normals))
    return {
        "distance": quantiles(distance_array),
        "within_threshold_fraction": {
            f"{threshold:.12g}": float(np.mean(distance_array <= threshold))
            for threshold in thresholds
        },
        "normal_abs_dot": quantiles(normal_dots),
        "normal_within_15deg_fraction": float(np.mean(normal_dots >= np.cos(np.deg2rad(15.0)))),
        "normal_within_30deg_fraction": float(np.mean(normal_dots >= np.cos(np.deg2rad(30.0)))),
    }


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.proximity_chunk_size < 1:
        raise ValueError("--proximity-chunk-size must be positive")
    if any(threshold <= 0.0 for threshold in args.thresholds):
        raise ValueError("--thresholds must all be positive")
    reference_path = args.reference.resolve()
    candidate_path = args.candidate.resolve()
    reference = load_mesh(reference_path)
    candidate = load_mesh(candidate_path)
    if (args.reference_middlebury_par is None) != (args.candidate_mvs_manifest is None):
        raise ValueError(
            "--reference-middlebury-par and --candidate-mvs-manifest must be specified together"
        )
    alignment = None
    if args.reference_middlebury_par is not None:
        alignment = align_candidate_from_cameras(
            candidate,
            args.reference_middlebury_par.resolve(),
            args.candidate_mvs_manifest.resolve(),
        )
    if args.aligned_candidate_output is not None:
        aligned_candidate_path = args.aligned_candidate_output.resolve()
        aligned_candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate.export(aligned_candidate_path)
    report = {
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "sample_count_per_direction": args.samples,
        "thresholds": args.thresholds,
        "alignment": alignment,
        "reference": mesh_statistics(reference),
        "candidate": mesh_statistics(candidate),
        "candidate_to_reference": closest_surface_metrics(
            candidate,
            reference,
            args.samples,
            args.seed,
            args.thresholds,
            args.proximity_chunk_size,
        ),
        "reference_to_candidate": closest_surface_metrics(
            reference,
            candidate,
            args.samples,
            args.seed + 1,
            args.thresholds,
            args.proximity_chunk_size,
        ),
    }
    forward = report["candidate_to_reference"]["distance"]
    backward = report["reference_to_candidate"]["distance"]
    report["symmetric"] = {
        "chamfer_l1_mean": 0.5 * (forward["mean"] + backward["mean"]),
        "hausdorff_sampled": max(forward["max"], backward["max"]),
        "p95_mean": 0.5 * (forward["p95"] + backward["p95"]),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "Compared meshes: "
        f"candidate faces={len(candidate.faces)}, reference faces={len(reference.faces)}, "
        f"Chamfer-L1={report['symmetric']['chamfer_l1_mean']:.8g}, "
        f"sampled Hausdorff={report['symmetric']['hausdorff_sampled']:.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
