"""Geometry sampling and topology metrics for an ETH3D scan/mesh pair."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from .mesh_io import TriangleMesh, load_ply_vertices, load_triangle_mesh, sample_surface


def _mesh_group(mlp_path: Path) -> list[tuple[Path, np.ndarray]]:
    root = ET.parse(mlp_path).getroot()
    entries: list[tuple[Path, np.ndarray]] = []
    for element in root.findall(".//MLMesh"):
        filename = element.get("filename")
        matrix_node = element.find("MLMatrix44")
        if not filename or matrix_node is None or not matrix_node.text:
            raise ValueError(f"incomplete MLMesh entry in {mlp_path}")
        values = np.fromstring(matrix_node.text, sep=" ", dtype=np.float64)
        if values.size != 16 or not np.isfinite(values).all():
            raise ValueError(f"invalid 4x4 transform for {filename}")
        entries.append((mlp_path.parent / filename, values.reshape(4, 4)))
    if not entries:
        raise ValueError(f"no MLMesh entries in {mlp_path}")
    return entries


def _load_scan_points(mlp_path: Path) -> np.ndarray:
    transformed: list[np.ndarray] = []
    for scan_path, transform in _mesh_group(mlp_path):
        vertices = load_ply_vertices(scan_path)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"scan has invalid vertices: {scan_path}")
        homogeneous = np.column_stack((vertices, np.ones(vertices.shape[0])))
        world = homogeneous @ transform.T
        valid = np.isfinite(world).all(axis=1) & (np.abs(world[:, 3]) > 1e-12)
        world = world[valid, :3] / world[valid, 3, np.newaxis]
        transformed.append(world)
    return np.concatenate(transformed, axis=0)


def _voxel_representatives(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    origin = np.min(points, axis=0)
    cells = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(first)]


def _distance_summary(distances: np.ndarray, thresholds: tuple[float, ...]) -> dict:
    if distances.size == 0 or not np.isfinite(distances).all():
        raise ValueError("distance samples must be non-empty and finite")
    result = {
        "sample_count": int(distances.size),
        "mean_m": float(np.mean(distances)),
        "rmse_m": float(np.sqrt(np.mean(np.square(distances)))),
        "p50_m": float(np.quantile(distances, 0.50)),
        "p95_m": float(np.quantile(distances, 0.95)),
    }
    result["within_threshold"] = {
        f"{threshold:.6g}": float(np.mean(distances <= threshold))
        for threshold in thresholds
    }
    return result


def _topology(mesh: TriangleMesh) -> dict:
    face_edges = np.concatenate(
        (mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]), axis=0
    )
    edges = np.sort(face_edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    parent = np.arange(len(mesh.vertices), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for first, second in unique_edges:
        first_root, second_root = find(int(first)), find(int(second))
        if first_root != second_root:
            parent[second_root] = first_root
    referenced = np.unique(mesh.faces)
    component_count = len({find(int(vertex)) for vertex in referenced})
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    euler = len(mesh.vertices) - len(unique_edges) + len(mesh.faces)
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "connected_component_count": int(component_count),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "euler_number": int(euler),
        "watertight": boundary_edges == 0 and nonmanifold_edges == 0,
    }


def evaluate_surface(
    mesh_path: Path,
    scan_alignment_path: Path,
    *,
    voxel_size_m: float = 0.02,
    mesh_sample_count: int = 1_000_000,
    thresholds_m: tuple[float, ...] = (0.01, 0.02, 0.05),
    seed: int = 0,
) -> dict:
    """Evaluate a mesh against aligned ETH3D scans using deterministic samples."""
    mesh_path = mesh_path.resolve(strict=True)
    scan_alignment_path = scan_alignment_path.resolve(strict=True)
    mesh = load_triangle_mesh(mesh_path)
    scan_points = _voxel_representatives(
        _load_scan_points(scan_alignment_path), voxel_size_m
    )
    sample_count = int(mesh_sample_count)
    if sample_count <= 0:
        raise ValueError("mesh_sample_count must be positive")
    mesh_points = sample_surface(mesh, sample_count, seed)
    scan_tree = cKDTree(scan_points)
    mesh_tree = cKDTree(mesh_points)
    accuracy = scan_tree.query(mesh_points, workers=1)[0]
    completeness = mesh_tree.query(scan_points, workers=1)[0]
    return {
        "schema": "plascan.eth3d_surface_evaluation.v1",
        "metric_semantics": {
            "accuracy": "sampled reconstruction surface to aligned scan points",
            "completeness": "voxelized aligned scan points to sampled reconstruction surface",
            "sampling": "deterministic bidirectional point distance; not the official ETH3D occlusion-aware score",
        },
        "inputs": {
            "mesh": str(mesh_path),
            "scan_alignment": str(scan_alignment_path),
        },
        "sampling": {
            "scan_voxel_size_m": voxel_size_m,
            "scan_point_count": int(scan_points.shape[0]),
            "mesh_surface_sample_count": int(mesh_points.shape[0]),
            "seed": seed,
        },
        "accuracy": _distance_summary(accuracy, thresholds_m),
        "completeness": _distance_summary(completeness, thresholds_m),
        "topology": _topology(mesh),
    }
