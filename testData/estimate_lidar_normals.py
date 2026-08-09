#!/usr/bin/env python3
"""Estimate scalable local-PCA normals for LiDAR PLY point clouds."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


PLY_NUMPY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "<u2",
    "uint16": "<u2",
    "short": "<i2",
    "int16": "<i2",
    "uint": "<u4",
    "uint32": "<u4",
    "int": "<i4",
    "int32": "<i4",
}


@dataclass(frozen=True)
class PlyVertexLayout:
    vertex_count: int
    fields: tuple[str, ...]
    types: tuple[str, ...]
    data_offset: int
    dtype: np.dtype


def parse_ply_layout(path: Path) -> PlyVertexLayout:
    """Read a binary little-endian PLY header without loading its body."""
    path = Path(path)
    with path.open("rb") as handle:
        first = handle.readline()
        if first.rstrip(b"\r\n") != b"ply":
            raise ValueError(f"{path} is not a PLY file")

        file_format = ""
        vertex_count = 0
        fields: list[str] = []
        types: list[str] = []
        in_vertex = False
        for _ in range(4096):
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"{path} has no end_header")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path} has a non-ASCII PLY header") from exc
            if line == "end_header":
                data_offset = handle.tell()
                break

            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                file_format = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if len(parts) < 3 or parts[1] == "list":
                    raise ValueError(f"{path} contains unsupported list vertex properties")
                if parts[1] not in PLY_NUMPY_TYPES:
                    raise ValueError(
                        f"{path} contains unsupported PLY property type: {parts[1]}"
                    )
                types.append(parts[1])
                fields.append(parts[2])
        else:
            raise ValueError(f"{path} PLY header is unexpectedly long")

    if file_format != "binary_little_endian":
        raise ValueError(f"{path} must be binary_little_endian, got {file_format}")
    if vertex_count <= 0:
        raise ValueError(f"{path} has no vertices")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"{path} is missing x/y/z fields")
    if len(set(fields)) != len(fields):
        raise ValueError(f"{path} contains duplicate vertex property names")

    dtype = np.dtype(
        [(name, PLY_NUMPY_TYPES[type_name]) for name, type_name in zip(fields, types)]
    )
    expected_size = data_offset + vertex_count * dtype.itemsize
    if path.stat().st_size < expected_size:
        raise ValueError(
            f"{path} body is truncated: {path.stat().st_size} bytes, expected at least {expected_size}"
        )
    return PlyVertexLayout(
        vertex_count=vertex_count,
        fields=tuple(fields),
        types=tuple(types),
        data_offset=data_offset,
        dtype=dtype,
    )


def read_ply_xyz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Memory-map a PLY body and copy only XYZ plus optional intensity."""
    path = Path(path)
    layout = parse_ply_layout(path)
    rows = np.memmap(
        path,
        dtype=layout.dtype,
        mode="r",
        offset=layout.data_offset,
        shape=(layout.vertex_count,),
    )
    points = np.column_stack((rows["x"], rows["y"], rows["z"])).astype(
        np.float64, copy=False
    )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path} contains non-finite XYZ coordinates")
    intensity = None
    if "intensity" in layout.fields:
        intensity = np.asarray(rows["intensity"], dtype=np.float32).copy()
    del rows
    return points, intensity


def deterministic_subsample(
    points: np.ndarray,
    max_samples: int,
    *attributes: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    if max_samples <= 0 or points.shape[0] <= max_samples:
        return (points, *attributes)
    indices = np.linspace(0, points.shape[0] - 1, max_samples, dtype=np.int64)
    sampled: list[np.ndarray | None] = [points[indices]]
    sampled.extend(None if values is None else values[indices] for values in attributes)
    return tuple(sampled)  # type: ignore[return-value]


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    *attributes: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    """Keep one deterministic representative per occupied voxel."""
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return (points, *attributes)
    if not np.isfinite(voxel_size):
        raise ValueError("voxel_size must be finite")

    keys = np.floor(points / voxel_size).astype(np.int64)
    packed = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    ).reshape(-1)
    _, first_indices = np.unique(packed, return_index=True)
    first_indices.sort()
    sampled: list[np.ndarray | None] = [points[first_indices]]
    sampled.extend(
        None if values is None else values[first_indices] for values in attributes
    )
    return tuple(sampled)  # type: ignore[return-value]


def estimate_normals(
    points: np.ndarray,
    k_neighbors: int,
    *,
    viewpoint: np.ndarray | None = None,
    chunk_size: int = 20000,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate point normals in bounded-memory batches using a cKDTree."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] < 4:
        raise ValueError("at least 4 points are required to estimate normals")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    neighbor_count = max(3, min(k_neighbors, points.shape[0] - 1))
    tree = cKDTree(points)
    normals = np.empty_like(points, dtype=np.float64)
    curvature = np.empty(points.shape[0], dtype=np.float64)
    orientation_center = points.mean(axis=0)
    resolved_viewpoint = None
    if viewpoint is not None:
        resolved_viewpoint = np.asarray(viewpoint, dtype=np.float64)
        if resolved_viewpoint.shape != (3,) or not np.all(np.isfinite(resolved_viewpoint)):
            raise ValueError("viewpoint must contain three finite coordinates")

    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        _, indices = tree.query(
            points[start:stop],
            k=neighbor_count + 1,
            workers=workers,
        )
        neighbors = points[indices[:, 1:]]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("bki,bkj->bij", centered, centered)
        covariance /= max(1, neighbor_count - 1)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        batch_normals = eigvecs[:, :, 0]
        lengths = np.linalg.norm(batch_normals, axis=1)
        valid = np.isfinite(lengths) & (lengths > 1.0e-12)
        batch_normals[valid] /= lengths[valid, None]
        batch_normals[~valid] = (0.0, 0.0, 1.0)

        if resolved_viewpoint is not None:
            direction = resolved_viewpoint - points[start:stop]
        else:
            direction = points[start:stop] - orientation_center
        flip = np.einsum("ij,ij->i", batch_normals, direction) < 0.0
        batch_normals[flip] *= -1.0

        safe_eigvals = np.maximum(eigvals, 0.0)
        denominator = safe_eigvals.sum(axis=1)
        batch_curvature = np.divide(
            safe_eigvals[:, 0],
            denominator,
            out=np.zeros(stop - start, dtype=np.float64),
            where=denominator > 1.0e-12,
        )
        batch_curvature[~valid] = 1.0
        normals[start:stop] = batch_normals
        curvature[start:stop] = batch_curvature

    return normals, curvature


def write_plane_ply(
    output_path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    generator: str = "estimate_lidar_normals.py",
) -> None:
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and normals must both have shape (N, 3)")
    if curvature.shape != (points.shape[0],):
        raise ValueError("curvature must have shape (N,)")
    if intensity is not None and intensity.shape != (points.shape[0],):
        raise ValueError("intensity must have shape (N,)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if intensity is not None:
        fields.append(("intensity", "<f4"))
    fields.extend(
        [
            ("normal_x", "<f4"),
            ("normal_y", "<f4"),
            ("normal_z", "<f4"),
            ("curvature", "<f4"),
        ]
    )
    rows = np.empty(points.shape[0], dtype=np.dtype(fields))
    rows["x"], rows["y"], rows["z"] = points.T
    if intensity is not None:
        rows["intensity"] = intensity
    rows["normal_x"], rows["normal_y"], rows["normal_z"] = normals.T
    rows["curvature"] = curvature

    property_lines = "".join(f"property float {name}\n" for name, _ in fields)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment generated_by {generator}\n"
        f"element vertex {points.shape[0]}\n"
        f"{property_lines}"
        "end_header\n"
    ).encode("ascii")
    with output_path.open("wb") as handle:
        handle.write(header)
        rows.tofile(handle)


def estimate_normals_for_ply(
    input_path: Path,
    output_path: Path,
    k_neighbors: int = 16,
    *,
    voxel_size: float = 0.0,
    max_samples: int = 0,
    viewpoint: np.ndarray | None = None,
    chunk_size: int = 20000,
    workers: int = 1,
) -> dict[str, Any]:
    points, intensity = read_ply_xyz(Path(input_path))
    input_count = points.shape[0]
    points, intensity = voxel_downsample(points, voxel_size, intensity)
    points, intensity = deterministic_subsample(points, max_samples, intensity)
    normals, curvature = estimate_normals(
        points,
        k_neighbors=k_neighbors,
        viewpoint=viewpoint,
        chunk_size=chunk_size,
        workers=workers,
    )
    write_plane_ply(Path(output_path), points, normals, curvature, intensity)

    normal_lengths = np.linalg.norm(normals, axis=1)
    return {
        "input_path": str(Path(input_path)),
        "output_path": str(Path(output_path)),
        "input_vertex_count": int(input_count),
        "vertex_count": int(points.shape[0]),
        "valid_normal_count": int(np.count_nonzero(normal_lengths > 0.99)),
        "k_neighbors": int(k_neighbors),
        "voxel_size": float(voxel_size),
        "max_samples": int(max_samples),
        "curvature_min": float(np.min(curvature)),
        "curvature_max": float(np.max(curvature)),
        "curvature_mean": float(np.mean(curvature)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k-neighbors", type=int, default=24)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--viewpoint", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = estimate_normals_for_ply(
        args.input,
        args.output,
        args.k_neighbors,
        voxel_size=args.voxel_size,
        max_samples=args.max_samples,
        viewpoint=None if args.viewpoint is None else np.asarray(args.viewpoint),
        chunk_size=args.chunk_size,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
