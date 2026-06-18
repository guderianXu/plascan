#!/usr/bin/env python3
"""Estimate local PCA normals for PLY point clouds used by LiDAR BA constraints."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


PLY_TYPE_FORMATS = {
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
    "uchar": "B",
    "uint8": "B",
    "char": "b",
    "int8": "b",
    "ushort": "H",
    "uint16": "H",
    "short": "h",
    "int16": "h",
    "uint": "I",
    "uint32": "I",
    "int": "i",
    "int32": "i",
}


def parse_ply(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    data = path.read_bytes()
    marker = b"end_header\n"
    if marker not in data:
        raise ValueError(f"{path} has no end_header")
    header_end = data.index(marker) + len(marker)
    header = data[:header_end].decode("ascii", errors="replace")

    file_format = ""
    vertex_count = 0
    fields: list[str] = []
    types: list[str] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            file_format = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex and len(parts) >= 3:
            if parts[1] == "list":
                raise ValueError(f"{path} contains list vertex properties")
            if parts[1] not in PLY_TYPE_FORMATS:
                raise ValueError(f"{path} contains unsupported PLY property type: {parts[1]}")
            types.append(parts[1])
            fields.append(parts[2])

    if file_format != "binary_little_endian":
        raise ValueError(f"{path} must be binary_little_endian, got {file_format}")
    if not {"x", "y", "z"}.issubset(set(fields)):
        raise ValueError(f"{path} is missing x/y/z fields")

    fmt = "<" + "".join(PLY_TYPE_FORMATS[item] for item in types)
    stride = struct.calcsize(fmt)
    body = data[header_end:]
    expected = vertex_count * stride
    if len(body) != expected:
        raise ValueError(f"{path} body has {len(body)} bytes, expected {expected}")

    rows = np.empty((vertex_count, len(fields)), dtype=np.float64)
    for index in range(vertex_count):
        rows[index, :] = struct.unpack_from(fmt, body, index * stride)
    return fields, types, rows


def estimate_normals(points: np.ndarray, k_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 3:
        raise ValueError("at least 3 points are required to estimate normals")

    k = max(3, min(k_neighbors, points.shape[0] - 1))
    centroid = points.mean(axis=0)
    normals = np.zeros_like(points)
    curvature = np.zeros(points.shape[0], dtype=np.float64)

    for index, point in enumerate(points):
        delta = points - point
        distances2 = np.einsum("ij,ij->i", delta, delta)
        neighbor_indices = np.argpartition(distances2, k)[: k + 1]
        neighbor_indices = neighbor_indices[neighbor_indices != index][:k]
        neighbors = points[neighbor_indices]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        norm = np.linalg.norm(normal)
        if not np.isfinite(norm) or norm <= 1e-12:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            curvature[index] = 1.0
        else:
            normal = normal / norm
            if np.dot(normal, point - centroid) < 0.0:
                normal = -normal
            denom = float(np.sum(np.maximum(eigvals, 0.0)))
            curvature[index] = float(max(eigvals[0], 0.0) / denom) if denom > 1e-12 else 0.0
        normals[index, :] = normal

    return normals, curvature


def write_normals_ply(output_path: Path,
                      fields: list[str],
                      rows: np.ndarray,
                      normals: np.ndarray,
                      curvature: np.ndarray) -> None:
    field_index = {name: index for index, name in enumerate(fields)}
    intensity = rows[:, field_index["intensity"]] if "intensity" in field_index else np.zeros(rows.shape[0])
    xyz = rows[:, [field_index["x"], field_index["y"], field_index["z"]]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated_by estimate_lidar_normals.py\n"
        f"element vertex {rows.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "property float normal_x\n"
        "property float normal_y\n"
        "property float normal_z\n"
        "property float curvature\n"
        "end_header\n"
    ).encode("ascii")
    with output_path.open("wb") as handle:
        handle.write(header)
        for index in range(rows.shape[0]):
            handle.write(
                struct.pack(
                    "<ffffffff",
                    float(xyz[index, 0]),
                    float(xyz[index, 1]),
                    float(xyz[index, 2]),
                    float(intensity[index]),
                    float(normals[index, 0]),
                    float(normals[index, 1]),
                    float(normals[index, 2]),
                    float(curvature[index]),
                )
            )


def estimate_normals_for_ply(input_path: Path,
                             output_path: Path,
                             k_neighbors: int = 16) -> dict[str, Any]:
    fields, _types, rows = parse_ply(Path(input_path))
    field_index = {name: index for index, name in enumerate(fields)}
    points = rows[:, [field_index["x"], field_index["y"], field_index["z"]]]
    normals, curvature = estimate_normals(points, k_neighbors=k_neighbors)
    write_normals_ply(Path(output_path), fields, rows, normals, curvature)

    normal_lengths = np.linalg.norm(normals, axis=1)
    return {
        "input_path": str(Path(input_path)),
        "output_path": str(Path(output_path)),
        "vertex_count": int(points.shape[0]),
        "valid_normal_count": int(np.count_nonzero(normal_lengths > 0.99)),
        "k_neighbors": int(k_neighbors),
        "curvature_min": float(np.min(curvature)),
        "curvature_max": float(np.max(curvature)),
        "curvature_mean": float(np.mean(curvature)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = estimate_normals_for_ply(args.input, args.output, args.k_neighbors)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
