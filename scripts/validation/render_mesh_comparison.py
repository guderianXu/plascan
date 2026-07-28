"""Render registered reference and candidate meshes from identical orthographic views."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a two-row front/side/top contact sheet for registered meshes."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-label", default="Reference")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--max-faces", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No triangle mesh found: {path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"No triangle mesh found: {path}")
    return loaded


def selected_face_indices(mesh: trimesh.Trimesh, maximum: int, seed: int) -> np.ndarray:
    face_count = len(mesh.faces)
    if face_count <= maximum:
        return np.arange(face_count)
    rng = np.random.default_rng(seed)
    areas = np.maximum(mesh.area_faces, np.finfo(np.float64).eps)
    probabilities = areas / np.sum(areas)
    return np.sort(rng.choice(face_count, size=maximum, replace=False, p=probabilities))


def render_mesh(
    axis: plt.Axes,
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    base_color: np.ndarray,
    elevation: float,
    azimuth: float,
    center: np.ndarray,
    radius: float,
) -> None:
    triangles = mesh.vertices[mesh.faces[face_indices]]
    normals = mesh.face_normals[face_indices]
    light = np.asarray([0.35, -0.45, 0.82], dtype=np.float64)
    light /= np.linalg.norm(light)
    intensity = np.clip(0.30 + 0.70 * np.abs(normals @ light), 0.0, 1.0)
    colors = np.clip(base_color[None, :] * intensity[:, None], 0.0, 1.0)
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors=(0.08, 0.08, 0.08, 0.10),
        linewidths=0.08,
        antialiased=False,
    )
    axis.add_collection3d(collection)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()


def main() -> int:
    args = parse_args()
    if args.max_faces < 100:
        raise ValueError("--max-faces must be at least 100")
    reference = load_mesh(args.reference.resolve())
    candidate = load_mesh(args.candidate.resolve())
    combined_minimum = np.minimum(reference.bounds[0], candidate.bounds[0])
    combined_maximum = np.maximum(reference.bounds[1], candidate.bounds[1])
    center = 0.5 * (combined_minimum + combined_maximum)
    radius = 0.55 * float(np.max(combined_maximum - combined_minimum))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Mesh bounds are invalid")

    reference_faces = selected_face_indices(reference, args.max_faces, args.seed)
    candidate_faces = selected_face_indices(candidate, args.max_faces, args.seed + 1)
    views = [
        ("Front", 0.0, -90.0),
        ("Side", 0.0, 0.0),
        ("Top", 90.0, -90.0),
    ]
    figure = plt.figure(figsize=(18, 11), dpi=140, facecolor="white")
    figure.suptitle("Registered mesh comparison — identical coordinates and scale", fontsize=18)
    for row, (mesh, indices, label, color) in enumerate(
        [
            (reference, reference_faces, args.reference_label, np.asarray([0.78, 0.82, 0.87])),
            (candidate, candidate_faces, args.candidate_label, np.asarray([0.78, 0.61, 0.39])),
        ]
    ):
        for column, (view_name, elevation, azimuth) in enumerate(views):
            axis = figure.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            render_mesh(
                axis,
                mesh,
                indices,
                color,
                elevation,
                azimuth,
                center,
                radius,
            )
            axis.set_title(f"{label} — {view_name}", fontsize=13, pad=8)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(
        f"Rendered {output}: reference_faces={len(reference_faces)}, "
        f"candidate_faces={len(candidate_faces)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
