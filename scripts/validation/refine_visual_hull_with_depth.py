#!/usr/bin/env python3
"""Refine a stable visual-hull mesh with bounded depth-surface displacements.

The source mesh topology is preserved.  Depth detail is transferred only as a
scalar displacement along the source vertex normal, which prevents tangential
vertex drift and the long spikes produced by unconstrained nearest-point moves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from analyze_depth_pose_alignment import load_frames, project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-hull", required=True, type=Path)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--depth-surface", type=Path)
    evidence.add_argument("--mvs-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--maximum-distance", type=float, default=0.012)
    parser.add_argument("--maximum-displacement", type=float, default=0.004)
    parser.add_argument("--minimum-normal-dot", type=float, default=0.60)
    parser.add_argument("--smoothing-iterations", type=int, default=12)
    parser.add_argument("--smoothing-lambda", type=float, default=0.45)
    parser.add_argument("--minimum-anchor-weight", type=float, default=0.05)
    parser.add_argument("--minimum-view-count", type=int, default=2)
    parser.add_argument("--minimum-depth-confidence", type=float, default=0.25)
    parser.add_argument("--maximum-view-mad", type=float, default=0.004)
    parser.add_argument("--propagation-decay", type=float, default=0.70)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise RuntimeError(f"Mesh is empty or unsupported: {path}")
    return loaded


def build_neighbor_average(
    vertex_count: int,
    edges: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    summed = np.zeros(vertex_count, dtype=np.float64)
    counts = np.zeros(vertex_count, dtype=np.float64)
    np.add.at(summed, edges[:, 0], values[edges[:, 1]])
    np.add.at(summed, edges[:, 1], values[edges[:, 0]])
    np.add.at(counts, edges[:, 0], 1.0)
    np.add.at(counts, edges[:, 1], 1.0)
    return np.divide(
        summed,
        np.maximum(counts, 1.0),
        out=np.zeros_like(summed),
        where=counts > 0.0,
    )


def mesh_displacement_evidence(
    hull_vertices: np.ndarray,
    hull_normals: np.ndarray,
    depth: trimesh.Trimesh,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    depth_vertices = np.asarray(depth.vertices, dtype=np.float64)
    depth_normals = np.asarray(depth.vertex_normals, dtype=np.float64)
    tree = cKDTree(depth_vertices)
    distances, indices = tree.query(hull_vertices, workers=-1)
    nearest = depth_vertices[indices]
    nearest_normals = depth_normals[indices]

    normal_dot = np.abs(np.einsum("ij,ij->i", hull_normals, nearest_normals))
    signed_displacement = np.einsum(
        "ij,ij->i", nearest - hull_vertices, hull_normals
    )
    distance_weight = np.square(
        np.clip(1.0 - distances / args.maximum_distance, 0.0, 1.0)
    )
    normal_weight = np.clip(
        (normal_dot - args.minimum_normal_dot)
        / max(1.0 - args.minimum_normal_dot, 1.0e-6),
        0.0,
        1.0,
    )
    return (
        signed_displacement,
        distance_weight * normal_weight,
        {
            "evidence_mode": "nearest_depth_surface",
            "depth_surface": str(args.depth_surface.resolve()),
        },
    )


def multiview_displacement_evidence(
    hull_vertices: np.ndarray,
    hull_normals: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    frames = load_frames(args.mvs_manifest.resolve())
    if len(frames) < args.minimum_view_count:
        raise RuntimeError(
            f"Not enough depth frames in manifest: {args.mvs_manifest}"
        )

    displacement_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    for frame in frames:
        projected_rows, projected_cols, projected_depth = project(
            frame, hull_vertices
        )
        nearest_rows = np.rint(projected_rows).astype(np.int64)
        nearest_cols = np.rint(projected_cols).astype(np.int64)
        inside = (
            (projected_depth > 0.0)
            & (nearest_rows >= 0)
            & (nearest_rows < frame.depth.shape[0])
            & (nearest_cols >= 0)
            & (nearest_cols < frame.depth.shape[1])
        )
        selected = np.flatnonzero(inside)
        displacement = np.full(len(hull_vertices), np.nan, dtype=np.float64)
        confidence = np.zeros(len(hull_vertices), dtype=np.float64)
        if selected.size > 0:
            rows = nearest_rows[selected]
            cols = nearest_cols[selected]
            observed_depth = frame.depth[rows, cols].astype(np.float64)
            observed_confidence = frame.confidence[rows, cols].astype(
                np.float64
            )
            valid = (
                np.isfinite(observed_depth)
                & (observed_depth > 0.0)
                & np.isfinite(observed_confidence)
                & (
                    observed_confidence
                    >= args.minimum_depth_confidence
                )
                & (
                    np.abs(observed_depth - projected_depth[selected])
                    <= args.maximum_distance
                )
            )
            selected = selected[valid]
            rows = rows[valid]
            cols = cols[valid]
            observed_depth = observed_depth[valid]
            observed_confidence = observed_confidence[valid]
            if selected.size > 0:
                camera_points = np.column_stack(
                    (
                        (projected_cols[selected] - frame.cx)
                        * observed_depth
                        / frame.fx,
                        (projected_rows[selected] - frame.cy)
                        * observed_depth
                        / frame.fy,
                        observed_depth,
                    )
                )
                target_world = (
                    frame.rotation_world_to_camera.T
                    @ (
                        camera_points
                        - frame.translation_world_to_camera
                    ).T
                ).T
                displacement[selected] = np.einsum(
                    "ij,ij->i",
                    target_world - hull_vertices[selected],
                    hull_normals[selected],
                )
                confidence[selected] = np.clip(
                    observed_confidence, 0.0, 1.0
                )
        displacement_rows.append(displacement)
        confidence_rows.append(confidence)

    displacement_matrix = np.stack(displacement_rows, axis=1)
    confidence_matrix = np.stack(confidence_rows, axis=1)
    valid = np.isfinite(displacement_matrix)
    view_count = np.count_nonzero(valid, axis=1)
    has_evidence = view_count > 0
    median = np.zeros(len(hull_vertices), dtype=np.float64)
    mad = np.full(len(hull_vertices), np.inf, dtype=np.float64)
    median[has_evidence] = np.nanmedian(
        displacement_matrix[has_evidence], axis=1
    )
    mad[has_evidence] = np.nanmedian(
        np.abs(
            displacement_matrix[has_evidence]
            - median[has_evidence, None]
        ),
        axis=1,
    )
    confidence_sum = np.sum(
        np.where(valid, confidence_matrix, 0.0), axis=1
    )
    mean_confidence = np.divide(
        confidence_sum,
        np.maximum(view_count, 1),
        out=np.zeros_like(confidence_sum),
        where=view_count > 0,
    )
    view_weight = np.clip(
        view_count / max(args.minimum_view_count + 1, 1), 0.0, 1.0
    )
    agreement_weight = np.clip(
        1.0 - mad / max(args.maximum_view_mad, 1.0e-9),
        0.0,
        1.0,
    )
    anchor_weight = mean_confidence * view_weight * agreement_weight
    anchor_weight[view_count < args.minimum_view_count] = 0.0
    return (
        median,
        anchor_weight,
        {
            "evidence_mode": "multiview_depth_projection",
            "mvs_manifest": str(args.mvs_manifest.resolve()),
            "frame_count": len(frames),
            "minimum_view_count": args.minimum_view_count,
            "median_supporting_view_count": float(np.median(view_count)),
            "p90_supporting_view_count": float(
                np.percentile(view_count, 90.0)
            ),
            "median_view_mad": float(
                np.median(mad[np.isfinite(mad)])
            ),
        },
    )


def refine_mesh(args: argparse.Namespace) -> dict[str, object]:
    hull = load_mesh(args.visual_hull)

    hull_vertices = np.asarray(hull.vertices, dtype=np.float64)
    hull_normals = np.asarray(hull.vertex_normals, dtype=np.float64)
    if args.depth_surface is not None:
        depth = load_mesh(args.depth_surface)
        signed_displacement, anchor_weight, evidence_report = (
            mesh_displacement_evidence(
                hull_vertices, hull_normals, depth, args
            )
        )
    else:
        signed_displacement, anchor_weight, evidence_report = (
            multiview_displacement_evidence(
                hull_vertices, hull_normals, args
            )
        )
    accepted = anchor_weight >= args.minimum_anchor_weight

    initial = np.zeros(len(hull_vertices), dtype=np.float64)
    initial[accepted] = np.clip(
        signed_displacement[accepted],
        -args.maximum_displacement,
        args.maximum_displacement,
    )

    values = initial.copy()
    confidence = anchor_weight.copy()
    edges = np.asarray(hull.edges_unique, dtype=np.int64)
    smoothing_lambda = float(np.clip(args.smoothing_lambda, 0.0, 20.0))
    propagation_decay = float(np.clip(args.propagation_decay, 0.0, 1.0))
    for _ in range(max(args.smoothing_iterations, 0)):
        neighbor_values = build_neighbor_average(len(values), edges, values)
        neighbor_confidence = build_neighbor_average(
            len(confidence), edges, confidence
        )
        propagated_weight = (
            smoothing_lambda * propagation_decay * neighbor_confidence
        )
        denominator = anchor_weight + propagated_weight
        updated = np.divide(
            anchor_weight * initial + propagated_weight * neighbor_values,
            np.maximum(denominator, 1.0e-12),
            out=np.zeros_like(values),
            where=denominator > 0.0,
        )
        values = np.clip(
            updated,
            -args.maximum_displacement,
            args.maximum_displacement,
        )
        confidence = np.maximum(
            anchor_weight, propagation_decay * neighbor_confidence
        )

    refined_vertices = hull_vertices + values[:, None] * hull_normals
    refined = trimesh.Trimesh(
        vertices=refined_vertices,
        faces=np.asarray(hull.faces),
        process=False,
    )
    if hull.visual.kind is not None:
        refined.visual = hull.visual.copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    refined.export(args.output)

    report = {
        "visual_hull": str(args.visual_hull.resolve()),
        "output": str(args.output.resolve()),
        "vertex_count": int(len(refined.vertices)),
        "face_count": int(len(refined.faces)),
        "accepted_anchor_count": int(np.count_nonzero(accepted)),
        "accepted_anchor_fraction": float(np.mean(accepted)),
        "propagated_vertex_count": int(np.count_nonzero(np.abs(values) > 1.0e-12)),
        "maximum_input_distance": float(args.maximum_distance),
        "maximum_allowed_displacement": float(args.maximum_displacement),
        "maximum_applied_displacement": float(np.max(np.abs(values))),
        "median_applied_displacement": float(np.median(np.abs(values))),
        "p90_applied_displacement": float(
            np.percentile(np.abs(values), 90.0)
        ),
        "bounds": refined.bounds.tolist(),
        "topology_preserved": bool(
            len(refined.vertices) == len(hull.vertices)
            and len(refined.faces) == len(hull.faces)
            and np.array_equal(refined.faces, hull.faces)
        ),
        **evidence_report,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    report = refine_mesh(args)
    print(
        "refined"
        f" vertices={report['vertex_count']}"
        f" faces={report['face_count']}"
        f" anchors={report['accepted_anchor_count']}"
        f" p90_displacement={report['p90_applied_displacement']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
