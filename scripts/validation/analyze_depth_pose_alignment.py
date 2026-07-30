#!/usr/bin/env python3
"""Derive read-only local pose corrections from PlaScan depth artifacts."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


FAST_DEPTH_HEADER = struct.Struct("<16siii4xQ")
FAST_DEPTH_MAGIC = b"PLASDEPTHMAT01\x00\x00"


@dataclass
class Frame:
    index: int
    image: str
    depth: np.ndarray
    confidence: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    center_world: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    source_indices: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate bounded per-camera SE(3) corrections from cross-view "
            "depth agreement without modifying the project."
        )
    )
    parser.add_argument("--mvs-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--corrected-manifest-output",
        type=Path,
        help=(
            "Optional diagnostic manifest copy with accepted local camera "
            "corrections applied. The source project is never modified."
        ),
    )
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-samples-per-camera", type=int, default=6000)
    parser.add_argument("--maximum-relative-depth-error", type=float, default=0.08)
    parser.add_argument("--maximum-translation", type=float, default=0.02)
    parser.add_argument("--maximum-rotation-degrees", type=float, default=2.0)
    parser.add_argument("--anchor-camera", type=int, default=0)
    return parser.parse_args()


def read_depth(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(FAST_DEPTH_HEADER.size)
        magic, rows, cols, cv_type, data_bytes = FAST_DEPTH_HEADER.unpack(header)
        if magic != FAST_DEPTH_MAGIC or cv_type != 5:
            raise ValueError(f"Unsupported PlaScan depth matrix: {path}")
        expected = rows * cols * np.dtype(np.float32).itemsize
        if data_bytes != expected:
            raise ValueError(f"Depth payload size mismatch: {path}")
        values = np.fromfile(stream, dtype="<f4", count=rows * cols)
    return values.reshape(rows, cols)


def load_frames(manifest_path: Path) -> list[Frame]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: list[Frame] = []
    for record in manifest.get("frames", []):
        depth_path = Path(record.get("raw_depth_path", ""))
        confidence_path = Path(record.get("raw_confidence_path", ""))
        camera = record.get("camera_model", {})
        if not depth_path.is_file():
            continue
        depth = read_depth(depth_path)
        confidence = (
            read_depth(confidence_path)
            if confidence_path.is_file()
            else np.ones_like(depth)
        )
        frames.append(
            Frame(
                index=int(record["ref_index"]),
                image=Path(record.get("ref_image", "")).name,
                depth=depth,
                confidence=confidence,
                rotation_world_to_camera=np.asarray(
                    camera["rotation_world_to_camera"], dtype=np.float64
                ).reshape(3, 3),
                translation_world_to_camera=np.asarray(
                    camera["translation_world_to_camera"], dtype=np.float64
                ),
                center_world=np.asarray(camera["camera_center"], dtype=np.float64),
                fx=float(camera["fx"]),
                fy=float(camera["fy"]),
                cx=float(camera["cx"]),
                cy=float(camera["cy"]),
                source_indices=[int(value) for value in record.get("source_indices", [])],
            )
        )
    frames.sort(key=lambda frame: frame.index)
    return frames


def unproject(frame: Frame, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    depth = frame.depth[rows, cols].astype(np.float64)
    camera_points = np.column_stack(
        (
            (cols - frame.cx) * depth / frame.fx,
            (rows - frame.cy) * depth / frame.fy,
            depth,
        )
    )
    return (
        frame.rotation_world_to_camera.T
        @ (camera_points - frame.translation_world_to_camera).T
    ).T


def project(frame: Frame, points_world: np.ndarray) -> tuple[np.ndarray, ...]:
    camera_points = (
        frame.rotation_world_to_camera @ points_world.T
    ).T + frame.translation_world_to_camera
    z = camera_points[:, 2]
    cols = frame.fx * camera_points[:, 0] / z + frame.cx
    rows = frame.fy * camera_points[:, 1] / z + frame.cy
    return rows, cols, z


def target_points_and_normals(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(frame.depth.shape)
    points = unproject(frame, rows.ravel(), cols.ravel()).reshape(
        frame.depth.shape + (3,)
    )
    horizontal = np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)
    vertical = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normals = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(normals, axis=2)
    valid = (
        np.isfinite(frame.depth)
        & (frame.depth > 0.0)
        & np.roll(frame.depth > 0.0, -1, axis=1)
        & np.roll(frame.depth > 0.0, 1, axis=1)
        & np.roll(frame.depth > 0.0, -1, axis=0)
        & np.roll(frame.depth > 0.0, 1, axis=0)
        & (lengths > np.finfo(np.float64).eps)
    )
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.nan
    return points, normals


def collect_correspondences(
    reference: Frame,
    frames_by_index: dict[int, Frame],
    stride: int,
    maximum_relative_depth_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices(reference.depth.shape)
    valid = (
        np.isfinite(reference.depth)
        & (reference.depth > 0.0)
        & np.isfinite(reference.confidence)
        & (reference.confidence > 0.0)
    )
    sampled = valid & (rows % stride == 0) & (cols % stride == 0)
    source_rows = rows[sampled]
    source_cols = cols[sampled]
    source_points = unproject(reference, source_rows, source_cols)
    source_confidence = reference.confidence[source_rows, source_cols]

    all_source: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_normal: list[np.ndarray] = []
    all_weight: list[np.ndarray] = []
    for target_index in reference.source_indices:
        target = frames_by_index.get(target_index)
        if target is None:
            continue
        target_points, target_normals = target_points_and_normals(target)
        projected_rows, projected_cols, projected_depth = project(
            target, source_points
        )
        nearest_rows = np.rint(projected_rows).astype(np.int64)
        nearest_cols = np.rint(projected_cols).astype(np.int64)
        inside = (
            (projected_depth > 0.0)
            & (nearest_rows >= 1)
            & (nearest_rows < target.depth.shape[0] - 1)
            & (nearest_cols >= 1)
            & (nearest_cols < target.depth.shape[1] - 1)
        )
        indices = np.flatnonzero(inside)
        if indices.size == 0:
            continue
        target_rows = nearest_rows[indices]
        target_cols = nearest_cols[indices]
        target_depth = target.depth[target_rows, target_cols]
        normals = target_normals[target_rows, target_cols]
        finite = (
            np.isfinite(target_depth)
            & (target_depth > 0.0)
            & np.all(np.isfinite(normals), axis=1)
        )
        relative_error = np.abs(projected_depth[indices] - target_depth) / np.maximum(
            np.maximum(projected_depth[indices], target_depth),
            np.finfo(np.float64).eps,
        )
        accepted = finite & (relative_error <= maximum_relative_depth_error)
        indices = indices[accepted]
        if indices.size == 0:
            continue
        target_rows = target_rows[accepted]
        target_cols = target_cols[accepted]
        all_source.append(source_points[indices])
        all_target.append(target_points[target_rows, target_cols])
        all_normal.append(target_normals[target_rows, target_cols])
        all_weight.append(
            np.sqrt(
                np.clip(source_confidence[indices], 0.0, 1.0)
                * np.clip(target.confidence[target_rows, target_cols], 0.0, 1.0)
            )
        )
    if not all_source:
        empty_points = np.empty((0, 3), dtype=np.float64)
        return empty_points, empty_points, empty_points, np.empty(0)
    return (
        np.concatenate(all_source),
        np.concatenate(all_target),
        np.concatenate(all_normal),
        np.concatenate(all_weight),
    )


def residual_quantiles(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "median": float(np.quantile(absolute, 0.5)),
        "p90": float(np.quantile(absolute, 0.9)),
        "p95": float(np.quantile(absolute, 0.95)),
    }


def refine_frame(
    frame: Frame,
    correspondences: tuple[np.ndarray, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source, target, normal, weight = correspondences
    if source.shape[0] > args.max_samples_per_camera:
        selection = np.linspace(
            0, source.shape[0] - 1, args.max_samples_per_camera, dtype=np.int64
        )
        source, target, normal, weight = (
            source[selection],
            target[selection],
            normal[selection],
            weight[selection],
        )
    if source.shape[0] < 24:
        return {
            "camera_index": frame.index,
            "image": frame.image,
            "accepted": False,
            "reason": "insufficient_correspondences",
            "correspondence_count": int(source.shape[0]),
        }

    pivot = np.mean(source, axis=0)

    def signed_residual(parameters: np.ndarray, weighted: bool = True) -> np.ndarray:
        rotation = Rotation.from_rotvec(parameters[3:]).as_matrix()
        corrected = (rotation @ (source - pivot).T).T + pivot + parameters[:3]
        residual = np.einsum("ij,ij->i", normal, corrected - target)
        return residual * np.sqrt(weight) if weighted else residual

    before = signed_residual(np.zeros(6), weighted=False)
    translation_bound = args.maximum_translation
    rotation_bound = np.deg2rad(args.maximum_rotation_degrees)
    lower = np.array(
        [-translation_bound] * 3 + [-rotation_bound] * 3, dtype=np.float64
    )
    upper = -lower
    optimization = least_squares(
        signed_residual,
        np.zeros(6),
        bounds=(lower, upper),
        loss="huber",
        f_scale=0.002,
        max_nfev=40,
    )
    after = signed_residual(optimization.x, weighted=False)
    before_metrics = residual_quantiles(before)
    after_metrics = residual_quantiles(after)
    translation = optimization.x[:3]
    rotation_vector = optimization.x[3:]
    safe = (
        np.linalg.norm(translation) <= translation_bound * 1.001
        and np.linalg.norm(rotation_vector) <= rotation_bound * 1.001
    )
    accepted = safe and after_metrics["p90"] < before_metrics["p90"] * 0.995

    correction_rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    camera_to_world = frame.rotation_world_to_camera.T
    derived_camera_to_world = correction_rotation @ camera_to_world
    derived_center = (
        correction_rotation @ (frame.center_world - pivot)
        + pivot
        + translation
    )
    derived_world_to_camera = derived_camera_to_world.T
    derived_translation = -derived_world_to_camera @ derived_center
    return {
        "camera_index": frame.index,
        "image": frame.image,
        "accepted": bool(accepted),
        "reason": "accepted" if accepted else "quality_or_safety_gate",
        "correspondence_count": int(source.shape[0]),
        "residual_before": before_metrics,
        "residual_after": after_metrics,
        "p90_improvement_ratio": float(
            1.0 - after_metrics["p90"] / before_metrics["p90"]
        ),
        "translation_increment_world": translation.tolist(),
        "translation_increment_norm": float(np.linalg.norm(translation)),
        "rotation_increment_axis_angle": rotation_vector.tolist(),
        "rotation_increment_degrees": float(
            np.rad2deg(np.linalg.norm(rotation_vector))
        ),
        "derived_camera_model": {
            "camera_center": derived_center.tolist(),
            "rotation_world_to_camera": derived_world_to_camera.reshape(-1).tolist(),
            "translation_world_to_camera": derived_translation.tolist(),
        },
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.mvs_manifest.resolve()
    frames = load_frames(manifest_path)
    if len(frames) < 3:
        raise ValueError(f"Need at least three depth frames: {manifest_path}")
    frames_by_index = {frame.index: frame for frame in frames}
    rows = []
    for frame in frames:
        if frame.index == args.anchor_camera:
            rows.append(
                {
                    "camera_index": frame.index,
                    "image": frame.image,
                    "accepted": False,
                    "reason": "anchor_camera",
                }
            )
            continue
        correspondences = collect_correspondences(
            frame,
            frames_by_index,
            max(1, args.stride),
            args.maximum_relative_depth_error,
        )
        rows.append(refine_frame(frame, correspondences, args))

    accepted = [row for row in rows if row.get("accepted")]
    report = {
        "schema": "plascan.depth_pose_alignment_diagnostic.v1",
        "read_only": True,
        "acceptance_scope": "point_to_plane_candidate_only",
        "project_pose_eligible": False,
        "missing_project_acceptance_gates": [
            "sparse_reprojection_rms",
            "silhouette_iou",
            "fixed_view_recall",
        ],
        "mvs_manifest": str(manifest_path),
        "settings": {
            "stride": args.stride,
            "max_samples_per_camera": args.max_samples_per_camera,
            "maximum_relative_depth_error": args.maximum_relative_depth_error,
            "maximum_translation": args.maximum_translation,
            "maximum_rotation_degrees": args.maximum_rotation_degrees,
            "anchor_camera": args.anchor_camera,
        },
        "frame_count": len(frames),
        "point_to_plane_candidate_count": len(accepted),
        "frames": rows,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.corrected_manifest_output is not None:
        corrected_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        corrections_by_index = {
            int(row["camera_index"]): row["derived_camera_model"]
            for row in accepted
        }
        applied_indices: list[int] = []
        for record in corrected_manifest.get("frames", []):
            frame_index = int(record.get("ref_index", -1))
            corrected_camera = corrections_by_index.get(frame_index)
            if corrected_camera is None:
                continue
            camera_model = record.setdefault("camera_model", {})
            camera_model["camera_center"] = corrected_camera["camera_center"]
            camera_model["rotation_world_to_camera"] = corrected_camera[
                "rotation_world_to_camera"
            ]
            camera_model["translation_world_to_camera"] = corrected_camera[
                "translation_world_to_camera"
            ]
            applied_indices.append(frame_index)
        corrected_manifest["depth_pose_alignment_diagnostic"] = {
            "source_report": str(output_path),
            "source_manifest": str(manifest_path),
            "applied_frame_indices": applied_indices,
            "project_pose_modified": False,
        }
        corrected_manifest_path = args.corrected_manifest_output.resolve()
        corrected_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        corrected_manifest_path.write_text(
            json.dumps(corrected_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report["corrected_manifest_output"] = str(corrected_manifest_path)
        report["corrected_manifest_frame_indices"] = applied_indices
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"Pose diagnostic: frames={len(frames)}, "
        f"point-to-plane candidates={len(accepted)}, "
        f"output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
