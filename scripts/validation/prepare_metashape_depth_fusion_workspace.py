#!/usr/bin/env python3
"""Prepare a PlaScan fusion workspace from exported Metashape depth maps.

This is an isolation tool: it preserves the PlaScan camera models from a known
MVS manifest while replacing only depth and per-pixel evidence. The
``recomputed`` mode derives support by reprojection into every other exported
depth map. The ``trusted`` mode keeps the same measured support but raises the
minimum support of valid pixels to two, providing an explicit mesher upper
bound rather than pretending that synthetic evidence is measured evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import struct

import cv2
import numpy as np


FAST_MATRIX_HEADER = struct.Struct("<16siii4xQ")
FAST_MATRIX_MAGIC = b"PLASDEPTHMAT01\x00\x00"
CV_16UC1 = 2
CV_32FC1 = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metashape-export", required=True, type=Path)
    parser.add_argument("--camera-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("recomputed", "trusted"), default="recomputed"
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.002)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0005)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Scale Metashape camera-Z depths into the PlaScan camera scale.",
    )
    return parser.parse_args()


def write_fast_matrix(path: Path, values: np.ndarray, cv_type: int) -> None:
    contiguous = np.ascontiguousarray(values)
    payload = contiguous.tobytes(order="C")
    with path.open("wb") as stream:
        stream.write(
            FAST_MATRIX_HEADER.pack(
                FAST_MATRIX_MAGIC,
                contiguous.shape[0],
                contiguous.shape[1],
                cv_type,
                len(payload),
            )
        )
        stream.write(payload)


def camera_arrays(frame: dict[str, object]) -> tuple[np.ndarray, ...]:
    camera = frame["camera_model"]
    rotation = np.asarray(camera["rotation_world_to_camera"], dtype=np.float64)
    rotation = rotation.reshape(3, 3)
    translation = np.asarray(camera["translation_world_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(
        [camera["fx"], camera["fy"], camera["cx"], camera["cy"]],
        dtype=np.float64,
    )
    return rotation, translation, intrinsics


def project_reference_depth(
    reference_depth: np.ndarray,
    reference_frame: dict[str, object],
    source_depth: np.ndarray,
    source_frame: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(np.isfinite(reference_depth) & (reference_depth > 0))
    depths = reference_depth[rows, columns].astype(np.float64)
    ref_rotation, ref_translation, ref_intrinsics = camera_arrays(reference_frame)
    fx, fy, cx, cy = ref_intrinsics
    rays = np.stack(
        ((columns + 0.5 - cx) / fx, (rows + 0.5 - cy) / fy, np.ones_like(depths)),
        axis=1,
    )
    world = (
        ref_rotation.T @ (rays * depths[:, None] - ref_translation).T
    ).T

    src_rotation, src_translation, src_intrinsics = camera_arrays(source_frame)
    fx, fy, cx, cy = src_intrinsics
    source_camera = (src_rotation @ world.T).T + src_translation
    projected_columns = np.rint(
        fx * source_camera[:, 0] / source_camera[:, 2] + cx - 0.5
    ).astype(np.int64)
    projected_rows = np.rint(
        fy * source_camera[:, 1] / source_camera[:, 2] + cy - 0.5
    ).astype(np.int64)
    inside = (
        (source_camera[:, 2] > 0)
        & (projected_columns >= 0)
        & (projected_columns < source_depth.shape[1])
        & (projected_rows >= 0)
        & (projected_rows < source_depth.shape[0])
    )
    observed = np.zeros_like(depths)
    observed[inside] = source_depth[
        projected_rows[inside], projected_columns[inside]
    ]
    inside &= np.isfinite(observed) & (observed > 0)
    return rows, columns, inside, source_camera[:, 2], observed


def compute_evidence(
    reference_index: int,
    depths: list[np.ndarray],
    frames: list[dict[str, object]],
    source_bit_indices: list[int],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = depths[reference_index]
    support = np.zeros(reference.shape, dtype=np.uint16)
    source_mask = np.zeros(reference.shape, dtype=np.uint16)
    for source_index, source in enumerate(depths):
        if source_index == reference_index:
            continue
        rows, columns, valid, predicted, observed = project_reference_depth(
            reference,
            frames[reference_index],
            source,
            frames[source_index],
        )
        tolerance = np.maximum(
            absolute_tolerance,
            relative_tolerance * np.maximum(predicted, observed),
        )
        agrees = valid & (np.abs(predicted - observed) <= tolerance)
        if not np.any(agrees):
            continue
        accepted_rows = rows[agrees]
        accepted_columns = columns[agrees]
        support[accepted_rows, accepted_columns] += 1
        source_mask[accepted_rows, accepted_columns] |= np.uint16(
            1 << source_bit_indices[source_index]
        )
    return support, source_mask


def write_preview(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.quantile(depth[valid], (0.01, 0.99))
        scale = np.clip((depth - low) / max(high - low, np.finfo(np.float32).eps), 0, 1)
        preview[valid] = np.asarray(scale[valid] * 255, dtype=np.uint8)
    color = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    cv2.imwrite(str(path), color)


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        raise ValueError("--depth-scale must be finite and greater than zero")
    export_dir = args.metashape_export.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = json.loads(
        (export_dir / "metashape_depth_manifest.json").read_text(encoding="utf-8")
    )
    source_manifest_path = args.camera_manifest.resolve()
    source_manifest_root = source_manifest_path.parent
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    exported_maps = exported["depth_maps"]
    source_frames = sorted(
        source_manifest["frames"], key=lambda item: item["ref_index"]
    )

    exported_by_label = {
        str(item["label"]).casefold(): item for item in exported_maps
    }
    if len(exported_by_label) != len(exported_maps):
        raise ValueError("Metashape export contains duplicate camera labels")

    ordered_exported_maps: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    depths: list[np.ndarray] = []
    missing_frames: list[dict[str, object]] = []
    for frame in source_frames:
        image_label = Path(str(frame["ref_image"])).stem.casefold()
        exported_map = exported_by_label.get(image_label)
        if exported_map is None:
            missing_frames.append(frame)
            print(
                f"skipped frame={frame['ref_index']} label={image_label}: "
                "Metashape has no depth map"
            )
            continue
        frames.append(frame)
        ordered_exported_maps.append(exported_map)
        depth = cv2.imread(
            str(export_dir / exported_map["depth_path"]), cv2.IMREAD_UNCHANGED
        )
        if depth is None or depth.dtype != np.float32 or depth.ndim != 2:
            raise ValueError(
                f"Expected one-channel float depth for camera {exported_map['camera_index']}"
            )
        depths.append(np.asarray(depth * args.depth_scale, dtype=np.float32))
    if len(frames) < 3:
        raise ValueError(
            f"Metashape export has only {len(frames)} matched depth maps; "
            "at least 3 are required for fusion"
        )

    source_bit_indices = [int(frame["ref_index"]) for frame in frames]

    manifest = copy.deepcopy(source_manifest)
    manifest["algorithm_revision"] = 12
    manifest["external_depth_provenance"] = {
        "kind": "metashape_reference_depth",
        "source_manifest": str(export_dir / "metashape_depth_manifest.json"),
        "evidence_mode": args.mode,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "depth_scale": args.depth_scale,
        "exported_depth_count": len(frames),
        "source_camera_count": len(source_frames),
        "missing_ref_indices": [
            int(frame["ref_index"]) for frame in missing_frames
        ],
    }
    output_frames: list[dict[str, object]] = []
    for index, (depth, frame, exported_map) in enumerate(
        zip(depths, frames, ordered_exported_maps, strict=True)
    ):
        valid = np.isfinite(depth) & (depth > 0)
        clean_depth = np.where(valid, depth, 0).astype(np.float32)
        support, source_mask = compute_evidence(
            index,
            depths,
            frames,
            source_bit_indices,
            args.relative_tolerance,
            args.absolute_tolerance,
        )
        measured_support = support.copy()
        if args.mode == "trusted":
            support[valid] = np.maximum(support[valid], 2)
        confidence = np.where(valid, 1.0, 0.0).astype(np.float32)
        inverse_mean = np.zeros(depth.shape, dtype=np.float32)
        inverse_mean[valid] = 1.0 / clean_depth[valid]
        inverse_spread = np.zeros(depth.shape, dtype=np.float32)
        mask = np.asarray(valid, dtype=np.uint8) * 255
        repaired_mask = np.zeros(depth.shape, dtype=np.uint8)

        ref_index = int(frame["ref_index"])
        prefix = output_dir / f"depth_{ref_index}"
        depth_path = prefix.with_suffix(".bin")
        confidence_path = output_dir / f"depth_{ref_index}_conf.bin"
        support_path = output_dir / f"depth_{ref_index}_geometry_support.bin"
        source_mask_path = output_dir / f"depth_{ref_index}_geometry_source_mask.bin"
        inverse_mean_path = output_dir / f"depth_{ref_index}_inverse_depth_mean.bin"
        inverse_spread_path = output_dir / f"depth_{ref_index}_inverse_depth_spread.bin"
        valid_mask_path = output_dir / f"depth_{ref_index}_mask.png"
        repaired_mask_path = (
            output_dir / f"depth_{ref_index}_cross_view_repaired_mask.png"
        )
        preview_path = output_dir / f"depth_{ref_index}.png"
        source_support_mask = Path(str(frame.get("support_mask_path", "")))
        if not source_support_mask.is_absolute():
            source_support_mask = source_manifest_root / source_support_mask
        effective_support_mask = (
            source_support_mask.resolve()
            if source_support_mask.is_file()
            else valid_mask_path
        )

        write_fast_matrix(depth_path, clean_depth, CV_32FC1)
        write_fast_matrix(confidence_path, confidence, CV_32FC1)
        write_fast_matrix(support_path, support, CV_16UC1)
        write_fast_matrix(source_mask_path, source_mask, CV_16UC1)
        write_fast_matrix(inverse_mean_path, inverse_mean, CV_32FC1)
        write_fast_matrix(inverse_spread_path, inverse_spread, CV_32FC1)
        cv2.imwrite(str(valid_mask_path), mask)
        cv2.imwrite(str(repaired_mask_path), repaired_mask)
        write_preview(preview_path, clean_depth)

        updated = copy.deepcopy(frame)
        updated.update(
            {
                "algorithm_revision": 12,
                "acceptance": "accepted",
                "status": "completed",
                "error": "",
                "raw_depth_path": str(depth_path),
                "raw_confidence_path": str(confidence_path),
                "raw_geometry_support_path": str(support_path),
                "raw_geometry_source_mask_path": str(source_mask_path),
                "raw_inverse_depth_mean_path": str(inverse_mean_path),
                "raw_inverse_depth_spread_path": str(inverse_spread_path),
                "valid_mask_path": str(valid_mask_path),
                "support_mask_path": str(effective_support_mask),
                "cross_view_repaired_mask_path": str(repaired_mask_path),
                "depth_png": str(preview_path),
                "valid_pixel_count": int(np.count_nonzero(valid)),
                "valid_coverage": float(np.mean(valid)),
                "depth_confidence_mean": 1.0,
                "external_depth_provenance": {
                    "kind": "metashape_reference_depth",
                    "camera_index": int(exported_map["camera_index"]),
                    "camera_label": str(exported_map["label"]),
                    "evidence_mode": args.mode,
                    "depth_scale": args.depth_scale,
                    "support_mask_kind": (
                        "source_object_support_mask"
                        if source_support_mask.is_file()
                        else "depth_valid_fallback"
                    ),
                    "measured_support_ge_1_fraction": float(
                        np.mean(measured_support[valid] >= 1)
                    ),
                    "measured_support_ge_2_fraction": float(
                        np.mean(measured_support[valid] >= 2)
                    ),
                    "measured_support_median": float(np.median(measured_support[valid])),
                },
            }
        )
        output_frames.append(updated)
        print(
            f"frame={ref_index} valid={np.mean(valid):.6f} "
            f"support>=1={np.mean(measured_support[valid] >= 1):.6f} "
            f"support>=2={np.mean(measured_support[valid] >= 2):.6f} "
            f"support_median={np.median(measured_support[valid]):.1f}"
        )

    manifest["frames"] = output_frames
    manifest_path = output_dir / "mvs_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
