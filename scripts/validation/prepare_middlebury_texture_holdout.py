#!/usr/bin/env python3
"""Prepare a reproducible Middlebury training/held-out texture benchmark split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--camera-parameters", required=True, type=Path)
    parser.add_argument("--input-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--held-out-indices",
        default="4,8,12,16",
        help="One-based view indices separated by commas.",
    )
    parser.add_argument("--silhouette-threshold", type=float, default=0.19)
    parser.add_argument("--dilate-pixels", type=float, default=10.0)
    parser.add_argument("--erode-pixels", type=float, default=7.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _parse_indices(value: str, view_count: int) -> list[int]:
    try:
        indices = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    except ValueError as exc:
        raise ValueError("--held-out-indices must contain integers") from exc
    if not indices or indices[0] < 1 or indices[-1] > view_count:
        raise ValueError(f"Held-out indices must be within 1..{view_count}")
    if len(indices) >= view_count:
        raise ValueError("At least one training view is required")
    return indices


def _load_cameras(path: Path) -> list[tuple[str, dict[str, object]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Camera parameter file is empty: {path}")
    expected_count = int(lines[0])
    if len(lines) != expected_count + 1:
        raise ValueError(
            f"Camera parameter count says {expected_count}, found {len(lines) - 1}: {path}"
        )
    cameras: list[tuple[str, dict[str, object]]] = []
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split()
        if len(fields) != 22:
            raise ValueError(f"Expected image + 21 camera values at {path}:{line_number}")
        values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite camera value at {path}:{line_number}")
        intrinsics = values[:9].reshape(3, 3)
        rotation = values[9:18].reshape(3, 3)
        translation = values[18:21]
        if (
            abs(intrinsics[0, 1]) > 1.0e-12
            or abs(intrinsics[1, 0]) > 1.0e-12
            or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1.0e-12)
        ):
            raise ValueError(f"Only zero-skew pinhole K is supported at {path}:{line_number}")
        orthogonality_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
        if orthogonality_error > 1.0e-5 or np.linalg.det(rotation) <= 0.0:
            raise ValueError(f"Invalid world-to-camera rotation at {path}:{line_number}")
        center = -(rotation.T @ translation)
        cameras.append(
            (
                fields[0],
                {
                    "fx": float(intrinsics[0, 0]),
                    "fy": float(intrinsics[1, 1]),
                    "cx": float(intrinsics[0, 2]),
                    "cy": float(intrinsics[1, 2]),
                    "rotation_world_to_camera": [float(value) for value in rotation.flat],
                    "camera_center": [float(value) for value in center],
                    "k1": 0.0,
                    "k2": 0.0,
                    "k3": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                },
            )
        )
    return cameras


def _resolve_input_lines(path: Path) -> dict[str, tuple[Path, Path]]:
    resolved: dict[str, tuple[Path, Path]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Expected image and camera path at {path}:{line_number}")
        image = (path.parent / fields[0]).resolve()
        camera = (path.parent / fields[1]).resolve()
        if not image.is_file() or not camera.is_file():
            raise FileNotFoundError(f"Input list entry does not exist at {path}:{line_number}")
        resolved[image.name] = (image, camera)
    return resolved


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _silhouette_mask(
    image_path: Path,
    threshold: float,
    dilation_radius: float,
    erosion_radius: float,
) -> np.ndarray:
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Middlebury image must be 8-bit RGB: {image_path}")
        srgb = np.asarray(image, dtype=np.float32) / 255.0
    luminance = srgb @ np.asarray([0.2989, 0.5870, 0.1140], dtype=np.float32)
    foreground = luminance >= threshold
    if dilation_radius > 0.0:
        foreground = distance_transform_edt(~foreground) <= dilation_radius
    if erosion_radius > 0.0:
        padding = int(np.ceil(erosion_radius)) + 1
        padded = np.pad(foreground, padding, mode="constant", constant_values=False)
        distance_inside = distance_transform_edt(padded)
        foreground = distance_inside[
            padding : padding + foreground.shape[0],
            padding : padding + foreground.shape[1],
        ] > erosion_radius
    return foreground


def main() -> int:
    args = parse_args()
    if not 0.0 < args.silhouette_threshold < 1.0:
        raise ValueError("--silhouette-threshold must be within (0,1)")
    if args.dilate_pixels < 0.0 or args.erode_pixels < 0.0:
        raise ValueError("Morphology radii must be non-negative")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    camera_parameters = args.camera_parameters.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "held_out_masks"
    masks_dir.mkdir(exist_ok=True)

    cameras = _load_cameras(camera_parameters)
    held_out_indices = set(_parse_indices(args.held_out_indices, len(cameras)))
    input_entries = _resolve_input_lines(input_list)
    training_images: list[str] = []
    training_lines: list[str] = []
    views: list[dict[str, object]] = []
    source_hashes = {
        str(camera_parameters): _sha256(camera_parameters),
        str(input_list): _sha256(input_list),
    }
    for index, (image_name, camera_model) in enumerate(cameras, 1):
        image_path = (dataset_dir / image_name).resolve()
        if image_name not in input_entries or input_entries[image_name][0] != image_path:
            raise ValueError(f"Camera/input-list image mismatch: {image_name}")
        listed_image, listed_camera = input_entries[image_name]
        source_hashes[str(listed_image)] = _sha256(listed_image)
        source_hashes[str(listed_camera)] = _sha256(listed_camera)
        if index not in held_out_indices:
            training_images.append(_relative(listed_image, output_dir))
            training_lines.append(
                f"{_relative(listed_image, output_dir)} {_relative(listed_camera, output_dir)}"
            )
            continue
        mask_path = masks_dir / f"{image_path.stem}_foreground.png"
        mask = _silhouette_mask(
            image_path,
            args.silhouette_threshold,
            args.dilate_pixels,
            args.erode_pixels,
        )
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(mask_path)
        views.append(
            {
                "id": image_path.stem,
                "source_id": image_path.stem,
                "image": _relative(image_path, output_dir),
                "mask": _relative(mask_path, output_dir),
                "camera_model": camera_model,
            }
        )

    training_list_path = output_dir / "image_camera_train12.lis"
    training_list_path.write_text("\n".join(training_lines) + "\n", encoding="utf-8")
    manifest_path = output_dir / "texture_holdout_manifest.json"
    manifest = {
        "schema": "plascan.texture_eval.v1",
        "protocol": {
            "name": "middlebury_dino_sparse_ring_12_train_4_holdout_v1",
            "pixel_convention": "Middlebury K[R|t], top-left pixel center (0,0)",
            "held_out_indices_one_based": sorted(held_out_indices),
            "silhouette_source": "Middlebury README recommended threshold/morphology recipe",
            "silhouette_threshold_srgb_luminance": args.silhouette_threshold,
            "silhouette_dilation_euclidean_pixels": args.dilate_pixels,
            "silhouette_erosion_euclidean_pixels": args.erode_pixels,
        },
        "training_images": training_images,
        "views": views,
    }
    _atomic_json(manifest_path, manifest)
    source_hashes[str(training_list_path)] = _sha256(training_list_path)
    source_hashes[str(manifest_path)] = _sha256(manifest_path)
    for mask_path in sorted(masks_dir.glob("*.png")):
        source_hashes[str(mask_path)] = _sha256(mask_path)
    _atomic_json(
        output_dir / "preparation_report.json",
        {
            "schema": "plascan.middlebury_texture_holdout_preparation.v1",
            "training_view_count": len(training_images),
            "held_out_view_count": len(views),
            "training_list": str(training_list_path),
            "texture_manifest": str(manifest_path),
            "sha256": source_hashes,
        },
    )
    print(f"Prepared {len(training_images)} training and {len(views)} held-out views: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
