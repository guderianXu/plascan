"""Strict COLMAP image-pose parsing for ETH3D frame identity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ColmapImagePose:
    image_id: int
    camera_id: int
    name: str
    rotation_world_to_camera: tuple[float, ...]
    translation_world_to_camera: tuple[float, ...]
    camera_center: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "camera_id": self.camera_id,
            "name": self.name,
            "rotation_world_to_camera": list(self.rotation_world_to_camera),
            "translation_world_to_camera": list(
                self.translation_world_to_camera
            ),
            "camera_center": list(self.camera_center),
        }


def read_colmap_image_poses(path: Path) -> list[ColmapImagePose]:
    """Read image header records while consuming each POINTS2D line exactly."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COLMAP images file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    poses: list[ColmapImagePose] = []
    image_ids: set[int] = set()
    index = 0
    while index < len(lines):
        header = lines[index].strip()
        index += 1
        if not header or header.startswith("#"):
            continue
        if index >= len(lines):
            raise ValueError(f"COLMAP image lacks POINTS2D line: {path}")
        index += 1
        fields = header.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(f"Malformed COLMAP image header: {header!r}")
        try:
            image_id = int(fields[0])
            quaternion = np.asarray(
                [float(value) for value in fields[1:5]], dtype=np.float64
            )
            translation = np.asarray(
                [float(value) for value in fields[5:8]], dtype=np.float64
            )
            camera_id = int(fields[8])
        except ValueError as error:
            raise ValueError(
                f"Malformed COLMAP image header: {header!r}: {error}"
            ) from error
        if image_id < 0 or camera_id < 0 or image_id in image_ids:
            raise ValueError(f"Invalid or duplicate COLMAP image id: {header!r}")
        name = fields[9].strip()
        if not name:
            raise ValueError(f"COLMAP image name is empty: {header!r}")
        rotation = _quaternion_to_rotation(quaternion)
        center = -(rotation.T @ translation)
        poses.append(
            ColmapImagePose(
                image_id=image_id,
                camera_id=camera_id,
                name=name,
                rotation_world_to_camera=tuple(float(item) for item in rotation.flat),
                translation_world_to_camera=tuple(float(item) for item in translation),
                camera_center=tuple(float(item) for item in center),
            )
        )
        image_ids.add(image_id)
    if not poses:
        raise ValueError(f"No COLMAP image records found: {path}")
    return poses


def select_colmap_image_pose(path: Path, image_name: str) -> ColmapImagePose:
    """Select one pose by exact normalized basename and reject ambiguity."""

    basename = Path(image_name).name
    matches = [
        pose for pose in read_colmap_image_poses(path)
        if Path(pose.name).name == basename
    ]
    if not matches:
        raise ValueError(f"Image {basename!r} is not present in {path}")
    if len(matches) != 1:
        raise ValueError(
            f"Image basename {basename!r} occurs {len(matches)} times in {path}"
        )
    return matches[0]


def validate_manifest_pose(
    official_pose: ColmapImagePose,
    rotation_world_to_camera: tuple[float, ...],
    translation_world_to_camera: tuple[float, ...],
    camera_center: tuple[float, ...],
    *,
    maximum_absolute_residual: float = 1.0e-5,
) -> dict[str, object]:
    """Require manifest and official fixed-camera extrinsics to agree."""

    if not np.isfinite(maximum_absolute_residual) or maximum_absolute_residual <= 0:
        raise ValueError("maximum_absolute_residual must be finite and positive")
    comparisons = {
        "rotation_world_to_camera": (
            np.asarray(rotation_world_to_camera, dtype=np.float64),
            np.asarray(official_pose.rotation_world_to_camera, dtype=np.float64),
        ),
        "translation_world_to_camera": (
            np.asarray(translation_world_to_camera, dtype=np.float64),
            np.asarray(official_pose.translation_world_to_camera, dtype=np.float64),
        ),
        "camera_center": (
            np.asarray(camera_center, dtype=np.float64),
            np.asarray(official_pose.camera_center, dtype=np.float64),
        ),
    }
    residuals = {
        name: float(np.max(np.abs(actual - expected)))
        for name, (actual, expected) in comparisons.items()
    }
    maximum = max(residuals.values())
    diagnostics = {
        "maximum_absolute_component_residual": maximum,
        "maximum_allowed_absolute_component_residual": (
            maximum_absolute_residual
        ),
        "component_residuals": residuals,
    }
    if maximum > maximum_absolute_residual:
        raise ValueError(
            "Prediction manifest pose does not match the official fixed "
            f"camera pose: maximum residual={maximum:.9g}, allowed="
            f"{maximum_absolute_residual:.9g}; residuals={residuals}"
        )
    return diagnostics


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("COLMAP quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("COLMAP quaternion norm must be positive")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
