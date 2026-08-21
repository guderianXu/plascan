"""Strict color-managed input loading for formal texture evaluation."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .model import srgb_to_linear


@dataclass(frozen=True)
class LinearReference:
    linear_rgb: np.ndarray
    original_size: tuple[int, int]
    scale: tuple[float, float]
    decoded_rgb_sha256: str


def _validate_common_image_metadata(image: Image.Image, path: Path) -> None:
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError(f"Formal evaluation requires a single-frame image: {path}")
    orientation = int(image.getexif().get(274, 1))
    if orientation != 1:
        raise ValueError(
            f"Formal evaluation requires pixels already normalized to EXIF orientation 1: {path}"
        )
    if image.info.get("icc_profile"):
        raise ValueError(
            f"Formal evaluation requires untagged 8-bit sRGB input; remove the ICC profile: {path}"
        )


def load_strict_srgb8(path: Path) -> np.ndarray:
    """Load an unambiguous 8-bit RGB image without implicit color/orientation transforms."""
    with Image.open(path) as image:
        _validate_common_image_metadata(image, path)
        if image.mode != "RGB":
            raise ValueError(
                f"Formal evaluation requires an 8-bit RGB image (mode RGB), got {image.mode}: {path}"
            )
        pixels = np.asarray(image, dtype=np.uint8).copy()
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"Decoded image is not HxWx3 RGB: {path}")
    return pixels


def decoded_rgb_sha256(path: Path) -> str:
    pixels = load_strict_srgb8(path)
    digest = hashlib.sha256()
    digest.update(b"plascan.decoded-srgb8.v1\0")
    digest.update(struct.pack("<II", pixels.shape[1], pixels.shape[0]))
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _resize_linear_rgb(linear_rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    channels: list[np.ndarray] = []
    for channel in range(3):
        image = Image.fromarray(linear_rgb[:, :, channel], mode="F")
        resized = image.resize(target_size, Image.Resampling.LANCZOS)
        channels.append(np.asarray(resized, dtype=np.float32))
    return np.clip(np.stack(channels, axis=2), 0.0, 1.0)


def load_linear_reference(path: Path, maximum_dimension: int) -> LinearReference:
    pixels = load_strict_srgb8(path)
    original_size = (pixels.shape[1], pixels.shape[0])
    linear_rgb = srgb_to_linear(pixels.astype(np.float32) / 255.0)
    if maximum_dimension > 0 and max(original_size) > maximum_dimension:
        scale = maximum_dimension / max(original_size)
        target_size = (
            max(1, round(original_size[0] * scale)),
            max(1, round(original_size[1] * scale)),
        )
        linear_rgb = _resize_linear_rgb(linear_rgb, target_size)
    evaluated_size = (linear_rgb.shape[1], linear_rgb.shape[0])
    scale = (evaluated_size[0] / original_size[0], evaluated_size[1] / original_size[1])
    return LinearReference(linear_rgb, original_size, scale, decoded_rgb_sha256(path))


def load_binary_mask(
    path: Path,
    original_size: tuple[int, int],
    evaluated_size: tuple[int, int],
) -> np.ndarray:
    with Image.open(path) as image:
        _validate_common_image_metadata(image, path)
        if image.mode not in {"1", "L"}:
            raise ValueError(f"Evaluation mask must be a single-channel 1-bit or 8-bit image: {path}")
        if image.size != original_size:
            raise ValueError(
                f"Evaluation mask size {image.size} does not match held-out image size "
                f"{original_size}: {path}"
            )
        mask = image.convert("L")
        if mask.size != evaluated_size:
            mask = mask.resize(evaluated_size, Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 127
