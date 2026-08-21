"""Strict depth readers used by the ETH3D evaluation CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


FAST_MATRIX_HEADER = struct.Struct("<16siii4xQ")
FAST_MATRIX_MAGIC = b"PLASDEPTHMAT01\x00\x00"
CV_32FC1 = 5
CV_8UC1 = 0


@dataclass(frozen=True)
class LoadedDepth:
    values: np.ndarray
    format: str
    validity: dict[str, int | float]


def read_eth3d_raw_depth(path: Path, width: int, height: int) -> LoadedDepth:
    """Read ETH3D's extension-less raw little-endian float32 depth payload."""

    path = path.resolve()
    _require_dimensions(width, height)
    expected_bytes = width * height * np.dtype("<f4").itemsize
    actual_bytes = _file_size(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"ETH3D depth file size mismatch for {path}: got {actual_bytes} bytes, "
            f"expected {expected_bytes} for {width}x{height} float32"
        )
    values = np.fromfile(path, dtype="<f4", count=width * height)
    values = values.reshape(height, width)
    return LoadedDepth(
        values=values,
        format="eth3d_raw_float32_le",
        validity=depth_validity_summary(values),
    )


def read_prediction_depth(
    path: Path,
    width: int,
    height: int,
    input_format: str = "auto",
) -> LoadedDepth:
    """Read NPY, PlaScan fast-matrix, or headerless float32 predictions."""

    path = path.resolve()
    _require_dimensions(width, height)
    supported_formats = {"auto", "npy", "plascan-fast-matrix", "raw-float32"}
    if input_format not in supported_formats:
        raise ValueError(f"Unsupported prediction format: {input_format}")
    if not path.is_file():
        raise FileNotFoundError(f"Prediction depth not found: {path}")

    resolved_format = input_format
    if resolved_format == "auto":
        if path.suffix.lower() == ".npy":
            resolved_format = "npy"
        elif _has_fast_matrix_magic(path):
            resolved_format = "plascan-fast-matrix"
        else:
            resolved_format = "raw-float32"

    if resolved_format == "npy":
        values = np.load(path, allow_pickle=False)
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError(
                f"Prediction NPY array must contain real numeric values: {path}"
            )
        values = np.asarray(values, dtype=np.float32)
    elif resolved_format == "plascan-fast-matrix":
        values = read_plascan_fast_matrix(path, CV_32FC1)
    else:
        expected_bytes = width * height * np.dtype("<f4").itemsize
        actual_bytes = _file_size(path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Raw prediction file size mismatch for {path}: got "
                f"{actual_bytes} bytes, expected {expected_bytes} for "
                f"{width}x{height} float32"
            )
        values = np.fromfile(path, dtype="<f4", count=width * height).reshape(
            height, width
        )

    expected_shape = (height, width)
    if values.shape != expected_shape:
        raise ValueError(
            f"Prediction shape mismatch for {path}: got {values.shape}, "
            f"expected {expected_shape}"
        )
    return LoadedDepth(
        values=values,
        format=resolved_format,
        validity=depth_validity_summary(values),
    )


def depth_validity_summary(values: np.ndarray) -> dict[str, int | float]:
    values = np.asarray(values)
    positive_finite = np.isfinite(values) & (values > 0.0)
    pixel_count = int(values.size)
    valid_count = int(np.count_nonzero(positive_finite))
    return {
        "pixel_count": pixel_count,
        "valid_positive_finite_count": valid_count,
        "valid_positive_finite_fraction": (
            float(valid_count / pixel_count) if pixel_count else 0.0
        ),
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        "positive_infinity_count": int(np.count_nonzero(np.isposinf(values))),
        "negative_infinity_count": int(np.count_nonzero(np.isneginf(values))),
        "zero_count": int(np.count_nonzero(values == 0.0)),
        "negative_finite_count": int(
            np.count_nonzero(np.isfinite(values) & (values < 0.0))
        ),
    }


def read_plascan_fast_matrix(path: Path, expected_cv_type: int) -> np.ndarray:
    """Read one deterministic PlaScan fast-matrix payload by exact CV type."""

    type_dtypes = {
        CV_8UC1: np.dtype("u1"),
        CV_32FC1: np.dtype("<f4"),
    }
    if expected_cv_type not in type_dtypes:
        raise ValueError(f"Unsupported expected OpenCV matrix type: {expected_cv_type}")
    dtype = type_dtypes[expected_cv_type]
    actual_bytes = _file_size(path)
    if actual_bytes < FAST_MATRIX_HEADER.size:
        raise ValueError(f"Incomplete PlaScan fast-matrix header: {path}")
    with path.open("rb") as stream:
        header = stream.read(FAST_MATRIX_HEADER.size)
        magic, rows, columns, cv_type, payload_bytes = FAST_MATRIX_HEADER.unpack(
            header
        )
        if magic != FAST_MATRIX_MAGIC:
            raise ValueError(f"Unexpected PlaScan fast-matrix magic: {path}")
        if rows <= 0 or columns <= 0:
            raise ValueError(f"Invalid PlaScan fast-matrix dimensions: {path}")
        if cv_type != expected_cv_type:
            raise ValueError(
                f"Expected PlaScan OpenCV type {expected_cv_type}, got "
                f"type {cv_type}: {path}"
            )
        expected_payload_bytes = rows * columns * dtype.itemsize
        if payload_bytes != expected_payload_bytes:
            raise ValueError(
                f"PlaScan fast-matrix payload mismatch for {path}: header "
                f"declares {payload_bytes}, expected {expected_payload_bytes}"
            )
        expected_file_bytes = FAST_MATRIX_HEADER.size + payload_bytes
        if actual_bytes != expected_file_bytes:
            raise ValueError(
                f"PlaScan fast-matrix file size mismatch for {path}: got "
                f"{actual_bytes}, expected {expected_file_bytes}"
            )
        values = np.fromfile(stream, dtype=dtype, count=rows * columns)
    return values.reshape(rows, columns)


def _has_fast_matrix_magic(path: Path) -> bool:
    if _file_size(path) < len(FAST_MATRIX_MAGIC):
        return False
    with path.open("rb") as stream:
        return stream.read(len(FAST_MATRIX_MAGIC)) == FAST_MATRIX_MAGIC


def _file_size(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Depth file not found: {path}")
    return path.stat().st_size


def _require_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Depth dimensions must be positive")
