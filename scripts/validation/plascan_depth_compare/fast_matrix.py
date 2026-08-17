"""PlaScan fast single-channel matrix reader."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


FAST_MATRIX_HEADER = struct.Struct("<16siii4xQ")
FAST_MATRIX_MAGIC = b"PLASDEPTHMAT01\x00\x00"
CV_16UC1 = 2
CV_32FC1 = 5
MATRIX_DTYPES = {
    CV_16UC1: np.dtype("<u2"),
    CV_32FC1: np.dtype("<f4"),
}


def read_fast_matrix(path: Path, expected_type: int | None = None) -> np.ndarray:
    """Read a supported PlaScan matrix as a read-only memory map.

    The 40-byte header layout matches ``FastDepthMatHeader`` on the supported
    MSVC/GCC 64-bit builds: 16-byte magic, three little-endian int32 values,
    four alignment bytes, and a little-endian uint64 payload length.
    """
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(FAST_MATRIX_HEADER.size)
    if len(header) != FAST_MATRIX_HEADER.size:
        raise ValueError(f"Incomplete PlaScan matrix header: {path}")

    magic, rows, columns, cv_type, payload_size = FAST_MATRIX_HEADER.unpack(header)
    if magic != FAST_MATRIX_MAGIC:
        raise ValueError(f"Unexpected PlaScan matrix magic: {path}")
    if rows <= 0 or columns <= 0:
        raise ValueError(f"Invalid PlaScan matrix dimensions {rows}x{columns}: {path}")
    if cv_type not in MATRIX_DTYPES:
        raise ValueError(
            f"Unsupported PlaScan matrix type {cv_type}; "
            f"expected CV_16UC1 or CV_32FC1: {path}"
        )
    if expected_type is not None and cv_type != expected_type:
        raise ValueError(
            f"Unexpected PlaScan matrix type {cv_type}; expected {expected_type}: {path}"
        )

    dtype = MATRIX_DTYPES[cv_type]
    expected_payload_size = rows * columns * dtype.itemsize
    if payload_size != expected_payload_size:
        raise ValueError(
            f"PlaScan matrix payload mismatch in {path}: "
            f"header={payload_size}, expected={expected_payload_size}"
        )
    expected_file_size = FAST_MATRIX_HEADER.size + expected_payload_size
    actual_file_size = path.stat().st_size
    if actual_file_size != expected_file_size:
        raise ValueError(
            f"PlaScan matrix file size mismatch in {path}: "
            f"actual={actual_file_size}, expected={expected_file_size}"
        )
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=FAST_MATRIX_HEADER.size,
        shape=(rows, columns),
        order="C",
    )
