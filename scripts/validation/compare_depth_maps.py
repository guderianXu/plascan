#!/usr/bin/env python3
"""Compare PlaScan and COLMAP depth maps generated from identical cameras."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


FAST_DEPTH_HEADER = struct.Struct("<16siii4xQ")
FAST_DEPTH_MAGIC = b"PLASDEPTHMAT01\x00\x00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PlaScan MVS depths with a COLMAP stereo workspace by image name."
    )
    parser.add_argument("--mvs-manifest", required=True, type=Path)
    parser.add_argument("--colmap-depth-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--input-type",
        choices=["photometric", "geometric"],
        default="geometric",
    )
    return parser.parse_args()


def read_plascan_depth(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(FAST_DEPTH_HEADER.size)
        if len(header) != FAST_DEPTH_HEADER.size:
            raise ValueError(f"Incomplete PlaScan depth header: {path}")
        magic, rows, cols, cv_type, data_bytes = FAST_DEPTH_HEADER.unpack(header)
        if magic != FAST_DEPTH_MAGIC:
            raise ValueError(f"Unexpected PlaScan depth magic: {path}")
        if cv_type != 5:
            raise ValueError(f"Expected CV_32FC1 (type 5), got type {cv_type}: {path}")
        expected_bytes = rows * cols * np.dtype(np.float32).itemsize
        if data_bytes != expected_bytes:
            raise ValueError(
                f"PlaScan depth payload mismatch in {path}: header={data_bytes}, expected={expected_bytes}"
            )
        values = np.fromfile(stream, dtype="<f4", count=rows * cols)
    return values.reshape(rows, cols)


def read_colmap_depth(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        dimensions: list[int] = []
        for _ in range(3):
            value = bytearray()
            while True:
                character = stream.read(1)
                if not character:
                    raise ValueError(f"Incomplete COLMAP depth header: {path}")
                if character == b"&":
                    break
                value.extend(character)
            dimensions.append(int(value.decode("ascii")))
        width, height, channels = dimensions
        values = np.fromfile(stream, dtype="<f4")
    expected = width * height * channels
    if values.size != expected:
        raise ValueError(f"COLMAP depth payload mismatch in {path}: got={values.size}, expected={expected}")
    matrix = values.reshape((width, height, channels), order="F").transpose(1, 0, 2)
    if channels != 1:
        raise ValueError(f"Expected a single-channel COLMAP depth map: {path}")
    return matrix[:, :, 0]


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {key: None for key in ("p50", "p90", "p95", "p99", "mean", "max")}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def compare_pair(plascan: np.ndarray, colmap: np.ndarray) -> dict[str, Any]:
    if plascan.shape != colmap.shape:
        raise ValueError(f"Depth shape mismatch: PlaScan={plascan.shape}, COLMAP={colmap.shape}")
    plascan_valid = np.isfinite(plascan) & (plascan > 0.0)
    colmap_valid = np.isfinite(colmap) & (colmap > 0.0)
    intersection = plascan_valid & colmap_valid
    union = plascan_valid | colmap_valid
    absolute_error = np.abs(plascan[intersection] - colmap[intersection])
    relative_error = absolute_error / np.maximum(
        np.maximum(plascan[intersection], colmap[intersection]),
        np.finfo(np.float32).eps,
    )
    pixel_count = plascan.size
    return {
        "plascan_valid_fraction": float(np.count_nonzero(plascan_valid) / pixel_count),
        "colmap_valid_fraction": float(np.count_nonzero(colmap_valid) / pixel_count),
        "valid_intersection_fraction": float(np.count_nonzero(intersection) / pixel_count),
        "valid_union_fraction": float(np.count_nonzero(union) / pixel_count),
        "valid_mask_iou": (
            float(np.count_nonzero(intersection) / np.count_nonzero(union))
            if np.any(union)
            else 1.0
        ),
        "plascan_only_fraction": float(np.count_nonzero(plascan_valid & ~colmap_valid) / pixel_count),
        "colmap_only_fraction": float(np.count_nonzero(colmap_valid & ~plascan_valid) / pixel_count),
        "absolute_depth_error": quantiles(absolute_error),
        "relative_depth_error": quantiles(relative_error),
    }


def median_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def main() -> int:
    args = parse_args()
    manifest_path = args.mvs_manifest.resolve()
    colmap_dir = args.colmap_depth_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for frame in manifest.get("frames", []):
        image_name = Path(frame.get("ref_image", "")).name
        plascan_path = Path(frame.get("raw_depth_path", ""))
        colmap_path = colmap_dir / f"{image_name}.{args.input_type}.bin"
        if not image_name or not plascan_path.is_file() or not colmap_path.is_file():
            continue
        row = {
            "image_name": image_name,
            **compare_pair(read_plascan_depth(plascan_path), read_colmap_depth(colmap_path)),
        }
        rows.append(row)
    if not rows:
        raise ValueError(
            f"No matching PlaScan/COLMAP depth pairs found for {manifest_path} and {colmap_dir}"
        )
    rows.sort(key=lambda row: row["image_name"])
    report = {
        "mvs_manifest": str(manifest_path),
        "colmap_depth_dir": str(colmap_dir),
        "input_type": args.input_type,
        "image_count": len(rows),
        "median_plascan_valid_fraction": median_metric(rows, "plascan_valid_fraction"),
        "median_colmap_valid_fraction": median_metric(rows, "colmap_valid_fraction"),
        "median_valid_mask_iou": median_metric(rows, "valid_mask_iou"),
        "median_plascan_only_fraction": median_metric(rows, "plascan_only_fraction"),
        "median_colmap_only_fraction": median_metric(rows, "colmap_only_fraction"),
        "per_image": rows,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Compared {len(rows)} depth pairs: PlaScan valid={report['median_plascan_valid_fraction']:.3f}, "
        f"COLMAP valid={report['median_colmap_valid_fraction']:.3f}, "
        f"mask IoU={report['median_valid_mask_iou']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
