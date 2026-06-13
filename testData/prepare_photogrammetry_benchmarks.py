#!/usr/bin/env python3
"""Prepare downloaded photogrammetry benchmark datasets for PlaScan.

The downloader keeps original benchmark data unchanged. This script creates a
PlaScan-ready view beside each dataset:

    testData/photogrammetry_benchmarks/<dataset>/prepared/plascan/
        image_camera.lis
        cameras/*.tsai
        summary.json

Supported input camera formats:
  - Middlebury sparse-ring *_par.txt files: projection K * [R t]
  - EPFL/Strecha *.camera files: projection K * [R^T | -R^T t]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_ROOT = REPO_ROOT / "testData" / "photogrammetry_benchmarks"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    input_format: str
    source_subdir: Path


@dataclass
class CameraRecord:
    image_name: str
    K: list[list[float]]
    rotation_camera_to_world: list[list[float]]
    center: list[float]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreparationResult:
    dataset_id: str
    output_dir: Path
    image_camera_list: Path
    camera_count: int


SUPPORTED_DATASETS: dict[str, DatasetSpec] = {
    "middlebury_dino_sparse_ring": DatasetSpec(
        dataset_id="middlebury_dino_sparse_ring",
        input_format="middlebury_par",
        source_subdir=Path("extracted") / "dinoSparseRing",
    ),
    "middlebury_temple_sparse_ring": DatasetSpec(
        dataset_id="middlebury_temple_sparse_ring",
        input_format="middlebury_par",
        source_subdir=Path("extracted") / "templeSparseRing",
    ),
    "epfl_rathaus_multiview": DatasetSpec(
        dataset_id="epfl_rathaus_multiview",
        input_format="epfl_camera",
        source_subdir=Path("extracted"),
    ),
}


def format_number(value: float) -> str:
    if abs(value) < 1e-14:
        value = 0.0
    return f"{value:.12g}"


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for row in range(3)] for col in range(3)]


def mat_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(matrix[row][col] * vector[col] for col in range(3))
        for row in range(3)
    ]


def parse_float_line(line: str, expected: int | None = None) -> list[float]:
    values = [float(token) for token in line.split()]
    if expected is not None and len(values) != expected:
        raise ValueError(f"expected {expected} numeric values, got {len(values)}: {line}")
    return values


def warn_if_unsupported_skew(K: list[list[float]], warnings: list[str]) -> None:
    skew_x = K[0][1]
    skew_y = K[1][0]
    if abs(skew_x) > 1e-12 or abs(skew_y) > 1e-12:
        warnings.append(
            "camera skew terms are not represented in PlaScan tsai output "
            f"(k01={format_number(skew_x)}, k10={format_number(skew_y)})"
        )


def parse_middlebury_line(line: str) -> CameraRecord:
    parts = line.split()
    if len(parts) != 22:
        raise ValueError(f"Middlebury camera line must contain image name plus 21 values: {line}")

    image_name = parts[0]
    values = [float(token) for token in parts[1:]]
    K = [values[0:3], values[3:6], values[6:9]]
    rotation_world_to_camera = [values[9:12], values[12:15], values[15:18]]
    translation_world_to_camera = values[18:21]

    rotation_camera_to_world = transpose(rotation_world_to_camera)
    center_raw = mat_vec_mul(rotation_camera_to_world, translation_world_to_camera)
    center = [-value for value in center_raw]

    warnings: list[str] = []
    warn_if_unsupported_skew(K, warnings)
    return CameraRecord(
        image_name=image_name,
        K=K,
        rotation_camera_to_world=rotation_camera_to_world,
        center=center,
        warnings=warnings,
    )


def parse_middlebury_par(path: Path) -> list[CameraRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Middlebury par file not found: {path}")

    data_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        data_lines.append(line)

    if data_lines and len(data_lines[0].split()) == 1:
        try:
            int(data_lines[0])
            data_lines = data_lines[1:]
        except ValueError:
            pass

    if not data_lines:
        raise ValueError(f"Middlebury par file contains no cameras: {path}")

    return [parse_middlebury_line(line) for line in data_lines]


def parse_epfl_camera_file(path: Path, image_name: str | None = None) -> CameraRecord:
    if not path.exists():
        raise FileNotFoundError(f"EPFL camera file not found: {path}")

    data_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(data_lines) < 8:
        raise ValueError(f"EPFL camera file must contain at least 8 numeric lines: {path}")

    K = [
        parse_float_line(data_lines[0], 3),
        parse_float_line(data_lines[1], 3),
        parse_float_line(data_lines[2], 3),
    ]
    # Line 4 contains radial distortion parameters. Current PlaScan tsai output
    # keeps distortion at zero because the Strecha README says zeros indicate
    # corrected images and the local files use zero values.
    parse_float_line(data_lines[3], 3)
    rotation_camera_to_world = [
        parse_float_line(data_lines[4], 3),
        parse_float_line(data_lines[5], 3),
        parse_float_line(data_lines[6], 3),
    ]
    center = parse_float_line(data_lines[7], 3)

    warnings: list[str] = []
    warn_if_unsupported_skew(K, warnings)
    resolved_image_name = image_name or path.with_suffix("").name
    return CameraRecord(
        image_name=resolved_image_name,
        K=K,
        rotation_camera_to_world=rotation_camera_to_world,
        center=center,
        warnings=warnings,
    )


def find_single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} under {directory}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple files matching {pattern!r} under {directory}: {matches}")
    return matches[0]


def load_dataset_cameras(spec: DatasetSpec, source_dir: Path) -> list[CameraRecord]:
    if not source_dir.exists():
        raise FileNotFoundError(f"dataset source directory not found: {source_dir}")

    if spec.input_format == "middlebury_par":
        return parse_middlebury_par(find_single_file(source_dir, "*_par.txt"))

    if spec.input_format == "epfl_camera":
        camera_paths = sorted(source_dir.glob("*.camera"))
        if not camera_paths:
            raise FileNotFoundError(f"no EPFL .camera files under {source_dir}")
        return [parse_epfl_camera_file(path) for path in camera_paths]

    raise RuntimeError(f"unsupported input format: {spec.input_format}")


def write_tsai(path: Path, camera: CameraRecord) -> None:
    K = camera.K
    R = camera.rotation_camera_to_world
    C = camera.center
    lines = [
        "VERSION_3",
        "PINHOLE",
        "TSAI",
        f"fu = {format_number(K[0][0])}",
        f"fv = {format_number(K[1][1])}",
        f"cu = {format_number(K[0][2])}",
        f"cv = {format_number(K[1][2])}",
        "u_direction = 1 0 0",
        "v_direction = 0 1 0",
        "w_direction = 0 0 1",
        "pitch = 1",
        "k1 = 0",
        "k2 = 0",
        "k3 = 0",
        "p1 = 0",
        "p2 = 0",
        "C = " + " ".join(format_number(value) for value in C),
        "R = " + " ".join(format_number(value) for row in R for value in row),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_token(path: Path, base_dir: Path) -> str:
    rel = os.path.relpath(path, base_dir)
    return rel.replace(os.sep, "/")


def write_image_camera_list(
    list_path: Path,
    source_dir: Path,
    cameras_dir: Path,
    records: Iterable[CameraRecord],
) -> None:
    list_dir = list_path.parent
    lines: list[str] = []
    for record in records:
        image_path = source_dir / record.image_name
        tsai_path = cameras_dir / f"{Path(record.image_name).stem}.tsai"
        if not image_path.exists():
            raise FileNotFoundError(f"image referenced by camera file does not exist: {image_path}")
        write_tsai(tsai_path, record)
        image_token = shlex.quote(relative_token(image_path, list_dir))
        camera_token = shlex.quote(relative_token(tsai_path, list_dir))
        lines.append(f"{image_token} {camera_token}")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(target_root: Path, dataset_id: str, overwrite: bool = False) -> PreparationResult:
    if dataset_id not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"unsupported dataset id: {dataset_id}. Supported: {supported}")

    spec = SUPPORTED_DATASETS[dataset_id]
    dataset_dir = target_root / dataset_id
    source_dir = dataset_dir / spec.source_subdir
    output_dir = dataset_dir / "prepared" / "plascan"
    cameras_dir = output_dir / "cameras"
    list_path = output_dir / "image_camera.lis"

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"prepared output already exists, pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cameras_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset_cameras(spec, source_dir)
    write_image_camera_list(list_path, source_dir, cameras_dir, records)

    warnings: list[str] = []
    for record in records:
        warnings.extend(f"{record.image_name}: {warning}" for warning in record.warnings)

    summary = {
        "dataset_id": dataset_id,
        "input_format": spec.input_format,
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "image_camera_list": str(list_path.resolve()),
        "camera_count": len(records),
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return PreparationResult(
        dataset_id=dataset_id,
        output_dir=output_dir,
        image_camera_list=list_path,
        camera_count=len(records),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert supported downloaded benchmark camera files to PlaScan tsai inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT, help="benchmark root directory")
    parser.add_argument("--dataset", action="append", default=[], help="dataset id to prepare; repeatable")
    parser.add_argument("--all", action="store_true", help="prepare all supported datasets")
    parser.add_argument("--list", action="store_true", help="list supported datasets and exit")
    parser.add_argument("--overwrite", action="store_true", help="overwrite generated prepared/plascan output")
    return parser.parse_args(argv)


def selected_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.list:
        return []
    if args.all:
        return list(SUPPORTED_DATASETS.keys())
    if args.dataset:
        unknown = sorted(set(args.dataset).difference(SUPPORTED_DATASETS.keys()))
        if unknown:
            supported = ", ".join(sorted(SUPPORTED_DATASETS))
            raise ValueError(f"unsupported dataset id(s): {', '.join(unknown)}. Supported: {supported}")
        return args.dataset
    return list(SUPPORTED_DATASETS.keys())


def list_supported() -> None:
    print(f"target_root: {DEFAULT_TARGET_ROOT}")
    print("supported prepared datasets:")
    for spec in SUPPORTED_DATASETS.values():
        print(f"  {spec.dataset_id:32s} {spec.input_format:16s} {spec.source_subdir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        list_supported()
        return 0

    try:
        for dataset_id in selected_dataset_ids(args):
            result = prepare_dataset(args.target_root, dataset_id, overwrite=args.overwrite)
            print(
                f"prepared: {result.dataset_id} "
                f"({result.camera_count} cameras) -> {result.image_camera_list}",
                flush=True,
            )
    except Exception as exc:
        print(f"prepare failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
