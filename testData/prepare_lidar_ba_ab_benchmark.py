#!/usr/bin/env python3
"""Prepare a small real-data A/B benchmark plan for LiDAR-constrained BA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    REPO_ROOT
    / "testData"
    / "photogrammetry_benchmarks"
    / "mun_frl_vil"
    / "lighthouse_benchmarking_bag"
    / "extracted"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "mun_frl_lidar_ba_ab_benchmark"


@dataclass(frozen=True)
class AssociationRow:
    image_index: int
    image_stamp_ns: int
    image_path: str
    lidar_index: int
    lidar_stamp_ns: int
    lidar_path: str
    dt_ms: float
    lidar_points: int

    @property
    def abs_dt_ms(self) -> float:
        return abs(self.dt_ms)


@dataclass(frozen=True)
class PlyMergeInfo:
    path: Path
    file_format: str
    vertex_count: int
    header_size_bytes: int
    property_lines: tuple[str, ...]
    fields: tuple[str, ...]
    vertex_stride_bytes: int

    @property
    def ba_ready(self) -> bool:
        field_set = set(self.fields)
        has_normals = {"normal_x", "normal_y", "normal_z"}.issubset(field_set) or {"nx", "ny", "nz"}.issubset(field_set)
        return {"x", "y", "z"}.issubset(field_set) and has_normals


PLY_SCALAR_SIZES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


def read_associations(path: Path) -> list[AssociationRow]:
    rows: list[AssociationRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_index",
            "image_stamp_ns",
            "image_path",
            "lidar_index",
            "lidar_stamp_ns",
            "lidar_path",
            "dt_ms",
            "lidar_points",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")

        for item in reader:
            rows.append(
                AssociationRow(
                    image_index=int(item["image_index"]),
                    image_stamp_ns=int(item["image_stamp_ns"]),
                    image_path=item["image_path"],
                    lidar_index=int(item["lidar_index"]),
                    lidar_stamp_ns=int(item["lidar_stamp_ns"]),
                    lidar_path=item["lidar_path"],
                    dt_ms=float(item["dt_ms"]),
                    lidar_points=int(item["lidar_points"]),
                )
            )
    return rows


def parse_ply_merge_info(path: Path) -> PlyMergeInfo:
    file_format = ""
    vertex_count = 0
    property_lines: list[str] = []
    fields: list[str] = []
    elements: list[str] = []
    in_vertex = False
    header_size = 0

    with path.open("rb") as handle:
        first_raw = handle.readline()
        header_size += len(first_raw)
        first = first_raw.decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"{path} has no end_header")
            header_size += len(raw_line)
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                file_format = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                elements.append(parts[1])
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex and len(parts) >= 3:
                if parts[1] == "list":
                    raise ValueError(f"{path} has list vertex properties; cannot merge safely")
                property_lines.append(line)
                fields.append(parts[2])

    if elements != ["vertex"]:
        raise ValueError(f"{path} has unsupported PLY elements: {', '.join(elements)}")
    if file_format not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"{path} has unsupported PLY format: {file_format}")

    vertex_stride = 0
    for line in property_lines:
        parts = line.split()
        scalar_type = parts[1]
        if scalar_type not in PLY_SCALAR_SIZES:
            raise ValueError(f"{path} has unsupported PLY property type: {scalar_type}")
        vertex_stride += PLY_SCALAR_SIZES[scalar_type]

    info = PlyMergeInfo(
        path=path,
        file_format=file_format,
        vertex_count=vertex_count,
        header_size_bytes=header_size,
        property_lines=tuple(property_lines),
        fields=tuple(fields),
        vertex_stride_bytes=vertex_stride,
    )
    if not info.ba_ready:
        raise ValueError(f"{path} is missing XYZ or normal fields for BA constraints")
    return info


def parse_ply_header_fields(path: Path) -> dict[str, object]:
    fields: list[str] = []
    file_format = ""
    vertex_count = 0
    in_vertex = False

    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"{path} has no end_header")
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                file_format = parts[1]
            elif parts[0] == "element":
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex and len(parts) >= 3:
                fields.append(parts[-1])

    field_set = set(fields)
    has_normals = {"normal_x", "normal_y", "normal_z"}.issubset(field_set) or {"nx", "ny", "nz"}.issubset(field_set)
    return {
        "file_format": file_format,
        "vertex_count": vertex_count,
        "fields": fields,
        "has_xyz": {"x", "y", "z"}.issubset(field_set),
        "has_normals": has_normals,
        "has_curvature": "curvature" in field_set,
        "ba_ready": {"x", "y", "z"}.issubset(field_set) and has_normals,
    }


def write_merged_header(handle, file_format: str, vertex_count: int, property_lines: tuple[str, ...]) -> None:
    header_lines = [
        "ply",
        f"format {file_format} 1.0",
        "comment generated_by prepare_lidar_ba_ab_benchmark.py",
        f"element vertex {vertex_count}",
        *property_lines,
        "end_header",
    ]
    handle.write(("\n".join(header_lines) + "\n").encode("ascii"))


def copy_binary_vertices(input_path: Path, output_handle, info: PlyMergeInfo) -> int:
    expected_bytes = info.vertex_count * info.vertex_stride_bytes
    with input_path.open("rb") as handle:
        handle.seek(info.header_size_bytes)
        body = handle.read()
    if len(body) != expected_bytes:
        raise ValueError(
            f"{input_path} body has {len(body)} bytes, expected {expected_bytes} "
            f"from {info.vertex_count} vertices"
        )
    output_handle.write(body)
    return len(body)


def copy_ascii_vertices(input_path: Path, output_handle, info: PlyMergeInfo) -> int:
    copied = 0
    with input_path.open("rb") as handle:
        handle.seek(info.header_size_bytes)
        for _ in range(info.vertex_count):
            line = handle.readline()
            if not line:
                raise ValueError(f"{input_path} ended before all vertices were read")
            output_handle.write(line)
            copied += len(line)
    return copied


def merge_lidar_clouds(input_paths: list[Path], output_path: Path) -> dict[str, object]:
    if not input_paths:
        raise ValueError("No LiDAR PLY files selected for merge")

    infos = [parse_ply_merge_info(path) for path in input_paths]
    first = infos[0]
    for info in infos[1:]:
        if info.file_format != first.file_format:
            raise ValueError(f"PLY format mismatch: {first.path} vs {info.path}")
        if info.property_lines != first.property_lines:
            raise ValueError(f"PLY vertex property mismatch: {first.path} vs {info.path}")

    total_vertices = sum(info.vertex_count for info in infos)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_copied = 0
    with output_path.open("wb") as out:
        write_merged_header(out, first.file_format, total_vertices, first.property_lines)
        for info in infos:
            if first.file_format == "binary_little_endian":
                bytes_copied += copy_binary_vertices(info.path, out, info)
            else:
                bytes_copied += copy_ascii_vertices(info.path, out, info)

    return {
        "path": str(output_path),
        "input_cloud_count": len(input_paths),
        "merged_vertex_count": total_vertices,
        "file_format": first.file_format,
        "fields": list(first.fields),
        "bytes_copied": bytes_copied,
    }


def select_window(rows: list[AssociationRow], start_index: int, window_size: int) -> tuple[list[AssociationRow], int]:
    requested = [
        row
        for row in rows
        if start_index <= row.image_index < start_index + window_size
    ]
    return requested, max(0, window_size - len(requested))


def row_to_image_item(dataset_root: Path, row: AssociationRow) -> dict[str, object]:
    return {
        "image_index": row.image_index,
        "stamp_ns": row.image_stamp_ns,
        "relative_path": row.image_path,
        "path": str(dataset_root / row.image_path),
        "nearest_lidar_index": row.lidar_index,
        "nearest_lidar_dt_ms": row.dt_ms,
    }


def unique_lidar_items(dataset_root: Path, rows: list[AssociationRow]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if row.lidar_path in seen:
            continue
        seen.add(row.lidar_path)
        lidar_file = dataset_root / row.lidar_path
        header = parse_ply_header_fields(lidar_file) if lidar_file.exists() else {}
        items.append(
            {
                "lidar_index": row.lidar_index,
                "stamp_ns": row.lidar_stamp_ns,
                "relative_path": row.lidar_path,
                "path": str(lidar_file),
                "points": row.lidar_points,
                "header": header,
            }
        )
    return items


def build_benchmark_plan(
    dataset_root: Path,
    association_csv: Path,
    start_index: int,
    window_size: int,
    max_abs_dt_ms: float,
) -> dict[str, object]:
    rows = read_associations(association_csv)
    requested_rows, missing_count = select_window(rows, start_index, window_size)
    selected_rows = [row for row in requested_rows if row.abs_dt_ms <= max_abs_dt_ms]
    rejected_rows = [row for row in requested_rows if row.abs_dt_ms > max_abs_dt_ms]
    lidar_clouds = unique_lidar_items(dataset_root, selected_rows)
    has_camera_info = (dataset_root / "camera" / "camera_info_first.yaml").exists()
    lidar_ba_ready = bool(lidar_clouds) and all(bool(item.get("header", {}).get("ba_ready")) for item in lidar_clouds)
    first_lidar_path = lidar_clouds[0]["path"] if lidar_clouds else ""

    return {
        "dataset_root": str(dataset_root),
        "association_csv": str(association_csv),
        "window": {
            "requested_start_index": start_index,
            "requested_size": window_size,
            "max_abs_dt_ms": max_abs_dt_ms,
            "requested_row_count": len(requested_rows),
            "selected_image_count": len(selected_rows),
            "rejected_by_dt_count": len(rejected_rows),
            "missing_row_count": missing_count,
            "unique_lidar_cloud_count": len(lidar_clouds),
        },
        "images": [row_to_image_item(dataset_root, row) for row in selected_rows],
        "lidar_clouds": lidar_clouds,
        "readiness": {
            "has_camera_info_first": has_camera_info,
            "has_selected_images": bool(selected_rows),
            "has_lidar_clouds": bool(lidar_clouds),
            "lidar_stream_ba_ready": lidar_ba_ready,
            "can_prepare_ab_inputs": has_camera_info and bool(selected_rows) and lidar_ba_ready,
        },
        "runs": {
            "baseline_ba": {
                "label": "普通 BA，不加入 LiDAR 约束",
                "options": {
                    "enable_laser_constraints": False,
                },
            },
            "lidar_ba": {
                "label": "LiDAR 点到面软约束 BA",
                "options": {
                    "enable_laser_constraints": True,
                    "laser_constraint_cloud_path": first_lidar_path,
                    "laser_association_max_distance_m": 1.0,
                    "laser_voxel_size_m": 0.0,
                    "laser_max_curvature": 0.2,
                    "laser_max_samples": 500000,
                    "laser_weight": 1.0,
                    "laser_huber_delta_m": 0.2,
                },
                "current_limitation": "BundleAdjustService currently accepts one PLY path; merge/window LiDAR frames before full A/B runs.",
            },
        },
        "metrics_to_compare": [
            "mean_reprojection_rms_before_px",
            "mean_reprojection_rms_after_px",
            "laser_rms_before_m",
            "laser_rms_after_m",
            "laser_median_before_m",
            "laser_median_after_m",
            "associated_tracks",
            "camera_center_delta_m",
            "track_displacement_m",
        ],
    }


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_benchmark_outputs(plan: dict[str, object], output_dir: Path, merge_lidar: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_plan = json.loads(json.dumps(plan, ensure_ascii=False))
    lidar_paths = [Path(item["path"]) for item in output_plan["lidar_clouds"]]
    if merge_lidar:
        merged_path = output_dir / "merged_lidar_cloud.ply"
        output_plan["merged_lidar_cloud"] = merge_lidar_clouds(lidar_paths, merged_path)
        output_plan["runs"]["lidar_ba"]["options"]["laser_constraint_cloud_path"] = str(merged_path)
        output_plan["runs"]["lidar_ba"]["current_limitation"] = (
            "LiDAR frames were merged into one PLY for the current BundleAdjustService input contract."
        )

    (output_dir / "benchmark_plan.json").write_text(
        json.dumps(output_plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_lines(output_dir / "images.lis", [item["path"] for item in output_plan["images"]])
    write_lines(output_dir / "lidar_clouds.lis", [item["path"] for item in output_plan["lidar_clouds"]])
    write_lines(
        output_dir / "README.md",
        [
            "# LiDAR BA A/B Benchmark Plan",
            "",
            "This directory is a reproducible input manifest for comparing ordinary BA with LiDAR-constrained BA.",
            "",
            "- `benchmark_plan.json`: selected image window, nearest LiDAR frames, and run options.",
            "- `images.lis`: selected image paths.",
            "- `lidar_clouds.lis`: unique LiDAR PLY paths for the selected window.",
            "",
            "Current note: PlaScan's service accepts one LiDAR PLY path, so full real-data A/B runs need a merge/window preprocessing step for `lidar_clouds.lis`.",
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a MUN-FRL real-data A/B plan for ordinary BA versus LiDAR-constrained BA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="MUN-FRL extracted root")
    parser.add_argument("--association-csv", type=Path, help="image-to-LiDAR nearest-neighbor CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="output benchmark plan directory")
    parser.add_argument("--start-index", type=int, default=5, help="first image index in the window")
    parser.add_argument("--window-size", type=int, default=20, help="number of consecutive image rows requested")
    parser.add_argument("--max-abs-dt-ms", type=float, default=100.0, help="reject image-LiDAR pairs above this time delta")
    parser.add_argument("--merge-lidar", action="store_true", help="merge selected LiDAR PLY frames into one BA-ready PLY")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root
    association_csv = args.association_csv or dataset_root / "associations" / "color_to_cloud_registered_nearest.csv"

    if not dataset_root.exists():
        print(f"dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2
    if not association_csv.exists():
        print(f"association CSV does not exist: {association_csv}", file=sys.stderr)
        return 2

    try:
        plan = build_benchmark_plan(
            dataset_root=dataset_root,
            association_csv=association_csv,
            start_index=args.start_index,
            window_size=args.window_size,
            max_abs_dt_ms=args.max_abs_dt_ms,
        )
    except (OSError, ValueError) as exc:
        print(f"failed to prepare benchmark plan: {exc}", file=sys.stderr)
        return 1

    try:
        write_benchmark_outputs(plan, args.output_dir, merge_lidar=args.merge_lidar)
    except (OSError, ValueError) as exc:
        print(f"failed to write benchmark outputs: {exc}", file=sys.stderr)
        return 1
    print(f"wrote: {args.output_dir / 'benchmark_plan.json'}")
    print(
        "selected images: "
        f"{plan['window']['selected_image_count']} / {plan['window']['requested_size']}, "
        f"unique LiDAR PLY: {plan['window']['unique_lidar_cloud_count']}"
    )
    if not plan["readiness"]["can_prepare_ab_inputs"]:
        print("warning: input manifest is not fully BA-ready; inspect readiness in benchmark_plan.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
