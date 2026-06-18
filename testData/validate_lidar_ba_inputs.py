#!/usr/bin/env python3
"""Validate LiDAR PLY streams for PlaScan laser-constrained bundle adjustment."""

from __future__ import annotations

import argparse
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

KNOWN_LIDAR_STREAMS = (
    Path("lidar/cloud_registered"),
    Path("lidar/velodyne_cloud_registered"),
    Path("lidar/velodyne_points"),
)


@dataclass(frozen=True)
class PlyProperty:
    kind: str
    name: str


@dataclass(frozen=True)
class PlyHeader:
    path: Path
    file_format: str
    vertex_count: int
    properties: tuple[PlyProperty, ...]
    comments: tuple[str, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.properties)

    @property
    def has_xyz(self) -> bool:
        fields = set(self.field_names)
        return {"x", "y", "z"}.issubset(fields)

    @property
    def has_normals(self) -> bool:
        fields = set(self.field_names)
        return {"normal_x", "normal_y", "normal_z"}.issubset(fields) or {"nx", "ny", "nz"}.issubset(fields)

    @property
    def has_curvature(self) -> bool:
        return "curvature" in set(self.field_names)

    @property
    def ba_ready(self) -> bool:
        return self.has_xyz and self.has_normals

    @property
    def fusion_reference(self) -> bool:
        return self.has_xyz and not self.ba_ready


def parse_ply_header(path: Path) -> PlyHeader:
    comments: list[str] = []
    properties: list[PlyProperty] = []
    file_format = ""
    vertex_count = 0
    in_vertex_element = False

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
            if not line:
                continue

            parts = line.split()
            if parts[0] == "format" and len(parts) >= 2:
                file_format = parts[1]
                continue
            if parts[0] == "comment":
                comments.append(line[len("comment"):].strip())
                continue
            if parts[0] == "element":
                in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
                continue
            if parts[0] == "property" and in_vertex_element and len(parts) >= 3:
                if parts[1] == "list" and len(parts) >= 5:
                    properties.append(PlyProperty(kind=" ".join(parts[1:4]), name=parts[4]))
                else:
                    properties.append(PlyProperty(kind=parts[1], name=parts[2]))

    return PlyHeader(
        path=path,
        file_format=file_format,
        vertex_count=vertex_count,
        properties=tuple(properties),
        comments=tuple(comments),
    )


def discover_stream_dirs(dataset_root: Path) -> list[tuple[str, Path]]:
    stream_dirs: list[tuple[str, Path]] = []
    for relative in KNOWN_LIDAR_STREAMS:
        stream_dir = dataset_root / relative
        if stream_dir.is_dir():
            stream_dirs.append((relative.as_posix(), stream_dir))

    if stream_dirs:
        return stream_dirs

    lidar_root = dataset_root / "lidar"
    if not lidar_root.is_dir():
        return []

    for stream_dir in sorted(path for path in lidar_root.iterdir() if path.is_dir()):
        stream_dirs.append((stream_dir.relative_to(dataset_root).as_posix(), stream_dir))
    return stream_dirs


def summarize_stream(dataset_root: Path, stream_name: str, stream_dir: Path, max_files_per_stream: int) -> dict[str, object]:
    ply_files = sorted(stream_dir.glob("*.ply"))
    sampled = ply_files[:max(1, max_files_per_stream)]
    headers: list[PlyHeader] = []
    errors: list[str] = []

    for ply_path in sampled:
        try:
            headers.append(parse_ply_header(ply_path))
        except (OSError, ValueError) as exc:
            errors.append(f"{ply_path.name}: {exc}")

    ba_ready = [header for header in headers if header.ba_ready]
    fusion_reference = [header for header in headers if header.fusion_reference]
    first_header = headers[0] if headers else None
    first_file = ply_files[0] if ply_files else None
    vertex_count_sampled = sum(header.vertex_count for header in headers)

    return {
        "path": str(stream_dir),
        "relative_path": stream_name,
        "ply_files": len(ply_files),
        "sampled_files": len(sampled),
        "first_file": str(first_file) if first_file is not None else "",
        "file_format": first_header.file_format if first_header is not None else "",
        "fields": list(first_header.field_names) if first_header is not None else [],
        "has_xyz": bool(first_header.has_xyz) if first_header is not None else False,
        "has_normals": bool(first_header.has_normals) if first_header is not None else False,
        "has_curvature": bool(first_header.has_curvature) if first_header is not None else False,
        "ba_ready_files": len(ba_ready),
        "fusion_reference_files": len(fusion_reference),
        "vertex_count_sampled": vertex_count_sampled,
        "errors": errors,
        "notes": build_stream_notes(len(ply_files), first_header),
    }


def build_stream_notes(file_count: int, first_header: PlyHeader | None) -> list[str]:
    notes: list[str] = []
    if file_count > 1:
        notes.append("Stream contains per-frame PLY files; current BA service accepts one PLY at a time.")
    if first_header is not None and first_header.has_xyz and not first_header.has_normals:
        notes.append("XYZ points are present but normals are missing; estimate normals before point-to-plane BA.")
    if first_header is not None and first_header.ba_ready and not first_header.has_curvature:
        notes.append("Normals are present; curvature is absent and will default to 0 in the current loader.")
    return notes


def summarize_dataset(dataset_root: Path, max_files_per_stream: int = 5) -> dict[str, object]:
    stream_dirs = discover_stream_dirs(dataset_root)
    streams = {
        stream_name: summarize_stream(dataset_root, stream_name, stream_dir, max_files_per_stream)
        for stream_name, stream_dir in stream_dirs
    }

    ba_stream = ""
    ba_cloud_path = ""
    for preferred in [path.as_posix() for path in KNOWN_LIDAR_STREAMS]:
        stream = streams.get(preferred)
        if stream and int(stream["ba_ready_files"]) > 0:
            ba_stream = preferred
            ba_cloud_path = str(stream["first_file"])
            break

    if not ba_stream:
        for stream_name, stream in streams.items():
            if int(stream["ba_ready_files"]) > 0:
                ba_stream = stream_name
                ba_cloud_path = str(stream["first_file"])
                break

    fusion_streams = [
        stream_name
        for stream_name, stream in streams.items()
        if bool(stream["has_xyz"])
    ]
    needs_normal_estimation = [
        stream_name
        for stream_name, stream in streams.items()
        if int(stream["fusion_reference_files"]) > 0 and int(stream["ba_ready_files"]) == 0
    ]

    return {
        "dataset_root": str(dataset_root),
        "streams": streams,
        "recommendation": {
            "ba_constraint_cloud_stream": ba_stream,
            "ba_constraint_cloud_path": ba_cloud_path,
            "fusion_reference_streams": fusion_streams,
            "needs_normal_estimation": needs_normal_estimation,
            "dialog_options": {
                "enable_laser_constraints": bool(ba_stream),
                "laser_constraint_cloud_path": ba_cloud_path,
                "laser_association_max_distance_m": 1.0,
                "laser_voxel_size_m": 0.0,
                "laser_max_curvature": 0.2,
                "laser_max_samples": 500000,
                "laser_weight": 1.0,
                "laser_huber_delta_m": 0.2,
            },
        },
    }


def print_summary(summary: dict[str, object]) -> None:
    recommendation = summary["recommendation"]
    ba_stream = recommendation["ba_constraint_cloud_stream"]
    if ba_stream:
        print(f"BA-ready LiDAR stream: {ba_stream}")
        print(f"  first PLY for current GUI/service: {recommendation['ba_constraint_cloud_path']}")
    else:
        print("No BA-ready LiDAR stream found.")

    print("Streams:")
    for stream_name, stream in summary["streams"].items():
        status = "BA-ready" if stream["ba_ready_files"] else "fusion/reference"
        if not stream["has_xyz"]:
            status = "not usable"
        print(
            f"  {stream_name}: {status}, "
            f"ply={stream['ply_files']}, sampled={stream['sampled_files']}, "
            f"fields={','.join(stream['fields'])}"
        )
        for note in stream["notes"]:
            print(f"    note: {note}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate extracted LiDAR PLY streams for PlaScan laser-constrained BA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="extracted dataset root")
    parser.add_argument("--summary-json", type=Path, help="optional JSON summary output")
    parser.add_argument("--max-files-per-stream", type=int, default=5, help="PLY headers sampled per stream")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root
    if not dataset_root.exists():
        print(f"dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    summary = summarize_dataset(dataset_root, max_files_per_stream=args.max_files_per_stream)
    print_summary(summary)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote: {args.summary_json}")

    return 0 if summary["recommendation"]["ba_constraint_cloud_stream"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
