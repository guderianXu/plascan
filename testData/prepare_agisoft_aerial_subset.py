#!/usr/bin/env python3
"""Prepare a deterministic, spatially connected Agisoft aerial subset."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CameraRecord:
    name: str
    latitude: float
    longitude: float
    source_line: str
    source_index: int
    east: float = 0.0
    north: float = 0.0
    texture_score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a compact, connected camera subset from the Agisoft aerial "
            "GCP benchmark and copy the matching images and TSAI cameras."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="source aerial_images_with_gcps directory")
    parser.add_argument("--output", type=Path, required=True, help="empty destination dataset directory")
    parser.add_argument("--count", type=int, default=100, help="number of spatially adjacent cameras")
    parser.add_argument("--neighbors", type=int, default=8, help="nearest-neighbor graph degree")
    parser.add_argument(
        "--minimum-texture-score",
        type=float,
        default=0.0,
        help="minimum 1024-pixel Laplacian variance; 0 disables filtering",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=0,
        help="resize copied images and TSAI intrinsics to this maximum dimension; 0 keeps originals",
    )
    return parser.parse_args()


def load_camera_records(path: Path) -> tuple[str, list[CameraRecord]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or not lines[0].lstrip().startswith("#"):
        raise ValueError(f"camera metadata header is missing: {path}")

    records: list[CameraRecord] = []
    for source_index, line in enumerate(lines[1:]):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 3:
            raise ValueError(f"invalid camera metadata line {source_index + 2}: {line}")
        records.append(
            CameraRecord(
                name=fields[0],
                latitude=float(fields[1]),
                longitude=float(fields[2]),
                source_line=line,
                source_index=source_index,
            )
        )
    if not records:
        raise ValueError(f"no camera records found: {path}")
    return lines[0], records


def project_to_local(records: list[CameraRecord]) -> list[CameraRecord]:
    latitude_origin = sum(record.latitude for record in records) / len(records)
    longitude_origin = sum(record.longitude for record in records) / len(records)
    latitude_scale = 111_320.0
    longitude_scale = latitude_scale * math.cos(math.radians(latitude_origin))
    return [
        CameraRecord(
            name=record.name,
            latitude=record.latitude,
            longitude=record.longitude,
            source_line=record.source_line,
            source_index=record.source_index,
            east=(record.longitude - longitude_origin) * longitude_scale,
            north=(record.latitude - latitude_origin) * latitude_scale,
            texture_score=record.texture_score,
        )
        for record in records
    ]


def filter_by_texture(
    source: Path, records: list[CameraRecord], minimum_score: float
) -> list[CameraRecord]:
    if minimum_score <= 0.0:
        return records
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV Python is required when --minimum-texture-score is enabled"
        ) from error

    accepted: list[CameraRecord] = []
    for record in records:
        image_path = source / "Images" / record.name
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"cannot read image for texture scoring: {image_path}")
        scale = min(1.0, 1024.0 / max(image.shape))
        if scale < 1.0:
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        score = float(cv2.Laplacian(image, cv2.CV_32F).var())
        if score >= minimum_score:
            accepted.append(
                CameraRecord(
                    name=record.name,
                    latitude=record.latitude,
                    longitude=record.longitude,
                    source_line=record.source_line,
                    source_index=record.source_index,
                    texture_score=score,
                )
            )
    return accepted


def filter_available_pairs(source: Path, records: list[CameraRecord]) -> list[CameraRecord]:
    accepted: list[CameraRecord] = []
    for record in records:
        stem = Path(record.name).stem
        image_path = source / "Images" / record.name
        camera_path = source / "cameras" / f"{stem}.tsai"
        if image_path.is_file() and camera_path.is_file():
            accepted.append(record)
            continue
        print(
            f"skipping metadata-only camera without a complete image/TSAI pair: {record.name}",
            file=sys.stderr,
        )
    return accepted


def squared_distance(first: CameraRecord, second: CameraRecord) -> float:
    east = first.east - second.east
    north = first.north - second.north
    return east * east + north * north


def build_neighbor_graph(records: list[CameraRecord], degree: int) -> list[list[int]]:
    graph: list[set[int]] = [set() for _ in records]
    for index, record in enumerate(records):
        ordered = sorted(
            (squared_distance(record, other), other_index)
            for other_index, other in enumerate(records)
            if other_index != index
        )
        for _, other_index in ordered[:degree]:
            graph[index].add(other_index)
            graph[other_index].add(index)
    return [sorted(neighbors) for neighbors in graph]


def select_connected_subset(
    records: list[CameraRecord], graph: list[list[int]], count: int
) -> list[int]:
    seed = min(
        range(len(records)),
        key=lambda index: (
            records[index].east * records[index].east
            + records[index].north * records[index].north,
            records[index].source_index,
        ),
    )
    selected = {seed}
    frontier: list[tuple[float, int, int]] = []

    def add_frontier(source: int) -> None:
        for destination in graph[source]:
            if destination in selected:
                continue
            heapq.heappush(
                frontier,
                (
                    squared_distance(records[source], records[destination]),
                    records[destination].source_index,
                    destination,
                ),
            )

    add_frontier(seed)
    while len(selected) < count:
        while frontier and frontier[0][2] in selected:
            heapq.heappop(frontier)
        if not frontier:
            raise RuntimeError("nearest-neighbor graph is disconnected before reaching requested count")
        _, _, destination = heapq.heappop(frontier)
        selected.add(destination)
        add_frontier(destination)
    return sorted(selected, key=lambda index: records[index].source_index)


def selected_edges(graph: list[list[int]], selected: set[int]) -> list[tuple[int, int]]:
    return [
        (first, second)
        for first in sorted(selected)
        for second in graph[first]
        if first < second and second in selected
    ]


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def scaled_tsai_text(path: Path, scale_x: float, scale_y: float) -> str:
    scales = {"fu": scale_x, "fv": scale_y, "cu": scale_x, "cv": scale_y}
    output: list[str] = []
    matched: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^(fu|fv|cu|cv)\s*=\s*(\S+)\s*$", line)
        if not match:
            output.append(line)
            continue
        name = match.group(1)
        value = float(match.group(2)) * scales[name]
        output.append(f"{name} = {value:.12g}")
        matched.add(name)
    if matched != set(scales):
        raise ValueError(f"TSAI intrinsics are incomplete: {path}")
    return "\n".join(output) + "\n"


def copy_or_resize_pair(
    image_source: Path,
    camera_source: Path,
    image_output: Path,
    camera_output: Path,
    max_image_dim: int,
) -> tuple[int, int, float, float]:
    if max_image_dim <= 0:
        shutil.copy2(image_source, image_output)
        shutil.copy2(camera_source, camera_output)
        return 0, 0, 1.0, 1.0
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV Python is required when --max-image-dim is enabled") from error

    image = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image for resizing: {image_source}")
    height, width = image.shape[:2]
    scale = min(1.0, max_image_dim / max(width, height))
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    if output_width != width or output_height != height:
        image = cv2.resize(
            image, (output_width, output_height), interpolation=cv2.INTER_AREA
        )
    if not cv2.imwrite(
        str(image_output), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ):
        raise OSError(f"cannot write resized image: {image_output}")
    scale_x = output_width / width
    scale_y = output_height / height
    camera_output.write_text(
        scaled_tsai_text(camera_source, scale_x, scale_y),
        encoding="utf-8",
        newline="\n",
    )
    return output_width, output_height, scale_x, scale_y


def copy_dataset(
    source: Path,
    output: Path,
    header: str,
    records: list[CameraRecord],
    graph: list[list[int]],
    selected_indices: list[int],
    neighbor_degree: int,
    minimum_texture_score: float,
    max_image_dim: int,
) -> None:
    require_empty_output(output)
    images_output = output / "Images"
    cameras_output = output / "cameras"
    metadata_output = output / "Metadata"
    images_output.mkdir()
    cameras_output.mkdir()
    metadata_output.mkdir()

    selected = set(selected_indices)
    list_lines: list[str] = []
    image_transform: dict[str, float | int] | None = None
    for index in selected_indices:
        record = records[index]
        stem = Path(record.name).stem
        image_source = source / "Images" / record.name
        camera_source = source / "cameras" / f"{stem}.tsai"
        if not image_source.is_file() or not camera_source.is_file():
            raise FileNotFoundError(f"missing image/camera pair: {image_source} / {camera_source}")
        width, height, scale_x, scale_y = copy_or_resize_pair(
            image_source,
            camera_source,
            images_output / image_source.name,
            cameras_output / camera_source.name,
            max_image_dim,
        )
        if max_image_dim > 0 and image_transform is None:
            image_transform = {
                "maximum_dimension": max_image_dim,
                "output_width": width,
                "output_height": height,
                "intrinsic_scale_x": scale_x,
                "intrinsic_scale_y": scale_y,
            }
        list_lines.append(f"Images/{record.name} cameras/{camera_source.name}")

    (output / "image_camera.lis").write_text(
        "\n".join(list_lines) + "\n", encoding="utf-8", newline="\n"
    )
    (metadata_output / "Cameras_WGS84.txt").write_text(
        header + "\n"
        + "\n".join(records[index].source_line for index in selected_indices)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for metadata_name in ("GCPs_WGS84.txt", "GNSS_offset.txt"):
        metadata_source = source / "Metadata" / metadata_name
        if metadata_source.is_file():
            shutil.copy2(metadata_source, metadata_output / metadata_name)
    if (source / "info.txt").is_file():
        shutil.copy2(source / "info.txt", output / "info.txt")

    edges = selected_edges(graph, selected)
    manifest = {
        "dataset_id": f"agisoft_aerial_gcps_{len(selected_indices)}",
        "source_dataset": str(source.resolve()),
        "selection": {
            "algorithm": "centroid_seed_connected_nearest_neighbor_growth",
            "camera_count": len(selected_indices),
            "neighbor_degree": neighbor_degree,
            "connected": True,
            "preserved_source_order": True,
            "seed_camera": records[min(
                selected_indices,
                key=lambda index: (
                    records[index].east * records[index].east
                    + records[index].north * records[index].north,
                    records[index].source_index,
                ),
            )].name,
            "adjacency_edge_count": len(edges),
            "minimum_texture_score": minimum_texture_score,
            "minimum_selected_texture_score": min(
                records[index].texture_score or 0.0 for index in selected_indices
            ),
        },
        "image_transform": image_transform,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cameras": [records[index].name for index in selected_indices],
        "adjacency_edges": [
            [records[first].name, records[second].name] for first, second in edges
        ],
    }
    (output / "subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if args.count < 2:
        raise ValueError("--count must be at least 2")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be at least 1")
    if args.max_image_dim < 0:
        raise ValueError("--max-image-dim must be non-negative")
    if source == output or source in output.parents:
        raise ValueError("output must not be inside the source dataset")

    header, records = load_camera_records(source / "Metadata" / "Cameras_WGS84.txt")
    if args.count > len(records):
        raise ValueError(f"requested {args.count} cameras but source contains {len(records)}")
    records = filter_available_pairs(source, records)
    records = filter_by_texture(source, records, args.minimum_texture_score)
    if args.count > len(records):
        raise ValueError(
            f"texture gate retained {len(records)} cameras, fewer than requested {args.count}"
        )
    records = project_to_local(records)
    graph = build_neighbor_graph(records, min(args.neighbors, len(records) - 1))
    selected_indices = select_connected_subset(records, graph, args.count)
    copy_dataset(
        source,
        output,
        header,
        records,
        graph,
        selected_indices,
        args.neighbors,
        args.minimum_texture_score,
        args.max_image_dim,
    )
    print(f"prepared {len(selected_indices)} connected cameras in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
