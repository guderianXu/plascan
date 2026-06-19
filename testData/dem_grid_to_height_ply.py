#!/usr/bin/env python3
"""Convert an ESRI ASCII DEM/DSM grid into XYZ-only height-plane PLY constraints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AsciiGrid:
    ncols: int
    nrows: int
    x_origin: float
    y_origin: float
    cellsize: float
    origin_mode: str
    nodata_value: float | None
    rows: list[list[float]]


def parse_ascii_grid(path: Path) -> AsciiGrid:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]

    header: dict[str, str] = {}
    data_start = 0
    for index, line in enumerate(raw_lines):
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"invalid ASCII grid header line {index + 1}: {line}")
        key = parts[0].lower()
        if key not in {
            "ncols",
            "nrows",
            "xllcorner",
            "yllcorner",
            "xllcenter",
            "yllcenter",
            "cellsize",
            "nodata_value",
        }:
            data_start = index
            break
        header[key] = parts[1]
    else:
        data_start = len(raw_lines)

    try:
        ncols = int(header["ncols"])
        nrows = int(header["nrows"])
        cellsize = float(header["cellsize"])
    except KeyError as exc:
        raise ValueError(f"missing ASCII grid header field: {exc.args[0]}") from exc
    except ValueError as exc:
        raise ValueError("invalid ncols/nrows/cellsize value") from exc

    if ncols <= 0 or nrows <= 0 or not math.isfinite(cellsize) or cellsize <= 0.0:
        raise ValueError("ASCII grid dimensions and cellsize must be positive")

    has_corner = "xllcorner" in header and "yllcorner" in header
    has_center = "xllcenter" in header and "yllcenter" in header
    if has_corner == has_center:
        raise ValueError("ASCII grid must specify either xllcorner/yllcorner or xllcenter/yllcenter")

    origin_mode = "corner" if has_corner else "center"
    x_origin = float(header["xllcorner" if has_corner else "xllcenter"])
    y_origin = float(header["yllcorner" if has_corner else "yllcenter"])
    nodata_value = float(header["nodata_value"]) if "nodata_value" in header else None

    data_lines = raw_lines[data_start:]
    if len(data_lines) != nrows:
        raise ValueError(f"ASCII grid expected {nrows} data rows, got {len(data_lines)}")

    rows: list[list[float]] = []
    for row_index, line in enumerate(data_lines):
        values = [float(token) for token in line.split()]
        if len(values) != ncols:
            raise ValueError(f"ASCII grid row {row_index + 1} expected {ncols} values, got {len(values)}")
        rows.append(values)

    return AsciiGrid(
        ncols=ncols,
        nrows=nrows,
        x_origin=x_origin,
        y_origin=y_origin,
        cellsize=cellsize,
        origin_mode=origin_mode,
        nodata_value=nodata_value,
        rows=rows,
    )


def cell_center_xy(grid: AsciiGrid, row: int, col: int) -> tuple[float, float]:
    if grid.origin_mode == "corner":
        x = grid.x_origin + (col + 0.5) * grid.cellsize
        y = grid.y_origin + (grid.nrows - row - 0.5) * grid.cellsize
    else:
        x = grid.x_origin + col * grid.cellsize
        y = grid.y_origin + (grid.nrows - row - 1) * grid.cellsize
    return x, y


def is_nodata(value: float, nodata_value: float | None) -> bool:
    if not math.isfinite(value):
        return True
    if nodata_value is None:
        return False
    return abs(value - nodata_value) <= 1e-12


def grid_points(grid: AsciiGrid, stride: int = 1) -> tuple[list[tuple[float, float, float]], int]:
    if stride <= 0:
        raise ValueError("stride must be positive")

    points: list[tuple[float, float, float]] = []
    skipped_nodata = 0
    for row_index, row_values in enumerate(grid.rows):
        if row_index % stride != 0:
            continue
        for col_index, value in enumerate(row_values):
            if col_index % stride != 0:
                continue
            if is_nodata(value, grid.nodata_value):
                skipped_nodata += 1
                continue
            x, y = cell_center_xy(grid, row_index, col_index)
            points.append((x, y, value))
    return points, skipped_nodata


def bounds_for_points(points: list[tuple[float, float, float]]) -> dict[str, float]:
    if not points:
        return {
            "min_x": 0.0,
            "max_x": 0.0,
            "min_y": 0.0,
            "max_y": 0.0,
            "min_z": 0.0,
            "max_z": 0.0,
        }
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "min_z": min(zs),
        "max_z": max(zs),
    }


def write_xyz_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment generated_by dem_grid_to_height_ply.py\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property double x\n")
        handle.write("property double y\n")
        handle.write("property double z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.9f} {y:.9f} {z:.9f}\n")


def convert_ascii_grid_to_height_ply(input_path: Path, output_path: Path, stride: int = 1) -> dict[str, Any]:
    grid = parse_ascii_grid(Path(input_path))
    points, skipped_nodata = grid_points(grid, stride=stride)
    if not points:
        raise ValueError("ASCII grid has no valid height samples after nodata/stride filtering")
    write_xyz_ply(Path(output_path), points)
    return {
        "input": str(Path(input_path)),
        "output": str(Path(output_path)),
        "valid_points": len(points),
        "skipped_nodata": skipped_nodata,
        "stride": stride,
        "grid": {
            "ncols": grid.ncols,
            "nrows": grid.nrows,
            "cellsize": grid.cellsize,
            "origin_mode": grid.origin_mode,
            "nodata_value": grid.nodata_value,
        },
        "bounds": bounds_for_points(points),
        "ba_usage": {
            "laser_missing_normals_as_height_planes": True,
            "recommended_cli_flag": "--laser-missing-normals-as-height-planes",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ESRI ASCII DEM/DSM grid heights to XYZ-only PLY BA constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="input ESRI ASCII grid .asc")
    parser.add_argument("--output", type=Path, required=True, help="output XYZ-only PLY")
    parser.add_argument("--stride", type=int, default=1, help="sample every Nth row/column")
    parser.add_argument("--summary-json", type=Path, help="optional summary JSON output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = convert_ascii_grid_to_height_ply(args.input, args.output, stride=args.stride)
    except (OSError, ValueError) as exc:
        print(f"failed to convert DEM grid to PLY: {exc}", file=sys.stderr)
        return 1
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
