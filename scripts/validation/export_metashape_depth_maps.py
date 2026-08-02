"""Export Metashape depth maps without interpreting its project cache format.

Run this script with Metashape's embedded Python runtime, for example::

    metashape.exe -r export_metashape_depth_maps.py project.psx output_dir

The project is opened read-only. Each available depth map is written as a
floating-point TIFF together with camera and calibration metadata used by the
PlaScan depth-to-surface isolation tests.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import Metashape


def matrix_rows(matrix: Metashape.Matrix) -> list[list[float]]:
    return [
        [float(matrix[row, column]) for column in range(matrix.size[1])]
        for row in range(matrix.size[0])
    ]


def calibration_record(calibration: Metashape.Calibration) -> dict[str, object]:
    return {
        "width": int(calibration.width),
        "height": int(calibration.height),
        "f": float(calibration.f),
        "cx": float(calibration.cx),
        "cy": float(calibration.cy),
        "b1": float(calibration.b1),
        "b2": float(calibration.b2),
        "k1": float(calibration.k1),
        "k2": float(calibration.k2),
        "k3": float(calibration.k3),
        "k4": float(calibration.k4),
        "p1": float(calibration.p1),
        "p2": float(calibration.p2),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: metashape.exe -r export_metashape_depth_maps.py "
            "PROJECT.psx OUTPUT_DIR"
        )
        return 2

    project_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    document = Metashape.Document()
    document.open(str(project_path), read_only=True)
    chunk = document.chunk
    if chunk is None:
        raise RuntimeError(f"Metashape project has no active chunk: {project_path}")

    records: list[dict[str, object]] = []
    for camera_index, camera in enumerate(chunk.cameras):
        if camera not in chunk.depth_maps:
            continue

        depth_map = chunk.depth_maps[camera]
        image = depth_map.image()
        depth_path = output_dir / f"depth_{camera_index:03d}.tif"
        image.save(str(depth_path))

        sensor = camera.sensor
        calibration = sensor.calibration if sensor is not None else None
        record: dict[str, object] = {
            "camera_index": camera_index,
            "camera_key": int(camera.key),
            "label": str(camera.label),
            "enabled": bool(camera.enabled),
            "depth_path": depth_path.name,
            "depth_width": int(image.width),
            "depth_height": int(image.height),
            "depth_channels": str(image.channels),
        }
        if camera.transform is not None:
            record["camera_to_chunk"] = matrix_rows(camera.transform)
        if calibration is not None:
            record["calibration"] = calibration_record(calibration)
        records.append(record)
        print(
            f"exported camera={camera_index} label={camera.label} "
            f"size={image.width}x{image.height} channels={image.channels}"
        )

    manifest = {
        "schema": "plascan.metashape_depth_export.v1",
        "project_path": str(project_path),
        "metashape_version": str(Metashape.app.version),
        "chunk_label": str(chunk.label),
        "chunk_transform": matrix_rows(chunk.transform.matrix),
        "depth_map_count": len(records),
        "depth_maps": records,
    }
    manifest_path = output_dir / "metashape_depth_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest={manifest_path}")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
