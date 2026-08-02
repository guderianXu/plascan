#!/usr/bin/env python3
"""Export accepted native depth-pose candidates to a manifest copy."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


EXTRINSIC_FIELDS = {
    "camera_center": 3,
    "rotation_world_to_camera": 9,
    "translation_world_to_camera": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply only accepted, candidate-only native depth-pose results to "
            "a new MVS manifest. The source manifest and project are unchanged."
        )
    )
    parser.add_argument("--mvs-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional audit report; defaults beside the output manifest.",
    )
    return parser.parse_args()


def finite_vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != size:
        return None
    converted = [float(item) for item in value]
    return converted if all(math.isfinite(item) for item in converted) else None


def main() -> int:
    args = parse_args()
    source_path = args.mvs_manifest.resolve()
    output_path = args.output_manifest.resolve()
    if source_path == output_path:
        raise ValueError("Output manifest must differ from the source manifest")
    if not source_path.is_file():
        raise FileNotFoundError(f"MVS manifest not found: {source_path}")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    frames = source.get("frames")
    if not isinstance(frames, list) or len(frames) < 3:
        raise ValueError(f"Expected at least three manifest frames: {source_path}")

    exported = copy.deepcopy(source)
    accepted_indices: list[int] = []
    skipped_reasons: dict[str, int] = {}
    seen_indices: set[int] = set()
    for frame in exported["frames"]:
        frame_index = int(frame.get("ref_index", -1))
        if frame_index < 0 or frame_index in seen_indices:
            raise ValueError(f"Invalid or duplicate ref_index: {frame_index}")
        seen_indices.add(frame_index)

        diagnostics = frame.get("pose_refinement_diagnostics", {})
        if not isinstance(diagnostics, dict) or not diagnostics.get("accepted", False):
            reason = str(diagnostics.get("reason", "missing_diagnostics"))
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue
        if not diagnostics.get("candidate_only", False):
            raise ValueError(
                f"Frame {frame_index} is accepted but is not marked candidate-only"
            )

        derived = frame.get("derived_camera_model", {})
        camera = frame.get("camera_model", {})
        if not isinstance(derived, dict) or not isinstance(camera, dict):
            raise ValueError(f"Frame {frame_index} has no usable camera model")
        for field, size in EXTRINSIC_FIELDS.items():
            values = finite_vector(derived.get(field), size)
            if values is None:
                raise ValueError(
                    f"Frame {frame_index} has invalid derived field: {field}"
                )
            camera[field] = values
        accepted_indices.append(frame_index)

    if not accepted_indices:
        raise ValueError(f"No accepted depth-pose candidates: {source_path}")

    accepted_indices.sort()
    exported["depth_pose_candidate_export"] = {
        "schema": "plascan.depth_pose_candidate_export.v1",
        "source_manifest": str(source_path),
        "project_camera_modified": False,
        "accepted_frame_indices": accepted_indices,
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(exported, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_path.with_suffix(".report.json")
    )
    report = {
        "schema": "plascan.depth_pose_candidate_export_report.v1",
        "source_manifest": str(source_path),
        "output_manifest": str(output_path),
        "project_camera_modified": False,
        "frame_count": len(frames),
        "accepted_frame_count": len(accepted_indices),
        "accepted_frame_indices": accepted_indices,
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(accepted_indices)}/{len(frames)} candidates: "
        f"{output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
