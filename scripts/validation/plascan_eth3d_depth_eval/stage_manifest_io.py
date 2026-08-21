"""Strict reader for non-authoritative MVS stage-snapshot manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .camera_io import ColmapCamera, validate_scaled_pinhole_camera
from .manifest_io import (
    PredictionManifestFrame,
    read_plascan_pinhole_camera_model,
)


STAGE_SNAPSHOT_SCHEMA = "plascan_mvs_stage_snapshots_v1"
STAGE_IDS = {
    "patchmatch_output",
    "cross_view_consistency",
    "confidence_postprocess",
    "final_admission",
}


@dataclass(frozen=True)
class StageSnapshotArtifact:
    path: Path
    size_bytes: int
    sha256: str
    rows: int
    columns: int
    opencv_type: int

    def provenance_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": self.columns,
            "opencv_type": self.opencv_type,
        }


@dataclass(frozen=True)
class StageSnapshotRecord:
    schema: str
    ref_index: int
    stage: str
    boundary: str
    original_width: int
    original_height: int
    snapshot_width: int
    snapshot_height: int
    valid_pixel_count: int
    depth: StageSnapshotArtifact
    confidence: StageSnapshotArtifact
    valid_mask: StageSnapshotArtifact
    manifest_camera_model: dict[str, float]
    rotation_world_to_camera: tuple[float, ...]
    camera_center: tuple[float, ...]
    translation_world_to_camera: tuple[float, ...]
    camera: ColmapCamera
    manifest_size_bytes: int
    manifest_sha256: str

    def provenance_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authoritative": False,
            "ref_index": self.ref_index,
            "stage": self.stage,
            "boundary": self.boundary,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "snapshot_width": self.snapshot_width,
            "snapshot_height": self.snapshot_height,
            "valid_pixel_count": self.valid_pixel_count,
            "depth": self.depth.provenance_dict(),
            "confidence": self.confidence.provenance_dict(),
            "valid_mask": self.valid_mask.provenance_dict(),
        }


def read_stage_snapshot_record(
    path: Path,
    ref_index: int,
    stage: str,
    *,
    camera_id: int,
    artifact_base_dir: Path | None = None,
) -> StageSnapshotRecord:
    """Read one captured record from a finalized stage-snapshot manifest."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage snapshot manifest not found: {path}")
    if isinstance(ref_index, bool) or not isinstance(ref_index, int) or ref_index < 0:
        raise ValueError("Stage snapshot ref index must be non-negative")
    if stage not in STAGE_IDS:
        raise ValueError(f"Unsupported MVS stage snapshot id: {stage!r}")
    manifest_bytes = path.read_bytes()
    try:
        root = json.loads(
            manifest_bytes.decode("utf-8"),
            parse_constant=_reject_non_finite_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid stage snapshot JSON {path}: {error}") from error
    if not isinstance(root, dict):
        raise ValueError(f"Stage snapshot manifest root must be an object: {path}")
    if _required_string(root, "schema", "stage manifest root") != STAGE_SNAPSHOT_SCHEMA:
        raise ValueError(f"Unsupported stage snapshot schema: {path}")
    if root.get("authoritative") is not False:
        raise ValueError("Stage snapshot manifest must explicitly be non-authoritative")
    if root.get("finalized") is not True or root.get("status") != "complete":
        raise ValueError("Stage snapshot manifest must be finalized with status='complete'")
    errors = root.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError("Stage snapshot manifest contains diagnostic errors")
    selected_refs = _integer_array(root, "selected_ref_indices", "stage manifest root")
    if ref_index not in selected_refs:
        raise ValueError(f"Stage snapshot ref {ref_index} was not selected")
    expected_stages = root.get("expected_stages")
    if not isinstance(expected_stages, list) or set(expected_stages) != STAGE_IDS:
        raise ValueError("Stage snapshot manifest has an invalid expected stage set")

    records = root.get("records")
    if not isinstance(records, list):
        raise ValueError("Stage snapshot records must be an array")
    matches = []
    for index, value in enumerate(records):
        if not isinstance(value, dict):
            raise ValueError(f"Stage snapshot record {index} must be an object")
        if value.get("ref_index") == ref_index and value.get("stage") == stage:
            matches.append((index, value))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one stage record for ref={ref_index}, stage={stage}; "
            f"found {len(matches)}"
        )
    record_index, record = matches[0]
    context = f"stage record {record_index} (ref={ref_index}, stage={stage})"
    if _required_string(record, "status", context) != "captured":
        raise ValueError(f"{context} is not captured")
    boundary = _required_string(record, "boundary", context)
    original_width = _required_positive_integer(record, "original_width", context)
    original_height = _required_positive_integer(record, "original_height", context)
    snapshot_width = _required_positive_integer(record, "snapshot_width", context)
    snapshot_height = _required_positive_integer(record, "snapshot_height", context)
    valid_pixel_count = _required_non_negative_integer(
        record, "valid_pixel_count", context
    )
    if valid_pixel_count > snapshot_width * snapshot_height:
        raise ValueError(f"{context} valid_pixel_count exceeds the snapshot raster")

    base_dir = artifact_base_dir.resolve() if artifact_base_dir else path.parent
    depth = _read_artifact(
        record, "depth", context, base_dir, snapshot_width, snapshot_height, 5
    )
    confidence = _read_artifact(
        record,
        "confidence",
        context,
        base_dir,
        snapshot_width,
        snapshot_height,
        5,
    )
    valid_mask = _read_artifact(
        record,
        "valid_mask",
        context,
        base_dir,
        snapshot_width,
        snapshot_height,
        0,
    )
    (
        camera,
        camera_model,
        rotation,
        center,
        translation,
    ) = read_plascan_pinhole_camera_model(
        record.get("camera_model"),
        camera_id=camera_id,
        width=snapshot_width,
        height=snapshot_height,
        context=context,
    )
    return StageSnapshotRecord(
        schema=STAGE_SNAPSHOT_SCHEMA,
        ref_index=ref_index,
        stage=stage,
        boundary=boundary,
        original_width=original_width,
        original_height=original_height,
        snapshot_width=snapshot_width,
        snapshot_height=snapshot_height,
        valid_pixel_count=valid_pixel_count,
        depth=depth,
        confidence=confidence,
        valid_mask=valid_mask,
        manifest_camera_model=camera_model,
        rotation_world_to_camera=rotation,
        camera_center=center,
        translation_world_to_camera=translation,
        camera=camera,
        manifest_size_bytes=len(manifest_bytes),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def validate_stage_snapshot_against_workspace(
    stage: StageSnapshotRecord,
    frame: PredictionManifestFrame,
) -> dict[str, object]:
    """Bind a diagnostic stage raster to its authoritative workspace frame."""

    if stage.ref_index != frame.ref_index:
        raise ValueError("Stage snapshot ref_index does not match workspace frame")
    if (stage.original_width, stage.original_height) != (
        frame.grid_width,
        frame.grid_height,
    ):
        raise ValueError(
            "Stage snapshot original raster does not match workspace depth grid"
        )
    for name, stage_values, frame_values in (
        (
            "rotation_world_to_camera",
            stage.rotation_world_to_camera,
            frame.rotation_world_to_camera,
        ),
        ("camera_center", stage.camera_center, frame.camera_center),
        (
            "translation_world_to_camera",
            stage.translation_world_to_camera,
            frame.translation_world_to_camera,
        ),
    ):
        if stage_values != frame_values:
            raise ValueError(f"Stage snapshot {name} does not match workspace frame")
    scaling = validate_scaled_pinhole_camera(frame.camera, stage.camera)
    return {
        "workspace_ref_index": frame.ref_index,
        "workspace_grid_width": frame.grid_width,
        "workspace_grid_height": frame.grid_height,
        "snapshot_scaling": scaling,
    }


def _read_artifact(
    record: dict[str, Any],
    key: str,
    context: str,
    base_dir: Path,
    width: int,
    height: int,
    expected_type: int,
) -> StageSnapshotArtifact:
    value = record.get(key)
    artifact_context = f"{context} {key}"
    if not isinstance(value, dict):
        raise ValueError(f"{artifact_context} must be an object")
    path = Path(_required_string(value, "path", artifact_context))
    if not path.is_absolute():
        path = base_dir / path
    columns = _required_positive_integer(value, "cols", artifact_context)
    rows = _required_positive_integer(value, "rows", artifact_context)
    opencv_type = _required_non_negative_integer(
        value, "opencv_type", artifact_context
    )
    if (columns, rows) != (width, height) or opencv_type != expected_type:
        raise ValueError(f"{artifact_context} raster/type contract is inconsistent")
    return StageSnapshotArtifact(
        path=path.resolve(),
        size_bytes=_required_positive_integer(value, "size_bytes", artifact_context),
        sha256=_required_sha256(value, "sha256", artifact_context),
        rows=rows,
        columns=columns,
        opencv_type=opencv_type,
    )


def _required_string(value: dict[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context} requires non-empty string field {key!r}")
    return result


def _required_positive_integer(
    value: dict[str, Any], key: str, context: str
) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise ValueError(f"{context} requires positive integer field {key!r}")
    return result


def _required_non_negative_integer(
    value: dict[str, Any], key: str, context: str
) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(f"{context} requires non-negative integer field {key!r}")
    return result


def _required_sha256(value: dict[str, Any], key: str, context: str) -> str:
    result = _required_string(value, key, context).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{context} requires a lowercase SHA-256 field {key!r}")
    return result


def _integer_array(value: dict[str, Any], key: str, context: str) -> list[int]:
    result = value.get(key)
    if not isinstance(result, list):
        raise ValueError(f"{context} requires integer array field {key!r}")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise ValueError(f"{context} field {key!r} must contain integers")
    if len(result) != len(set(result)):
        raise ValueError(f"{context} field {key!r} contains duplicates")
    return result


def _reject_non_finite_json_number(value: str) -> None:
    raise ValueError(f"Stage snapshot manifest contains non-finite JSON: {value}")
