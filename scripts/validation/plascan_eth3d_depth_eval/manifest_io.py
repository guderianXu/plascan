"""Strict PlaScan MVS manifest reader for depth-evaluation provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .camera_io import ColmapCamera


MVS_WORKSPACE_SCHEMA = "plascan.mvs.workspace.v2"


@dataclass(frozen=True)
class PredictionManifestFrame:
    schema: str
    algorithm_revision: int
    config_hash: str
    frame_position: int
    ref_index: int
    status: str
    acceptance: str
    fusion_eligible: bool
    grid_width: int
    grid_height: int
    ref_image: Path
    raw_depth_path: Path
    manifest_camera_model: dict[str, float]
    rotation_world_to_camera: tuple[float, ...]
    camera_center: tuple[float, ...]
    translation_world_to_camera: tuple[float, ...]
    effective_native_final_depth_grid: bool
    pixel_domain_diagnostics: dict[str, object]
    manifest_size_bytes: int
    manifest_sha256: str
    camera: ColmapCamera

    def provenance_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "algorithm_revision": self.algorithm_revision,
            "config_hash": self.config_hash,
            "frame_position": self.frame_position,
            "ref_index": self.ref_index,
            "status": self.status,
            "acceptance": self.acceptance,
            "fusion_eligible": self.fusion_eligible,
            "effective_native_final_depth_grid": (
                self.effective_native_final_depth_grid
            ),
            "raw_depth_path": str(self.raw_depth_path),
            "ref_image": str(self.ref_image),
        }


def read_prediction_manifest_frame(
    path: Path,
    ref_index: int,
    *,
    camera_id: int = 0,
    artifact_base_dir: Path | None = None,
) -> PredictionManifestFrame:
    """Read one completed final frame record from an MVS workspace manifest."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction manifest not found: {path}")
    if isinstance(ref_index, bool) or not isinstance(ref_index, int) or ref_index < 0:
        raise ValueError("Prediction manifest ref index must be non-negative")
    try:
        manifest_bytes = path.read_bytes()
        root = json.loads(
            manifest_bytes.decode("utf-8"),
            parse_constant=_reject_non_finite_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid prediction manifest JSON {path}: {error}") from error
    if not isinstance(root, dict):
        raise ValueError(f"Prediction manifest root must be an object: {path}")

    schema = _required_string(root, "schema", "manifest root")
    if schema != MVS_WORKSPACE_SCHEMA:
        raise ValueError(
            f"Unsupported prediction manifest schema {schema!r}: {path}"
        )
    algorithm_revision = _required_positive_integer(
        root, "algorithm_revision", "manifest root"
    )
    config_hash = _required_string(root, "config_hash", "manifest root")
    frames = root.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Prediction manifest frames must be an array: {path}")

    matches: list[tuple[int, dict[str, Any]]] = []
    for position, value in enumerate(frames):
        if not isinstance(value, dict):
            raise ValueError(
                f"Prediction manifest frame {position} must be an object: {path}"
            )
        value_ref_index = value.get("ref_index")
        if isinstance(value_ref_index, bool) or not isinstance(value_ref_index, int):
            raise ValueError(
                f"Prediction manifest frame {position} has invalid ref_index: {path}"
            )
        if value_ref_index == ref_index:
            matches.append((position, value))
    if not matches:
        raise ValueError(
            f"Prediction ref_index {ref_index} is not present in manifest: {path}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Prediction ref_index {ref_index} occurs {len(matches)} times in "
            f"manifest; final frame record is ambiguous: {path}"
        )

    frame_position, record = matches[0]
    context = f"prediction manifest frame {frame_position} (ref_index={ref_index})"
    status = _required_string(record, "status", context)
    if status != "completed":
        raise ValueError(f"{context} is not completed; status={status!r}")
    acceptance = _required_string(record, "acceptance", context)
    if acceptance not in {"accepted", "validation_only", "rejected"}:
        raise ValueError(f"{context} has invalid acceptance={acceptance!r}")
    fusion_eligible = _required_boolean(record, "fusion_eligible", context)
    record_revision = _required_positive_integer(
        record, "algorithm_revision", context
    )
    if record_revision != algorithm_revision:
        raise ValueError(
            f"{context} algorithm_revision {record_revision} does not match "
            f"manifest revision {algorithm_revision}"
        )
    record_config_hash = _required_string(record, "config_hash", context)
    if record_config_hash != config_hash:
        raise ValueError(
            f"{context} config_hash does not match the manifest config_hash"
        )

    width = _required_positive_integer(record, "grid_width", context)
    height = _required_positive_integer(record, "grid_height", context)
    artifact_directory = (
        artifact_base_dir.resolve()
        if artifact_base_dir is not None
        else path.parent
    )
    ref_image = _resolved_artifact_path(
        _required_string(record, "ref_image", context), artifact_directory
    )
    raw_depth_path = _resolved_artifact_path(
        _required_string(record, "raw_depth_path", context), artifact_directory
    )
    (
        camera,
        manifest_camera_model,
        rotation_world_to_camera,
        camera_center,
        translation_world_to_camera,
    ) = read_plascan_pinhole_camera_model(
        record.get("camera_model"),
        camera_id=camera_id,
        width=width,
        height=height,
        context=context,
    )
    (
        effective_native_final_depth_grid,
        pixel_domain_diagnostics,
    ) = _read_pixel_domain_contract(
        record,
        algorithm_revision=algorithm_revision,
        width=width,
        height=height,
        context=context,
    )
    return PredictionManifestFrame(
        schema=schema,
        algorithm_revision=algorithm_revision,
        config_hash=config_hash,
        frame_position=frame_position,
        ref_index=ref_index,
        status=status,
        acceptance=acceptance,
        fusion_eligible=fusion_eligible,
        grid_width=width,
        grid_height=height,
        ref_image=ref_image,
        raw_depth_path=raw_depth_path,
        manifest_camera_model=manifest_camera_model,
        rotation_world_to_camera=rotation_world_to_camera,
        camera_center=camera_center,
        translation_world_to_camera=translation_world_to_camera,
        effective_native_final_depth_grid=(
            effective_native_final_depth_grid
        ),
        pixel_domain_diagnostics=pixel_domain_diagnostics,
        manifest_size_bytes=len(manifest_bytes),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        camera=camera,
    )


def validate_manifest_pixel_domain_against_official_camera(
    frame: PredictionManifestFrame,
    official_camera: ColmapCamera,
) -> dict[str, object]:
    """Bind revision-40 pixel-domain provenance to the official raster."""

    reduced_grid = (
        frame.grid_width != official_camera.width
        or frame.grid_height != official_camera.height
    )
    if frame.algorithm_revision < 40:
        if reduced_grid:
            raise ValueError(
                "Legacy prediction manifest revisions may only be evaluated "
                "on the full official raster"
            )
        return {
            "contract": "legacy_full_raster",
            "algorithm_revision": frame.algorithm_revision,
        }

    diagnostics = frame.pixel_domain_diagnostics
    raster_width = _required_positive_integer(
        diagnostics, "raster_width", "pixel_domain_diagnostics"
    )
    raster_height = _required_positive_integer(
        diagnostics, "raster_height", "pixel_domain_diagnostics"
    )
    if (raster_width, raster_height) != (
        official_camera.width,
        official_camera.height,
    ):
        raise ValueError(
            "Prediction pixel_domain_diagnostics raster does not match the "
            "selected official camera: "
            f"{raster_width}x{raster_height} != "
            f"{official_camera.width}x{official_camera.height}"
        )
    return {
        "contract": "revision_40_strict",
        "algorithm_revision": frame.algorithm_revision,
        "official_raster_width": official_camera.width,
        "official_raster_height": official_camera.height,
    }


def _read_pixel_domain_contract(
    record: dict[str, Any],
    *,
    algorithm_revision: int,
    width: int,
    height: int,
    context: str,
) -> tuple[bool, dict[str, object]]:
    if algorithm_revision < 40:
        effective = record.get("effective_native_final_depth_grid", False)
        if not isinstance(effective, bool):
            raise ValueError(
                f"{context} field 'effective_native_final_depth_grid' "
                "must be boolean"
            )
        return effective, _optional_object(
            record, "pixel_domain_diagnostics", context
        )

    effective = _required_boolean(
        record, "effective_native_final_depth_grid", context
    )
    diagnostics = _required_object(
        record, "pixel_domain_diagnostics", context
    )
    diagnostics_context = f"{context} pixel_domain_diagnostics"
    configured_domain = _required_string(
        diagnostics, "configured_pixel_domain", diagnostics_context
    )
    if configured_domain != "prepared_full_raster":
        raise ValueError(
            f"{diagnostics_context} configured_pixel_domain must be "
            f"'prepared_full_raster', got {configured_domain!r}"
        )
    effective_domain = _required_string(
        diagnostics, "effective_pixel_domain", diagnostics_context
    )
    if effective_domain != "depth_grid":
        raise ValueError(
            f"{diagnostics_context} effective_pixel_domain must be "
            f"'depth_grid', got {effective_domain!r}"
        )
    requested_native = _required_boolean(
        diagnostics,
        "requested_native_final_depth_grid",
        diagnostics_context,
    )
    diagnostic_effective = _required_boolean(
        diagnostics,
        "effective_native_final_depth_grid",
        diagnostics_context,
    )
    if diagnostic_effective != effective:
        raise ValueError(
            f"{diagnostics_context} effective_native_final_depth_grid "
            "does not match the frame field"
        )
    if effective and not requested_native:
        raise ValueError(
            f"{diagnostics_context} effective_native_final_depth_grid=true "
            "requires requested_native_final_depth_grid=true"
        )

    raster_width = _required_positive_integer(
        diagnostics, "raster_width", diagnostics_context
    )
    raster_height = _required_positive_integer(
        diagnostics, "raster_height", diagnostics_context
    )
    grid_width = _required_positive_integer(
        diagnostics, "grid_width", diagnostics_context
    )
    grid_height = _required_positive_integer(
        diagnostics, "grid_height", diagnostics_context
    )
    if (grid_width, grid_height) != (width, height):
        raise ValueError(
            f"{diagnostics_context} grid does not match frame grid: "
            f"{grid_width}x{grid_height} != {width}x{height}"
        )

    scale_x = _required_positive_finite_number(
        diagnostics, "scale_x", diagnostics_context
    )
    scale_y = _required_positive_finite_number(
        diagnostics, "scale_y", diagnostics_context
    )
    linear_scale = _required_positive_finite_number(
        diagnostics, "linear_scale", diagnostics_context
    )
    area_scale = _required_positive_finite_number(
        diagnostics, "area_scale", diagnostics_context
    )
    expected_scale_x = grid_width / raster_width
    expected_scale_y = grid_height / raster_height
    expected_area_scale = expected_scale_x * expected_scale_y
    expected_linear_scale = math.sqrt(expected_area_scale)
    for name, actual, expected in (
        ("scale_x", scale_x, expected_scale_x),
        ("scale_y", scale_y, expected_scale_y),
        ("area_scale", area_scale, expected_area_scale),
        ("linear_scale", linear_scale, expected_linear_scale),
    ):
        _require_close(name, actual, expected, diagnostics_context)

    grid_matches_raster = _required_boolean(
        diagnostics, "grid_matches_raster", diagnostics_context
    )
    expected_grid_matches_raster = (
        grid_width == raster_width and grid_height == raster_height
    )
    if grid_matches_raster != expected_grid_matches_raster:
        raise ValueError(
            f"{diagnostics_context} grid_matches_raster is inconsistent "
            "with the declared dimensions"
        )
    _required_object(diagnostics, "parameters", diagnostics_context)

    if not expected_grid_matches_raster and (
        not requested_native or not effective or not diagnostic_effective
    ):
        raise ValueError(
            f"{diagnostics_context} reduced grid requires requested and "
            "effective native-final-depth-grid flags"
        )
    return effective, diagnostics


def read_plascan_pinhole_camera_model(
    value: object,
    *,
    camera_id: int,
    width: int,
    height: int,
    context: str,
) -> tuple[
    ColmapCamera,
    dict[str, float],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} camera_model must be an object")
    declared_model = value.get("model")
    if declared_model is not None and declared_model != "PINHOLE":
        raise ValueError(
            f"{context} camera_model must be PINHOLE, got {declared_model!r}"
        )
    for dimension_name, expected in (("width", width), ("height", height)):
        if dimension_name in value and value[dimension_name] != expected:
            raise ValueError(
                f"{context} camera_model {dimension_name} does not match "
                f"grid_{dimension_name}: {value[dimension_name]!r} != {expected}"
            )
    distortion_model = value.get("distortion_model")
    if distortion_model is not None and str(distortion_model).strip().lower() not in {
        "",
        "none",
        "pinhole",
    }:
        raise ValueError(
            f"{context} camera_model is distorted: "
            f"distortion_model={distortion_model!r}"
        )
    distortion = value.get("distortion_coefficients")
    if distortion is not None:
        if not isinstance(distortion, list) or not distortion:
            raise ValueError(
                f"{context} distortion_coefficients must be a non-empty array"
            )
        coefficients = np.asarray(distortion, dtype=np.float64)
        if coefficients.ndim != 1 or not np.all(np.isfinite(coefficients)):
            raise ValueError(
                f"{context} distortion_coefficients must all be finite"
            )
        if np.any(coefficients != 0.0):
            raise ValueError(f"{context} camera_model must be undistorted")
    for coefficient_name in (
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
        "k6",
        "p1",
        "p2",
        "s1",
        "s2",
        "s3",
        "s4",
    ):
        if coefficient_name not in value:
            continue
        coefficient = _required_finite_number(
            value, coefficient_name, f"{context} camera_model"
        )
        if coefficient != 0.0:
            raise ValueError(
                f"{context} camera_model must be undistorted; "
                f"{coefficient_name}={coefficient}"
            )

    raster_params = tuple(
        _required_finite_number(value, name, f"{context} camera_model")
        for name in ("fx", "fy", "cx", "cy")
    )
    rotation_world_to_camera = _required_finite_number_array(
        value, "rotation_world_to_camera", 9, f"{context} camera_model"
    )
    camera_center = _required_finite_number_array(
        value, "camera_center", 3, f"{context} camera_model"
    )
    translation_world_to_camera = _required_finite_number_array(
        value, "translation_world_to_camera", 3, f"{context} camera_model"
    )
    # PlaScan/OpenCV projects the centre of array element [0, 0] at (0, 0),
    # while COLMAP stores the same pixel centre at (0.5, 0.5).  Remapping uses
    # COLMAP cameras throughout, so convert only at this manifest boundary.
    camera = ColmapCamera(
        camera_id=camera_id,
        model="PINHOLE",
        width=width,
        height=height,
        params=(
            raster_params[0],
            raster_params[1],
            raster_params[2] + 0.5,
            raster_params[3] + 0.5,
        ),
    )
    manifest_camera_model = {
        name: float(parameter)
        for name, parameter in zip(
            ("fx", "fy", "cx", "cy"), raster_params, strict=True
        )
    }
    return (
        camera,
        manifest_camera_model,
        rotation_world_to_camera,
        camera_center,
        translation_world_to_camera,
    )


def _resolved_artifact_path(value: str, manifest_directory: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_directory / path
    return path.resolve()


def _required_string(value: dict[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context} requires non-empty string field {key!r}")
    return result


def _required_boolean(value: dict[str, Any], key: str, context: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"{context} requires boolean field {key!r}")
    return result


def _optional_object(
    value: dict[str, Any], key: str, context: str
) -> dict[str, object]:
    result = value.get(key)
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise ValueError(f"{context} field {key!r} must be an object")
    return result


def _required_object(
    value: dict[str, Any], key: str, context: str
) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{context} requires object field {key!r}")
    return result


def _required_positive_integer(
    value: dict[str, Any], key: str, context: str
) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise ValueError(f"{context} requires positive integer field {key!r}")
    return result


def _required_finite_number(
    value: dict[str, Any], key: str, context: str
) -> float:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"{context} requires numeric field {key!r}")
    result = float(result)
    if not np.isfinite(result):
        raise ValueError(f"{context} field {key!r} must be finite")
    return result


def _required_positive_finite_number(
    value: dict[str, Any], key: str, context: str
) -> float:
    result = _required_finite_number(value, key, context)
    if result <= 0.0:
        raise ValueError(f"{context} field {key!r} must be positive")
    return result


def _require_close(
    name: str,
    actual: float,
    expected: float,
    context: str,
) -> None:
    tolerance = max(1.0e-12, abs(expected) * 1.0e-12)
    if abs(actual - expected) > tolerance:
        raise ValueError(
            f"{context} field {name!r} is inconsistent: "
            f"{actual:.17g} != {expected:.17g}"
        )


def _required_finite_number_array(
    value: dict[str, Any], key: str, count: int, context: str
) -> tuple[float, ...]:
    result = value.get(key)
    if not isinstance(result, list) or len(result) != count:
        raise ValueError(
            f"{context} requires {count}-element numeric array field {key!r}"
        )
    array = np.asarray(result, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{context} field {key!r} must contain finite numbers")
    return tuple(float(item) for item in array)


def _reject_non_finite_json_number(value: str) -> None:
    raise ValueError(f"Prediction manifest contains non-finite JSON number: {value}")
