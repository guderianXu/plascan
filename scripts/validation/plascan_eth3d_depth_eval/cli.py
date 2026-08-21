"""Command-line entry point for reproducible ETH3D depth evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import platform
from pathlib import Path
from typing import Sequence

import numpy as np

from .camera_io import select_colmap_camera, validate_scaled_pinhole_camera
from .depth_io import (
    CV_8UC1,
    CV_32FC1,
    read_eth3d_raw_depth,
    read_plascan_fast_matrix,
    read_prediction_depth,
)
from .evaluation import evaluate_depth_prediction
from .image_io import select_colmap_image_pose, validate_manifest_pose
from .manifest_io import (
    PredictionManifestFrame,
    read_prediction_manifest_frame,
    validate_manifest_pixel_domain_against_official_camera,
)
from .output_io import (
    PreparedOutput,
    prepare_npy_output,
    prepare_text_output,
    publish_outputs,
)
from .remap import remap_eth3d_depth_to_undistorted
from .snapshot_io import InputSnapshotSet
from .stage_manifest_io import (
    STAGE_IDS,
    StageSnapshotRecord,
    read_stage_snapshot_record,
    validate_stage_snapshot_against_workspace,
)


@dataclass(frozen=True)
class _FrozenOutputPaths:
    report: Path
    remapped_ground_truth: Path | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map raw ETH3D THIN_PRISM_FISHEYE ground-truth depth into the "
            "official undistorted PINHOLE image domain and evaluate a prediction."
        )
    )
    parser.add_argument("--raw-depth", required=True, type=Path)
    parser.add_argument("--raw-cameras", required=True, type=Path)
    parser.add_argument("--undistorted-cameras", required=True, type=Path)
    parser.add_argument(
        "--undistorted-images",
        type=Path,
        help=(
            "Official COLMAP images.txt used to bind a manifest frame to its "
            "image name and fixed pose. Required with --prediction-manifest."
        ),
    )
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        help=(
            "Optional PlaScan mvs_manifest.json containing the prediction's "
            "native grid and camera. Requires --prediction-ref-index."
        ),
    )
    parser.add_argument(
        "--prediction-ref-index",
        type=int,
        help=(
            "Reference-frame index to select from --prediction-manifest. "
            "Requires --prediction-manifest."
        ),
    )
    parser.add_argument(
        "--stage-snapshot-manifest",
        type=Path,
        help=(
            "Optional finalized diagnostic stage-snapshot manifest. Requires "
            "--prediction-manifest, --prediction-ref-index and "
            "--stage-snapshot-stage."
        ),
    )
    parser.add_argument(
        "--stage-snapshot-stage",
        choices=sorted(STAGE_IDS),
        help="Captured MVS stage to evaluate as a non-publishable diagnostic.",
    )
    parser.add_argument(
        "--allow-non-publishable-frame",
        action="store_true",
        help=(
            "Permit diagnostic scoring of validation-only or rejected frames. "
            "The report marks this as non-publishable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing report/remapped-GT outputs.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-camera-id", type=int)
    parser.add_argument("--undistorted-camera-id", type=int)
    parser.add_argument(
        "--prediction-format",
        choices=["auto", "npy", "plascan-fast-matrix", "raw-float32"],
        default="auto",
    )
    parser.add_argument(
        "--maximum-relative-depth-spread",
        type=float,
        default=0.02,
        help=(
            "Reject a raw measurement when an adjacent valid depth differs "
            "by more than this fraction of the nearer measurement."
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument(
        "--maximum-reprojection-error",
        type=float,
        default=1.0e-4,
        help="Strict raw/undistorted camera round-trip tolerance in pixels.",
    )
    parser.add_argument(
        "--remapped-ground-truth-output",
        type=Path,
        help="Optional NPY file for the remapped float32 ground truth.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_paths = _freeze_output_paths(args)
    _validate_arguments(args)
    _validate_output_paths(args, output_paths)
    prepared_report: PreparedOutput | None = None
    prepared_remapped: PreparedOutput | None = None
    output_path = output_paths.report
    try:
        with InputSnapshotSet() as snapshots:
            raw_depth_input = snapshots.capture(args.raw_depth)
            raw_cameras_input = snapshots.capture(args.raw_cameras)
            undistorted_cameras_input = snapshots.capture(
                args.undistorted_cameras
            )
            prediction_input = snapshots.capture(args.prediction)
            manifest_input = (
                snapshots.capture(args.prediction_manifest)
                if args.prediction_manifest is not None
                else None
            )
            stage_manifest_input = (
                snapshots.capture(args.stage_snapshot_manifest)
                if args.stage_snapshot_manifest is not None
                else None
            )
            images_input = (
                snapshots.capture(args.undistorted_images)
                if args.undistorted_images is not None
                else None
            )

            raw_camera = select_colmap_camera(
                raw_cameras_input.snapshot_path,
                "THIN_PRISM_FISHEYE",
                args.raw_camera_id,
            )
            official_camera = select_colmap_camera(
                undistorted_cameras_input.snapshot_path,
                "PINHOLE",
                args.undistorted_camera_id,
            )
            manifest_frame: PredictionManifestFrame | None = None
            manifest_ref_image_input = None
            camera_scaling: dict[str, object] | None = None
            pixel_domain_validation: dict[str, object] | None = None
            pose_validation: dict[str, object] | None = None
            official_pose: dict[str, object] | None = None
            stage_record: StageSnapshotRecord | None = None
            stage_validation: dict[str, object] | None = None
            stage_confidence_input = None
            stage_valid_mask_input = None
            stage_valid_mask_values: np.ndarray | None = None
            stage_confidence_values: np.ndarray | None = None
            target_camera = official_camera
            if manifest_input is not None:
                manifest_frame = read_prediction_manifest_frame(
                    manifest_input.snapshot_path,
                    args.prediction_ref_index,
                    camera_id=official_camera.camera_id,
                    artifact_base_dir=manifest_input.source_path.parent,
                )
                _validate_parsed_manifest_snapshot(
                    manifest_frame,
                    manifest_input.size_bytes,
                    manifest_input.sha256,
                )
                target_camera = manifest_frame.camera
                expected_prediction_path = manifest_frame.raw_depth_path
                expected_prediction_label = "raw_depth_path"
                if stage_manifest_input is not None:
                    stage_record = read_stage_snapshot_record(
                        stage_manifest_input.snapshot_path,
                        args.prediction_ref_index,
                        args.stage_snapshot_stage,
                        camera_id=official_camera.camera_id,
                        artifact_base_dir=(
                            stage_manifest_input.source_path.parent
                        ),
                    )
                    _validate_parsed_stage_snapshot(
                        stage_record,
                        stage_manifest_input.size_bytes,
                        stage_manifest_input.sha256,
                    )
                    stage_validation = validate_stage_snapshot_against_workspace(
                        stage_record, manifest_frame
                    )
                    target_camera = stage_record.camera
                    expected_prediction_path = stage_record.depth.path
                    expected_prediction_label = "stage depth artifact path"
                    stage_confidence_input = snapshots.capture(
                        stage_record.confidence.path
                    )
                    stage_valid_mask_input = snapshots.capture(
                        stage_record.valid_mask.path
                    )
                    _validate_stage_artifact_snapshot(
                        stage_confidence_input, stage_record.confidence
                    )
                    _validate_stage_artifact_snapshot(
                        stage_valid_mask_input, stage_record.valid_mask
                    )
                    _validate_stage_artifact_snapshot(
                        prediction_input, stage_record.depth
                    )
                    _validate_dynamic_input_output_paths(
                        output_paths,
                        (
                            stage_manifest_input.source_path,
                            stage_confidence_input.source_path,
                            stage_valid_mask_input.source_path,
                        ),
                    )
                    stage_confidence_values = read_plascan_fast_matrix(
                        stage_confidence_input.snapshot_path, CV_32FC1
                    )
                    stage_valid_mask_values = read_plascan_fast_matrix(
                        stage_valid_mask_input.snapshot_path, CV_8UC1
                    )
                    _validate_stage_companion_payloads(
                        stage_record,
                        stage_confidence_values,
                        stage_valid_mask_values,
                    )
                if prediction_input.source_path != expected_prediction_path:
                    raise ValueError(
                        "--prediction does not match the selected manifest "
                        f"{expected_prediction_label}: "
                        f"{prediction_input.source_path} != "
                        f"{expected_prediction_path}"
                    )
                if args.raw_depth.name != manifest_frame.ref_image.name:
                    raise ValueError(
                        "Raw ETH3D depth filename does not match the selected "
                        f"manifest ref_image: {args.raw_depth.name!r} "
                        f"!= {manifest_frame.ref_image.name!r}"
                    )
                manifest_ref_image_input = snapshots.capture(
                    manifest_frame.ref_image
                )
                _validate_manifest_ref_image_output_paths(
                    output_paths,
                    manifest_ref_image_input.source_path,
                )
                if stage_record is None:
                    _validate_publishable_scope(args, manifest_frame)
                selected_pose = select_colmap_image_pose(
                    images_input.snapshot_path,
                    manifest_frame.ref_image.name,
                )
                if selected_pose.camera_id != official_camera.camera_id:
                    raise ValueError(
                        "Official image pose camera_id does not match the "
                        "selected official camera: "
                        f"{selected_pose.camera_id} != "
                        f"{official_camera.camera_id}"
                    )
                pose_validation = validate_manifest_pose(
                    selected_pose,
                    manifest_frame.rotation_world_to_camera,
                    manifest_frame.translation_world_to_camera,
                    manifest_frame.camera_center,
                )
                official_pose = selected_pose.as_dict()
                camera_scaling = validate_scaled_pinhole_camera(
                    official_camera,
                    target_camera,
                )
                pixel_domain_validation = (
                    validate_manifest_pixel_domain_against_official_camera(
                        manifest_frame,
                        official_camera,
                    )
                )

            raw_depth = read_eth3d_raw_depth(
                raw_depth_input.snapshot_path,
                raw_camera.width,
                raw_camera.height,
            )
            prediction = read_prediction_depth(
                prediction_input.snapshot_path,
                target_camera.width,
                target_camera.height,
                args.prediction_format,
            )
            if stage_record is not None:
                prediction_valid = np.isfinite(prediction.values) & (
                    prediction.values > 0.0
                )
                if not np.array_equal(
                    prediction_valid, stage_valid_mask_values != 0
                ):
                    raise ValueError(
                        "Stage snapshot valid mask does not exactly match the "
                        "positive finite depth support"
                    )
            remapped = remap_eth3d_depth_to_undistorted(
                raw_depth.values,
                raw_camera,
                target_camera,
                maximum_relative_depth_spread=(
                    args.maximum_relative_depth_spread
                ),
                chunk_rows=args.chunk_rows,
                maximum_reprojection_error=args.maximum_reprojection_error,
            )
            metrics = evaluate_depth_prediction(
                prediction.values, remapped.depth
            )

            remapped_output_path: str | None = None
            remapped_output_artifact: dict[str, object] | None = None
            if output_paths.remapped_ground_truth is not None:
                prepared_remapped = prepare_npy_output(
                    output_paths.remapped_ground_truth,
                    remapped.depth,
                )
                remapped_output_path = str(prepared_remapped.final_path)
                remapped_output_artifact = prepared_remapped.report_record()

            input_records = {
                "raw_depth": raw_depth_input.report_record(),
                "raw_cameras": raw_cameras_input.report_record(),
                "undistorted_cameras": (
                    undistorted_cameras_input.report_record()
                ),
                "prediction": {
                    **prediction_input.report_record(),
                    "format": prediction.format,
                },
            }
            conventions = _evaluation_conventions()
            options = {
                "maximum_relative_depth_spread": (
                    args.maximum_relative_depth_spread
                ),
                "chunk_rows": args.chunk_rows,
                "maximum_reprojection_error_pixels": (
                    args.maximum_reprojection_error
                ),
            }
            if manifest_frame is not None:
                input_records["prediction_manifest"] = (
                    manifest_input.report_record()
                )
                input_records["undistorted_images"] = images_input.report_record()
                input_records["manifest_ref_image"] = (
                    manifest_ref_image_input.report_record()
                )
                conventions["official_to_prediction_camera_scaling"] = (
                    "COLMAP f'=f*scale and c'=c*scale; PlaScan manifest "
                    "c_manifest'=c_colmap*scale-0.5"
                )
                options["prediction_ref_index"] = args.prediction_ref_index
            if stage_record is not None:
                input_records["stage_snapshot_manifest"] = (
                    stage_manifest_input.report_record()
                )
                input_records["stage_snapshot_confidence"] = (
                    stage_confidence_input.report_record()
                )
                input_records["stage_snapshot_valid_mask"] = (
                    stage_valid_mask_input.report_record()
                )
                options["stage_snapshot_stage"] = stage_record.stage

            report = {
                "schema": "plascan.eth3d_depth_evaluation.v1",
                "runtime": {
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                },
                "inputs": input_records,
                "raw_camera": raw_camera.as_dict(),
                "undistorted_camera": official_camera.as_dict(),
                "conventions": conventions,
                "options": options,
                "raw_depth_validity": raw_depth.validity,
                "prediction_validity": prediction.validity,
                "remap": remapped.diagnostics,
                "metrics": metrics,
                "remapped_ground_truth_output": remapped_output_path,
            }
            if remapped_output_artifact is not None:
                report["remapped_ground_truth_artifact"] = (
                    remapped_output_artifact
                )
            if manifest_frame is not None:
                _add_manifest_report_fields(
                    report,
                    manifest_frame,
                    target_camera.as_dict(),
                    camera_scaling,
                    pixel_domain_validation,
                    official_pose,
                    pose_validation,
                )
            if stage_record is not None:
                report["evaluation_scope"] = "diagnostic_stage_snapshot"
                report["stage_snapshot_record"] = stage_record.provenance_dict()
                report["stage_snapshot_workspace_validation"] = stage_validation
                report["stage_snapshot_payload_diagnostics"] = (
                    _stage_payload_diagnostics(
                        stage_confidence_values, stage_valid_mask_values
                    )
                )

            report_text = json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n"
            prepared_report = prepare_text_output(output_path, report_text)
            snapshots.verify_unchanged()
            publish_outputs(
                prepared_report,
                prepared_remapped,
                overwrite=args.overwrite,
            )
    finally:
        if prepared_report is not None:
            prepared_report.cleanup()
        if prepared_remapped is not None:
            prepared_remapped.cleanup()

    relative_p95 = metrics["relative_absolute_depth_error"]["p95"]
    print(
        "ETH3D depth evaluation: "
        f"GT coverage={metrics['prediction_coverage_of_ground_truth']:.6f}, "
        f"common={metrics['common_valid_pixel_count']}, "
        f"relative p95={relative_p95 if relative_p95 is not None else 'n/a'}"
    )
    print(f"Report: {output_path}")
    return 0


def _validate_parsed_manifest_snapshot(
    frame: PredictionManifestFrame,
    snapshot_size_bytes: int,
    snapshot_sha256: str,
) -> None:
    if (
        frame.manifest_size_bytes != snapshot_size_bytes
        or frame.manifest_sha256 != snapshot_sha256
    ):
        raise ValueError(
            "Parsed prediction manifest does not match its immutable input "
            "snapshot"
        )


def _validate_parsed_stage_snapshot(
    record: StageSnapshotRecord,
    snapshot_size_bytes: int,
    snapshot_sha256: str,
) -> None:
    if (
        record.manifest_size_bytes != snapshot_size_bytes
        or record.manifest_sha256 != snapshot_sha256
    ):
        raise ValueError(
            "Parsed stage snapshot manifest does not match its immutable input "
            "snapshot"
        )


def _validate_stage_artifact_snapshot(snapshot, artifact) -> None:
    if (
        snapshot.source_path != artifact.path
        or snapshot.size_bytes != artifact.size_bytes
        or snapshot.sha256 != artifact.sha256
    ):
        raise ValueError(
            "Stage snapshot artifact does not match its manifest fingerprint: "
            f"{artifact.path}"
        )


def _validate_stage_companion_payloads(
    record: StageSnapshotRecord,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    expected_shape = (record.snapshot_height, record.snapshot_width)
    if confidence.shape != expected_shape or valid_mask.shape != expected_shape:
        raise ValueError("Stage snapshot companion raster dimensions are inconsistent")
    if np.any(~np.isfinite(confidence)):
        raise ValueError("Stage snapshot confidence contains non-finite values")
    if np.any((valid_mask != 0) & (valid_mask != 255)):
        raise ValueError("Stage snapshot valid mask must be binary 0/255")
    if int(np.count_nonzero(valid_mask)) != record.valid_pixel_count:
        raise ValueError("Stage snapshot valid mask count does not match its manifest")


def _stage_payload_diagnostics(
    confidence: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, object]:
    valid_confidence = confidence[valid_mask != 0].astype(np.float64)
    if valid_confidence.size == 0:
        confidence_summary = {
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    else:
        confidence_summary = {
            "mean": float(np.mean(valid_confidence)),
            "median": float(np.median(valid_confidence)),
            "p05": float(np.percentile(valid_confidence, 5.0)),
            "p95": float(np.percentile(valid_confidence, 95.0)),
        }
    return {
        "valid_mask_nonzero_count": int(np.count_nonzero(valid_mask)),
        "valid_mask_fraction": float(np.count_nonzero(valid_mask) / valid_mask.size),
        "confidence_on_valid_support": confidence_summary,
    }


def _validate_dynamic_input_output_paths(
    output_paths: _FrozenOutputPaths,
    input_paths: tuple[Path, ...],
) -> None:
    for input_path in input_paths:
        if output_paths.report == input_path:
            raise ValueError("--output must not overwrite a stage snapshot input")
        if output_paths.remapped_ground_truth == input_path:
            raise ValueError(
                "--remapped-ground-truth-output must not overwrite a stage "
                "snapshot input"
            )


def _validate_publishable_scope(
    args: argparse.Namespace,
    frame: PredictionManifestFrame,
) -> None:
    if args.allow_non_publishable_frame:
        return
    if frame.acceptance != "accepted" or not frame.fusion_eligible:
        raise ValueError(
            "Selected manifest frame is not publishable: acceptance="
            f"{frame.acceptance!r}, fusion_eligible={frame.fusion_eligible}; "
            "use --allow-non-publishable-frame only for explicit diagnostics"
        )


def _validate_manifest_ref_image_output_paths(
    output_paths: _FrozenOutputPaths,
    ref_image_path: Path,
) -> None:
    if output_paths.report == ref_image_path:
        raise ValueError("--output must not overwrite the manifest ref_image")
    if (
        output_paths.remapped_ground_truth is not None
        and output_paths.remapped_ground_truth == ref_image_path
    ):
        raise ValueError(
            "--remapped-ground-truth-output must not overwrite the manifest "
            "ref_image"
        )


def _evaluation_conventions() -> dict[str, object]:
    return {
        "raw_depth_dtype": "little-endian float32",
        "raw_invalid_depth": "non-finite or non-positive and therefore unscored",
        "ground_truth_missing_semantics": (
            "unobserved, not evidence that a prediction is invalid"
        ),
        "camera_model_equations": "COLMAP THIN_PRISM_FISHEYE and PINHOLE",
        "raster_pixel_center_to_colmap": "(column + 0.5, row + 0.5)",
        "colmap_to_raster_pixel_center": "(x - 0.5, y - 0.5)",
        "depth_value_transform": "unchanged_on_corresponding_camera_ray",
        "relative_error_denominator": "remapped_ground_truth_depth",
    }


def _add_manifest_report_fields(
    report: dict[str, object],
    frame: PredictionManifestFrame,
    target_camera: dict[str, object],
    camera_scaling: dict[str, object],
    pixel_domain_validation: dict[str, object],
    official_pose: dict[str, object],
    pose_validation: dict[str, object],
) -> None:
    report["evaluation_scope"] = (
        "publishable_frame"
        if frame.acceptance == "accepted" and frame.fusion_eligible
        else "diagnostic_non_publishable_frame"
    )
    report["prediction_manifest_frame"] = frame.provenance_dict()
    report["prediction_manifest_camera_model"] = {
        "coordinate_convention": "PlaScan/OpenCV raster index origin",
        "width": frame.grid_width,
        "height": frame.grid_height,
        **frame.manifest_camera_model,
    }
    report["prediction_camera"] = target_camera
    report["prediction_camera_scaling_validation"] = camera_scaling
    report["prediction_pixel_domain_validation"] = pixel_domain_validation
    report["official_image_pose"] = official_pose
    report["prediction_pose_validation"] = pose_validation
    report["prediction_pixel_domain_diagnostics"] = (
        frame.pixel_domain_diagnostics
    )


def _validate_arguments(args: argparse.Namespace) -> None:
    if (args.prediction_manifest is None) != (args.prediction_ref_index is None):
        raise ValueError(
            "--prediction-manifest and --prediction-ref-index must be used together"
        )
    if args.prediction_ref_index is not None and args.prediction_ref_index < 0:
        raise ValueError("--prediction-ref-index must be non-negative")
    if (args.prediction_manifest is None) != (args.undistorted_images is None):
        raise ValueError(
            "--undistorted-images is required exactly when "
            "--prediction-manifest is used"
        )
    if args.allow_non_publishable_frame and args.prediction_manifest is None:
        raise ValueError(
            "--allow-non-publishable-frame requires --prediction-manifest"
        )
    if (args.stage_snapshot_manifest is None) != (
        args.stage_snapshot_stage is None
    ):
        raise ValueError(
            "--stage-snapshot-manifest and --stage-snapshot-stage must be used "
            "together"
        )
    if (
        args.stage_snapshot_manifest is not None
        and args.prediction_manifest is None
    ):
        raise ValueError(
            "Stage snapshot evaluation requires --prediction-manifest and "
            "--prediction-ref-index"
        )
    if (
        not np.isfinite(args.maximum_relative_depth_spread)
        or args.maximum_relative_depth_spread < 0.0
    ):
        raise ValueError(
            "--maximum-relative-depth-spread must be finite and non-negative"
        )
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    if (
        not np.isfinite(args.maximum_reprojection_error)
        or args.maximum_reprojection_error <= 0.0
    ):
        raise ValueError(
            "--maximum-reprojection-error must be finite and positive"
        )


def _freeze_output_paths(args: argparse.Namespace) -> _FrozenOutputPaths:
    return _FrozenOutputPaths(
        report=args.output.resolve(),
        remapped_ground_truth=(
            args.remapped_ground_truth_output.resolve()
            if args.remapped_ground_truth_output is not None
            else None
        ),
    )


def _validate_output_paths(
    args: argparse.Namespace,
    output_paths: _FrozenOutputPaths,
) -> None:
    input_paths = {
        args.raw_depth.resolve(),
        args.raw_cameras.resolve(),
        args.undistorted_cameras.resolve(),
        args.prediction.resolve(),
    }
    if args.prediction_manifest is not None:
        input_paths.add(args.prediction_manifest.resolve())
        input_paths.add(args.undistorted_images.resolve())
    if args.stage_snapshot_manifest is not None:
        input_paths.add(args.stage_snapshot_manifest.resolve())
    output_path = output_paths.report
    if output_path in input_paths:
        raise ValueError("--output must not overwrite an input file")
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"Output report path is not a file: {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output report already exists; use --overwrite: {output_path}"
        )
    if output_paths.remapped_ground_truth is None:
        return
    remapped_path = output_paths.remapped_ground_truth
    if remapped_path in input_paths:
        raise ValueError(
            "--remapped-ground-truth-output must not overwrite an input file"
        )
    if remapped_path == output_path:
        raise ValueError(
            "--remapped-ground-truth-output and --output must be different files"
        )
    if remapped_path.exists() and not remapped_path.is_file():
        raise ValueError(
            "Remapped ground-truth output path is not a file: "
            f"{remapped_path}"
        )
    if remapped_path.exists() and not args.overwrite:
        raise FileExistsError(
            "Remapped ground-truth output already exists; use --overwrite: "
            f"{remapped_path}"
        )
