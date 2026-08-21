from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import scripts.validation.plascan_eth3d_depth_eval.cli as evaluator_cli
import scripts.validation.plascan_eth3d_depth_eval.output_io as evaluator_output_io

from scripts.validation.plascan_eth3d_depth_eval import (
    CV_8UC1,
    CV_32FC1,
    FAST_MATRIX_HEADER,
    FAST_MATRIX_MAGIC,
    ColmapCamera,
    ProjectionError,
    colmap_pixels_to_raster_coordinates,
    depth_boundary_safe_mask,
    evaluate_depth_prediction,
    project_pinhole,
    project_thin_prism_fisheye,
    raster_indices_to_colmap_pixels,
    read_eth3d_raw_depth,
    read_prediction_depth,
    read_prediction_manifest_frame,
    remap_eth3d_depth_to_undistorted,
    scale_pinhole_camera,
    unproject_pinhole,
    unproject_thin_prism_fisheye,
    validate_camera_roundtrip,
    validate_scaled_pinhole_camera,
)
from scripts.validation.plascan_eth3d_depth_eval.depth_io import (
    read_plascan_fast_matrix,
)
from scripts.validation.plascan_eth3d_depth_eval.stage_manifest_io import (
    STAGE_IDS,
    read_stage_snapshot_record,
)
from scripts.validation.plascan_eth3d_depth_eval.cli import main


OFFICE_RAW_CAMERA = ColmapCamera(
    camera_id=0,
    model="THIN_PRISM_FISHEYE",
    width=6048,
    height=4032,
    params=(
        3437.84,
        3435.95,
        3040.23,
        2010.15,
        0.209024,
        0.193056,
        0.000466489,
        -7.84834e-05,
        -0.108072,
        0.356239,
        0.00105437,
        -0.000409613,
    ),
)
OFFICE_UNDISTORTED_CAMERA = ColmapCamera(
    camera_id=0,
    model="PINHOLE",
    width=6221,
    height=4146,
    params=(3437.84, 3435.95, 3127.19, 2066.98),
)


def small_cameras() -> tuple[ColmapCamera, ColmapCamera]:
    raw = ColmapCamera(
        camera_id=0,
        model="THIN_PRISM_FISHEYE",
        width=12,
        height=10,
        params=(10.0, 10.0, 6.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    target = ColmapCamera(
        camera_id=0,
        model="PINHOLE",
        width=12,
        height=10,
        params=(10.0, 10.0, 6.0, 5.0),
    )
    return raw, target


class Eth3dCameraModelTest(unittest.TestCase):
    def test_raster_colmap_half_pixel_convention_is_exactly_reversible(self):
        pixels = raster_indices_to_colmap_pixels(
            np.asarray([0, 11]), np.asarray([0, 9])
        )
        np.testing.assert_array_equal(pixels, [[0.5, 0.5], [11.5, 9.5]])
        np.testing.assert_array_equal(
            colmap_pixels_to_raster_coordinates(pixels), [[0.0, 0.0], [11.0, 9.0]]
        )

    def test_office_camera_projection_matches_frozen_colmap_golden_values(self):
        raster_coordinates = np.asarray(
            [[0, 0], [1000, 500], [3110, 2072], [5000, 3000], [6220, 4145]],
            dtype=np.float64,
        )
        target_pixels = raster_indices_to_colmap_pixels(
            raster_coordinates[:, 0], raster_coordinates[:, 1]
        )
        rays = unproject_pinhole(OFFICE_UNDISTORTED_CAMERA, target_pixels)
        actual = project_thin_prism_fisheye(OFFICE_RAW_CAMERA, rays)
        expected = np.asarray(
            [
                [5.019322402279158, 2.741963230051851],
                [995.2342321028068, 502.8674469209959],
                [3023.540104248116, 2015.669999647127],
                [4858.385430404789, 2915.743371239427],
                [6047.338137574483, 4029.343465989360],
            ]
        )
        self.assertTrue(np.all(actual.valid))
        np.testing.assert_allclose(actual.pixels, expected, rtol=0.0, atol=1.0e-9)

    def test_office_raw_pixels_map_to_frozen_undistorted_golden_values(self):
        raw_raster = np.asarray(
            [[0, 0], [1000, 500], [3040, 2010], [5000, 3000], [6047, 4031]],
            dtype=np.float64,
        )
        raw_pixels = raster_indices_to_colmap_pixels(
            raw_raster[:, 0], raw_raster[:, 1]
        )
        inverse = unproject_thin_prism_fisheye(
            OFFICE_RAW_CAMERA,
            raw_pixels,
            maximum_reprojection_error=1.0e-8,
        )
        target = project_pinhole(OFFICE_UNDISTORTED_CAMERA, inverse.rays)
        actual = colmap_pixels_to_raster_coordinates(target.pixels)
        expected = np.asarray(
            [
                [-4.240245997342299, -2.033614402937474],
                [1005.528059264846, 497.5792598531746],
                [3126.959999922718, 2066.829999968538],
                [5152.410345321754, 3090.424305251087],
                [6220.058320209529, 4147.147775945718],
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-9)

    def test_office_camera_inverse_has_strict_forward_projection_parity(self):
        report = validate_camera_roundtrip(
            OFFICE_RAW_CAMERA,
            OFFICE_UNDISTORTED_CAMERA,
            grid_size=17,
            maximum_reprojection_error=1.0e-6,
        )
        self.assertGreater(report["sample_count"], 200)
        self.assertLess(report["raw_reprojection_error_max_pixels"], 1.0e-8)
        self.assertLess(report["target_reprojection_error_max_pixels"], 1.0e-8)

    def test_inverse_rejects_unconverged_solution(self):
        with self.assertRaisesRegex(ProjectionError, "inversion failed"):
            unproject_thin_prism_fisheye(
                OFFICE_RAW_CAMERA,
                np.asarray([[0.5, 0.5]]),
                maximum_iterations=1,
                maximum_reprojection_error=1.0e-12,
            )


class Eth3dDepthIoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.temp_path = Path(self._temporary_directory.name)

    def test_raw_depth_accepts_infinity_and_reports_all_invalid_classes(self):
        path = self.temp_path / "depth.JPG"
        values = np.asarray(
            [[1.0, np.inf, np.nan], [0.0, -2.0, -np.inf]], dtype="<f4"
        )
        path.write_bytes(values.tobytes())
        loaded = read_eth3d_raw_depth(path, width=3, height=2)
        self.assertEqual(loaded.values.shape, (2, 3))
        self.assertEqual(loaded.validity["valid_positive_finite_count"], 1)
        self.assertEqual(loaded.validity["positive_infinity_count"], 1)
        self.assertEqual(loaded.validity["negative_infinity_count"], 1)
        self.assertEqual(loaded.validity["nan_count"], 1)
        self.assertEqual(loaded.validity["zero_count"], 1)
        self.assertEqual(loaded.validity["negative_finite_count"], 1)

    def test_raw_depth_rejects_wrong_file_size(self):
        path = self.temp_path / "truncated.JPG"
        path.write_bytes(np.asarray([1.0], dtype="<f4").tobytes())
        with self.assertRaisesRegex(ValueError, "file size mismatch"):
            read_eth3d_raw_depth(path, width=2, height=2)

    def test_plascan_fast_matrix_rejects_trailing_payload(self):
        path = self.temp_path / "depth.bin"
        values = np.asarray([[1.0, 2.0]], dtype="<f4")
        payload = values.tobytes()
        path.write_bytes(
            FAST_MATRIX_HEADER.pack(
                FAST_MATRIX_MAGIC, 1, 2, CV_32FC1, len(payload)
            )
            + payload
            + b"unexpected"
        )
        with self.assertRaisesRegex(ValueError, "file size mismatch"):
            read_prediction_depth(path, width=2, height=1)

    def test_prediction_npy_rejects_complex_values(self):
        path = self.temp_path / "complex.npy"
        with path.open("wb") as stream:
            np.save(stream, np.asarray([[1.0 + 2.0j]], dtype=np.complex64))
        with self.assertRaisesRegex(ValueError, "real numeric values"):
            read_prediction_depth(path, width=1, height=1)


class Eth3dDepthRemapTest(unittest.TestCase):
    def test_boundary_mask_preserves_isolated_measurement_but_rejects_edge(self):
        depth = np.asarray(
            [
                [np.inf, np.inf, np.inf, np.inf],
                [np.inf, 1.0, np.inf, np.inf],
                [np.inf, np.inf, 2.0, 4.0],
            ],
            dtype=np.float32,
        )
        valid, safe = depth_boundary_safe_mask(
            depth,
            maximum_relative_depth_spread=0.05,
        )
        self.assertTrue(valid[1, 1])
        self.assertTrue(safe[1, 1])
        self.assertTrue(valid[2, 2])
        self.assertTrue(valid[2, 3])
        self.assertFalse(safe[2, 2])
        self.assertFalse(safe[2, 3])

    def test_constant_synthetic_scene_has_exact_zero_error(self):
        raw_camera, target_camera = small_cameras()
        raw_depth = np.full(
            (raw_camera.height, raw_camera.width), 3.25, dtype=np.float32
        )
        remapped = remap_eth3d_depth_to_undistorted(
            raw_depth,
            raw_camera,
            target_camera,
            maximum_relative_depth_spread=0.0,
            chunk_rows=3,
        )
        prediction = np.full(remapped.depth.shape, 3.25, dtype=np.float32)
        metrics = evaluate_depth_prediction(prediction, remapped.depth)
        self.assertEqual(
            metrics["common_valid_pixel_count"],
            remapped.diagnostics["valid_remapped_pixel_count"],
        )
        self.assertGreater(metrics["common_valid_pixel_count"], 0)
        self.assertEqual(
            remapped.diagnostics["raw_valid_pixel_count"],
            remapped.diagnostics["depth_boundary_safe_raw_pixel_count"]
            + remapped.diagnostics["depth_boundary_rejected_raw_pixel_count"],
        )
        self.assertEqual(
            remapped.diagnostics["inverse_projection"]["sample_count"],
            remapped.diagnostics["projected_candidate_count"]
            + remapped.diagnostics["outside_target_sample_count"],
        )
        self.assertEqual(metrics["absolute_depth_error"]["maximum"], 0.0)
        self.assertEqual(
            metrics["relative_absolute_depth_error"]["maximum"], 0.0
        )
        self.assertEqual(metrics["prediction_coverage_of_ground_truth"], 1.0)

    def test_cli_writes_hashed_reproducible_report_and_remapped_depth(self):
        raw_camera, target_camera = small_cameras()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_depth_path = root / "raw.JPG"
            raw_cameras_path = root / "raw_cameras.txt"
            target_cameras_path = root / "target_cameras.txt"
            prediction_path = root / "prediction.npy"
            report_path = root / "report.json"
            remapped_path = root / "ground_truth.npy"
            raw_depth_path.write_bytes(
                np.full(
                    (raw_camera.height, raw_camera.width), 2.0, dtype="<f4"
                ).tobytes()
            )
            raw_cameras_path.write_text(
                "0 THIN_PRISM_FISHEYE 12 10 "
                "10 10 6 5 0 0 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            target_cameras_path.write_text(
                "0 PINHOLE 12 10 10 10 6 5\n", encoding="utf-8"
            )
            with prediction_path.open("wb") as stream:
                np.save(
                    stream,
                    np.full(
                        (target_camera.height, target_camera.width),
                        2.0,
                        dtype=np.float32,
                    ),
                    allow_pickle=False,
                )

            exit_code = main(
                [
                    "--raw-depth",
                    str(raw_depth_path),
                    "--raw-cameras",
                    str(raw_cameras_path),
                    "--undistorted-cameras",
                    str(target_cameras_path),
                    "--prediction",
                    str(prediction_path),
                    "--output",
                    str(report_path),
                    "--remapped-ground-truth-output",
                    str(remapped_path),
                    "--maximum-relative-depth-spread",
                    "0",
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "plascan.eth3d_depth_evaluation.v1")
            self.assertNotIn("prediction_manifest", report["inputs"])
            self.assertNotIn("prediction_manifest_frame", report)
            self.assertNotIn("prediction_camera", report)
            self.assertEqual(len(report["inputs"]["raw_depth"]["sha256"]), 64)
            self.assertEqual(
                report["metrics"]["relative_absolute_depth_error"]["maximum"],
                0.0,
            )
            self.assertEqual(
                report["remapped_ground_truth_output"],
                str(remapped_path.resolve()),
            )
            self.assertEqual(
                report["remapped_ground_truth_artifact"]["path"],
                str(remapped_path.resolve()),
            )
            self.assertEqual(
                len(report["remapped_ground_truth_artifact"]["sha256"]),
                64,
            )
            self.assertEqual(
                report["remapped_ground_truth_artifact"]["sha256"],
                hashlib.sha256(remapped_path.read_bytes()).hexdigest(),
            )
            remapped_depth = np.load(remapped_path, allow_pickle=False)
            self.assertEqual(remapped_depth.shape, (10, 12))


class Eth3dPredictionManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.raw_camera = ColmapCamera(
            camera_id=0,
            model="THIN_PRISM_FISHEYE",
            width=13,
            height=11,
            params=(
                10.0,
                9.0,
                6.25,
                5.125,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        self.official_camera = ColmapCamera(
            camera_id=0,
            model="PINHOLE",
            width=13,
            height=11,
            params=(10.0, 9.0, 6.25, 5.125),
        )

    def test_odd_grid_scaling_uses_colmap_corner_origin_per_axis(self):
        scaled = scale_pinhole_camera(self.official_camera, width=7, height=5)
        scale_x = 7.0 / 13.0
        scale_y = 5.0 / 11.0
        self.assertEqual(scaled.width, 7)
        self.assertEqual(scaled.height, 5)
        np.testing.assert_allclose(
            scaled.params,
            (
                10.0 * scale_x,
                9.0 * scale_y,
                6.25 * scale_x,
                5.125 * scale_y,
            ),
            rtol=0.0,
            atol=1.0e-15,
        )
        validation = validate_scaled_pinhole_camera(
            self.official_camera, scaled
        )
        self.assertEqual(validation["scale_x"], scale_x)
        self.assertEqual(validation["scale_y"], scale_y)
        self.assertEqual(
            validation["maximum_absolute_intrinsic_residual_pixels"], 0.0
        )

    def test_manifest_reader_rejects_missing_contract_fields(self):
        camera = scale_pinhole_camera(self.official_camera, 7, 5)
        for missing_field in ("grid_height", "camera_model"):
            with self.subTest(missing_field=missing_field):
                manifest = self._manifest(camera, ref_index=8)
                del manifest["frames"][0][missing_field]
                path = self.root / f"missing_{missing_field}.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, missing_field):
                    read_prediction_manifest_frame(path, 8)

    def test_manifest_reader_rejects_missing_reference_index(self):
        camera = scale_pinhole_camera(self.official_camera, 7, 5)
        path = self._write_manifest(camera, ref_index=8)
        with self.assertRaisesRegex(ValueError, "ref_index 9 is not present"):
            read_prediction_manifest_frame(path, 9)

    def test_scaled_camera_validation_rejects_wrong_intrinsics(self):
        expected = scale_pinhole_camera(self.official_camera, 7, 5)
        wrong = ColmapCamera(
            camera_id=expected.camera_id,
            model=expected.model,
            width=expected.width,
            height=expected.height,
            params=(
                expected.fx,
                expected.fy,
                expected.cx + 0.01,
                expected.cy,
            ),
        )
        with self.assertRaisesRegex(ValueError, "not the corner-origin"):
            validate_scaled_pinhole_camera(self.official_camera, wrong)

    def test_manifest_reader_rejects_distorted_camera(self):
        camera = scale_pinhole_camera(self.official_camera, 7, 5)
        manifest = self._manifest(camera, ref_index=8)
        manifest["frames"][0]["camera_model"]["k1"] = 0.001
        path = self.root / "distorted_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be undistorted"):
            read_prediction_manifest_frame(path, 8)

    def test_cli_requires_manifest_and_reference_index_as_a_pair(self):
        prediction_camera = scale_pinhole_camera(
            self.official_camera, width=7, height=5
        )
        paths = self._write_cli_inputs(prediction_camera, ref_index=8)
        for incomplete_arguments in (
            ["--prediction-manifest", str(paths["manifest"])],
            ["--prediction-ref-index", "8"],
        ):
            with self.subTest(arguments=incomplete_arguments):
                with self.assertRaisesRegex(ValueError, "must be used together"):
                    main(self._cli_arguments(paths) + incomplete_arguments)

    def test_cli_remaps_raw_ground_truth_directly_to_odd_native_grid(self):
        prediction_camera = scale_pinhole_camera(
            self.official_camera, width=7, height=5
        )
        paths = self._write_cli_inputs(prediction_camera, ref_index=8)
        exit_code = main(
            self._manifest_cli_arguments(paths)
            + [
                "--prediction-manifest",
                str(paths["manifest"]),
                "--prediction-ref-index",
                "8",
            ]
        )

        self.assertEqual(exit_code, 0)
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        self.assertEqual(report["prediction_camera"]["width"], 7)
        self.assertEqual(report["prediction_camera"]["height"], 5)
        self.assertEqual(
            report["prediction_manifest_camera_model"]["cx"],
            prediction_camera.cx - 0.5,
        )
        self.assertEqual(report["remap"]["target_pixel_count"], 35)
        self.assertEqual(report["prediction_manifest_frame"]["ref_index"], 8)
        self.assertEqual(
            report["prediction_manifest_frame"]["config_hash"], "a" * 64
        )
        self.assertEqual(
            len(report["inputs"]["prediction_manifest"]["sha256"]), 64
        )
        self.assertEqual(
            report["inputs"]["manifest_ref_image"]["path"],
            str(paths["ref_image"].resolve()),
        )
        self.assertEqual(
            report["inputs"]["manifest_ref_image"]["sha256"],
            hashlib.sha256(paths["ref_image"].read_bytes()).hexdigest(),
        )
        self.assertEqual(
            report["inputs"]["prediction"]["path"],
            str(paths["prediction"].resolve()),
        )
        self.assertEqual(
            report["prediction_camera_scaling_validation"][
                "maximum_absolute_intrinsic_residual_pixels"
            ],
            0.0,
        )
        self.assertEqual(
            report["metrics"]["relative_absolute_depth_error"]["maximum"],
            0.0,
        )

    def test_full_size_manifest_matches_legacy_remap_and_metrics(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        legacy_report = self.root / "legacy_report.json"
        legacy_remap = self.root / "legacy_remap.npy"
        native_remap = self.root / "manifest_remap.npy"
        arguments = self._cli_arguments(paths)
        arguments[arguments.index(str(paths["report"]))] = str(legacy_report)
        arguments.extend(
            ["--remapped-ground-truth-output", str(legacy_remap)]
        )
        self.assertEqual(main(arguments), 0)

        manifest_arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
            "--remapped-ground-truth-output",
            str(native_remap),
        ]
        self.assertEqual(main(manifest_arguments), 0)

        legacy = json.loads(legacy_report.read_text(encoding="utf-8"))
        manifest = json.loads(paths["report"].read_text(encoding="utf-8"))
        self.assertEqual(legacy["metrics"], manifest["metrics"])
        np.testing.assert_array_equal(
            np.load(legacy_remap, allow_pickle=False),
            np.load(native_remap, allow_pickle=False),
        )
        self.assertEqual(
            manifest["prediction_camera_scaling_validation"]["scale_x"], 1.0
        )
        self.assertEqual(
            manifest["prediction_camera_scaling_validation"]["scale_y"], 1.0
        )

    def test_unconverted_colmap_principal_point_is_rejected(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["frames"][0]["camera_model"]["cx"] = self.official_camera.cx
        manifest["frames"][0]["camera_model"]["cy"] = self.official_camera.cy
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "corner-origin scaling"):
            main(
                self._manifest_cli_arguments(paths)
                + [
                    "--prediction-manifest",
                    str(paths["manifest"]),
                    "--prediction-ref-index",
                    "4",
                ]
            )

    def test_nonconstant_depth_detects_manifest_coordinate_shift(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        rows, columns = np.indices(
            (self.official_camera.height, self.official_camera.width)
        )
        values = (2.0 + columns * 0.02 + rows * 0.001).astype(np.float32)
        paths["raw_depth"].write_bytes(values.astype("<f4").tobytes())
        expected = remap_eth3d_depth_to_undistorted(
            values,
            self.raw_camera,
            self.official_camera,
            maximum_relative_depth_spread=1.0,
        )
        with paths["prediction"].open("wb") as stream:
            np.save(stream, expected.depth, allow_pickle=False)

        arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
        ]
        spread_index = arguments.index("--maximum-relative-depth-spread") + 1
        arguments[spread_index] = "1"
        self.assertEqual(main(arguments), 0)
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        self.assertEqual(
            report["metrics"]["absolute_depth_error"]["maximum"], 0.0
        )

    def test_manifest_binds_prediction_and_ground_truth_image(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        other_prediction = self.root / "other.npy"
        other_prediction.write_bytes(paths["prediction"].read_bytes())
        arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
        ]
        prediction_index = arguments.index("--prediction") + 1
        arguments[prediction_index] = str(other_prediction)
        with self.assertRaisesRegex(ValueError, "raw_depth_path"):
            main(arguments)

        other_raw = self.root / "other.JPG"
        other_raw.write_bytes(paths["raw_depth"].read_bytes())
        arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
        ]
        raw_index = arguments.index("--raw-depth") + 1
        arguments[raw_index] = str(other_raw)
        with self.assertRaisesRegex(ValueError, "filename does not match"):
            main(arguments)

    def test_rejected_frame_requires_explicit_diagnostic_flag(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["frames"][0]["acceptance"] = "rejected"
        manifest["frames"][0]["fusion_eligible"] = False
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
        ]
        with self.assertRaisesRegex(ValueError, "not publishable"):
            main(arguments)
        self.assertEqual(
            main(arguments + ["--allow-non-publishable-frame"]), 0
        )
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        self.assertEqual(
            report["evaluation_scope"],
            "diagnostic_non_publishable_frame",
        )

    def test_reduced_grid_requires_effective_native_contract(self):
        camera = scale_pinhole_camera(self.official_camera, 7, 5)
        paths = self._write_cli_inputs(camera, ref_index=4)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["frames"][0]["effective_native_final_depth_grid"] = False
        manifest["frames"][0]["pixel_domain_diagnostics"][
            "effective_native_final_depth_grid"
        ] = False
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reduced grid requires"):
            main(
                self._manifest_cli_arguments(paths)
                + [
                    "--prediction-manifest",
                    str(paths["manifest"]),
                    "--prediction-ref-index",
                    "4",
                ]
            )

    def test_revision_40_requires_complete_pixel_domain_contract(self):
        camera = scale_pinhole_camera(self.official_camera, 7, 5)
        required_fields = (
            "configured_pixel_domain",
            "effective_pixel_domain",
            "requested_native_final_depth_grid",
            "effective_native_final_depth_grid",
            "raster_width",
            "raster_height",
            "grid_width",
            "grid_height",
            "scale_x",
            "scale_y",
            "linear_scale",
            "area_scale",
            "grid_matches_raster",
            "parameters",
        )
        for missing_field in required_fields:
            with self.subTest(missing_field=missing_field):
                manifest = self._manifest(camera, ref_index=8)
                del manifest["frames"][0]["pixel_domain_diagnostics"][
                    missing_field
                ]
                path = self.root / f"missing_pixel_domain_{missing_field}.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, missing_field):
                    read_prediction_manifest_frame(path, 8)

    def test_reduced_grid_requires_requested_native_contract(self):
        camera = scale_pinhole_camera(self.official_camera, 7, 5)
        manifest = self._manifest(camera, ref_index=8)
        manifest["frames"][0]["pixel_domain_diagnostics"][
            "requested_native_final_depth_grid"
        ] = False
        path = self.root / "native_not_requested.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires requested_native"):
            read_prediction_manifest_frame(path, 8)

    def test_full_grid_effective_native_requires_requested_native(self):
        manifest = self._manifest(self.official_camera, ref_index=8)
        frame = manifest["frames"][0]
        frame["effective_native_final_depth_grid"] = True
        frame["pixel_domain_diagnostics"][
            "effective_native_final_depth_grid"
        ] = True
        path = self.root / "full_effective_not_requested.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires requested_native"):
            read_prediction_manifest_frame(path, 8)

    def test_revision_40_requires_top_level_native_flag_and_diagnostics(self):
        camera = scale_pinhole_camera(self.official_camera, 7, 5)
        for missing_field in (
            "effective_native_final_depth_grid",
            "pixel_domain_diagnostics",
        ):
            with self.subTest(missing_field=missing_field):
                manifest = self._manifest(camera, ref_index=8)
                del manifest["frames"][0][missing_field]
                path = self.root / f"missing_frame_{missing_field}.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, missing_field):
                    read_prediction_manifest_frame(path, 8)

    def test_revision_40_rejects_inconsistent_pixel_domain_values(self):
        camera = scale_pinhole_camera(self.official_camera, 7, 5)
        mutations = {
            "grid_width": 6,
            "scale_x": 0.25,
            "scale_y": 0.25,
            "linear_scale": 0.25,
            "area_scale": 0.25,
            "grid_matches_raster": True,
            "effective_native_final_depth_grid": False,
        }
        expected_errors = {
            "grid_width": "grid does not match frame grid",
            "scale_x": "scale_x.*inconsistent",
            "scale_y": "scale_y.*inconsistent",
            "linear_scale": "linear_scale.*inconsistent",
            "area_scale": "area_scale.*inconsistent",
            "grid_matches_raster": "grid_matches_raster is inconsistent",
            "effective_native_final_depth_grid": "does not match the frame",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                manifest = self._manifest(camera, ref_index=8)
                manifest["frames"][0]["pixel_domain_diagnostics"][field] = value
                path = self.root / f"inconsistent_{field}.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected_errors[field]):
                    read_prediction_manifest_frame(path, 8)

    def test_revision_40_diagnostic_raster_must_match_official_camera(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        frame = manifest["frames"][0]
        diagnostics = frame["pixel_domain_diagnostics"]
        diagnostics["raster_width"] = self.official_camera.width + 1
        diagnostics["requested_native_final_depth_grid"] = True
        diagnostics["effective_native_final_depth_grid"] = True
        frame["effective_native_final_depth_grid"] = True
        diagnostics["grid_matches_raster"] = False
        diagnostics["scale_x"] = (
            self.official_camera.width / diagnostics["raster_width"]
        )
        diagnostics["scale_y"] = 1.0
        diagnostics["area_scale"] = diagnostics["scale_x"]
        diagnostics["linear_scale"] = diagnostics["scale_x"] ** 0.5
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "selected official camera"):
            main(
                self._manifest_cli_arguments(paths)
                + [
                    "--prediction-manifest",
                    str(paths["manifest"]),
                    "--prediction-ref-index",
                    "4",
                ]
            )

    def test_revision_39_full_raster_manifest_remains_compatible(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["algorithm_revision"] = 39
        frame = manifest["frames"][0]
        frame["algorithm_revision"] = 39
        del frame["effective_native_final_depth_grid"]
        del frame["pixel_domain_diagnostics"]
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

        self.assertEqual(
            main(
                self._manifest_cli_arguments(paths)
                + [
                    "--prediction-manifest",
                    str(paths["manifest"]),
                    "--prediction-ref-index",
                    "4",
                ]
            ),
            0,
        )
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        self.assertEqual(
            report["prediction_pixel_domain_validation"]["contract"],
            "legacy_full_raster",
        )

    def test_manifest_ref_image_must_exist_and_is_independently_hashed(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        paths["ref_image"].unlink()
        with self.assertRaises(FileNotFoundError):
            main(
                self._manifest_cli_arguments(paths)
                + [
                    "--prediction-manifest",
                    str(paths["manifest"]),
                    "--prediction-ref-index",
                    "4",
                ]
            )

    def test_report_output_must_not_overwrite_manifest_ref_image(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        original_image = paths["ref_image"].read_bytes()
        arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
            "--overwrite",
        ]
        output_index = arguments.index("--output") + 1
        arguments[output_index] = str(paths["ref_image"])

        with self.assertRaisesRegex(ValueError, "manifest ref_image"):
            main(arguments)
        self.assertEqual(paths["ref_image"].read_bytes(), original_image)

    def test_remapped_output_must_not_overwrite_manifest_ref_image(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        original_image = paths["ref_image"].read_bytes()
        arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
            "--remapped-ground-truth-output",
            str(paths["ref_image"]),
            "--overwrite",
        ]

        with self.assertRaisesRegex(ValueError, "manifest ref_image"):
            main(arguments)
        self.assertEqual(paths["ref_image"].read_bytes(), original_image)

    def test_late_report_symlink_cannot_redirect_overwrite_to_raw_depth(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        raw_depth_path = paths["raw_depth"].resolve()
        report_path = paths["report"].absolute()
        original_raw_depth = raw_depth_path.read_bytes()
        original_reader = evaluator_cli.read_prediction_depth

        def reader_after_symlink_race(snapshot_path, *args, **kwargs):
            try:
                report_path.symlink_to(raw_depth_path)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"Symbolic links are unavailable: {error}")
            return original_reader(snapshot_path, *args, **kwargs)

        with mock.patch.object(
            evaluator_cli,
            "read_prediction_depth",
            side_effect=reader_after_symlink_race,
        ):
            self.assertEqual(main(self._cli_arguments(paths) + ["--overwrite"]), 0)

        self.assertEqual(raw_depth_path.read_bytes(), original_raw_depth)
        self.assertFalse(report_path.is_symlink())
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8"))["schema"],
            "plascan.eth3d_depth_evaluation.v1",
        )

    def test_late_npy_symlink_cannot_redirect_overwrite_to_raw_depth(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        raw_depth_path = paths["raw_depth"].resolve()
        remapped_path = (self.root / "late_remapped.npy").absolute()
        original_raw_depth = raw_depth_path.read_bytes()
        original_reader = evaluator_cli.read_prediction_depth

        def reader_after_symlink_race(snapshot_path, *args, **kwargs):
            try:
                remapped_path.symlink_to(raw_depth_path)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"Symbolic links are unavailable: {error}")
            return original_reader(snapshot_path, *args, **kwargs)

        arguments = self._cli_arguments(paths) + [
            "--remapped-ground-truth-output",
            str(remapped_path),
            "--overwrite",
        ]
        with mock.patch.object(
            evaluator_cli,
            "read_prediction_depth",
            side_effect=reader_after_symlink_race,
        ):
            self.assertEqual(main(arguments), 0)

        self.assertEqual(raw_depth_path.read_bytes(), original_raw_depth)
        self.assertFalse(remapped_path.is_symlink())
        self.assertEqual(
            np.load(remapped_path, allow_pickle=False).shape,
            (self.raw_camera.height, self.raw_camera.width),
        )

    def test_official_image_pose_camera_id_must_match_selected_camera(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        paths["undistorted_images"].write_text(
            "1 1 0 0 0 0 0 0 1 view.JPG\n\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "camera_id does not match"):
            main(
                self._manifest_cli_arguments(paths)
                + [
                    "--prediction-manifest",
                    str(paths["manifest"]),
                    "--prediction-ref-index",
                    "4",
                ]
            )

    def test_manifest_parse_hash_is_checked_against_snapshot_immediately(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        original_reader = evaluator_cli.read_prediction_manifest_frame

        def reader_with_wrong_hash(*args, **kwargs):
            frame = original_reader(*args, **kwargs)
            return replace(frame, manifest_sha256="0" * 64)

        with mock.patch.object(
            evaluator_cli,
            "read_prediction_manifest_frame",
            side_effect=reader_with_wrong_hash,
        ):
            with self.assertRaisesRegex(ValueError, "immutable input snapshot"):
                main(
                    self._manifest_cli_arguments(paths)
                    + [
                        "--prediction-manifest",
                        str(paths["manifest"]),
                        "--prediction-ref-index",
                        "4",
                    ]
                )

    def test_aba_source_change_is_detected_while_reading_snapshot(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        remapped_path = self.root / "aba_remapped.npy"
        source_prediction = paths["prediction"].resolve()
        original_reader = evaluator_cli.read_prediction_depth

        def reader_after_aba(snapshot_path, *args, **kwargs):
            self.assertNotEqual(Path(snapshot_path), source_prediction)
            self.assertEqual(Path(snapshot_path).suffix, source_prediction.suffix)
            original_bytes = source_prediction.read_bytes()
            changed_bytes = bytearray(original_bytes)
            changed_bytes[-1] ^= 1
            source_prediction.write_bytes(changed_bytes)
            source_prediction.write_bytes(original_bytes)
            return original_reader(snapshot_path, *args, **kwargs)

        arguments = self._cli_arguments(paths) + [
            "--remapped-ground-truth-output",
            str(remapped_path),
        ]
        with mock.patch.object(
            evaluator_cli,
            "read_prediction_depth",
            side_effect=reader_after_aba,
        ):
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                main(arguments)
        self.assertFalse(paths["report"].exists())
        self.assertFalse(remapped_path.exists())

    def test_no_clobber_report_race_retains_auditable_remapped_orphan(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        remapped_path = self.root / "raced_remapped.npy"
        report_path = paths["report"].resolve()
        real_link = os.link

        def racing_link(source, target, *args, **kwargs):
            if Path(target) == report_path:
                report_path.write_text("raced report", encoding="utf-8")
                raise FileExistsError("injected report race")
            return real_link(source, target, *args, **kwargs)

        arguments = self._cli_arguments(paths) + [
            "--remapped-ground-truth-output",
            str(remapped_path),
        ]
        with mock.patch.object(
            evaluator_output_io.os,
            "link",
            side_effect=racing_link,
        ):
            with self.assertRaisesRegex(RuntimeError, "orphan was retained"):
                main(arguments)
        self.assertEqual(report_path.read_text(encoding="utf-8"), "raced report")
        self.assertEqual(
            np.load(remapped_path, allow_pickle=False).shape,
            (self.raw_camera.height, self.raw_camera.width),
        )
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_overwrite_report_failure_restores_previous_remapped_depth(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        remapped_path = self.root / "existing_remapped.npy"
        old_report = b"old report"
        old_remapped = b"old remapped"
        paths["report"].write_bytes(old_report)
        remapped_path.write_bytes(old_remapped)
        report_path = paths["report"].resolve()
        real_replace = os.replace

        def failing_report_replace(source, target, *args, **kwargs):
            if Path(target) == report_path:
                raise OSError("injected report publication failure")
            return real_replace(source, target, *args, **kwargs)

        arguments = self._cli_arguments(paths) + [
            "--remapped-ground-truth-output",
            str(remapped_path),
            "--overwrite",
        ]
        with mock.patch.object(
            evaluator_output_io.os,
            "replace",
            side_effect=failing_report_replace,
        ):
            with self.assertRaisesRegex(OSError, "injected report"):
                main(arguments)
        self.assertEqual(paths["report"].read_bytes(), old_report)
        self.assertEqual(remapped_path.read_bytes(), old_remapped)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_manifest_pose_must_match_official_images_file(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["frames"][0]["camera_model"]["camera_center"][0] = 0.1
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pose does not match"):
            main(
                self._manifest_cli_arguments(paths)
                + [
                    "--prediction-manifest",
                    str(paths["manifest"]),
                    "--prediction-ref-index",
                    "4",
                ]
            )

    def test_existing_outputs_require_explicit_overwrite(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        arguments = self._cli_arguments(paths)
        self.assertEqual(main(arguments), 0)
        with self.assertRaises(FileExistsError):
            main(arguments)
        self.assertEqual(main(arguments + ["--overwrite"]), 0)

    def test_stage_snapshot_cli_binds_and_scores_all_three_payloads(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        stage_paths = self._write_stage_snapshot(paths, ref_index=4)
        arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
            "--stage-snapshot-manifest",
            str(stage_paths["manifest"]),
            "--stage-snapshot-stage",
            "cross_view_consistency",
        ]
        prediction_index = arguments.index("--prediction") + 1
        arguments[prediction_index] = str(stage_paths["depth"])

        self.assertEqual(main(arguments), 0)
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        self.assertEqual(report["evaluation_scope"], "diagnostic_stage_snapshot")
        self.assertEqual(
            report["stage_snapshot_record"]["stage"],
            "cross_view_consistency",
        )
        self.assertEqual(report["prediction_camera"]["width"], 7)
        self.assertEqual(report["prediction_camera"]["height"], 5)
        self.assertGreater(report["metrics"]["common_valid_pixel_count"], 0)
        self.assertEqual(
            report["inputs"]["stage_snapshot_confidence"]["sha256"],
            hashlib.sha256(stage_paths["confidence"].read_bytes()).hexdigest(),
        )

    def test_stage_snapshot_rejects_tampered_companion_payload(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        stage_paths = self._write_stage_snapshot(paths, ref_index=4)
        stage_paths["confidence"].write_bytes(
            stage_paths["confidence"].read_bytes()[:-1] + b"x"
        )
        arguments = self._manifest_cli_arguments(paths) + [
            "--prediction-manifest",
            str(paths["manifest"]),
            "--prediction-ref-index",
            "4",
            "--stage-snapshot-manifest",
            str(stage_paths["manifest"]),
            "--stage-snapshot-stage",
            "cross_view_consistency",
        ]
        arguments[arguments.index("--prediction") + 1] = str(stage_paths["depth"])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            main(arguments)

    def test_stage_snapshot_rejects_workspace_grid_mismatch(self):
        paths = self._write_cli_inputs(self.official_camera, ref_index=4)
        stage_paths = self._write_stage_snapshot(paths, ref_index=4)
        manifest = json.loads(stage_paths["manifest"].read_text(encoding="utf-8"))
        captured = next(
            record for record in manifest["records"]
            if record.get("status") == "captured"
        )
        captured["original_width"] -= 1
        stage_paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        record = read_stage_snapshot_record(
            stage_paths["manifest"], 4, "cross_view_consistency", camera_id=0
        )
        frame = read_prediction_manifest_frame(paths["manifest"], 4)
        with self.assertRaisesRegex(ValueError, "original raster"):
            evaluator_cli.validate_stage_snapshot_against_workspace(record, frame)

    def _manifest(
        self, camera: ColmapCamera, *, ref_index: int
    ) -> dict[str, object]:
        scale_x = camera.width / self.official_camera.width
        scale_y = camera.height / self.official_camera.height
        area_scale = scale_x * scale_y
        reduced_grid = (
            camera.width != self.official_camera.width
            or camera.height != self.official_camera.height
        )
        return {
            "schema": "plascan.mvs.workspace.v2",
            "algorithm_revision": 40,
            "config_hash": "a" * 64,
            "frames": [
                {
                    "ref_index": ref_index,
                    "status": "completed",
                    "acceptance": "accepted",
                    "fusion_eligible": True,
                    "algorithm_revision": 40,
                    "config_hash": "a" * 64,
                    "grid_width": camera.width,
                    "grid_height": camera.height,
                    "ref_image": "reference/view.JPG",
                    "raw_depth_path": str(self.root / "prediction.npy"),
                    "effective_native_final_depth_grid": reduced_grid,
                    "pixel_domain_diagnostics": {
                        "configured_pixel_domain": "prepared_full_raster",
                        "effective_pixel_domain": "depth_grid",
                        "requested_native_final_depth_grid": reduced_grid,
                        "effective_native_final_depth_grid": reduced_grid,
                        "raster_width": self.official_camera.width,
                        "raster_height": self.official_camera.height,
                        "grid_width": camera.width,
                        "grid_height": camera.height,
                        "scale_x": scale_x,
                        "scale_y": scale_y,
                        "linear_scale": area_scale ** 0.5,
                        "area_scale": area_scale,
                        "grid_matches_raster": not reduced_grid,
                        "parameters": {},
                    },
                    "camera_model": {
                        "fx": camera.fx,
                        "fy": camera.fy,
                        "cx": camera.cx - 0.5,
                        "cy": camera.cy - 0.5,
                        "rotation_world_to_camera": [
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0,
                        ],
                        "translation_world_to_camera": [0.0, 0.0, 0.0],
                        "camera_center": [0.0, 0.0, 0.0],
                    },
                }
            ],
        }

    def _write_manifest(
        self, camera: ColmapCamera, *, ref_index: int
    ) -> Path:
        path = self.root / "mvs_manifest.json"
        path.write_text(
            json.dumps(self._manifest(camera, ref_index=ref_index)),
            encoding="utf-8",
        )
        return path

    def _write_cli_inputs(
        self, prediction_camera: ColmapCamera, *, ref_index: int
    ) -> dict[str, Path]:
        paths = {
            "raw_depth": self.root / "view.JPG",
            "raw_cameras": self.root / "raw_cameras.txt",
            "undistorted_cameras": self.root / "undistorted_cameras.txt",
            "undistorted_images": self.root / "images.txt",
            "prediction": self.root / "prediction.npy",
            "manifest": self.root / "mvs_manifest.json",
            "report": self.root / "report.json",
            "ref_image": self.root / "reference" / "view.JPG",
        }
        paths["raw_depth"].write_bytes(
            np.full((11, 13), 2.5, dtype="<f4").tobytes()
        )
        paths["raw_cameras"].write_text(
            "0 THIN_PRISM_FISHEYE 13 11 "
            "10 9 6.25 5.125 0 0 0 0 0 0 0 0\n",
            encoding="utf-8",
        )
        paths["undistorted_cameras"].write_text(
            "0 PINHOLE 13 11 10 9 6.25 5.125\n", encoding="utf-8"
        )
        paths["undistorted_images"].write_text(
            "1 1 0 0 0 0 0 0 0 view.JPG\n\n", encoding="utf-8"
        )
        paths["ref_image"].parent.mkdir()
        paths["ref_image"].write_bytes(b"synthetic undistorted image")
        with paths["prediction"].open("wb") as stream:
            np.save(
                stream,
                np.full(
                    (prediction_camera.height, prediction_camera.width),
                    2.5,
                    dtype=np.float32,
                ),
                allow_pickle=False,
            )
        self._write_manifest(prediction_camera, ref_index=ref_index)
        return paths

    def _write_stage_snapshot(
        self, paths: dict[str, Path], *, ref_index: int
    ) -> dict[str, Path]:
        stage_root = self.root / "stage_snapshots"
        frame_root = stage_root / f"ref_{ref_index:04d}"
        frame_root.mkdir(parents=True)
        snapshot_camera = scale_pinhole_camera(self.official_camera, 7, 5)
        depth = np.full((5, 7), 2.5, dtype="<f4")
        confidence = np.full((5, 7), 0.75, dtype="<f4")
        valid_mask = np.full((5, 7), 255, dtype="u1")
        result = {
            "manifest": stage_root / "manifest.json",
            "depth": frame_root / "cross_view_consistency_depth.bin",
            "confidence": frame_root / "cross_view_consistency_confidence.bin",
            "valid_mask": frame_root / "cross_view_consistency_valid_mask.bin",
        }
        self._write_fast_matrix(result["depth"], depth, CV_32FC1)
        self._write_fast_matrix(result["confidence"], confidence, CV_32FC1)
        self._write_fast_matrix(result["valid_mask"], valid_mask, CV_8UC1)

        def artifact(path: Path, values: np.ndarray, cv_type: int):
            payload = path.read_bytes()
            return {
                "path": str(path),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "rows": values.shape[0],
                "cols": values.shape[1],
                "opencv_type": cv_type,
            }

        captured = {
            "ref_index": ref_index,
            "stage": "cross_view_consistency",
            "boundary": "synthetic_cross_view_boundary",
            "status": "captured",
            "original_width": self.official_camera.width,
            "original_height": self.official_camera.height,
            "snapshot_width": snapshot_camera.width,
            "snapshot_height": snapshot_camera.height,
            "valid_pixel_count": int(valid_mask.size),
            "effective_native_final_depth_grid": False,
            "pixel_domain_diagnostics": {},
            "quality_metrics": {},
            "quality_decision": {},
            "depth_completeness": {},
            "camera_model": {
                "fx": snapshot_camera.fx,
                "fy": snapshot_camera.fy,
                "cx": snapshot_camera.cx - 0.5,
                "cy": snapshot_camera.cy - 0.5,
                "rotation_world_to_camera": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
                "translation_world_to_camera": [0.0, 0.0, 0.0],
                "camera_center": [0.0, 0.0, 0.0],
            },
            "depth": artifact(result["depth"], depth, CV_32FC1),
            "confidence": artifact(
                result["confidence"], confidence, CV_32FC1
            ),
            "valid_mask": artifact(result["valid_mask"], valid_mask, CV_8UC1),
        }
        records = [captured]
        for stage in sorted(STAGE_IDS - {"cross_view_consistency"}):
            records.append(
                {
                    "ref_index": ref_index,
                    "stage": stage,
                    "boundary": "synthetic_missing_boundary",
                    "status": "missing",
                    "reason": "stage_not_reached",
                }
            )
        manifest = {
            "schema": "plascan_mvs_stage_snapshots_v1",
            "status": "complete",
            "authoritative": False,
            "selected_ref_indices": [ref_index],
            "expected_stages": sorted(STAGE_IDS),
            "maximum_long_edge": 7,
            "budget_bytes": 1024 * 1024,
            "used_bytes": sum(
                result[key].stat().st_size
                for key in ("depth", "confidence", "valid_mask")
            ),
            "records": records,
            "finalized": True,
            "errors": [],
        }
        result["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        return result

    @staticmethod
    def _write_fast_matrix(path: Path, values: np.ndarray, cv_type: int) -> None:
        payload = values.tobytes(order="C")
        path.write_bytes(
            FAST_MATRIX_HEADER.pack(
                FAST_MATRIX_MAGIC,
                values.shape[0],
                values.shape[1],
                cv_type,
                len(payload),
            )
            + payload
        )

    @staticmethod
    def _cli_arguments(paths: dict[str, Path]) -> list[str]:
        return [
            "--raw-depth",
            str(paths["raw_depth"]),
            "--raw-cameras",
            str(paths["raw_cameras"]),
            "--undistorted-cameras",
            str(paths["undistorted_cameras"]),
            "--prediction",
            str(paths["prediction"]),
            "--output",
            str(paths["report"]),
            "--maximum-relative-depth-spread",
            "0",
        ]

    @staticmethod
    def _manifest_cli_arguments(paths: dict[str, Path]) -> list[str]:
        return Eth3dPredictionManifestTest._cli_arguments(paths) + [
            "--undistorted-images",
            str(paths["undistorted_images"]),
        ]


if __name__ == "__main__":
    unittest.main()
