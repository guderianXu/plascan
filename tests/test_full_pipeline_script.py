import importlib.util
import contextlib
import io
import tempfile
import types
import unittest
from unittest import mock
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "legacy" / "run_full_pipeline_test.py"
EXTRACT_FEATURES_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "workflows" / "extract_features.py"
SPEC = importlib.util.spec_from_file_location("run_full_pipeline_test", SCRIPT_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class FullPipelineScriptTest(unittest.TestCase):
    def test_default_dense_algorithm_prefers_opencv_sgbm(self):
        with mock.patch.object(sys, "argv", ["run_full_pipeline_test.py", "input.lis"]):
            args = pipeline.parse_args()

        self.assertEqual(args.dense_algorithm, "opencv_sgbm")
        self.assertEqual(args.dense_cost, "census")

    def test_default_disk_feature_settings_are_not_limited_to_1200(self):
        with mock.patch.object(sys, "argv", ["run_full_pipeline_test.py", "input.lis"]):
            args = pipeline.parse_args()

        self.assertEqual(args.max_image_dim, 0)
        self.assertEqual(args.max_keypoints, 0)
        self.assertEqual(pipeline.max_keypoints_for_algorithm(args, "disk"), 8192)
        self.assertEqual(pipeline.MODEL_CANDIDATES["disk"]["cpu"][0], "disk_extractor_cpu_8192.torchscript")

    def test_disk_and_aliked_lightglue_candidates_use_dedicated_torchscript_models(self):
        self.assertEqual(pipeline.lightglue_model_kind_for_algorithm("disk"), "lightglue_disk")
        self.assertEqual(pipeline.lightglue_model_kind_for_algorithm("aliked"), "lightglue_aliked")
        self.assertEqual(pipeline.lightglue_model_kind_for_algorithm("sift"), "lightglue_sift")
        self.assertEqual(pipeline.MODEL_CANDIDATES["lightglue_disk"]["cpu"][0],
                         "lightglue_disk_cpu.torchscript")
        self.assertEqual(pipeline.MODEL_CANDIDATES["lightglue_aliked"]["cuda"][0],
                         "lightglue_aliked_cuda.torchscript")
        self.assertEqual(pipeline.MODEL_CANDIDATES["lightglue_sift"]["cpu"],
                         ["lightglue_sift_cpu.torchscript"])
        for kind in ("lightglue", "lightglue_disk", "lightglue_aliked", "lightglue_sift"):
            for names in pipeline.MODEL_CANDIDATES[kind].values():
                self.assertFalse(any(name.endswith(".pt") for name in names))

    def test_auto_dense_max_disparity_uses_camera_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left_camera = root / "left.tsai"
            right_camera = root / "right.tsai"
            common = "\n".join([
                "VERSION_3",
                "fu = 4.943",
                "fv = 4.943",
                "cu = 1.136",
                "cv = 0.815",
                "u_direction = 1 0 0",
                "v_direction = 0 1 0",
                "w_direction = 0 0 1",
                "R = 1 0 0 0 1 0 0 0 1",
                "pitch = 0.00185",
                "",
            ])
            left_camera.write_text(common + "C = 0 0 -7900\n", encoding="utf-8")
            right_camera.write_text(common + "C = 765 0 -7900\n", encoding="utf-8")

            max_disp = pipeline.estimate_dense_max_disp_for_pair(
                left_camera,
                right_camera,
                max_image_dim=1200,
            )

            self.assertEqual(max_disp % 16, 0)
            self.assertGreaterEqual(max_disp, 512)
            self.assertLessEqual(max_disp, 640)

    def test_run_terrain_uses_image_camera_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_images = root / "images"
            work_cameras = root / "cameras"
            work_images.mkdir()
            work_cameras.mkdir()

            items = []
            for index in range(2):
                image = work_images / f"{index + 1:03d}.png"
                camera = work_cameras / f"{index + 1:03d}.tsai"
                image.write_bytes(b"image")
                camera.write_text("pitch = 1\n", encoding="utf-8")
                items.append(
                    pipeline.InputItem(
                        image=image,
                        camera=camera,
                        work_image=image,
                        work_camera=camera,
                    )
                )

            dense_cloud = root / "dense.ply"
            dense_cloud.write_text("ply\n", encoding="utf-8")
            output_dir = root / "out"
            log_path = output_dir / "commands.log"
            args = types.SimpleNamespace(dem_resolution=2.5)
            tools = {"terrain_dem_dom_tool": Path("/usr/bin/terrain_dem_dom_tool")}
            captured = {}

            original_run_command = pipeline.run_command

            def fake_run_command(cmd, log):
                captured["cmd"] = cmd
                products = output_dir / "terrain" / "products"
                products.mkdir(parents=True, exist_ok=True)
                (products / "dem.tif").write_bytes(b"dem")
                (products / "dom.png").write_bytes(b"dom")
                return pipeline.CommandResult(
                    0,
                    '{"dom": {"camera_projected": true, "filled_pixel_count": 42}}',
                    "",
                    0.01,
                )

            pipeline.run_command = fake_run_command
            original_evaluate_dom_quality = pipeline.evaluate_dom_quality
            original_evaluate_dem_quality = pipeline.evaluate_dem_quality
            pipeline.evaluate_dom_quality = lambda path: {"passed": True, "reason": "ok"}
            pipeline.evaluate_dem_quality = lambda path: {"passed": True, "reason": "ok"}
            try:
                result = pipeline.run_terrain(
                    args,
                    items,
                    tools,
                    {"point_cloud": str(dense_cloud)},
                    output_dir,
                    log_path,
                )
            finally:
                pipeline.run_command = original_run_command
                pipeline.evaluate_dom_quality = original_evaluate_dom_quality
                pipeline.evaluate_dem_quality = original_evaluate_dem_quality

            self.assertEqual(result["status"], "ok")
            self.assertEqual(captured["cmd"][1], "--list")
            self.assertEqual(captured["cmd"][2], str(dense_cloud))

            image_camera_list = Path(captured["cmd"][3])
            self.assertTrue(image_camera_list.exists())
            self.assertEqual(captured["cmd"][4], str(output_dir / "terrain"))
            self.assertEqual(captured["cmd"][5], "2.5")

            lines = image_camera_list.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], f"{items[0].work_image} {items[0].work_camera}")
            self.assertEqual(lines[1], f"{items[1].work_image} {items[1].work_camera}")

    def write_ascii_ply(self, path, points):
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {len(points)}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write("property float error\n")
            handle.write("end_header\n")
            for point in points:
                handle.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")

    def write_ascii_ply_with_intensity(self, path, points):
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {len(points)}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write("property float error\n")
            handle.write("property uchar intensity\n")
            handle.write("end_header\n")
            for point in points:
                handle.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]}\n")

    def test_quality_filtered_merge_removes_far_outliers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ply_a = root / "a.ply"
            ply_b = root / "b.ply"
            output = root / "merged.ply"
            cluster = [
                (-2.0, -1.0, 0.0, 0.01),
                (-1.0, 0.0, 0.0, 0.01),
                (0.0, 0.0, 0.0, 0.01),
                (1.0, 0.0, 0.0, 0.01),
                (2.0, 1.0, 0.0, 0.01),
            ]
            self.write_ascii_ply(ply_a, cluster[:3])
            self.write_ascii_ply(ply_b, [*cluster[3:], (1000000.0, 0.0, 0.0, 0.01)])

            result = pipeline.write_quality_filtered_ascii_ply([ply_a, ply_b], output)

            self.assertEqual(result["raw_point_count"], 6)
            self.assertEqual(result["point_count"], 5)
            self.assertGreater(result["removed_outlier_count"], 0)
            self.assertTrue(result["quality"]["passed"])
            self.assertNotIn("1000000.0", output.read_text(encoding="utf-8"))

    def test_quality_filtered_merge_preserves_intensity_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ply_a = root / "a.ply"
            output = root / "merged.ply"
            self.write_ascii_ply_with_intensity(
                ply_a,
                [
                    (-1.0, 0.0, 0.0, 0.01, 51),
                    (0.0, 0.0, 0.0, 0.01, 87),
                    (1.0, 0.0, 0.0, 0.01, 86),
                ],
            )

            result = pipeline.write_quality_filtered_ascii_ply([ply_a], output)

            self.assertEqual(result["point_count"], 3)
            text = output.read_text(encoding="utf-8")
            self.assertIn("property uchar intensity", text)
            self.assertIn("-1.0 0.0 0.0 0.01 51", text)
            self.assertIn("0.0 0.0 0.0 0.01 87", text)

    def test_terrain_local_frame_transforms_cloud_and_cameras_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense_cloud = root / "dense.ply"
            self.write_ascii_ply(dense_cloud, [(11.0, 22.0, 27.0, 0.25)])

            image = root / "image.png"
            image.write_bytes(b"image")
            camera = root / "camera.tsai"
            camera.write_text(
                "\n".join([
                    "VERSION_4",
                    "PINHOLE",
                    "fu = 10",
                    "fv = 10",
                    "cu = 5",
                    "cv = 5",
                    "u_direction = 1 0 0",
                    "v_direction = 0 1 0",
                    "w_direction = 0 0 1",
                    "C = 11 22 27",
                    "R = 1 0 0 0 1 0 0 0 1",
                    "pitch = 1",
                    "",
                ]),
                encoding="utf-8",
            )
            item = pipeline.InputItem(image=image, camera=camera, work_image=image, work_camera=camera)
            frame = pipeline.LocalFrame(
                origin=(10.0, 20.0, 30.0),
                axes=(
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, -1.0),
                ),
            )

            result = pipeline.prepare_terrain_local_frame(dense_cloud, [item], root / "local", frame=frame)

            self.assertTrue(result["enabled"])
            local_rows = pipeline.read_ascii_ply_vertex_rows(Path(result["point_cloud"]))
            self.assertEqual(local_rows, ["2 1 3 0.25"])
            local_camera = Path(result["items"][0].work_camera).read_text(encoding="utf-8")
            self.assertIn("C = 2 1 3", local_camera)
            self.assertIn("R = 0 1 0 1 0 0 0 0 -1", local_camera)

    def test_terrain_local_frame_preserves_intensity_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense_cloud = root / "dense.ply"
            self.write_ascii_ply_with_intensity(dense_cloud, [(11.0, 22.0, 27.0, 0.25, 51)])

            image = root / "image.png"
            image.write_bytes(b"image")
            camera = root / "camera.tsai"
            camera.write_text(
                "\n".join([
                    "VERSION_4",
                    "PINHOLE",
                    "fu = 10",
                    "fv = 10",
                    "cu = 5",
                    "cv = 5",
                    "u_direction = 1 0 0",
                    "v_direction = 0 1 0",
                    "w_direction = 0 0 1",
                    "C = 11 22 27",
                    "R = 1 0 0 0 1 0 0 0 1",
                    "pitch = 1",
                    "",
                ]),
                encoding="utf-8",
            )
            item = pipeline.InputItem(image=image, camera=camera, work_image=image, work_camera=camera)
            frame = pipeline.LocalFrame(
                origin=(10.0, 20.0, 30.0),
                axes=(
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, -1.0),
                ),
            )

            result = pipeline.prepare_terrain_local_frame(dense_cloud, [item], root / "local", frame=frame)

            self.assertTrue(result["enabled"])
            local_rows = pipeline.read_ascii_ply_vertex_rows(Path(result["point_cloud"]))
            self.assertEqual(local_rows, ["2 1 3 0.25 51"])

    def test_dom_mask_quality_rejects_fragmented_large_components(self):
        mask = [
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ]

        quality = pipeline.evaluate_dom_mask_quality(mask)

        self.assertFalse(quality["passed"])
        self.assertEqual(quality["component_count"], 4)
        self.assertEqual(quality["large_component_count"], 4)
        self.assertLess(quality["largest_component_ratio"], 0.8)


class ExtractFeaturesScriptTest(unittest.TestCase):
    def load_script_with_dependency_stubs(self):
        spec = importlib.util.spec_from_file_location("extract_features_under_test", EXTRACT_FEATURES_SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.device = lambda name: name

        with mock.patch.dict(sys.modules, {
            "cv2": types.ModuleType("cv2"),
            "numpy": types.ModuleType("numpy"),
            "torch": fake_torch,
        }):
            spec.loader.exec_module(module)
        return module

    def load_script_with_real_numpy_dependency_stubs(self):
        spec = importlib.util.spec_from_file_location("extract_features_under_test", EXTRACT_FEATURES_SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.device = lambda name: name

        with mock.patch.dict(sys.modules, {
            "cv2": types.ModuleType("cv2"),
            "torch": fake_torch,
        }):
            spec.loader.exec_module(module)
        return module

    def assert_missing_lightglue_is_reported_cleanly(self, algo):
        module = self.load_script_with_dependency_stubs()

        original_import = __import__

        def import_without_lightglue(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lightglue" or name.startswith("lightglue."):
                raise ModuleNotFoundError("No module named 'lightglue'")
            return original_import(name, globals, locals, fromlist, level)

        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "extract_features.py",
            "--algo", algo,
            "--images", "missing.png",
            "--output", tempfile.gettempdir(),
            "--device", "cpu",
        ]

        with mock.patch.object(sys, "argv", argv), \
                mock.patch("builtins.__import__", side_effect=import_without_lightglue), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            try:
                module.main()
            except ModuleNotFoundError as exc:
                self.fail(f"裸 ModuleNotFoundError 泄漏到调用方: {exc}")
            except SystemExit as exc:
                self.assertNotEqual(exc.code, 0)
            else:
                self.fail("缺少 lightglue 时脚本不应成功退出")

        self.assertIn("lightglue", stderr.getvalue().lower())
        self.assertIn("git+https://github.com/cvg/LightGlue.git", stderr.getvalue())
        self.assertNotIn("traceback", stderr.getvalue().lower())

    def test_vendored_lightglue_directory_is_added_to_import_path(self):
        module = self.load_script_with_dependency_stubs()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "3rdparty" / "LightGlue-main" / "lightglue"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")

            original_path = list(sys.path)
            try:
                sys.path = [p for p in sys.path if str(root) not in p and "LightGlue-main" not in p]

                added = module.add_vendored_lightglue_to_path(root)

                self.assertEqual(added, package_dir.parent)
                self.assertEqual(sys.path[0], str(package_dir.parent))
            finally:
                sys.path = original_path

    def test_disk_reports_missing_lightglue_without_traceback(self):
        self.assert_missing_lightglue_is_reported_cleanly("disk")

    def test_aliked_reports_missing_lightglue_without_traceback(self):
        self.assert_missing_lightglue_is_reported_cleanly("aliked")

    def test_grayscale_range_filter_keeps_only_points_inside_image_intensity_window(self):
        module = self.load_script_with_real_numpy_dependency_stubs()

        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[0, 0, :] = 0.10
        image[1, 1, :] = 0.45
        image[2, 2, :] = 0.80
        image[3, 3, :] = 0.95
        keypoints = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [99.0, 99.0],
        ], dtype=np.float32)
        descriptors = np.arange(10, dtype=np.float32).reshape(5, 2)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        filtered_keypoints, filtered_descriptors, filtered_scores = module.filter_features_by_grayscale_range(
            keypoints,
            descriptors,
            scores,
            image,
            0.40,
            0.85,
        )

        self.assertEqual(filtered_keypoints.tolist(), [[1.0, 1.0], [2.0, 2.0]])
        self.assertEqual(filtered_descriptors.tolist(), [[2.0, 3.0], [4.0, 5.0]])
        np.testing.assert_allclose(filtered_scores, np.array([0.2, 0.3], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
