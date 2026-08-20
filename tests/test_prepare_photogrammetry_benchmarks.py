import importlib.util
import json
import math
import shlex
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "testData" / "prepare_photogrammetry_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("prepare_photogrammetry_benchmarks", SCRIPT_PATH)
preparer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = preparer
SPEC.loader.exec_module(preparer)


def project_with_middlebury(K, R_wc, t_wc, point):
    cam = [
        sum(R_wc[row][col] * point[col] for col in range(3)) + t_wc[row]
        for row in range(3)
    ]
    return [
        (K[0][0] * cam[0] + K[0][1] * cam[1] + K[0][2] * cam[2]) / cam[2],
        (K[1][0] * cam[0] + K[1][1] * cam[1] + K[1][2] * cam[2]) / cam[2],
    ]


def project_with_plascan(camera, point):
    R = camera.rotation_camera_to_world
    C = camera.center
    delta = [point[i] - C[i] for i in range(3)]
    cam = [
        R[0][row] * delta[0] + R[1][row] * delta[1] + R[2][row] * delta[2]
        for row in range(3)
    ]
    return [
        (camera.K[0][0] * cam[0] + camera.K[0][1] * cam[1] + camera.K[0][2] * cam[2]) / cam[2],
        (camera.K[1][0] * cam[0] + camera.K[1][1] * cam[1] + camera.K[1][2] * cam[2]) / cam[2],
    ]


class PreparePhotogrammetryBenchmarksTest(unittest.TestCase):
    def test_middlebury_pose_conversion_matches_original_projection(self):
        row = (
            "dinoSR0001.png "
            "120 0 40 0 130 50 0 0 1 "
            "0 -1 0 1 0 0 0 0 1 "
            "2 -3 4"
        )

        camera = preparer.parse_middlebury_line(row)

        self.assertEqual(camera.image_name, "dinoSR0001.png")
        self.assertEqual(camera.rotation_camera_to_world, [[0.0, 1.0, 0.0],
                                                           [-1.0, 0.0, 0.0],
                                                           [0.0, 0.0, 1.0]])
        self.assertEqual(camera.center, [3.0, 2.0, -4.0])

        point = [4.0, 5.0, 8.0]
        original_uv = project_with_middlebury(camera.K, [[0.0, -1.0, 0.0],
                                                        [1.0, 0.0, 0.0],
                                                        [0.0, 0.0, 1.0]],
                                             [2.0, -3.0, 4.0],
                                             point)
        converted_uv = project_with_plascan(camera, point)

        self.assertAlmostEqual(converted_uv[0], original_uv[0], places=9)
        self.assertAlmostEqual(converted_uv[1], original_uv[1], places=9)

    def test_epfl_pose_conversion_matches_original_projection_and_records_skew(self):
        text = "\n".join([
            "3954.75 -8.5 1619.9",
            "0 3948.0 1151.4",
            "0 0 1",
            "0 0 0",
            "0 -1 0",
            "1 0 0",
            "0 0 1",
            "60 -11 -35",
            "3072 2048",
        ])

        with tempfile.TemporaryDirectory() as tmp:
            camera_path = Path(tmp) / "rdimage.000.ppm.camera"
            camera_path.write_text(text + "\n", encoding="utf-8")

            camera = preparer.parse_epfl_camera_file(camera_path, image_name="rdimage.000.ppm")

        self.assertEqual(camera.image_name, "rdimage.000.ppm")
        self.assertEqual(camera.rotation_camera_to_world, [[0.0, -1.0, 0.0],
                                                           [1.0, 0.0, 0.0],
                                                           [0.0, 0.0, 1.0]])
        self.assertEqual(camera.center, [60.0, -11.0, -35.0])
        self.assertTrue(any("skew" in warning for warning in camera.warnings))

        point = [62.0, -10.0, -25.0]
        converted_uv = project_with_plascan(camera, point)

        R = camera.rotation_camera_to_world
        delta = [point[i] - camera.center[i] for i in range(3)]
        cam = [
            R[0][row] * delta[0] + R[1][row] * delta[1] + R[2][row] * delta[2]
            for row in range(3)
        ]
        original_uv = [
            (camera.K[0][0] * cam[0] + camera.K[0][1] * cam[1] + camera.K[0][2] * cam[2]) / cam[2],
            (camera.K[1][0] * cam[0] + camera.K[1][1] * cam[1] + camera.K[1][2] * cam[2]) / cam[2],
        ]

        self.assertAlmostEqual(converted_uv[0], original_uv[0], places=9)
        self.assertAlmostEqual(converted_uv[1], original_uv[1], places=9)

    def test_prepare_middlebury_writes_tsai_lis_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            scene_dir = root / "middlebury_dino_sparse_ring" / "extracted" / "dinoSparseRing"
            scene_dir.mkdir(parents=True)
            (scene_dir / "dinoSR0001.png").write_bytes(b"fake png")
            (scene_dir / "dinoSR0002.png").write_bytes(b"fake png")
            (scene_dir / "dinoSR_par.txt").write_text(
                "\n".join([
                    "2",
                    "dinoSR0001.png 120 0 40 0 130 50 0 0 1 1 0 0 0 1 0 0 0 1 1 2 3",
                    "dinoSR0002.png 120 0 40 0 130 50 0 0 1 1 0 0 0 1 0 0 0 1 4 5 6",
                ]) + "\n",
                encoding="utf-8",
            )

            result = preparer.prepare_dataset(root, "middlebury_dino_sparse_ring", overwrite=True)

            self.assertEqual(result.dataset_id, "middlebury_dino_sparse_ring")
            self.assertEqual(result.camera_count, 2)
            self.assertTrue(result.image_camera_list.exists())

            lis_lines = result.image_camera_list.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lis_lines), 2)
            first_image, first_camera = shlex.split(lis_lines[0])
            self.assertEqual((result.image_camera_list.parent / first_image).resolve(),
                             (scene_dir / "dinoSR0001.png").resolve())
            self.assertEqual((result.image_camera_list.parent / first_camera).resolve(),
                             (result.output_dir / "cameras" / "dinoSR0001.tsai").resolve())

            tsai_text = (result.output_dir / "cameras" / "dinoSR0001.tsai").read_text(encoding="utf-8")
            self.assertIn("fu = 120", tsai_text)
            self.assertIn("C = -1 -2 -3", tsai_text)

            summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["dataset_id"], "middlebury_dino_sparse_ring")
            self.assertEqual(summary["camera_count"], 2)
            self.assertEqual(summary["input_format"], "middlebury_par")

    def test_prepare_epfl_writes_summary_warning_for_skew(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            scene_dir = root / "epfl_rathaus_multiview" / "extracted"
            scene_dir.mkdir(parents=True)
            (scene_dir / "rdimage.000.ppm").write_bytes(b"fake ppm")
            (scene_dir / "rdimage.000.ppm.camera").write_text(
                "\n".join([
                    "3954.75 -8.5 1619.9",
                    "0 3948.0 1151.4",
                    "0 0 1",
                    "0 0 0",
                    "1 0 0",
                    "0 1 0",
                    "0 0 1",
                    "60 -11 -35",
                ]) + "\n",
                encoding="utf-8",
            )

            result = preparer.prepare_dataset(root, "epfl_rathaus_multiview", overwrite=True)

            self.assertEqual(result.camera_count, 1)
            summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input_format"], "epfl_camera")
            self.assertTrue(any("skew" in warning for warning in summary["warnings"]))

    def test_main_prepares_selected_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            scene_dir = root / "middlebury_temple_sparse_ring" / "extracted" / "templeSparseRing"
            scene_dir.mkdir(parents=True)
            (scene_dir / "templeSR0001.png").write_bytes(b"fake png")
            (scene_dir / "templeSR_par.txt").write_text(
                "1\n"
                "templeSR0001.png 120 0 40 0 130 50 0 0 1 1 0 0 0 1 0 0 0 1 1 2 3\n",
                encoding="utf-8",
            )

            exit_code = preparer.main([
                "--target-root", str(root),
                "--dataset", "middlebury_temple_sparse_ring",
                "--overwrite",
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "middlebury_temple_sparse_ring" / "prepared" / "plascan" /
                             "image_camera.lis").exists())


if __name__ == "__main__":
    unittest.main()
