import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "testData" / "compare_lidar_ba_ab_results.py"
SPEC = importlib.util.spec_from_file_location("compare_lidar_ba_ab_results", SCRIPT_PATH)
comparator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparator
SPEC.loader.exec_module(comparator)


class CompareLidarBaAbResultsTest(unittest.TestCase):
    def test_compare_summaries_reports_reprojection_and_lidar_improvements(self):
        baseline = {
            "track_count": 100,
            "optimized_count": 95,
            "mean_rms_before": 1.8,
            "mean_rms_after": 1.2,
        }
        lidar = {
            "track_count": 100,
            "optimized_count": 94,
            "mean_rms_before": 1.8,
            "mean_rms_after": 1.25,
            "laser_constraints_summary": {
                "enabled": True,
                "associated_tracks": 42,
                "laser_constraint_count": 42,
                "laser_rms_before_m": 0.75,
                "laser_rms_after_m": 0.18,
                "laser_median_before_m": 0.5,
                "laser_median_after_m": 0.12,
            },
        }

        comparison = comparator.compare_summaries(baseline, lidar)

        self.assertEqual(comparison["baseline"]["optimized_count"], 95)
        self.assertEqual(comparison["lidar"]["optimized_count"], 94)
        self.assertAlmostEqual(comparison["deltas"]["reprojection_rms_after_px"], 0.05)
        self.assertAlmostEqual(comparison["deltas"]["laser_rms_reduction_m"], 0.57)
        self.assertAlmostEqual(comparison["deltas"]["laser_rms_reduction_percent"], 76.0)
        self.assertEqual(comparison["lidar"]["associated_tracks"], 42)
        self.assertIn("LiDAR", comparison["verdict"])

    def test_compare_summaries_reports_common_tracks_and_camera_pose_deltas(self):
        baseline = {
            "track_count": 3,
            "optimized_count": 2,
            "mean_rms_before": 2.0,
            "mean_rms_after": 1.0,
            "points": [
                {"index": 0, "valid": True, "rms_after": 1.0},
                {"index": 1, "valid": False, "rms_after": 99.0},
                {"index": 2, "valid": True, "rms_after": 3.0},
            ],
            "refined_cameras": [
                {"image_path": "a.jpg", "camera": {"C": [0.0, 0.0, 0.0], "R": [1, 0, 0, 0, 1, 0, 0, 0, 1]}},
                {"image_path": "b.jpg", "camera": {"C": [1.0, 0.0, 0.0], "R": [1, 0, 0, 0, 1, 0, 0, 0, 1]}},
            ],
        }
        lidar = {
            "track_count": 3,
            "optimized_count": 3,
            "mean_rms_before": 2.0,
            "mean_rms_after": 1.2,
            "points": [
                {"index": 0, "valid": True, "rms_after": 1.5},
                {"index": 1, "valid": True, "rms_after": 9.0},
                {"index": 2, "valid": True, "rms_after": 2.5},
            ],
            "refined_cameras": [
                {"image_path": "a.jpg", "camera": {"C": [3.0, 4.0, 0.0], "R": [1, 0, 0, 0, 1, 0, 0, 0, 1]}},
                {"image_path": "b.jpg", "camera": {"C": [1.0, 0.0, 0.0], "R": [1, 0, 0, 0, 1, 0, 0, 0, 1]}},
            ],
            "laser_constraints_summary": {
                "enabled": True,
                "associated_tracks": 2,
                "laser_constraint_count": 2,
                "laser_rms_before_m": 1.0,
                "laser_rms_after_m": 0.5,
            },
        }

        comparison = comparator.compare_summaries(baseline, lidar)

        common = comparison["common_tracks"]
        self.assertEqual(common["common_valid_count"], 2)
        self.assertEqual(common["baseline_valid_count"], 2)
        self.assertEqual(common["lidar_valid_count"], 3)
        self.assertAlmostEqual(common["baseline_mean_rms_after_px"], 2.0)
        self.assertAlmostEqual(common["lidar_mean_rms_after_px"], 2.0)
        self.assertAlmostEqual(common["delta_mean_rms_after_px"], 0.0)

        camera_delta = comparison["camera_pose_delta"]
        self.assertEqual(camera_delta["common_camera_count"], 2)
        self.assertAlmostEqual(camera_delta["mean_center_shift_m"], 2.5)
        self.assertAlmostEqual(camera_delta["max_center_shift_m"], 5.0)

    def test_main_writes_json_and_markdown_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            lidar_path = root / "lidar.json"
            out_json = root / "comparison.json"
            out_md = root / "comparison.md"
            baseline_path.write_text(
                json.dumps({
                    "track_count": 10,
                    "optimized_count": 10,
                    "mean_rms_before": 2.0,
                    "mean_rms_after": 1.0,
                }),
                encoding="utf-8",
            )
            lidar_path.write_text(
                json.dumps({
                    "track_count": 10,
                    "optimized_count": 10,
                    "mean_rms_before": 2.0,
                    "mean_rms_after": 1.1,
                    "laser_constraints_summary": {
                        "enabled": True,
                        "associated_tracks": 5,
                        "laser_constraint_count": 5,
                        "laser_rms_before_m": 1.0,
                        "laser_rms_after_m": 0.2,
                    },
                }),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = comparator.main([
                    "--baseline-json",
                    str(baseline_path),
                    "--lidar-json",
                    str(lidar_path),
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                ])

            self.assertEqual(exit_code, 0)
            comparison = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertAlmostEqual(comparison["deltas"]["laser_rms_reduction_m"], 0.8)
            self.assertIn("LiDAR BA A/B Comparison", out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
