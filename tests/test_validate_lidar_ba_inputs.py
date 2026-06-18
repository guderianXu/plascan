import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "testData" / "validate_lidar_ba_inputs.py"
SPEC = importlib.util.spec_from_file_location("validate_lidar_ba_inputs", SCRIPT_PATH)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def write_ply(path: Path, properties: list[tuple[str, str]], vertex_count: int = 3) -> None:
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment test fixture",
        f"element vertex {vertex_count}",
    ]
    header.extend(f"property {kind} {name}" for kind, name in properties)
    header.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + b"\x00" * 64)


class ValidateLidarBaInputsTest(unittest.TestCase):
    def test_parse_ply_header_detects_ba_ready_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "frame_000000.ply"
            write_ply(
                ply,
                [
                    ("float", "x"),
                    ("float", "y"),
                    ("float", "z"),
                    ("float", "intensity"),
                    ("float", "normal_x"),
                    ("float", "normal_y"),
                    ("float", "normal_z"),
                    ("float", "curvature"),
                ],
                vertex_count=275,
            )

            header = validator.parse_ply_header(ply)

            self.assertEqual(header.vertex_count, 275)
            self.assertTrue(header.has_xyz)
            self.assertTrue(header.has_normals)
            self.assertTrue(header.has_curvature)
            self.assertTrue(header.ba_ready)

    def test_scan_dataset_recommends_cloud_registered_for_ba_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mun_frl" / "extracted"
            write_ply(
                root / "lidar" / "cloud_registered" / "frame_000000.ply",
                [
                    ("float", "x"),
                    ("float", "y"),
                    ("float", "z"),
                    ("float", "intensity"),
                    ("float", "normal_x"),
                    ("float", "normal_y"),
                    ("float", "normal_z"),
                    ("float", "curvature"),
                ],
            )
            write_ply(
                root / "lidar" / "velodyne_points" / "frame_000000.ply",
                [
                    ("float", "x"),
                    ("float", "y"),
                    ("float", "z"),
                    ("float", "intensity"),
                    ("ushort", "ring"),
                    ("float", "time"),
                ],
            )

            summary = validator.summarize_dataset(root, max_files_per_stream=5)

            self.assertEqual(
                summary["recommendation"]["ba_constraint_cloud_stream"],
                "lidar/cloud_registered",
            )
            self.assertIn("lidar/velodyne_points", summary["recommendation"]["needs_normal_estimation"])
            self.assertEqual(summary["streams"]["lidar/cloud_registered"]["ba_ready_files"], 1)
            self.assertEqual(summary["streams"]["lidar/velodyne_points"]["fusion_reference_files"], 1)

    def test_main_writes_json_summary_and_returns_success_for_ba_ready_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "extracted"
            out_json = Path(tmp) / "lidar_ba_summary.json"
            write_ply(
                root / "lidar" / "cloud_registered" / "frame_000000.ply",
                [
                    ("float", "x"),
                    ("float", "y"),
                    ("float", "z"),
                    ("float", "normal_x"),
                    ("float", "normal_y"),
                    ("float", "normal_z"),
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = validator.main([
                    "--dataset-root",
                    str(root),
                    "--summary-json",
                    str(out_json),
                    "--max-files-per-stream",
                    "1",
                ])

            self.assertEqual(exit_code, 0)
            summary = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["recommendation"]["ba_constraint_cloud_stream"], "lidar/cloud_registered")

    def test_main_returns_nonzero_when_only_fusion_reference_clouds_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "extracted"
            write_ply(
                root / "lidar" / "velodyne_points" / "frame_000000.ply",
                [
                    ("float", "x"),
                    ("float", "y"),
                    ("float", "z"),
                    ("float", "intensity"),
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = validator.main(["--dataset-root", str(root), "--max-files-per-stream", "1"])

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
