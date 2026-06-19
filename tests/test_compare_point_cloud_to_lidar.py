import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "testData" / "compare_point_cloud_to_lidar.py"


def load_comparator():
    if not SCRIPT_PATH.exists():
        raise AssertionError(f"missing comparison helper: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("compare_point_cloud_to_lidar", SCRIPT_PATH)
    comparator = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = comparator
    spec.loader.exec_module(comparator)
    return comparator


def write_ascii_ply(path: Path, rows: list[str]) -> None:
    path.write_text(
        "\n".join([
            "ply",
            "format ascii 1.0",
            f"element vertex {len(rows)}",
            "property double x",
            "property double y",
            "property double z",
            "property uchar classification",
            "end_header",
            *rows,
        ]),
        encoding="utf-8",
    )


class ComparePointCloudToLidarTest(unittest.TestCase):
    def test_compare_ascii_ply_to_reference_reports_nearest_neighbor_metrics(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reconstruction.ply"
            reference = root / "lidar_reference.ply"
            write_ascii_ply(source, [
                "0.300000 0.000000 0.000000 2",
                "9.000000 0.000000 0.000000 2",
            ])
            write_ascii_ply(reference, [
                "0.000000 0.000000 0.000000 5",
                "10.000000 0.000000 0.000000 5",
            ])

            comparison = comparator.compare_point_clouds(
                source,
                reference,
                max_rmse_m=1.0,
                max_p95_m=1.0,
            )

            self.assertEqual(comparison["source_points"], 2)
            self.assertEqual(comparison["reference_points"], 2)
            metrics = comparison["distance_m"]
            self.assertAlmostEqual(metrics["mean"], 0.65)
            self.assertAlmostEqual(metrics["rmse"], (0.09 + 1.0) ** 0.5 / (2 ** 0.5))
            self.assertAlmostEqual(metrics["median"], 0.65)
            self.assertAlmostEqual(metrics["p95"], 1.0)
            self.assertAlmostEqual(metrics["max"], 1.0)
            self.assertTrue(comparison["quality_gate"]["passed"])

    def test_compare_text_xyz_and_csv_reference_points(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reconstruction.xyz"
            reference = root / "lidar_reference.csv"
            source.write_text(
                "\n".join([
                    "# x y z",
                    "0.25 0.00 0.00",
                    "9.50 0.00 0.00",
                ]),
                encoding="utf-8",
            )
            reference.write_text(
                "\n".join([
                    "x,y,z,intensity",
                    "0.00,0.00,0.00,128",
                    "10.00,0.00,0.00,96",
                ]),
                encoding="utf-8",
            )

            comparison = comparator.compare_point_clouds(source, reference)

            self.assertEqual(comparison["source_points"], 2)
            self.assertEqual(comparison["reference_points"], 2)
            metrics = comparison["distance_m"]
            self.assertAlmostEqual(metrics["mean"], 0.375)
            self.assertAlmostEqual(metrics["p95"], 0.5)

    def test_kd_tree_nearest_neighbor_matches_bruteforce(self):
        comparator = load_comparator()
        source_points = [
            (-3.0, 0.0, 0.0),
            (4.9, 0.0, 0.0),
            (8.0, 2.0, 0.0),
            (0.0, 0.0, 5.0),
        ]
        reference_points = [
            (-10.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (5.1, 0.0, 0.0),
            (9.0, 2.0, 0.0),
            (0.0, 0.0, 8.0),
        ]

        brute = comparator.nearest_neighbor_distances(source_points, reference_points, method="brute")
        kd_tree = comparator.nearest_neighbor_distances(source_points, reference_points, method="kd-tree")

        self.assertEqual(len(kd_tree), len(brute))
        for left, right in zip(kd_tree, brute):
            self.assertAlmostEqual(left, right)

    def test_reference_coverage_gate_catches_partial_lidar_overlap(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "partial_reconstruction.xyz"
            reference = root / "lidar_reference.xyz"
            source.write_text(
                "\n".join([
                    "0.00 0.00 0.00",
                    "0.10 0.00 0.00",
                ]),
                encoding="utf-8",
            )
            reference.write_text(
                "\n".join([
                    "0.00 0.00 0.00",
                    "10.00 0.00 0.00",
                ]),
                encoding="utf-8",
            )

            comparison = comparator.compare_point_clouds(
                source,
                reference,
                coverage_radius_m=0.5,
                min_reference_coverage_percent=75.0,
            )

            coverage = comparison["reference_coverage"]
            self.assertEqual(coverage["covered_points"], 1)
            self.assertEqual(coverage["total_points"], 2)
            self.assertAlmostEqual(coverage["covered_percent"], 50.0)
            gate = comparison["quality_gate"]
            self.assertFalse(gate["passed"])
            self.assertIn("reference_coverage_below_threshold", gate["failure_codes"])

    def test_cli_accepts_explicit_kd_tree_nearest_neighbor_method(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reconstruction.xyz"
            reference = root / "lidar_reference.xyz"
            output_json = root / "quality.json"
            source.write_text("0.2 0 0\n2.0 0 0\n", encoding="utf-8")
            reference.write_text("0 0 0\n3 0 0\n", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = comparator.main([
                    "--source",
                    str(source),
                    "--reference",
                    str(reference),
                    "--output-json",
                    str(output_json),
                    "--nearest-neighbor-method",
                    "kd-tree",
                ])

            self.assertEqual(exit_code, 0)
            self.assertIn("wrote:", stdout.getvalue())
            comparison = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(comparison["nearest_neighbor_method"], "kd-tree")
            self.assertAlmostEqual(comparison["distance_m"]["max"], 1.0)

    def test_cli_writes_json_and_fails_when_quality_gate_fails(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reconstruction.ply"
            reference = root / "lidar_reference.ply"
            output_json = root / "quality.json"
            write_ascii_ply(source, [
                "3.000000 0.000000 0.000000 2",
                "4.000000 0.000000 0.000000 2",
            ])
            write_ascii_ply(reference, [
                "0.000000 0.000000 0.000000 5",
            ])

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = comparator.main([
                    "--source",
                    str(source),
                    "--reference",
                    str(reference),
                    "--output-json",
                    str(output_json),
                    "--max-rmse-m",
                    "1.0",
                    "--max-p95-m",
                    "1.0",
                    "--fail-on-quality-gate",
                ])

            self.assertEqual(exit_code, 2)
            self.assertIn("wrote:", stdout.getvalue())
            self.assertIn("quality gate failed", stderr.getvalue())
            comparison = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertFalse(comparison["quality_gate"]["passed"])
            self.assertIn("rmse_above_threshold", comparison["quality_gate"]["failure_codes"])
            self.assertIn("p95_above_threshold", comparison["quality_gate"]["failure_codes"])

    def test_cli_rejects_reference_coverage_gate_without_radius(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reconstruction.xyz"
            reference = root / "lidar_reference.xyz"
            output_json = root / "quality.json"
            source.write_text("0 0 0\n", encoding="utf-8")
            reference.write_text("0 0 0\n", encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = comparator.main([
                    "--source",
                    str(source),
                    "--reference",
                    str(reference),
                    "--output-json",
                    str(output_json),
                    "--min-reference-coverage-percent",
                    "80",
                ])

            self.assertEqual(exit_code, 1)
            self.assertFalse(output_json.exists())
            self.assertIn("coverage radius", stderr.getvalue())

    def test_parser_rejects_binary_ply(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binary.ply"
            path.write_text(
                "\n".join([
                    "ply",
                    "format binary_little_endian 1.0",
                    "element vertex 1",
                    "property double x",
                    "property double y",
                    "property double z",
                    "end_header",
                ]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "only ASCII PLY"):
                comparator.read_ascii_ply_points(path)


if __name__ == "__main__":
    unittest.main()
