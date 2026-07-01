import contextlib
import importlib.util
import io
import json
import struct
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


def write_binary_ply(path: Path, vertices: list[tuple[float, float, float, int, int, int]]) -> None:
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ]).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for vertex in vertices:
            handle.write(struct.pack("<fffBBB", *vertex))


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

    def test_compare_reports_signed_vertical_error_to_reference_surface(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "rough_reconstruction.xyz"
            reference = root / "flat_reference.xyz"
            source.write_text(
                "\n".join([
                    "0.00 0.00 1.00",
                    "1.00 0.00 -2.00",
                ]),
                encoding="utf-8",
            )
            reference.write_text(
                "\n".join([
                    "0.00 0.00 0.00",
                    "1.00 0.00 0.00",
                ]),
                encoding="utf-8",
            )

            comparison = comparator.compare_point_clouds(source, reference, nearest_neighbor_method="kd-tree")

            vertical = comparison["vertical_error_m"]
            self.assertAlmostEqual(vertical["mean_signed"], -0.5)
            self.assertAlmostEqual(vertical["mean_abs"], 1.5)
            self.assertAlmostEqual(vertical["rmse"], (1.0 + 4.0) ** 0.5 / (2 ** 0.5))
            self.assertAlmostEqual(vertical["p95_abs"], 2.0)

    def test_vertical_error_quality_gate_catches_rough_surface(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "rough_reconstruction.xyz"
            reference = root / "flat_reference.xyz"
            source.write_text("0 0 1\n1 0 -2\n", encoding="utf-8")
            reference.write_text("0 0 0\n1 0 0\n", encoding="utf-8")

            comparison = comparator.compare_point_clouds(
                source,
                reference,
                max_vertical_rmse_m=1.0,
                max_vertical_p95_m=1.5,
            )

            gate = comparison["quality_gate"]
            self.assertFalse(gate["passed"])
            self.assertIn("vertical_rmse_above_threshold", gate["failure_codes"])
            self.assertIn("vertical_p95_above_threshold", gate["failure_codes"])
            self.assertEqual(gate["thresholds"]["max_vertical_rmse_m"], 1.0)
            self.assertEqual(gate["thresholds"]["max_vertical_p95_m"], 1.5)

    def test_local_roughness_gate_catches_thick_terrain_cloud(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "thick_reconstruction.xyz"
            reference = root / "flat_metashape_reference.xyz"
            source.write_text(
                "\n".join([
                    "0.00 0.00 0.00",
                    "0.10 0.00 0.02",
                    "0.00 0.10 0.04",
                    "0.10 0.10 1.20",
                    "1.00 0.00 0.00",
                    "1.10 0.00 0.01",
                    "1.00 0.10 0.03",
                    "1.10 0.10 0.04",
                ]),
                encoding="utf-8",
            )
            reference.write_text(
                "\n".join([
                    "0.00 0.00 0.00",
                    "0.10 0.00 0.01",
                    "0.00 0.10 0.02",
                    "0.10 0.10 0.03",
                    "1.00 0.00 0.00",
                    "1.10 0.00 0.01",
                    "1.00 0.10 0.02",
                    "1.10 0.10 0.03",
                ]),
                encoding="utf-8",
            )

            comparison = comparator.compare_point_clouds(
                source,
                reference,
                roughness_grid_cells=1,
                roughness_min_cell_points=3,
                max_local_z_range_p95_m=0.25,
            )

            source_roughness = comparison["source_local_roughness"]
            reference_roughness = comparison["reference_local_roughness"]
            self.assertGreater(source_roughness["z_range_in_cell"]["p95"], 1.0)
            self.assertLess(reference_roughness["z_range_in_cell"]["p95"], 0.05)
            gate = comparison["quality_gate"]
            self.assertFalse(gate["passed"])
            self.assertIn("local_z_range_p95_above_threshold", gate["failure_codes"])
            self.assertEqual(gate["thresholds"]["max_local_z_range_p95_m"], 0.25)

    def test_improvement_report_compares_raw_refined_and_reference_clouds(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "dense_cloud_raw.xyz"
            refined = root / "dense_cloud_refined.xyz"
            reference = root / "metashape_reference.xyz"
            raw.write_text(
                "\n".join([
                    "0.00 0.00 0.00",
                    "0.10 0.00 0.03",
                    "0.00 0.10 0.05",
                    "0.10 0.10 1.20",
                    "1.00 0.00 0.00",
                    "1.10 0.00 0.04",
                    "1.00 0.10 0.06",
                    "1.10 0.10 1.10",
                ]),
                encoding="utf-8",
            )
            refined.write_text(
                "\n".join([
                    "0.00 0.00 0.00",
                    "0.10 0.00 0.03",
                    "0.00 0.10 0.05",
                    "0.10 0.10 0.07",
                    "1.00 0.00 0.00",
                    "1.10 0.00 0.04",
                    "1.00 0.10 0.06",
                    "1.10 0.10 0.08",
                ]),
                encoding="utf-8",
            )
            reference.write_text(
                "\n".join([
                    "0.00 0.00 0.00",
                    "0.10 0.00 0.02",
                    "0.00 0.10 0.04",
                    "0.10 0.10 0.06",
                    "1.00 0.00 0.00",
                    "1.10 0.00 0.03",
                    "1.00 0.10 0.05",
                    "1.10 0.10 0.07",
                ]),
                encoding="utf-8",
            )

            report = comparator.compare_point_cloud_improvement(
                raw,
                refined,
                reference,
                roughness_grid_cells=1,
                roughness_min_cell_points=3,
                max_local_z_range_p95_m=0.25,
                min_local_z_range_p95_improvement_percent=80.0,
            )

            self.assertEqual(report["baseline"]["source"], str(raw))
            self.assertEqual(report["candidate"]["source"], str(refined))
            self.assertFalse(report["baseline"]["quality_gate"]["passed"])
            self.assertTrue(report["candidate"]["quality_gate"]["passed"])
            self.assertGreater(report["improvement"]["local_z_range_p95_reduction_percent"], 90.0)
            self.assertTrue(report["quality_gate"]["passed"])

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

    def test_binary_ply_reader_supports_sampling_and_extra_properties(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binary.ply"
            write_binary_ply(path, [
                (0.0, 0.0, 0.0, 255, 0, 0),
                (10.0, 0.0, 0.0, 0, 255, 0),
                (20.0, 0.0, 0.0, 0, 0, 255),
            ])

            points = comparator.read_ply_points(path, max_points=2)

            self.assertEqual(points, [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)])

    def test_compare_binary_ply_clouds_can_limit_large_inputs(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "candidate_binary.ply"
            reference = root / "reference_binary.ply"
            write_binary_ply(source, [
                (0.25, 0.0, 0.0, 255, 0, 0),
                (9.50, 0.0, 0.0, 0, 255, 0),
                (30.0, 0.0, 0.0, 0, 0, 255),
            ])
            write_binary_ply(reference, [
                (0.0, 0.0, 0.0, 255, 255, 255),
                (10.0, 0.0, 0.0, 255, 255, 255),
                (30.0, 0.0, 0.0, 255, 255, 255),
            ])

            comparison = comparator.compare_point_clouds(
                source,
                reference,
                max_source_points=2,
                max_reference_points=2,
            )

            self.assertEqual(comparison["source_points"], 2)
            self.assertEqual(comparison["reference_points"], 2)
            self.assertEqual(comparison["sampling"]["max_source_points"], 2)
            self.assertEqual(comparison["sampling"]["max_reference_points"], 2)
            self.assertAlmostEqual(comparison["distance_m"]["p95"], 0.25)


if __name__ == "__main__":
    unittest.main()
