import unittest
import json
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DenseCloudRefineCliTest(unittest.TestCase):
    def test_cli_target_exists_and_exposes_quality_filter_options(self):
        cmake = (ROOT / "src/cli/CMakeLists.txt").read_text(encoding="utf-8")
        source_path = ROOT / "src/cli/cli_dense_cloud_refine.cpp"

        self.assertIn("dense_cloud_refine_cli", cmake)
        self.assertIn("cli_dense_cloud_refine.cpp", cmake)
        self.assertTrue(source_path.exists())

        source = source_path.read_text(encoding="utf-8")
        self.assertIn("--input", source)
        self.assertIn("--output", source)
        self.assertIn("--report-json", source)
        self.assertIn("--terrain-grid-cells", source)
        self.assertIn("--terrain-min-cell-points", source)
        self.assertIn("--terrain-min-height-threshold", source)
        self.assertIn("--terrain-mad-multiplier", source)
        self.assertIn("--terrain-local-plane-filter", source)
        self.assertIn("--terrain-local-plane-min-points", source)
        self.assertIn("--terrain-local-plane-min-residual-threshold", source)
        self.assertIn("--terrain-local-plane-mad-multiplier", source)
        self.assertIn("--terrain-filter-passes", source)
        self.assertIn("--streaming-chunk-mb", source)
        self.assertIn("int terrain_grid_cells = 260;", source)
        self.assertIn("int terrain_min_cell_points = 32;", source)
        self.assertIn("float terrain_min_height_threshold = 0.25f;", source)
        self.assertIn("float terrain_mad_multiplier = 3.0f;", source)
        self.assertIn("bool terrain_local_plane_filter = true;", source)
        self.assertIn("int terrain_local_plane_min_points = 12;", source)
        self.assertIn("float terrain_local_plane_min_residual_threshold = 0.12f;", source)
        self.assertIn("float terrain_local_plane_mad_multiplier = 4.0f;", source)
        self.assertIn("int terrain_filter_passes = 2;", source)
        self.assertIn("parseBinaryPlyVertexStreamHeader", source)
        self.assertIn("readPlyVertexChunk", source)
        self.assertIn("filterTerrainHeightSpikes", source)
        self.assertIn("terrain_spike_filter", source)
        self.assertIn("terrain_filter_passes", source)
        self.assertIn("pass_reports", source)
        self.assertIn("local_plane_removed_points", source)
        self.assertNotIn("streamingOptions.localPlaneFilterEnabled = false", source)

    def test_streaming_cli_applies_local_plane_filter(self):
        exe = ROOT / "build/windows-vcpkg-cuda-release/bin/dense_cloud_refine_cli.exe"
        if not exe.exists():
            self.skipTest("dense_cloud_refine_cli.exe has not been built")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_ply = tmp_path / "sloped_spike_binary.ply"
            output_ply = tmp_path / "refined.ply"
            report_json = tmp_path / "report.json"

            points = []
            for y in range(8):
                for x in range(8):
                    z = 0.15 * x + 0.08 * y
                    points.append((float(x), float(y), float(z)))
            points.append((3.25, 4.25, 0.15 * 3.25 + 0.08 * 4.25 + 0.42))

            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            ).encode("ascii")
            with input_ply.open("wb") as f:
                f.write(header)
                for point in points:
                    f.write(struct.pack("<fff", *point))

            result = subprocess.run(
                [
                    str(exe),
                    "--input",
                    str(input_ply),
                    "--output",
                    str(output_ply),
                    "--report-json",
                    str(report_json),
                    "--terrain-grid-cells",
                    "1",
                    "--terrain-min-cell-points",
                    "12",
                    "--terrain-min-height-threshold",
                    "2.0",
                    "--terrain-mad-multiplier",
                    "20.0",
                    "--terrain-local-plane-filter",
                    "--terrain-local-plane-min-points",
                    "12",
                    "--terrain-local-plane-min-residual-threshold",
                    "0.10",
                    "--terrain-local-plane-mad-multiplier",
                    "4.0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(report_json.read_text(encoding="utf-8"))

            self.assertEqual(report["mode"], "streaming")
            self.assertEqual(report["terrain_filter_passes"], 2)
            self.assertEqual(len(report["pass_reports"]), 2)
            self.assertEqual(report["input_points"], 65)
            self.assertEqual(report["output_points"], 64)
            self.assertEqual(report["terrain_spike_filter"]["local_plane_removed_points"], 1)
            self.assertTrue(output_ply.exists())


if __name__ == "__main__":
    unittest.main()
