import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "testData" / "dem_grid_to_height_ply.py"
SPEC = importlib.util.spec_from_file_location("dem_grid_to_height_ply", SCRIPT_PATH)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


class DemGridToHeightPlyTest(unittest.TestCase):
    def test_convert_esri_ascii_grid_to_height_plane_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid_path = root / "dem.asc"
            ply_path = root / "height_constraints.ply"
            grid_path.write_text(
                "\n".join([
                    "ncols 3",
                    "nrows 2",
                    "xllcorner 100",
                    "yllcorner 200",
                    "cellsize 2",
                    "NODATA_value -9999",
                    "10 -9999 12",
                    "13 14 15",
                ]),
                encoding="utf-8",
            )

            summary = converter.convert_ascii_grid_to_height_ply(grid_path, ply_path)

            self.assertEqual(summary["valid_points"], 5)
            self.assertEqual(summary["skipped_nodata"], 1)
            self.assertEqual(summary["bounds"], {
                "min_x": 101.0,
                "max_x": 105.0,
                "min_y": 201.0,
                "max_y": 203.0,
                "min_z": 10.0,
                "max_z": 15.0,
            })

            lines = ply_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("element vertex 5", lines)
            self.assertEqual(lines[-5:], [
                "101.000000000 203.000000000 10.000000000",
                "105.000000000 203.000000000 12.000000000",
                "101.000000000 201.000000000 13.000000000",
                "103.000000000 201.000000000 14.000000000",
                "105.000000000 201.000000000 15.000000000",
            ])

    def test_cli_writes_summary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid_path = root / "dem.asc"
            ply_path = root / "height_constraints.ply"
            summary_path = root / "summary.json"
            grid_path.write_text(
                "\n".join([
                    "ncols 2",
                    "nrows 2",
                    "xllcenter 10",
                    "yllcenter 20",
                    "cellsize 1",
                    "NODATA_value -9999",
                    "1 2",
                    "3 -9999",
                ]),
                encoding="utf-8",
            )

            exit_code = converter.main([
                "--input",
                str(grid_path),
                "--output",
                str(ply_path),
                "--summary-json",
                str(summary_path),
            ])

            self.assertEqual(exit_code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["valid_points"], 3)
            self.assertEqual(summary["grid"]["origin_mode"], "center")
            self.assertTrue(ply_path.exists())


if __name__ == "__main__":
    unittest.main()
