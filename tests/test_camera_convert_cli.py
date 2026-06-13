import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = Path(
    os.environ.get(
        "PLASCAN_CAMERA_CONVERT_CLI",
        REPO_ROOT / "build" / "bin" / "camera_convert_cli",
    )
)


class CameraConvertCliTest(unittest.TestCase):
    def run_cli(self, args):
        if not CLI_PATH.exists():
            self.skipTest(f"camera_convert_cli not found: {CLI_PATH}")
        return subprocess.run(
            [str(CLI_PATH), *map(str, args)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_list_formats(self):
        result = self.run_cli(["--list-formats"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("middlebury-par", result.stdout)
        self.assertIn("epfl-camera", result.stdout)

    def test_middlebury_par_conversion_writes_plascan_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dinoSparseRing"
            source.mkdir()
            (source / "dinoSR0001.png").write_bytes(b"fake")
            (source / "dinoSR0002.png").write_bytes(b"fake")
            (source / "dinoSR_par.txt").write_text(
                "\n".join(
                    [
                        "2",
                        "dinoSR0001.png 120 0 40 0 130 50 0 0 1 1 0 0 0 1 0 0 0 1 1 2 3",
                        "dinoSR0002.png 121 0 41 0 131 51 0 0 1 1 0 0 0 1 0 0 0 1 4 5 6",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_dir = root / "plascan"
            result = self.run_cli(
                [
                    "--format",
                    "middlebury-par",
                    "--input",
                    source,
                    "--output-dir",
                    output_dir,
                    "--overwrite",
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("image_camera.lis", result.stdout)
            self.assertTrue((output_dir / "image_camera.lis").exists())
            self.assertTrue((output_dir / "cameras" / "dinoSR0001.tsai").exists())

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input_format"], "middlebury-par")
            self.assertEqual(summary["camera_count"], 2)

    def test_existing_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "epfl"
            source.mkdir()
            (source / "rdimage.000.ppm").write_bytes(b"fake")
            (source / "rdimage.000.ppm.camera").write_text(
                "\n".join(
                    [
                        "100 0 50",
                        "0 100 50",
                        "0 0 1",
                        "0 0 0",
                        "1 0 0",
                        "0 1 0",
                        "0 0 1",
                        "0 0 0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "out"
            output_dir.mkdir()
            (output_dir / "keep.txt").write_text("keep", encoding="utf-8")

            result = self.run_cli(["--format", "epfl-camera", "--input", source, "--output-dir", output_dir])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("非空", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
