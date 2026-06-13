import json
import os
import subprocess
import tempfile
import unittest
import zipfile
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
        self.assertIn("colmap-text", result.stdout)
        self.assertIn("metashape-xml", result.stdout)

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

    def test_colmap_text_conversion_writes_plascan_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "south-building"
            source = dataset / "sparse"
            images = dataset / "images"
            source.mkdir(parents=True)
            images.mkdir(parents=True)
            (images / "P1180141.JPG").write_bytes(b"fake")
            (source / "cameras.txt").write_text(
                "1 SIMPLE_RADIAL 3072 2304 2559.68 1536 1152 -0.0204997\n",
                encoding="utf-8",
            )
            (source / "images.txt").write_text(
                "1 1 0 0 0 10 20 30 1 P1180141.JPG\n"
                "0 0 -1\n",
                encoding="utf-8",
            )
            (source / "points3D.txt").write_text("# unused\n", encoding="utf-8")

            output_dir = root / "plascan"
            result = self.run_cli(
                [
                    "--format",
                    "colmap-text",
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
            self.assertTrue((output_dir / "cameras" / "P1180141.tsai").exists())

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input_format"], "colmap-text")
            self.assertEqual(summary["camera_count"], 1)

    def test_metashape_chunk_zip_conversion_writes_plascan_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "depth_images"
            images = dataset / "Depthimages"
            project = dataset / "Metashape" / "Project_depthimages.files" / "0"
            images.mkdir(parents=True)
            project.mkdir(parents=True)
            (images / "IMG_0262.JPG").write_bytes(b"fake")
            doc_xml = """<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" label="RGB" type="frame">
        <resolution width="1000" height="800"/>
        <calibration type="frame" class="adjusted">
          <f>500</f><cx>10</cx><cy>-20</cy>
        </calibration>
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" sensor_id="0" component_id="0" label="IMG_0262_0">
        <transform>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</transform>
      </camera>
    </cameras>
  </chunk>
</document>
"""
            with zipfile.ZipFile(project / "chunk.zip", "w") as zf:
                zf.writestr("doc.xml", doc_xml)

            output_dir = root / "plascan"
            result = self.run_cli(
                [
                    "--format",
                    "auto",
                    "--input",
                    dataset,
                    "--output-dir",
                    output_dir,
                    "--overwrite",
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("image_camera.lis", result.stdout)
            self.assertTrue((output_dir / "image_camera.lis").exists())
            self.assertTrue((output_dir / "cameras" / "IMG_0262.tsai").exists())

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input_format"], "metashape-xml")
            self.assertEqual(summary["camera_count"], 1)

    def test_metashape_repeated_warnings_are_summarized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "depth_images"
            images = dataset / "Depthimages"
            project = dataset / "Metashape" / "Project_depthimages.files" / "0"
            images.mkdir(parents=True)
            project.mkdir(parents=True)
            (images / "IMG_0262.JPG").write_bytes(b"fake")
            (images / "IMG_0263.JPG").write_bytes(b"fake")
            doc_xml = """<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" label="RGB" type="frame">
        <resolution width="1000" height="800"/>
        <calibration type="frame" class="adjusted">
          <f>500</f><cx>10</cx><cy>-20</cy><k1>0.01</k1>
        </calibration>
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" sensor_id="0" component_id="0" label="IMG_0262_0">
        <transform>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</transform>
      </camera>
      <camera id="1" sensor_id="0" component_id="0" label="IMG_0263_0">
        <transform>1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1</transform>
      </camera>
    </cameras>
  </chunk>
</document>
"""
            with zipfile.ZipFile(project / "chunk.zip", "w") as zf:
                zf.writestr("doc.xml", doc_xml)

            output_dir = root / "plascan"
            result = self.run_cli(
                [
                    "--format",
                    "auto",
                    "--input",
                    dataset,
                    "--output-dir",
                    output_dir,
                    "--overwrite",
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr.count("Metashape distortion terms"), 1)
            self.assertIn("2 个相机", result.stderr)
            self.assertIn("IMG_0262.JPG", result.stderr)


if __name__ == "__main__":
    unittest.main()
