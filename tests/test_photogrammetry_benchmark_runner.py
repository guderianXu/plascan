import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "bench" / "run_photogrammetry_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_photogrammetry_benchmarks", SCRIPT_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class PhotogrammetryBenchmarkRunnerTest(unittest.TestCase):
    def test_explicit_cli_without_suffix_resolves_windows_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / "three_d_reconstruction_cli"
            executable = requested.with_suffix(".exe")
            executable.write_bytes(b"")

            self.assertEqual(
                runner.resolve_explicit_executable(requested),
                executable,
            )

    def test_default_cli_resolves_multiconfig_release_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / "bin" / "three_d_reconstruction_cli"
            executable = (
                requested.parent
                / "Release"
                / "three_d_reconstruction_cli.exe"
            )
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"")

            self.assertEqual(
                runner.resolve_explicit_executable(requested),
                executable,
            )

    def test_dry_run_discovers_prepared_dataset_and_writes_planned_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            prepared = root / "middlebury_dino_sparse_ring" / "prepared" / "plascan"
            prepared.mkdir(parents=True)
            list_path = prepared / "image_camera.lis"
            list_path.write_text("a.png cameras/a.tsai\nb.png cameras/b.tsai\n", encoding="utf-8")

            output_root = Path(tmp) / "runs"
            report_path = output_root / "summary.json"

            exit_code = runner.main([
                "--root", str(root),
                "--output-dir", str(output_root),
                "--cli", "/tmp/three_d_reconstruction_cli",
                "--stage", "sfm",
                "--dry-run",
                "--summary", str(report_path),
            ])

            self.assertEqual(exit_code, 0)
            summary = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "planned")
            self.assertEqual(summary["stage"], "sfm")
            self.assertEqual(len(summary["datasets"]), 1)

            dataset = summary["datasets"][0]
            self.assertEqual(dataset["dataset_id"], "middlebury_dino_sparse_ring")
            self.assertEqual(dataset["image_count"], 2)
            self.assertEqual(Path(dataset["list_file"]), list_path.resolve())
            self.assertIn("--stop-after-sfm", dataset["command"])
            self.assertIn("--force", dataset["command"])

    def test_stage_mvs_command_skips_mesh_but_keeps_mvs(self):
        list_path = Path("/tmp/example/image_camera.lis")
        command = runner.build_reconstruction_command(
            cli_path=Path("/tmp/three_d_reconstruction_cli"),
            list_file=list_path,
            output_dir=Path("/tmp/out"),
            stage="mvs",
            device="cpu",
            quality=2,
            threads=4,
            timeout=10,
        )

        self.assertIn("--skip-mesh", command)
        self.assertNotIn("--stop-after-sfm", command)
        self.assertNotIn("--skip-mvs", command)
        self.assertIn("--quality", command)
        self.assertIn("2", command)


if __name__ == "__main__":
    unittest.main()
