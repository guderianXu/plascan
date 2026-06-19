import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_full_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_full_pipeline", SCRIPT_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class FullPipelineEntrypointTest(unittest.TestCase):
    def test_default_command_uses_gui_reconstruction_cli(self):
        args = pipeline.parse_args([
            "input.lis",
            "--build-dir",
            "/tmp/plascan-build",
            "--output-dir",
            "/tmp/out",
            "--device",
            "cpu",
        ])

        with mock.patch.object(Path, "resolve", lambda self: self):
            cmd = pipeline.build_command(args)

        expected_build_dir = Path("/tmp/plascan-build")
        if not expected_build_dir.is_absolute():
            expected_build_dir = pipeline.repo_root() / expected_build_dir
        expected_tool = expected_build_dir / "bin" / "reconstruct_pipeline_cli"
        expected_output_dir = Path("/tmp/out")
        if not expected_output_dir.is_absolute():
            expected_output_dir = Path.cwd() / expected_output_dir

        self.assertEqual(Path(cmd[0]), expected_tool)
        self.assertIn("--output-dir", cmd)
        self.assertIn(str(expected_output_dir), cmd)
        self.assertNotIn("dense_match_cli", " ".join(cmd))
        self.assertNotIn("triangulate_cli", " ".join(cmd))

    def test_force_flag_is_forwarded_to_reconstruction_cli(self):
        args = pipeline.parse_args([
            "input.lis",
            "--build-dir",
            "/tmp/plascan-build",
            "--output-dir",
            "/tmp/out",
            "--force",
        ])

        with mock.patch.object(Path, "resolve", lambda self: self):
            cmd = pipeline.build_command(args)

        self.assertIn("--force", cmd)

    def test_legacy_flag_dispatches_old_validation_script(self):
        args = [
            "input.lis",
            "--legacy-stereo-test",
        ]
        fake_module = types.SimpleNamespace(main=lambda: 42)
        with mock.patch.dict(sys.modules, {"run_full_pipeline_test": fake_module}):
            self.assertEqual(pipeline.main(args), 42)
