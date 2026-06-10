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

        self.assertEqual(cmd[0], "/tmp/plascan-build/bin/reconstruct_pipeline_cli")
        self.assertIn("--output-dir", cmd)
        self.assertIn("/tmp/out", cmd)
        self.assertNotIn("dense_match_cli", " ".join(cmd))
        self.assertNotIn("triangulate_cli", " ".join(cmd))

    def test_legacy_flag_dispatches_old_validation_script(self):
        args = [
            "input.lis",
            "--legacy-stereo-test",
        ]
        fake_module = types.SimpleNamespace(main=lambda: 42)
        with mock.patch.dict(sys.modules, {"run_full_pipeline_test": fake_module}):
            self.assertEqual(pipeline.main(args), 42)
