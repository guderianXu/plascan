import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "workflows" / "run_full_pipeline.py"
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
        executable_name = (
            "reconstruct_pipeline_cli.exe"
            if os.name == "nt"
            else "reconstruct_pipeline_cli"
        )
        expected_tool = expected_build_dir / "bin" / executable_name
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

    def test_windows_multiconfig_executable_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            executable = (
                build_dir
                / "bin"
                / "Release"
                / "reconstruct_pipeline_cli.exe"
            )
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"")
            args = pipeline.parse_args([
                "input.lis",
                "--build-dir",
                str(build_dir),
                "--output-dir",
                str(build_dir / "out"),
            ])

            cmd = pipeline.build_command(args)

            self.assertEqual(Path(cmd[0]), executable)

    def test_removed_legacy_flag_is_rejected(self):
        with self.assertRaises(SystemExit):
            pipeline.parse_args(["input.lis", "--legacy-stereo-test"])
