import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = Path(
    os.environ.get(
        "PLASCAN_BUNDLE_ADJUST_CLI",
        REPO_ROOT / "build" / "bin" / "bundle_adjust_cli",
    )
)


class BundleAdjustCliTest(unittest.TestCase):
    def test_cmake_registers_bundle_adjust_cli_target(self):
        cmake = (REPO_ROOT / "src" / "cli" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("bundle_adjust_cli", cmake)
        self.assertIn("cli_bundle_adjust.cpp", cmake)

    def test_cli_source_exposes_lidar_ab_compare_options(self):
        source_path = REPO_ROOT / "src" / "cli" / "cli_bundle_adjust.cpp"
        self.assertTrue(source_path.exists(), f"missing {source_path}")
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("--laser-cloud", source)
        self.assertIn("--laser-missing-normals-as-height-planes", source)
        self.assertIn("options.laserUseMissingNormalsAsHeightPlanes", source)
        self.assertIn("--ab-compare", source)
        self.assertIn("ba_ab_compare.json", source)
        self.assertIn("quality_gate", source)
        self.assertIn("reprojection_rms_regressed", source)
        self.assertIn("--fail-on-quality-gate", source)
        self.assertIn("failOnQualityGate", source)
        self.assertIn("cli::EXIT_ALGO_ERR", source)

    def test_cli_disables_eval_plot_by_default_for_headless_runs(self):
        source = (REPO_ROOT / "src" / "cli" / "cli_bundle_adjust.cpp").read_text(encoding="utf-8")
        self.assertIn("bool exportEvalPlot = false;", source)
        self.assertIn("--export-eval-plot", source)
        self.assertIn("options.exportEvalPlot = exportEvalPlot;", source)

    def test_missing_project_fails_before_creating_output(self):
        if not CLI_PATH.exists():
            self.skipTest(f"bundle_adjust_cli not found: {CLI_PATH}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            missing_project = root / "missing.plascan"

            result = subprocess.run(
                [
                    str(CLI_PATH),
                    str(missing_project),
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out_dir.exists())
            self.assertIn("项目文件不存在", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
