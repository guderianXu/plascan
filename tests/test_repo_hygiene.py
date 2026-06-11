import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepoHygieneTest(unittest.TestCase):
    def test_full_pipeline_entrypoint_is_registered_in_ctest(self):
        cmake = (ROOT / "tests" / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("FullPipelineEntrypointTest", cmake)
        self.assertIn("tests.test_full_pipeline_entrypoint", cmake)

    def test_license_file_matches_readme_license(self):
        license_path = ROOT / "LICENSE"

        self.assertTrue(license_path.exists(), "README declares MIT, but LICENSE is missing")
        text = license_path.read_text(encoding="utf-8")
        self.assertIn("MIT License", text)
        self.assertIn("Permission is hereby granted", text)

    def test_github_actions_ci_covers_configure_build_and_ctest(self):
        workflow_path = ROOT / ".github" / "workflows" / "ci.yml"

        self.assertTrue(workflow_path.exists(), "GitHub Actions CI workflow is missing")
        text = workflow_path.read_text(encoding="utf-8")
        self.assertIn("submodules: recursive", text)
        self.assertIn("cmake -S . -B build", text)
        self.assertIn("cmake --build build", text)
        self.assertIn("ctest --test-dir build", text)

    def test_triangulate_cli_test_uses_process_api_and_unique_temp_dirs(self):
        source = (ROOT / "tests" / "test_triangulate_cli.cpp").read_text(encoding="utf-8")

        self.assertNotIn("std::system", source)
        self.assertNotIn("plascan_triangulate_cli_regression", source)
        self.assertNotIn("plascan_triangulate_cli_intensity", source)
        self.assertIn("QProcess", source)
        self.assertIn("makeUniqueTempDir", source)


if __name__ == "__main__":
    unittest.main()
