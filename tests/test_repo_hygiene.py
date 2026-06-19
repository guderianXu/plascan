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

    def test_windows_runtime_scripts_force_utf8_console_for_native_logs(self):
        script_paths = [
            ROOT / "scripts" / "env" / "env_common.py",
            ROOT / "scripts" / "build_win" / "build_windows_cuda.ps1",
        ]

        for script_path in script_paths:
            with self.subTest(script=str(script_path.relative_to(ROOT))):
                text = script_path.read_text(encoding="utf-8")
                self.assertIn("[Console]::InputEncoding", text)
                self.assertIn("[Console]::OutputEncoding", text)
                self.assertIn("chcp.com 65001", text)
                self.assertIn("PYTHONUTF8", text)
                self.assertIn("PYTHONIOENCODING", text)

    def test_windows_cuda_build_keeps_vs_compiler_path_for_nvcc_device_link(self):
        script_path = ROOT / "scripts" / "build_win" / "build_windows_cuda.ps1"
        text = script_path.read_text(encoding="utf-8")

        self.assertIn("Capture-VsDevPathEntries", text)
        self.assertIn("$vsDevPathEntries", text)
        self.assertIn("$prepend + $vsDevPathEntries + $filtered", text)
        self.assertIn("$vsDevPathValue", text)
        self.assertIn("-ieq \"Path\"", text)

    def test_laser_photogrammetry_dataset_notes_capture_future_ba_inputs(self):
        doc_path = ROOT / "docs" / "design" / "LASER_PHOTOGRAMMETRY_DATASETS.md"

        self.assertTrue(doc_path.exists(), "Laser/photogrammetry dataset research notes are missing")
        text = doc_path.read_text(encoding="utf-8")

        required_terms = [
            "MUN-FRL",
            "Hessigheim 3D",
            "NTU VIRAL",
            "LiDAR",
            "UAV",
            "license",
            "Bundle Adjustment",
            "control points",
            "point cloud fusion",
            "First Small Fixture Plan",
            "source_manifest.json",
            "LaserControlPoint",
            "LaserObservation",
            "LiDARFrame",
            "Residual Diagnostics",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, text)

        self.assertGreaterEqual(text.count("|"), 40, "Dataset notes should include a structured comparison table")

    def test_release_1_1_5_metadata_is_synchronized(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        core_cmake = (ROOT / "src" / "core" / "CMakeLists.txt").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        release_doc = ROOT / "docs" / "releases" / "v1.1.5.md"

        self.assertIn("project(PlaScan VERSION 1.1.5", root_cmake)
        self.assertIn("project(PlaScanCore VERSION 1.1.5", core_cmake)
        self.assertIn("## v1.1.5 - 2026-06-19", changelog)
        self.assertTrue(release_doc.exists(), "v1.1.5 release notes are missing")

        release_text = release_doc.read_text(encoding="utf-8")
        for required in [
            "MVS",
            "CUDA",
            "LiDAR",
            "test_gui_project_utils",
            "GitHub Actions",
            "v1.1.5",
        ]:
            with self.subTest(required=required):
                self.assertIn(required, release_text)

    def test_reconstruction_stage_docs_cover_new_pipeline_modules(self):
        docs_to_terms = {
            ROOT / "README.md": [
                "MvsWorkspaceManifest",
                "MvsSourcePlanner",
                "TerrainProductManifest",
                "ReferenceTerrainPrior",
                "Windows CUDA",
                "libtorch-cu130",
            ],
            ROOT / "src" / "core" / "mvs" / "README.md": [
                "MvsWorkspaceManifest",
                "MvsSourcePlanner",
                "source plan",
                "valid mask",
                "confidence",
                "streaming fusion",
                "cancel",
            ],
            ROOT / "src" / "core" / "terrain" / "README.md": [
                "DemGridAggregator",
                "TerrainProductManifest",
                "DemMosaic",
                "dem_error.tif",
                "dem_count.tif",
                "dem_confidence.tif",
                "dem_coverage.tif",
            ],
            ROOT / "docs" / "PROJECT_ARCHITECTURE.md": [
                "MvsWorkspaceManifest",
                "MvsSourcePlanner",
                "TerrainProductManifest",
                "ReconstructionQualityReport",
                "ReferenceTerrainPrior",
            ],
        }

        for path, required_terms in docs_to_terms.items():
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertTrue(path.exists(), f"{path.relative_to(ROOT)} is missing")
                text = path.read_text(encoding="utf-8")
                for term in required_terms:
                    self.assertIn(term, text)


if __name__ == "__main__":
    unittest.main()
