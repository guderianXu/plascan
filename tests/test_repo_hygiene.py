import json
import os
import subprocess
import sys
import tempfile
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
        self.assertIn("qt6-base-private-dev", text)
        self.assertIn("qt6-shadertools-dev", text)

    def test_github_actions_uses_current_checkout_action(self):
        workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
        text = workflow_path.read_text(encoding="utf-8")

        self.assertIn("uses: actions/checkout@v7", text)
        self.assertNotIn("uses: actions/checkout@v4", text)

    def test_triangulate_cli_test_uses_process_api_and_unique_temp_dirs(self):
        source = (ROOT / "src" / "cli" / "dense" / "tests" / "test_dense_cli.cpp").read_text(
            encoding="utf-8"
        )
        support = (
            ROOT / "src" / "cli" / "common" / "tests" / "CliTestSupport.h"
        ).read_text(encoding="utf-8")

        self.assertNotIn("std::system", source)
        self.assertNotIn("plascan_triangulate_cli_regression", source)
        self.assertNotIn("plascan_triangulate_cli_intensity", source)
        self.assertIn("QTemporaryDir", source)
        self.assertIn("runCli", source)
        self.assertIn("QProcess", support)

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
        self.assertIn("Resolve-MsvcCompilerPathEntries", text)
        self.assertIn("cl.exe", text)
        self.assertIn("$msvcCompilerPathEntries", text)
        self.assertIn("CMAKE_CUDA_HOST_COMPILER", text)
        self.assertIn("$msvcCudaHostCompiler", text)
        self.assertIn("CUDAHOSTCXX", text)
        self.assertIn("--compiler-bindir", text)

    def test_cuda_arches_are_propagated_to_native_backends(self):
        cmake_path = ROOT / "cmake" / "PlascanPackages.cmake"
        text = cmake_path.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^\s*set\s*\(\s*CMAKE_CUDA_ARCHITECTURES\b")
        self.assertIn("PLAMATRIX_CUDA_ARCHITECTURES", text)
        self.assertIn("PLAPOINT_CUDA_ARCHITECTURES", text)
        self.assertIn("FORCE", text)
        self.assertNotIn("find_package(Torch", text)

    def test_cmake_preserves_command_line_cuda_flags_on_windows(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        dependency_paths = (ROOT / "cmake" / "PlascanDependencyPaths.cmake").read_text(encoding="utf-8")

        self.assertNotIn('set(CMAKE_CUDA_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")', root_cmake)
        self.assertIn('${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1', root_cmake)
        self.assertIn('string(REPLACE "-B/usr/bin"', root_cmake)
        self.assertIn("WIN32 OR APPLE OR NOT conda_prefix", dependency_paths)
        self.assertIn("if(WIN32 OR NOT PLASCAN_USE_SYSTEM_BINUTILS_FOR_MIXED_TOOLCHAIN)", dependency_paths)

    def test_plapoint_cuda_warning_sentinels_use_device_safe_values(self):
        knn_source = (ROOT / "3rdparty" / "plapoint" / "src" / "knn_gpu.cu").read_text(encoding="utf-8")
        distance_key_source = (
            ROOT / "3rdparty" / "plapoint" / "include" / "plapoint" / "gpu" /
            "detail" / "distance_key.cuh"
        ).read_text(encoding="utf-8")
        icp_source = (ROOT / "3rdparty" / "plapoint" / "src" / "icp_gpu.cu").read_text(encoding="utf-8")

        self.assertNotIn("HUGE_VAL", knn_source)
        self.assertNotRegex(icp_source, r"\bINFINITY\b")
        self.assertIn("detail::finiteDistanceKey", knn_source)
        self.assertIn("detail::squaredOutputDistance", knn_source)
        self.assertIn("Scalar(FLT_MAX)", knn_source)
        self.assertIn("Scalar(DBL_MAX)", knn_source)
        self.assertIn("return {INT_MAX, DBL_MAX};", distance_key_source)
        self.assertIn("return static_cast<Scalar>(maximum);", distance_key_source)
        self.assertIn("std::numeric_limits<double>::infinity()", icp_source)
        self.assertIn("markSharedTransformMaybeUnused", icp_source)
        self.assertIn("(void)min_z;", icp_source)
        self.assertIn("(void)max_z;", icp_source)

    def test_vcpkg_manifest_has_optional_opencv_dnn_cuda_feature(self):
        manifest = json.loads((ROOT / "vcpkg.json").read_text(encoding="utf-8"))
        base_opencv_dependency = next(
            dependency for dependency in manifest.get("dependencies", [])
            if isinstance(dependency, dict) and dependency.get("name") == "opencv"
        )
        self.assertIn("dnn", set(base_opencv_dependency.get("features", [])))

        optional_features = manifest.get("features", {})
        self.assertIn("opencv-dnn-cuda", optional_features)
        feature_dependencies = optional_features["opencv-dnn-cuda"].get("dependencies", [])
        opencv_dependency = next(
            dependency for dependency in feature_dependencies
            if isinstance(dependency, dict) and dependency.get("name") == "opencv"
        )
        opencv_features = set(opencv_dependency.get("features", []))

        self.assertIn("cuda", opencv_features)
        self.assertIn("dnn", opencv_features)
        self.assertIn("dnn-cuda", opencv_features)

    def test_windows_cuda_build_script_exposes_opencv_dnn_cuda_switch(self):
        script_path = ROOT / "scripts" / "build_win" / "build_windows_cuda.ps1"
        text = script_path.read_text(encoding="utf-8")

        self.assertIn("Sync-VcpkgRuntime", text)
        self.assertIn("$TripletRoot \"bin\"", text)
        self.assertIn("-Filter \"*.dll\"", text)
        self.assertIn("vcpkg runtime DLLs", text)
        self.assertIn("Sync-MsvcRuntime", text)
        self.assertIn("Resolve-MsvcRedistRuntimeDlls", text)
        self.assertIn("vcomp140.dll", text)
        self.assertIn("MSVC runtime DLLs", text)
        self.assertIn("Sync-CudaRuntime", text)
        self.assertIn("$CudaPath \"bin\\x64\"", text)
        self.assertIn("CUDA runtime DLLs", text)
        self.assertIn("VCPKG_APPLOCAL_DEPS=OFF", text)
        self.assertIn("EnableOpenCvDnnCuda", text)
        self.assertIn("EnableCeresCudaBa", text)
        self.assertIn("-UVCPKG_MANIFEST_FEATURES", text)
        self.assertIn("VCPKG_MANIFEST_FEATURES=opencv-dnn-cuda", text)
        self.assertIn("manifestFeaturesValue", text)
        self.assertIn('$manifestFeaturesValue = "$manifestFeaturesValue;ceres-cuda"', text)
        self.assertIn("Assert-OpenCvDnnCudaFeatures", text)
        self.assertIn("Assert-CeresCudaFeatures", text)
        self.assertIn(
            "if ($EnableCeresCudaBa)\n{\n    $vcpkgOverlayPortsCMake = Convert-ToCMakePath "
            "(Ensure-CeresCuda13OverlayPort",
            text,
        )
        self.assertIn("CudnnRoot", text)
        self.assertIn("CUDNN_ROOT_DIR", text)
        self.assertIn("build\\env\\cudnn-cu13", text)
        self.assertIn("VCPKG_OVERLAY_TRIPLETS", text)
        self.assertIn(
            "VCPKG_ENV_PASSTHROUGH CUDNN_ROOT_DIR CUDNN CUDNN_PATH "
            "CUDNN_INCLUDE_DIR CUDNN_LIBRARY cudnn",
            text,
        )
        self.assertIn("Ensure-CeresCuda13OverlayPort", text)
        self.assertIn("CMAKE_CUDA_STANDARD=17", text)
        self.assertIn("CMAKE_CUDA_FLAGS=--std=c++17", text)
        self.assertIn("CMAKE_CUDA_ARCHITECTURES=75\\;86\\;89\\;120", text)
        self.assertIn("Ensure-OpenCvCuda13OverlayPort", text)
        self.assertIn("0024-cuda13-device-props.patch", text)
        self.assertIn("VCPKG_OVERLAY_PORTS", text)
        self.assertIn('if (-not [string]::IsNullOrWhiteSpace($vcpkgOverlayPortsCMake))', text)
        self.assertIn("CUDA_ARCH_BIN=75\\;86\\;89\\;120", text)
        self.assertIn("BUILD_opencv_videostab=OFF", text)
        self.assertIn("OpenCV DNN CUDA", text)

    def test_release_1_1_6_metadata_is_synchronized(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        core_cmake = (ROOT / "src" / "core" / "CMakeLists.txt").read_text(encoding="utf-8")
        manifest = json.loads((ROOT / "vcpkg.json").read_text(encoding="utf-8"))
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        release_doc = ROOT / "docs" / "releases" / "v1.1.6.md"

        self.assertIn("project(PlaScan VERSION 1.1.6", root_cmake)
        self.assertIn("project(PlaScanCore VERSION 1.1.6", core_cmake)
        self.assertEqual(manifest.get("version-string"), "1.1.6")
        self.assertIn("## v1.1.6 - 2026-06-21", changelog)
        self.assertTrue(release_doc.exists(), "v1.1.6 release notes are missing")

        release_text = release_doc.read_text(encoding="utf-8")
        for required in [
            "MVS",
            "CUDA",
            "LiDAR",
            "test_gui_project_utils",
            "GitHub Actions",
            "v1.1.6",
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
                "TensorRT",
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

    def test_linux_install_rules_have_single_plascan_launcher_owner(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        gui_install = (ROOT / "src" / "gui" / "cmake" / "GuiInstall.cmake").read_text(encoding="utf-8")

        self.assertNotIn("resources/plascan.sh", root_cmake)
        self.assertIn("plascan_gui_launcher.sh.in", gui_install)
        self.assertIn('install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan" DESTINATION bin)', gui_install)
        self.assertIn('install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan" DESTINATION /usr/bin)', gui_install)

    def test_configure_with_env_rejects_foreign_platform_paths(self):
        script = ROOT / "scripts" / "env" / "configure_with_env.py"
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "plascan-env.json"
            env_file.write_text(
                json.dumps(
                    {
                        "CUDAToolkit_ROOT": r"E:\CUDA\v13.1",
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--env-file",
                    str(env_file),
                    "--vcpkg-file",
                    str(Path(tmp) / "missing-vcpkg.json"),
                    "--dry-run",
                ],
                cwd=ROOT,
                env={**os.environ, "_PLASCAN_TEST_HOST_PLATFORM": "linux"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("foreign platform path", result.stderr)
        self.assertIn("CUDAToolkit_ROOT", result.stderr)

    def test_camera_preview_memory_detection_supports_linux(self):
        source = (
            ROOT / "src" / "gui" / "dialogs" / "camera" / "CameraModel3DDialog.cpp"
        ).read_text(encoding="utf-8")

        self.assertRegex(source, r"#elif\s+defined\(Q_OS_LINUX\)")
        self.assertIn("/proc/meminfo", source)
        self.assertIn("MemAvailable:", source)
        self.assertRegex(source, r"return\s+availableKb\s*\*\s*1024")


if __name__ == "__main__":
    unittest.main()
