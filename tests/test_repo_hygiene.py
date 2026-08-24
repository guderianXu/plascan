import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def workflow_job_block(workflow_text, job_name):
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow_text,
    )
    if match is None:
        raise AssertionError(f"GitHub Actions job is missing: {job_name}")
    return match.group(0)


class RepoHygieneTest(unittest.TestCase):
    def test_full_pipeline_entrypoint_is_registered_in_ctest(self):
        cmake = (ROOT / "tests" / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn(
            "plascan_add_python_unittest(FullPipelineEntrypointTest "
            "test_full_pipeline_entrypoint)",
            cmake,
        )

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
        self.assertIn("submodules: false", text)
        self.assertNotIn("submodules: recursive", text)
        self.assertIn(
            "git submodule update --init --depth 1 3rdparty/plamatrix "
            "3rdparty/plapoint 3rdparty/opencv 3rdparty/PoissonRecon",
            text,
        )
        self.assertIn("cmake -S . -B build", text)
        self.assertIn("cmake --build build", text)
        self.assertIn("python3 scripts/env/run_tests.py --test-dir build", text)
        self.assertIn("cmake -S . -B build-headless", text)
        self.assertIn("-DPLASCAN_BUILD_GUI=OFF", text)
        self.assertIn("-DPLASCAN_BUILD_GUI_TESTS=OFF", text)
        self.assertIn("CONDA_PREFIX: ${{ runner.temp }}/plascan-poison-conda", text)
        self.assertIn('grep -F "${CONDA_PREFIX}" headless-configure.log', text)
        self.assertIn('Path("build-headless/CMakeCache.txt")', text)
        self.assertIn('"PLASCAN_EFFECTIVE_CONDA_PREFIX",', text)
        self.assertIn("Removed CMake option remains:", text)
        for forbidden_qt_cache_entry in [
            "Qt6Test_DIR",
            "Qt6Widgets_DIR",
            "Qt6ShaderTools_DIR",
            "Qt6GuiPrivate_DIR",
        ]:
            self.assertIn(forbidden_qt_cache_entry, text)
        self.assertNotIn("runs-on: windows-2025", text)
        self.assertNotIn("jurplel/install-qt-action", text)
        self.assertIn("Configure pinned vcpkg", text)
        self.assertIn("3c5d90a305ff00ca841f085a74a7ce74ee777dee", text)
        self.assertIn("VCPKG_TARGET_TRIPLET=x64-linux-dynamic", text)
        self.assertIn('VCPKG_OVERLAY_PORTS="${GITHUB_WORKSPACE}/cmake/vcpkg-overlays"', text)
        self.assertNotIn("qt6-base-dev", text)
        self.assertNotIn("libapriltag-dev", text)
        self.assertNotIn("libopenmesh-dev", text)
        self.assertIn("gperf", text)
        self.assertIn("'^libxcb.*-dev'", text)
        self.assertIn("python3 -m pip install --disable-pip-version-check numpy pillow scipy", text)
        self.assertIn("--timeout 120", text)
        self.assertIn("uses: actions/cache@v4", text)
        self.assertIn("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", text)
        self.assertIn("cancel-in-progress: true", text)
        self.assertIn("paths-ignore:", text)
        self.assertNotIn("libtorch", text.lower())
        self.assertNotIn("Torch_DIR", text)

        packages_text = (ROOT / "cmake" / "PlascanPackages.cmake").read_text(encoding="utf-8")
        self.assertIn("find_package(Qt6 6.7 REQUIRED", packages_text)
        self.assertIn("pkg_check_modules(APRILTAG REQUIRED IMPORTED_TARGET GLOBAL apriltag)", packages_text)

        windows_ci_triplet = (
            ROOT / "cmake" / "vcpkg-triplets" / "x64-windows-ci-release.cmake"
        )
        self.assertTrue(windows_ci_triplet.exists())
        triplet_text = windows_ci_triplet.read_text(encoding="utf-8")
        self.assertIn("set(VCPKG_TARGET_ARCHITECTURE x64)", triplet_text)
        self.assertIn("set(VCPKG_CRT_LINKAGE dynamic)", triplet_text)
        self.assertIn("set(VCPKG_LIBRARY_LINKAGE dynamic)", triplet_text)
        self.assertIn("set(VCPKG_BUILD_TYPE release)", triplet_text)

        manifest = json.loads((ROOT / "vcpkg.json").read_text(encoding="utf-8"))
        linux_vulkan_loader = next(
            dependency
            for dependency in manifest["dependencies"]
            if isinstance(dependency, dict) and dependency.get("name") == "vulkan-loader"
        )
        self.assertEqual("linux", linux_vulkan_loader["platform"])
        self.assertEqual(["xcb"], linux_vulkan_loader["features"])
        production_dependency_names = {
            dependency if isinstance(dependency, str) else dependency.get("name")
            for dependency in manifest["dependencies"]
        }
        self.assertNotIn("lapack", production_dependency_names)
        self.assertNotIn("openblas", production_dependency_names)
        self.assertNotIn("openmesh", production_dependency_names)
        self.assertNotIn("PLAMATRIX_WITH_SYSTEM_LINALG", packages_text)

        meshing_cmake_text = (
            ROOT / "src" / "core" / "mesh" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        self.assertNotIn("find_package(OpenMesh", meshing_cmake_text)
        self.assertNotIn("OpenMeshSimplifier", meshing_cmake_text)
        plamatrix_cmake_text = (
            ROOT / "3rdparty" / "plamatrix" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        self.assertNotIn("PLAMATRIX_WITH_SYSTEM_LINALG", plamatrix_cmake_text)
        self.assertNotIn("find_package(BLAS", plamatrix_cmake_text)
        self.assertNotIn("find_package(LAPACK", plamatrix_cmake_text)
        self.assertFalse(
            (ROOT / "3rdparty" / "plamatrix" / "src" / "ops" / "fortran_linalg.h").exists()
        )
        root_cmake_text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("libblas3", root_cmake_text)
        self.assertNotIn("liblapack3", root_cmake_text)
        self.assertNotIn("libgfortran5", root_cmake_text)
        self.assertEqual([], manifest["default-features"])
        self.assertNotIn("ceres", manifest["dependencies"])
        self.assertNotIn("features", manifest)

        presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        linux_cuda_opencl = next(
            preset
            for preset in presets["configurePresets"]
            if preset["name"] == "linux-vcpkg-cuda-opencl-release"
        )
        self.assertEqual(
            "/usr/local/cuda-13.1/bin/nvcc",
            linux_cuda_opencl["environment"]["CUDACXX"],
        )
        self.assertEqual("/usr/bin/g++-13", linux_cuda_opencl["environment"]["CUDAHOSTCXX"])
        cuda_cache = linux_cuda_opencl["cacheVariables"]
        self.assertEqual("/usr/local/cuda-13.1", cuda_cache["CUDAToolkit_ROOT"])
        self.assertEqual("89", cuda_cache["PLASCAN_CUDA_ARCHITECTURES"])
        self.assertEqual("ON", cuda_cache["PLASCAN_ENABLE_CUDA"])
        self.assertEqual("ON", cuda_cache["PLASCAN_ENABLE_OPENCL"])
        self.assertEqual("OFF", cuda_cache["PLASCAN_ENABLE_TENSORRT"])
        self.assertNotIn("VCPKG_MANIFEST_FEATURES", cuda_cache)
        self.assertNotIn("PLASCAN_ENABLE_CERES_REFERENCE", cuda_cache)
        self.assertEqual(
            "${sourceDir}/build/linux-vcpkg-cuda-opencl-release/vcpkg_installed",
            cuda_cache["VCPKG_INSTALLED_DIR"],
        )

        cuda_overlay = ROOT / "cmake" / "vcpkg-overlays" / "cuda"
        self.assertTrue(cuda_overlay.is_dir())
        cuda_find = (cuda_overlay / "vcpkg_find_cuda.cmake").read_text(encoding="utf-8")
        self.assertIn("ENV{CUDACXX}", cuda_find)
        self.assertIn("Using CUDA compiler from CUDACXX", cuda_find)
        self.assertIn("CUDACXX points to a missing CUDA compiler", cuda_find)
        self.assertIn("OUT_CUDA_COMPILER", cuda_find)
        cuda_manifest = json.loads((cuda_overlay / "vcpkg.json").read_text(encoding="utf-8"))
        self.assertEqual(14, cuda_manifest["port-version"])

        self.assertFalse((ROOT / "cmake" / "vcpkg-overlays" / "ceres").exists())

    def test_pinned_qt_opencv_source_build_is_self_contained(self):
        modules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
        for path in (
            "3rdparty/qt",
            "3rdparty/opencv",
            "3rdparty/gdal",
            "3rdparty/apriltag",
            "3rdparty/PoissonRecon",
        ):
            self.assertIn(f"path = {path}", modules)

        versions = (ROOT / "cmake" / "PlascanSourceDependencyVersions.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn('PLASCAN_SOURCE_QT_VERSION "6.11.2"', versions)
        self.assertIn('PLASCAN_SOURCE_OPENCV_VERSION "5.0.0"', versions)
        self.assertIn('PLASCAN_SOURCE_APRILTAG_VERSION "3.4.5"', versions)
        self.assertIn('PLASCAN_SOURCE_GDAL_VERSION "3.12.4"', versions)
        self.assertIn("713a36536903d172f9e6737584d428753c119496", versions)
        self.assertIn("40738fb16ceddb5fb3fea747585f7ce6abb0605b", versions)
        self.assertIn("94be783968e5091bcc9972c72c84fd63efce2935", versions)
        self.assertIn("0e3e27c90f57130232d215d783ff49cc332cd950", versions)
        self.assertIn("262b0f539d404057d1f36e1adc07fc9388678899", versions)

        source_manifest = json.loads(
            (ROOT / "cmake" / "source-deps" / "vcpkg.json").read_text(encoding="utf-8")
        )
        source_dependency_names = {
            dependency if isinstance(dependency, str) else dependency["name"]
            for dependency in source_manifest["dependencies"]
        }
        self.assertTrue(
            {
                "json-c",
                "libgeotiff",
                "proj",
                "sqlite3",
                "tiff",
                "zlib",
                "libzip",
            }.issubset(
                source_dependency_names
            )
        )
        sqlite_dependency = next(
            dependency
            for dependency in source_manifest["dependencies"]
            if isinstance(dependency, dict) and dependency["name"] == "sqlite3"
        )
        self.assertIn("rtree", sqlite_dependency.get("features", []))
        self.assertTrue(
            {"qtbase", "opencv", "gdal", "apriltag"}.isdisjoint(source_dependency_names)
        )

        presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        configure_preset_names = {
            preset["name"] for preset in presets["configurePresets"]
        }
        for platform in ("linux", "macos", "windows"):
            self.assertIn(f"{platform}-source-deps-release", configure_preset_names)
            self.assertIn(f"{platform}-source-release", configure_preset_names)

        superbuild = (ROOT / "cmake" / "PlascanSourceDependencies.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn("ExternalProject_Add(plascan_qt_source", superbuild)
        self.assertIn("ExternalProject_Add(plascan_opencv_source", superbuild)
        self.assertIn("ExternalProject_Add(plascan_apriltag_source", superbuild)
        self.assertIn("ExternalProject_Add(plascan_gdal_source", superbuild)
        self.assertIn('"-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON"', superbuild)
        self.assertIn('"-UGDAL_ENABLE_DRIVER_*"', superbuild)
        self.assertIn('"-UOGR_ENABLE_DRIVER_*"', superbuild)
        self.assertIn('"-DGDAL_BUILD_OPTIONAL_DRIVERS:BOOL=OFF"', superbuild)
        self.assertIn('"-DOGR_BUILD_OPTIONAL_DRIVERS:BOOL=OFF"', superbuild)
        self.assertIn('"-DGDAL_USE_HDF5:BOOL=OFF"', superbuild)
        self.assertIn('"-DGDAL_USE_NETCDF:BOOL=OFF"', superbuild)
        self.assertIn('"-DGDAL_ENABLE_DRIVER_HDF5:BOOL=OFF"', superbuild)
        self.assertIn('"-DGDAL_ENABLE_DRIVER_NETCDF:BOOL=OFF"', superbuild)
        self.assertIn('"-DGDAL_ENABLE_DRIVER_JP2OPENJPEG:BOOL=ON"', superbuild)
        self.assertIn('"-DGDAL_ENABLE_DRIVER_PDS:BOOL=ON"', superbuild)
        self.assertIn('"-DOGR_ENABLE_DRIVER_GPKG:BOOL=OFF"', superbuild)
        self.assertIn("-submodules qtbase,qtshadertools", superbuild)
        self.assertIn("-qt-libpng", superbuild)
        self.assertIn('"-UFEATURE_system_png"', superbuild)
        self.assertIn('"-UQT_FEATURE_system_png"', superbuild)
        self.assertIn('"-UOPENCV_EXTRA_MODULES_PATH"', superbuild)
        self.assertIn('"-UBUILD_opencv_x*"', superbuild)
        self.assertIn('"-DWITH_AVIF:BOOL=OFF"', superbuild)

        mesh_cmake = (ROOT / "src" / "core" / "mesh" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${CMAKE_SOURCE_DIR}/3rdparty/PoissonRecon"', mesh_cmake)
        self.assertIn("CUDA_RESOLVE_DEVICE_SYMBOLS ON", mesh_cmake)
        self.assertNotIn("FetchContent_Declare(plascan_poisson_recon", mesh_cmake)

        onnxruntime = (ROOT / "cmake" / "PlascanOnnxRuntime.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn('PLASCAN_ONNXRUNTIME_VERSION "1.29.0"', onnxruntime)
        self.assertIn('PLASCAN_ONNXRUNTIME_API_VERSION "29"', onnxruntime)
        self.assertIn("PLASCAN_ONNXRUNTIME_CACHE_DIR", onnxruntime)
        self.assertIn("PLASCAN_ONNXRUNTIME_ARCHIVE", onnxruntime)

        test_runtime = (ROOT / "cmake" / "PlascanTestRuntime.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn("PLASCAN_ONNXRUNTIME_RUNTIME_LIBRARY", test_runtime)
        self.assertIn("copy_if_different", test_runtime)

        gui_install = (ROOT / "src" / "gui" / "cmake" / "GuiInstall.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn('DESTINATION share/gdal', gui_install)
        self.assertIn('DESTINATION share/proj', gui_install)

        gui_main = (ROOT / "src" / "gui" / "main.cpp").read_text(encoding="utf-8")
        self.assertIn("configureGeospatialDataPaths", gui_main)
        geospatial_runtime = (
            ROOT / "src" / "gui" / "runtime" / "GeospatialRuntimePaths.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn('qputenv("GDAL_DATA"', geospatial_runtime)
        self.assertIn('qputenv("PROJ_DATA"', geospatial_runtime)
        self.assertIn('QStringLiteral("vcpkg_installed")', geospatial_runtime)
        self.assertIn('QStringLiteral("proj.db")', geospatial_runtime)

        configure_script = (ROOT / "scripts" / "env" / "configure_with_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--source-deps"', configure_script)
        self.assertIn('"--build-dir"', configure_script)
        self.assertIn('dependencies_dir = main_dir / "source-deps"', configure_script)
        self.assertIn('middle = "source-deps" if dependencies else "source"', configure_script)
        self.assertIn('"--accept-tensorrt-license"', configure_script)
        self.assertIn("prepare_tensorrt_sdk", configure_script)

        tensorrt_setup = (ROOT / "scripts" / "env" / "setup_tensorrt_sdk.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('TENSORRT_VERSION = "10.15.1.29"', tensorrt_setup)
        self.assertIn("TensorRT-10.15.1.29.Windows.amd64.cuda-13.1.zip", tensorrt_setup)
        self.assertIn("83304c1f9ab86534f083bc4864691b38", tensorrt_setup)
        self.assertIn('"--continue-at"', tensorrt_setup)
        self.assertIn("PLASCAN_ACCEPT_TENSORRT_LICENSE", tensorrt_setup)

        windows_source = next(
            preset
            for preset in presets["configurePresets"]
            if preset["name"] == "windows-source-release"
        )
        self.assertEqual("ON", windows_source["cacheVariables"]["PLASCAN_ENABLE_CUDA"])
        self.assertEqual("ON", windows_source["cacheVariables"]["PLASCAN_ENABLE_TENSORRT"])
        self.assertIn("build/env/sdk/tensorrt/10.15.1.29/bin", windows_source["environment"]["PATH"])
        self.assertIn("$penv{CUDA_PATH}/bin", windows_source["environment"]["PATH"])

        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("option(PLASCAN_ENABLE_TENSORRT", root_cmake)
        self.assertIn("find_package(TensorRT QUIET)", root_cmake)
        self.assertIn("falling back to CUDA backends plus ONNX Runtime CPU inference", root_cmake)
        core_cmake = (ROOT / "src" / "core" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("find_package(TensorRT REQUIRED)", core_cmake)

    def test_release_packages_are_gated_by_platform_tests(self):
        workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
        text = workflow_path.read_text(encoding="utf-8")

        linux_package = workflow_job_block(text, "linux-package-deb")
        self.assertRegex(linux_package, r"(?m)^    needs: build-test$")
        self.assertIn(
            "startsWith(github.ref, 'refs/tags/v') || "
            "github.event_name == 'workflow_dispatch'",
            linux_package,
        )
        self.assertIn("cmake --workflow --preset linux-package-deb", linux_package)
        self.assertNotIn("gfortran", linux_package)
        self.assertIn("models-v1.1.0", linux_package)
        self.assertIn("--pattern U2Net_v1.onnx", linux_package)
        self.assertIn("--pattern lightglue_sift_bucket4096.onnx", linux_package)
        self.assertNotIn("models-v1.2.0", linux_package)
        self.assertNotIn("BiRefNet_dynamic_1024.onnx", linux_package)
        self.assertNotIn("BiRefNet_dynamic_1024.provenance.json", linux_package)
        self.assertIn("*.deb.sha256", linux_package)
        self.assertIn("if-no-files-found: error", linux_package)

        windows_smoke = workflow_job_block(text, "windows-cuda-smoke")
        self.assertIn(
            "github.event_name == 'workflow_dispatch' && "
            "inputs.enable_windows_cuda == true",
            windows_smoke,
        )
        self.assertIn(
            "runs-on: [self-hosted, windows, x64, cuda, tensorrt]",
            windows_smoke,
        )
        self.assertIn("scripts/build_win/build_windows_cuda.ps1", windows_smoke)
        self.assertIn("RunTests = $true", windows_smoke)
        expected_cuda_ctest_regex = (
            'CTestRegex = "TensorRt|U2Net|BiRefNet|SuperPoint|Feature|Match|DenseMatch|Mvs|Sfm"'
        )
        self.assertIn(expected_cuda_ctest_regex, windows_smoke)
        self.assertIn("RunU2NetTensorRtDeploymentTest = $true", windows_smoke)
        self.assertIn("RunBiRefNetTensorRtDeploymentTest = $true", windows_smoke)
        self.assertIn('$buildParameters["CudaRoot"]', windows_smoke)
        self.assertIn('$buildParameters["TensorRtRoot"]', windows_smoke)
        self.assertNotIn("$buildParameters.CudaRoot", windows_smoke)
        self.assertNotIn("$buildParameters.TensorRtRoot", windows_smoke)

        windows_package = workflow_job_block(text, "windows-package-release")
        self.assertRegex(windows_package, r"(?m)^    needs: windows-cuda-smoke$")
        self.assertIn("cmake --workflow --preset windows-package-release", windows_package)
        self.assertIn("scripts/build_win/build_windows_cuda.ps1", windows_package)
        self.assertIn("RunTests = $true", windows_package)
        self.assertIn(expected_cuda_ctest_regex, windows_package)
        self.assertIn("RunU2NetTensorRtDeploymentTest = $true", windows_package)
        self.assertIn("RunBiRefNetTensorRtDeploymentTest = $true", windows_package)
        self.assertIn('$buildParameters["CudaRoot"]', windows_package)
        self.assertIn('$buildParameters["TensorRtRoot"]', windows_package)
        self.assertNotIn("$buildParameters.CudaRoot", windows_package)
        self.assertNotIn("$buildParameters.TensorRtRoot", windows_package)
        for model_pattern in [
            "U2Net_v1.onnx",
            "lightglue_sift_bucket4096.onnx",
            '"loma_r_*"',
        ]:
            self.assertIn(model_pattern, windows_package)
        self.assertIn("if-no-files-found: error", windows_package)

        birefnet_assets = [
            "BiRefNet_dynamic_1024.onnx",
            "BiRefNet_dynamic_1024.provenance.json",
        ]
        for cuda_job in [windows_smoke, windows_package]:
            self.assertEqual(cuda_job.count("gh release download models-v1.2.0"), 2)
            self.assertIn("resources/models/birefnet_dynamic", cuda_job)
            for asset in birefnet_assets:
                with self.subTest(cuda_job=cuda_job.splitlines()[0], asset=asset):
                    self.assertEqual(cuda_job.count(f"--pattern {asset}"), 1)

        non_cuda_workflow = text.replace(windows_smoke, "").replace(windows_package, "")
        self.assertNotIn("models-v1.2.0", non_cuda_workflow)
        for asset in birefnet_assets:
            self.assertNotIn(asset, non_cuda_workflow)

        self.assertIn("enable_windows_cuda:", text)
        self.assertIn("type: boolean", text)
        self.assertIn(
            "A Linux CUDA/TensorRT gate is intentionally deferred until the repository",
            text,
        )
        self.assertNotIn("vcpkg-configuration.json", text)

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
        self.assertIn('"-UCMAKE_CUDA_FLAGS"', text)
        self.assertNotIn("-DCMAKE_CUDA_FLAGS=--compiler-bindir", text)
        self.assertIn("$env:SystemRoot", text)
        self.assertIn('$system32Dir = Join-Path $systemRoot "System32"', text)
        self.assertIn("$system32Dir", text)
        self.assertIn('$chcpProbe = & (Join-Path $system32Dir "cmd.exe")', text)
        self.assertIn('$tensorRtRuntimeDir = if (', text)
        self.assertIn('$env:TENSORRT_ROOT "bin"', text)

    def test_cuda_arches_are_propagated_to_native_backends(self):
        cmake_path = ROOT / "cmake" / "PlascanPackages.cmake"
        text = cmake_path.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^\s*set\s*\(\s*CMAKE_CUDA_ARCHITECTURES\b")
        self.assertIn("PLAMATRIX_CUDA_ARCHITECTURES", text)
        self.assertIn("PLAPOINT_CUDA_ARCHITECTURES", text)
        self.assertIn("FORCE", text)
        self.assertNotIn("find_package(Torch", text)

    def test_cmake_requires_vcpkg_without_conda_dependency_discovery(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        dependency_paths = (ROOT / "cmake" / "PlascanDependencyPaths.cmake").read_text(encoding="utf-8")

        self.assertNotIn("CONDA", root_cmake.upper())
        self.assertNotIn("CONDA", dependency_paths.upper())
        self.assertNotIn("PLASCAN_ENABLE_VCPKG", root_cmake)
        self.assertIn("requires the vcpkg toolchain", dependency_paths)
        self.assertIn("VCPKG_MANIFEST_MODE", dependency_paths)
        self.assertIn("VCPKG_TARGET_TRIPLET", dependency_paths)
        self.assertIn("vcpkg(manifest,${VCPKG_TARGET_TRIPLET})", dependency_paths)

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

    def test_opencv5_is_the_only_supported_api(self):
        manifest = json.loads((ROOT / "vcpkg.json").read_text(encoding="utf-8"))
        dependency_names = {
            dependency if isinstance(dependency, str) else dependency["name"]
            for dependency in manifest.get("dependencies", [])
        }
        self.assertNotIn("opencv", dependency_names)

        package_config = (ROOT / "cmake" / "PlascanPackages.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn("find_package(OpenCV 5.0 REQUIRED COMPONENTS", package_config)
        for component in ("geometry", "stereo", "features"):
            self.assertIn(component, package_config)
        self.assertNotIn("CV_VERSION_MAJOR", package_config)
        self.assertNotIn("calib3d features2d", package_config)

        source_config = (ROOT / "cmake" / "PlascanSourceDependencies.cmake").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("opencv_contrib", source_config)
        self.assertNotIn("xfeatures2d", source_config)
        self.assertNotIn("ximgproc", source_config)
        self.assertFalse((ROOT / "src" / "common" / "OpenCvCompat.h").exists())

        forbidden = ("OpenCvCompat.h", "CV_VERSION_MAJOR", "opencv_compat")
        for path in (ROOT / "src").rglob("*"):
            if path.suffix.lower() not in {".h", ".hpp", ".cpp", ".cu", ".cuh"}:
                continue
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, str(path))

        presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        for preset in presets.get("configurePresets", []):
            manifest_features = preset.get("cacheVariables", {}).get(
                "VCPKG_MANIFEST_FEATURES", ""
            )
            self.assertNotIn("opencv-dnn-cuda", manifest_features)

    def test_windows_cuda_build_script_deploys_u2net_tensorrt_without_cudnn(self):
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
        self.assertIn("Sync-TensorRtRuntime", text)
        self.assertIn("nvinfer_builder_resource_*.dll", text)
        self.assertIn("Assert-U2NetTensorRtDeployment", text)
        self.assertIn("RunU2NetTensorRtDeploymentTest", text)
        self.assertIn("RunU2NetCudaDeploymentTest", text)
        self.assertIn("RunBiRefNetTensorRtDeploymentTest", text)
        self.assertIn("-DPLASCAN_BUNDLE_BIREFNET_DYNAMIC=ON", text)
        self.assertIn('Join-Path $BuildDir "package-smoke\\PlaScan"', text)
        self.assertIn('"test_mask_generation",', text)
        self.assertIn('"plascan_package_smoke"', text)
        self.assertIn("test_birefnet_tensorrt_deployment.ps1", text)
        self.assertIn("VCPKG_APPLOCAL_DEPS=OFF", text)
        self.assertIn("Resolve-ReparseTargetPath", text)
        self.assertIn("CMAKE_MAKE_PROGRAM=$(Convert-ToCMakePath $ninjaExe)", text)
        self.assertIn("CMAKE_CXX_COMPILER=$msvcCudaHostCompiler", text)
        self.assertIn("CMAKE_RC_COMPILER=$(Convert-ToCMakePath $windowsSdkRc)", text)
        self.assertIn("CMAKE_MT=$(Convert-ToCMakePath $windowsSdkMt)", text)
        self.assertNotIn("EnableOpenCvDnnCuda", text)
        self.assertNotIn("Sync-CudnnRuntime", text)
        self.assertNotIn("CUDNN_ROOT_DIR", text)
        self.assertNotIn("Ceres", text)
        self.assertIn("-UVCPKG_MANIFEST_FEATURES", text)
        self.assertIn("-UVCPKG_OVERLAY_PORTS", text)
        self.assertIn("Assert-OpenCv5CpuOnly", text)
        self.assertIn("TensorRtRoot", text)
        self.assertIn("TensorRT_ROOT", text)
        self.assertNotIn('"build\\env\\vcpkg-overlay-ports"', text)
        self.assertIn("PLASCAN_CUDA_ARCHITECTURES=75;86;89;120", text)
        self.assertNotIn("CMAKE_CUDA_FLAGS=--std=c++17", text)
        self.assertNotIn("CMAKE_CUDA_ARCHITECTURES=75\\;86\\;89\\;120", text)
        self.assertNotIn("Ensure-OpenCvCuda13OverlayPort", text)
        self.assertIn("VCPKG_OVERLAY_PORTS", text)
        self.assertNotIn("vcpkgOverlayPortsCMake", text)
        self.assertIn("OpenCV DNN: CPU-only", text)

        deployment_test = (
            ROOT / "scripts" / "build_win" / "test_u2net_tensorrt_deployment.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("EnvironmentVariables.Clear()", deployment_test)
        self.assertIn('EnvironmentVariables["PATH"]', deployment_test)
        self.assertIn("OnnxModelRunsOnTensorRtWhenAvailable", deployment_test)
        self.assertIn("TensorRT available", deployment_test)
        self.assertIn("nvinfer_builder_resource_*.dll", deployment_test)
        self.assertIn('Name -like "cudnn*.dll"', deployment_test)
        self.assertIn("it may have been skipped", deployment_test)

        birefnet_deployment_test = (
            ROOT
            / "scripts"
            / "build_win"
            / "test_birefnet_tensorrt_deployment.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("EnvironmentVariables.Clear()", birefnet_deployment_test)
        self.assertIn(
            "BiRefNetMaskGeneratorIntegrationTest."
            "OnnxModelRunsOnTensorRtWhenExplicitlyEnabled",
            birefnet_deployment_test,
        )
        self.assertIn(
            'EnvironmentVariables["PLASCAN_BIREFNET_INTEGRATION"] = "1"',
            birefnet_deployment_test,
        )
        self.assertIn(
            'EnvironmentVariables["PLASCAN_BIREFNET_MODEL"] = $ModelPath',
            birefnet_deployment_test,
        )
        self.assertIn(
            'EnvironmentVariables["PLASCAN_BIREFNET_ENGINE_CACHE"] = $EngineCache',
            birefnet_deployment_test,
        )
        self.assertIn('"package-smoke\\PlaScan"', birefnet_deployment_test)
        self.assertIn(
            '"resources\\models\\birefnet_dynamic\\BiRefNet_dynamic_1024.onnx"',
            birefnet_deployment_test,
        )
        self.assertIn(
            '"resources\\models\\birefnet_dynamic\\BiRefNet_dynamic_1024.provenance.json"',
            birefnet_deployment_test,
        )
        self.assertIn("Assert-NoEngineArtifacts $InstallRoot", birefnet_deployment_test)
        self.assertIn("engine_reused\\s*=\\s*$ExpectedReuse", birefnet_deployment_test)
        self.assertIn('-ExpectedReuse "no"', birefnet_deployment_test)
        self.assertIn('-ExpectedReuse "yes"', birefnet_deployment_test)

    def test_gui_build_deploys_vcpkg_runtime_dlls(self):
        module = (ROOT / "cmake" / "PlascanWindowsRuntime.cmake").read_text(
            encoding="utf-8"
        )
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        gui_cmake = (ROOT / "src" / "gui" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        install_bundle = (
            ROOT
            / "src"
            / "gui"
            / "packaging"
            / "InstallBundledRuntimeWindows.cmake.in"
        ).read_text(encoding="utf-8")
        tensorrt_runtime = (
            ROOT / "cmake" / "PlascanTensorRtRuntime.cmake"
        ).read_text(encoding="utf-8")

        self.assertIn("function(plascan_deploy_vcpkg_runtime target_name)", module)
        self.assertIn("VCPKG_INSTALLED_DIR", module)
        self.assertIn('"${_plascan_vcpkg_runtime_dir}/*.dll"', module)
        self.assertIn("copy_if_different", module)
        self.assertIn("COMMAND_EXPAND_LISTS", module)
        self.assertIn("include(PlascanWindowsRuntime)", root_cmake)
        self.assertIn("plascan_deploy_vcpkg_runtime(plascan_gui)", gui_cmake)
        self.assertNotIn('  "cudnn*.dll"', install_bundle)
        self.assertIn('"nvinfer_builder_resource_*.dll"', install_bundle)
        self.assertIn("must not bundle cuDNN", install_bundle)
        self.assertIn("_plascan_dynamic_runtime_patterns", install_bundle)
        self.assertIn('"${CUDAToolkit_BIN_DIR}/x64"', tensorrt_runtime)
        self.assertIn('"${_plascan_cuda_runtime_dir}/cublas64_*.dll"', tensorrt_runtime)
        self.assertIn('"${_plascan_cuda_runtime_dir}/nvrtc64_*.dll"', tensorrt_runtime)
        self.assertIn("nvinfer_builder_resource_", tensorrt_runtime)

    def test_release_1_1_7_metadata_is_synchronized(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        core_cmake = (ROOT / "src" / "core" / "CMakeLists.txt").read_text(encoding="utf-8")
        manifest = json.loads((ROOT / "vcpkg.json").read_text(encoding="utf-8"))
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        release_doc = ROOT / "docs" / "releases" / "v1.1.7.md"

        self.assertIn("project(PlaScan VERSION 1.1.7", root_cmake)
        self.assertIn("project(PlaScanCore VERSION 1.1.7", core_cmake)
        self.assertEqual(manifest.get("version-string"), "1.1.7")
        self.assertIn("## v1.1.7 - 2026-08-10", changelog)
        self.assertTrue(release_doc.exists(), "v1.1.7 release notes are missing")

        release_text = release_doc.read_text(encoding="utf-8")
        for required in [
            "MVS",
            "CUDA",
            "LoMa-R",
            "U2Net",
            "GitHub Actions",
            "v1.1.7",
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

    def test_linux_install_rules_keep_launcher_separate_from_plascan_binary(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        gui_install = (ROOT / "src" / "gui" / "cmake" / "GuiInstall.cmake").read_text(encoding="utf-8")
        normalized_gui_install = " ".join(gui_install.split())

        self.assertNotIn("resources/plascan.sh", root_cmake)
        self.assertIn("plascan_gui_launcher.sh.in", gui_install)
        self.assertIn(
            'configure_file("${PLASCAN_LAUNCHER_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan-launcher" @ONLY)',
            normalized_gui_install,
        )
        self.assertIn(
            'install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan-launcher" DESTINATION /usr/bin RENAME plascan',
            normalized_gui_install,
        )
        self.assertNotIn(
            'install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan-launcher" DESTINATION bin',
            normalized_gui_install,
        )

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

    def test_camera_scene_uses_full_ply_loader_without_preview_memory_probe(self):
        source = (
            ROOT / "src" / "gui" / "views" / "CameraSceneWidget.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("plapoint::io::readPly<float>", source)
        self.assertNotIn("readBinaryPlyPreview", source)
        self.assertNotIn("/proc/meminfo", source)
        self.assertNotIn("MemAvailable:", source)

    def test_gui_gtest_discovery_does_not_initialize_qapplication(self):
        gui_test_sources = [
            "test_gui_project_utils.cpp",
            "test_map_project_dialog.cpp",
            "test_workspace_section_icons.cpp",
            "test_workflow_settings_dialog.cpp",
            "test_workflow_parameter_dialog_style.cpp",
        ]

        for source_name in gui_test_sources:
            with self.subTest(source=source_name):
                source = (ROOT / "tests" / source_name).read_text(encoding="utf-8")
                self.assertIn('"--gtest_list_tests"', source)
                discovery_guard = source.rfind("if (lists_tests)")
                application_start = max(
                    source.rfind("QApplication app("),
                    source.rfind("QApplication application("),
                )
                self.assertGreater(discovery_guard, 0)
                self.assertGreater(application_start, discovery_guard)

    def test_build_warning_exceptions_are_capability_scoped(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        dense_costs = (
            ROOT / "src" / "core" / "dense_match" / "CostFunctions.cpp"
        ).read_text(encoding="utf-8")
        mvs_cmake = (ROOT / "src" / "core" / "mvs" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        packages = (ROOT / "cmake" / "PlascanPackages.cmake").read_text(
            encoding="utf-8"
        )

        self.assertIn("-Xcompiler=/Zc:preprocessor", root_cmake)
        self.assertIn("defined(_OPENMP) && _OPENMP >= 200805", dense_costs)
        self.assertIn("-diag-suppress=940,1394", mvs_cmake)
        self.assertIn("QT_NO_PRIVATE_MODULE_WARNING ON", packages)
        self.assertIn(
            'OpenMP_CXX_FLAGS "/openmp:llvm /openmp:experimental"', packages
        )
        self.assertIn("target_compile_options(plamatrix PRIVATE", packages)
        self.assertIn("$<$<COMPILE_LANGUAGE:CXX>:/wd4849>", packages)
        self.assertIn("target_compile_options(plapoint PRIVATE", packages)
        self.assertIn("--expt-relaxed-constexpr -diag-suppress=550", packages)

        windows_cuda_build = (
            ROOT / "scripts" / "build_win" / "build_windows_cuda.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('"-UCMAKE_CUDA_FLAGS"', windows_cuda_build)
        self.assertNotIn(
            "-DCMAKE_CUDA_FLAGS=--compiler-bindir", windows_cuda_build
        )


if __name__ == "__main__":
    unittest.main()
