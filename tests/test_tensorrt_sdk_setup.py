import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "env"))

from setup_tensorrt_sdk import (  # noqa: E402
    TENSORRT_ARCHIVE_NAME,
    TENSORRT_ARCHIVE_SHA256,
    TENSORRT_VERSION,
    find_sdk_root,
    is_valid_sdk,
    prepare_tensorrt_sdk,
)
from configure_with_env import (  # noqa: E402
    cmake_path,
    compute_capability_architectures,
    custom_build_layout,
    default_build_jobs,
    default_vcpkg_triplet,
)


def create_fake_sdk(root: Path) -> None:
    for relative in (
        "include/NvInferRuntime.h",
        "include/NvOnnxParser.h",
        "lib/nvinfer_10.lib",
        "lib/nvonnxparser_10.lib",
        "bin/nvinfer_10.dll",
        "bin/nvonnxparser_10.dll",
        "bin/nvinfer_plugin_10.dll",
        "bin/nvinfer_builder_resource_120.dll",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


class TensorRtSdkSetupTest(unittest.TestCase):
    def test_compute_capabilities_are_converted_to_cmake_architectures(self):
        self.assertEqual("89;120", compute_capability_architectures("12.0\n8.9\n12.0\n"))

    def test_build_defaults_to_all_logical_cpus(self):
        with patch("configure_with_env.os.cpu_count", return_value=32):
            self.assertEqual(32, default_build_jobs({}))

    def test_build_parallelism_environment_override_is_preserved(self):
        self.assertEqual(6, default_build_jobs({"CMAKE_BUILD_PARALLEL_LEVEL": "6"}))

    def test_custom_build_layout_keeps_dependencies_below_main_build_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            main_dir, dependencies_dir, prefix, vcpkg_dir = custom_build_layout(directory)

            self.assertEqual(Path(directory).resolve(), main_dir)
            self.assertEqual(main_dir / "source-deps", dependencies_dir)
            self.assertEqual(dependencies_dir / "install", prefix)
            self.assertEqual(dependencies_dir / "vcpkg_installed", vcpkg_dir)
            self.assertNotIn("\\", cmake_path(vcpkg_dir))

    def test_default_vcpkg_triplet_matches_host_platform(self):
        with patch("configure_with_env.host_platform", return_value="windows"):
            self.assertEqual("x64-windows", default_vcpkg_triplet())
        with patch("configure_with_env.host_platform", return_value="linux"):
            self.assertEqual("x64-linux-dynamic", default_vcpkg_triplet())
        with patch("configure_with_env.host_platform", return_value="macos"):
            self.assertEqual("arm64-osx", default_vcpkg_triplet())

    def test_pinned_windows_sdk_metadata_is_complete(self):
        self.assertEqual("10.15.1.29", TENSORRT_VERSION)
        self.assertIn("cuda-13.1", TENSORRT_ARCHIVE_NAME)
        self.assertRegex(TENSORRT_ARCHIVE_SHA256, r"^[0-9a-f]{64}$")

    def test_complete_sdk_is_found_beneath_extract_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            extract_root = Path(directory)
            sdk_root = extract_root / "TensorRT-10.15.1.29"
            create_fake_sdk(sdk_root)

            self.assertTrue(is_valid_sdk(sdk_root))
            self.assertEqual(sdk_root.resolve(), find_sdk_root(extract_root))

    def test_existing_sdk_does_not_require_download_or_license_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            sdk_root = repo_root / "sdk"
            create_fake_sdk(sdk_root)

            result = prepare_tensorrt_sdk(
                repo_root,
                {},
                requested_root=str(sdk_root),
                auto_install=True,
                accept_license=False,
            )

            self.assertEqual(sdk_root.resolve(), result)


if __name__ == "__main__":
    unittest.main()
