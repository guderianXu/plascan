import json
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DRIVER = runpy.run_path(str(ROOT / "scripts/bench/run_ba_backend_benchmark.py"))


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class BaCudaContractsTest(unittest.TestCase):
    def read_text(self, relative_path: str) -> str:
        return read_text(relative_path)

    def test_build_configuration_has_no_ceres_backend(self):
        script = read_text("scripts/build_win/build_windows_cuda.ps1")
        manifest = json.loads(read_text("vcpkg.json"))

        self.assertNotIn("ceres", script.lower())
        self.assertEqual([], manifest["default-features"])
        self.assertNotIn("features", manifest)

    def test_ba_auto_backend_thresholds_have_one_default_source(self):
        options = read_text("3rdparty/plabundle/include/plabundle/options.h")
        cli = read_text("src/cli/reconstruction/cli_bundle_adjust.cpp")
        source = read_text(
            "src/core/aerial_triangulation/reconstruction/SfmAttemptRunner.cpp"
        )
        line_scan = read_text("src/core/lidar/PlanetaryLineScanBundleAdjust.h")

        self.assertIn("kAutoPolicyVersion = 3;", options)
        self.assertIn("kDefaultMinCudaCameras = 128;", options)
        self.assertIn("kDefaultMinCudaObservations = 30000;", options)
        self.assertIn("kDefaultMinOpenClCameras = 160;", options)
        self.assertIn("kDefaultMinOpenClObservations = 50000;", options)
        self.assertIn("kDefaultMinVulkanCameras = 160;", options)
        self.assertIn("kDefaultMinVulkanObservations = 50000;", options)
        self.assertIn("kDefaultMinDenseCameras = 120;", options)
        self.assertIn("kDefaultMinCudaDenseObservations = 150000;", options)
        self.assertIn("kDefaultMinOpenClDenseObservations = 200000;", options)
        self.assertIn("kDefaultMinVulkanDenseObservations = 200000;", options)
        self.assertIn(
            "kDefaultMinPlaMatrixCudaCameras = BackendOptions::kDefaultMinCudaCameras;",
            options,
        )
        self.assertIn("plabundle::BackendOptions::kDefaultMinCudaCameras", cli)
        self.assertIn("plabundle::BackendOptions::kDefaultMinOpenClCameras", cli)
        self.assertIn("plabundle::BackendOptions::kDefaultMinVulkanCameras", cli)
        self.assertIn("options->baOptions.backend.requested = plabundle::Backend::Auto;", source)
        self.assertIn("options->baOptions.backend.requested = plabundle::Backend::PlaMatrixCpu;", source)
        self.assertNotIn("minPlaMatrixCudaObservations = 300000", source)
        self.assertIn("options->baOptions.quality.enabled = true;", source)
        self.assertIn("options->baOptions.backend.allowFallback = true;", source)
        self.assertIn("plabundle::BackendOptions::kDefaultMinCudaCameras", line_scan)
        self.assertIn("plabundle::BackendOptions::kDefaultMinOpenClCameras", line_scan)
        self.assertIn("plabundle::BackendOptions::kDefaultMinVulkanCameras", line_scan)

    def test_adaptive_camera_model_declares_full_model_then_filters_parameters(self):
        source = read_text(
            "src/core/aerial_triangulation/reconstruction/SfmAttemptRunner.cpp"
        )

        self.assertIn("options->adaptiveCameraModelFitting = true;", source)
        self.assertIn("options->baOptions.calibration.refineSharedFocalAspectRatio = true;", source)
        self.assertIn("options->baOptions.calibration.refineSharedPrincipalPoint = true;", source)
        self.assertIn("options->baOptions.calibration.refineSharedRadialDistortion = true;", source)

        coordinator = read_text(
            "src/core/sfm/pipeline/SfmBundleAdjustCoordinator.cpp"
        )
        self.assertIn("assessAdaptiveCameraModel", coordinator)
        self.assertIn("applyAdaptiveCameraModel", coordinator)

    def test_bundle_adjust_service_records_requested_and_used_backend(self):
        source = read_text("src/gui/project/services/BundleAdjustService.cpp")

        self.assertIn('saveObj[QStringLiteral("ba_requested_backend")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_used_backend")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_used_gpu")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_backend_fallback")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_backend_selection_reason")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_quality_gate_rejected")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_valid_track_ratio")]', source)

    def test_bundle_adjust_execution_defaults_to_auto_backend(self):
        controller = read_text(
            "src/gui/project/tasks/ProjectBundleAdjustController.cpp"
        )

        self.assertIn('toString(QStringLiteral("auto"))', controller)
        self.assertIn('options->baOpt.backend.requested = plabundle::Backend::Auto;', controller)
        self.assertIn('options->baOpt.backend.minPlaMatrixCudaObservations', controller)
        self.assertIn('options->baOpt.backend.minPlaMatrixOpenClObservations', controller)
        self.assertIn('options->baOpt.backend.minPlaMatrixVulkanObservations', controller)
        self.assertNotIn('kLegacyMinPlaMatrixGpuCameras', controller)
        self.assertNotIn('kLegacyMinPlaMatrixGpuObservations', controller)
        self.assertNotIn('legacy_cpu', controller)
        self.assertIn('ProjectConfigManager::validateBundleAdjustSettings', controller)
        self.assertIn('options->baOpt.solver.maxInitialTrackRms', controller)
        self.assertIn('options->baOpt.quality.enabled', controller)

    def test_plamatrix_backend_is_exposed_with_comparison_metrics(self):
        header = self.read_text("3rdparty/plabundle/include/plabundle/backend.h")
        benchmark = self.read_text(
            "3rdparty/plabundle/benchmark/plabundle_benchmark.cpp"
        )
        service = self.read_text("src/gui/project/services/BundleAdjustService.cpp")

        self.assertIn("PlaMatrixCpu", header)
        self.assertIn("PlaMatrixCuda", header)
        self.assertIn("PlaMatrixOpenCl", header)
        self.assertIn("PlaMatrixVulkan", header)
        self.assertIn("plamatrix_cpu", benchmark)
        self.assertIn("plamatrix_cuda", benchmark)
        self.assertIn("plamatrix_opencl", benchmark)
        self.assertIn("plamatrix_vulkan", benchmark)
        self.assertIn('",initial_cost="', benchmark)
        self.assertIn('",final_cost="', benchmark)
        self.assertIn('",linear_solver="', benchmark)
        self.assertIn('",device="', benchmark)
        self.assertIn("ba_plamatrix_initial_cost", service)
        self.assertIn("ba_plamatrix_final_cost", service)
        self.assertIn("ba_plamatrix_linear_solver", service)
        self.assertIn("ba_plamatrix_device_name", service)

    def test_backend_benchmark_driver_parses_plabundle_run_records(self):
        parse_metric_line = BENCHMARK_DRIVER["parse_metric_line"]
        row = parse_metric_line(
            "run,backend=plamatrix_cpu,repetition=2,phase=warm,status=success,"
            "linear_solver=block_sparse_cholesky,total_seconds=0.25,wall_seconds=0.27"
        )

        self.assertIsNotNone(row)
        self.assertEqual("plamatrix_cpu", row["backend"])
        self.assertEqual("2", row["repetition"])
        self.assertEqual("warm", row["phase"])
        self.assertEqual("0.27", row["wall_seconds"])
        self.assertIsNone(parse_metric_line("dataset,cameras=80,tracks=3000"))


if __name__ == "__main__":
    unittest.main()
