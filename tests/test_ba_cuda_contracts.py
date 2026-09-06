import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


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
        options = read_text("src/core/bundle_adjust/BundleAdjustOptions.h")
        cli = read_text("src/cli/reconstruction/cli_bundle_adjust.cpp")
        source = read_text(
            "src/core/aerial_triangulation/reconstruction/SfmAttemptRunner.cpp"
        )
        line_scan = read_text("src/core/lidar/PlanetaryLineScanBundleAdjust.h")

        self.assertIn("kAutoBackendPolicyVersion = 2;", options)
        self.assertIn("kDefaultMinPlaMatrixCudaCameras = 128;", options)
        self.assertIn("kDefaultMinPlaMatrixCudaObservations = 30000;", options)
        self.assertIn("kDefaultMinPlaMatrixOpenClCameras = 160;", options)
        self.assertIn("kDefaultMinPlaMatrixOpenClObservations = 50000;", options)
        self.assertIn("kDefaultMinPlaMatrixDenseCameras = 120;", options)
        self.assertIn("kDefaultMinPlaMatrixCudaDenseObservations = 150000;", options)
        self.assertIn("kDefaultMinPlaMatrixOpenClDenseObservations = 200000;", options)
        self.assertIn("BAOptions::kDefaultMinPlaMatrixCudaCameras", cli)
        self.assertIn("BAOptions::kDefaultMinPlaMatrixOpenClCameras", cli)
        self.assertIn("options->baOptions.backend = BABackend::Auto;", source)
        self.assertIn("options->baOptions.backend = BABackend::PlaMatrixCpu;", source)
        self.assertNotIn("minPlaMatrixCudaObservations = 300000", source)
        self.assertIn("options->baOptions.enableBackendQualityGate = true;", source)
        self.assertIn("options->baOptions.allowBackendFallback = true;", source)
        self.assertIn("BAOptions::kDefaultMinPlaMatrixCudaCameras", line_scan)
        self.assertIn("BAOptions::kDefaultMinPlaMatrixOpenClCameras", line_scan)

    def test_adaptive_camera_model_declares_full_model_then_filters_parameters(self):
        source = read_text(
            "src/core/aerial_triangulation/reconstruction/SfmAttemptRunner.cpp"
        )

        self.assertIn("options->adaptiveCameraModelFitting = true;", source)
        self.assertIn("options->baOptions.refineSharedFocalAspectRatio = true;", source)
        self.assertIn("options->baOptions.refineSharedPrincipalPoint = true;", source)
        self.assertIn("options->baOptions.refineSharedRadialDistortion = true;", source)

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
        project_manager = read_text("src/gui/project/manager/ProjectManager.cpp")

        self.assertIn('toString(QStringLiteral("auto"))', project_manager)
        self.assertIn('opts.baOpt.backend = xjw::BABackend::Auto;', project_manager)
        self.assertIn('opts.baOpt.minPlaMatrixCudaObservations', project_manager)
        self.assertIn('opts.baOpt.minPlaMatrixOpenClObservations', project_manager)
        self.assertIn('ba_auto_backend_policy_version', project_manager)
        self.assertIn('opts.baOpt.maxInitialTrackRms', project_manager)
        self.assertIn('opts.baOpt.enableBackendQualityGate', project_manager)

    def test_plamatrix_backend_is_exposed_with_comparison_metrics(self):
        header = self.read_text("src/core/bundle_adjust/BundleAdjustTypes.h")
        benchmark = self.read_text(
            "src/core/bundle_adjust/tools/ba_backend_benchmark.cpp"
        )
        service = self.read_text("src/gui/project/services/BundleAdjustService.cpp")

        self.assertIn("PlaMatrixCpu", header)
        self.assertIn("PlaMatrixCuda", header)
        self.assertIn("PlaMatrixOpenCl", header)
        self.assertIn("plamatrix_cpu", benchmark)
        self.assertIn("plamatrix_cuda", benchmark)
        self.assertIn("plamatrix_opencl", benchmark)
        self.assertIn("plamatrix_initial_cost", benchmark)
        self.assertIn("plamatrix_final_cost", benchmark)
        self.assertIn("plamatrix_linear_solver", benchmark)
        self.assertIn("plamatrix_device", benchmark)
        self.assertIn("ba_plamatrix_initial_cost", service)
        self.assertIn("ba_plamatrix_final_cost", service)
        self.assertIn("ba_plamatrix_linear_solver", service)
        self.assertIn("ba_plamatrix_device_name", service)


if __name__ == "__main__":
    unittest.main()
