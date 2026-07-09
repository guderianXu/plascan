from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class BaCudaContractsTest(unittest.TestCase):
    def read_text(self, relative_path: str) -> str:
        return read_text(relative_path)

    def test_windows_cuda_build_enables_ceres_cuda_manifest_feature(self):
        script = read_text("scripts/build_win/build_windows_cuda.ps1")

        self.assertIn("EnableCeresCudaBa", script)
        self.assertIn("ceres-cuda", script)
        self.assertIn("manifestFeaturesValue", script)
        self.assertIn("VCPKG_MANIFEST_FEATURES=$manifestFeaturesValue", script)

    def test_aerial_triangulation_cuda_requests_auto_ba_with_cuda_thresholds(self):
        source = read_text("src/core/aerial_triangulation/AerialTriangulationService.cpp")

        self.assertIn("sfmOpts.baOptions.backend = xjw::BABackend::Auto;", source)
        self.assertIn("sfmOpts.baOptions.ceresCudaDevice = 0;", source)
        self.assertIn("sfmOpts.baOptions.minCeresCudaObservations = 500000;", source)
        self.assertIn("sfmOpts.baOptions.minCeresCpuObservations = 50000;", source)
        self.assertIn("sfmOpts.baOptions.enableBackendQualityGate = true;", source)
        self.assertIn("BundleAdjust::backendName(sfmOpts.baOptions.backend)", source)

    def test_bundle_adjust_service_records_requested_and_used_backend(self):
        source = read_text("src/gui/project/services/BundleAdjustService.cpp")

        self.assertIn('saveObj[QStringLiteral("ba_requested_backend")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_used_backend")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_used_gpu")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_backend_fallback")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_backend_selection_reason")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_quality_gate_rejected")]', source)
        self.assertIn('saveObj[QStringLiteral("ba_valid_track_ratio")]', source)

    def test_bundle_adjust_cli_exposes_ba_backend_controls(self):
        source = read_text("src/cli/cli_bundle_adjust.cpp")

        self.assertIn('--ba-backend', source)
        self.assertIn('--ba-cuda-device', source)
        self.assertIn('--ba-min-cuda-cameras', source)
        self.assertIn('--ba-min-cuda-observations', source)
        self.assertIn('--ba-min-cpu-observations', source)
        self.assertIn('--ba-max-ceres-point-only-observations', source)
        self.assertIn('--ba-quality-gate', source)
        self.assertIn('--ba-max-rms-growth', source)
        self.assertIn('--ba-min-valid-track-ratio', source)
        self.assertIn('--ba-compare-legacy', source)
        self.assertIn('parseBaBackendName(toQString(baBackendRaw))', source)
        self.assertIn('baOptions.backend', source)

    def test_standalone_bundle_adjust_defaults_to_auto_backend(self):
        dialog = read_text("src/gui/dialogs/BundleAdjustDialog.cpp")
        project_manager = read_text("src/gui/project/manager/ProjectManager.cpp")

        self.assertIn('options[QStringLiteral("ba_backend")] = QStringLiteral("auto");', dialog)
        self.assertIn('options[QStringLiteral("ba_min_cpu_observations")] = 50000;', dialog)
        self.assertIn('options[QStringLiteral("ba_max_ceres_point_only_observations")] = 100000;', dialog)
        self.assertIn('options[QStringLiteral("ba_enable_backend_quality_gate")] = true;', dialog)
        self.assertIn('toString(QStringLiteral("auto"))', project_manager)
        self.assertIn('opts.baOpt.backend = xjw::BABackend::Auto;', project_manager)
        self.assertIn('opts.baOpt.minCeresCudaObservations', project_manager)
        self.assertIn('opts.baOpt.minCeresCpuObservations', project_manager)
        self.assertIn('opts.baOpt.maxCeresPointOnlyObservations', project_manager)
        self.assertIn('opts.baOpt.enableBackendQualityGate', project_manager)

    def test_native_cuda_backend_is_exposed(self):
        header = self.read_text("src/core/bundle_adjust/BundleAdjust.h")
        self.assertIn("NativeCuda", header)
        self.assertIn("minNativeCudaCameras", header)
        self.assertIn("minNativeCudaObservations", header)
        self.assertIn("nativeCudaPcgIterations", header)

    def test_service_reports_native_cuda_metrics(self):
        source = self.read_text("src/gui/project/services/BundleAdjustService.cpp")
        self.assertIn("ba_native_cuda_pcg_iterations", source)
        self.assertIn("ba_native_cuda_linear_residual", source)
        self.assertIn("ba_native_cuda_active_observations", source)

    def test_benchmark_supports_native_cuda(self):
        source = self.read_text("src/core/bundle_adjust/tools/ba_backend_benchmark.cpp")
        self.assertIn("native_cuda", source)
        self.assertIn("BABackend::NativeCuda", source)


if __name__ == "__main__":
    unittest.main()
