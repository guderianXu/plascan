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
        source = read_text(
            "src/core/aerial_triangulation/reconstruction/SfmAttemptRunner.cpp"
        )

        self.assertIn("options->baOptions.backend = BABackend::Auto;", source)
        self.assertIn("options->baOptions.minNativeCudaCameras = 50;", source)
        self.assertIn("options->baOptions.minNativeCudaObservations = 500000;", source)
        self.assertIn("options->baOptions.minCeresCudaObservations = 500000;", source)
        self.assertIn("options->baOptions.minCeresCpuObservations = 50000;", source)
        self.assertIn("options->baOptions.enableBackendQualityGate = true;", source)
        self.assertIn("options->baOptions.allowBackendFallback = true;", source)

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
        self.assertIn("ba_native_cuda_kernel_seconds", source)
        self.assertIn("ba_native_cuda_upload_seconds", source)
        self.assertIn("ba_native_cuda_staging_seconds", source)
        self.assertIn("ba_native_cuda_release_seconds", source)

    def test_benchmark_supports_native_cuda(self):
        source = self.read_text("src/core/bundle_adjust/tools/ba_backend_benchmark.cpp")
        self.assertIn("native_cuda", source)
        self.assertIn("BABackend::NativeCuda", source)
        self.assertIn("native_kernel_seconds", source)
        self.assertIn("native_staging_seconds", source)


if __name__ == "__main__":
    unittest.main()
