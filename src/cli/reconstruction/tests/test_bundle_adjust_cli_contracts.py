from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]


class BundleAdjustCliContractsTest(unittest.TestCase):
    def test_exposes_ba_backend_controls(self):
        source = (ROOT / "src/cli/reconstruction/cli_bundle_adjust.cpp").read_text(
            encoding="utf-8"
        )

        for option in (
            "--ba-backend",
            "--ba-min-cuda-cameras",
            "--ba-min-cuda-observations",
            "--ba-min-opencl-cameras",
            "--ba-min-opencl-observations",
            "--ba-min-dense-cameras",
            "--ba-min-cuda-dense-observations",
            "--ba-min-opencl-dense-observations",
            "--ba-max-initial-track-rms",
            "--ba-quality-gate",
            "--ba-max-rms-growth",
            "--ba-min-valid-track-ratio",
        ):
            self.assertIn(option, source)

        self.assertIn(
            "parseBaBackendName(xjw::cli::fromStdString(baBackendRaw))", source
        )
        self.assertNotIn("QString toQString(", source)
        self.assertIn("baOptions.backend", source)

    def test_removed_solver_alias_and_options_are_rejected(self):
        source = (ROOT / "src/cli/reconstruction/cli_bundle_adjust.cpp").read_text(encoding="utf-8")
        for option in ("legacy_cpu", "--ba-compare-legacy", "--no-ba-compare-legacy",
                       "--max-point-iterations", "--max-camera-iterations", "--huber-delta",
                       "--damping", "--finite-diff-eps", "--step-tolerance", "--ba-max-dense-schur-cameras"):
            self.assertNotIn(option, source)
        benchmark = (ROOT / "src/core/bundle_adjust/tools/ba_backend_benchmark.cpp").read_text(encoding="utf-8")
        self.assertNotIn("legacy_cpu", benchmark)
        self.assertNotIn("--max-dense-schur-cameras", benchmark)

    def test_default_output_uses_current_chunk_bundle_adjust_directory(self):
        source = (ROOT / "src/cli/reconstruction/cli_bundle_adjust.cpp").read_text(
            encoding="utf-8"
        )

        self.assertIn("ProjectIO::projectBundleAdjustDir(projectPath)", source)
        self.assertIn(
            "未指定时写入当前 Chunk 的 bundle_adjust/<timestamp>", source
        )

    def test_exposes_and_forwards_planetary_laser_range_controls(self):
        source = (ROOT / "src/cli/reconstruction/cli_bundle_adjust.cpp").read_text(
            encoding="utf-8"
        )

        for option in (
            "--laser-range-data",
            "--laser-range-camera-frame",
            "--laser-range-camera-sensor-frame",
            "--laser-range-image-alias",
            "--laser-range-allow-unmapped-shots",
            "--laser-range-allow-unmapped-measures",
            "--laser-range-isis-target",
            "--laser-range-isis-body-frame",
            "--laser-range-isis-laser-frame",
            "--laser-range-isis-sensor-model",
            "--laser-range-isis-range-type",
            "--laser-range-isis-lever-arm",
        ):
            self.assertIn(option, source)

        self.assertIn("parsePlanetaryLaserImageAliases", source)
        self.assertIn("mergePlanetaryLaserProjectImageAliases", source)
        self.assertIn("planetaryLaserAllowUnmappedMeasuredImages", source)
        self.assertIn("planetaryLaserParseOptions", source)
        self.assertIn("planetaryLaserRangeWeightOption->count() > 0", source)
        self.assertIn("planetaryLaserHuberOption->count() > 0", source)


if __name__ == "__main__":
    unittest.main()
