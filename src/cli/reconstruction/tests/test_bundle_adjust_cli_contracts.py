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
            "--ba-cuda-device",
            "--ba-min-cuda-cameras",
            "--ba-min-cuda-observations",
            "--ba-min-cpu-observations",
            "--ba-max-ceres-point-only-observations",
            "--ba-quality-gate",
            "--ba-max-rms-growth",
            "--ba-min-valid-track-ratio",
            "--ba-compare-legacy",
        ):
            self.assertIn(option, source)

        self.assertIn(
            "parseBaBackendName(xjw::cli::fromStdString(baBackendRaw))", source
        )
        self.assertNotIn("QString toQString(", source)
        self.assertIn("baOptions.backend", source)


if __name__ == "__main__":
    unittest.main()
