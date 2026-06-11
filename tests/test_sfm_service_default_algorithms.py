import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SfmServiceDefaultAlgorithmsTest(unittest.TestCase):
    def test_one_click_sfm_defaults_to_disk_lightglue(self):
        header = (ROOT / "src/core/pipeline/SFMService.h").read_text(encoding="utf-8")

        self.assertRegex(header, r'featureAlgorithm\s*=\s*QStringLiteral\("disk"\)')
        self.assertRegex(header, r'matchAlgorithm\s*=\s*QStringLiteral\("lightglue"\)')

    def test_one_click_sfm_uses_algorithm_aware_feature_and_match_pipeline(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")

        self.assertNotIn("ProjectIO::findSpForImage", source)
        self.assertNotIn('#include "SuperPoint.h"', source)
        self.assertNotIn('#include "SuperGlueMatcher.h"', source)
        self.assertIn("createExtractor(featureAlgorithm.toStdString(), extractorCfg)", source)
        self.assertIn("LightGlueMatcher", source)
        self.assertIn("feature_algorithm", source)
        self.assertIn("match_algorithm", source)


if __name__ == "__main__":
    unittest.main()
