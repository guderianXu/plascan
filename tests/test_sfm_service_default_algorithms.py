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

    def test_disk_and_aliked_lightglue_use_dedicated_torchscript_models(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")
        runner_source = (ROOT / "src/core/pipeline/FeatureMatchRunner.cpp").read_text(encoding="utf-8")
        start = source.index("QStringList lightGlueModelCandidates")
        end = source.index("QString findScriptFile", start)
        body = source[start:end]

        disk_start = body.index('featureAlgorithm == QStringLiteral("disk")')
        aliked_start = body.index('featureAlgorithm == QStringLiteral("aliked")')
        generic_start = body.index('QStringLiteral("lightglue_matcher_%1.torchscript")')

        disk_branch = body[disk_start:aliked_start]
        aliked_branch = body[aliked_start:generic_start]
        self.assertIn("lightglue_disk_%1.torchscript", disk_branch)
        self.assertIn("lightglue_aliked_%1.torchscript", aliked_branch)
        self.assertIn("lightglue_disk_%1.pt", disk_branch)
        self.assertIn("lightglue_aliked_%1.pt", aliked_branch)
        self.assertNotIn("lightglue_matcher", disk_branch)
        self.assertNotIn("lightglue_matcher", aliked_branch)

        self.assertIn("PLASCAN_ALLOW_PYTHON_LIGHTGLUE_FALLBACK", source)
        self.assertIn("PLASCAN_ALLOW_PYTHON_LIGHTGLUE_FALLBACK", runner_source)
        self.assertNotIn(
            "const bool usePythonLightGlue = lgModelPath.isEmpty() && canUsePythonLightGlue;",
            source,
        )
        self.assertIn("runPythonLightGlue", source)
        self.assertIn("run_lightglue.py", source)
        self.assertIn("默认流程不再自动回退 Python LightGlue", source)

    def test_disk_feature_extraction_uses_high_capacity_model_and_quality_cap(self):
        header = (ROOT / "src/core/pipeline/SFMService.h").read_text(encoding="utf-8")
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")
        factory = (ROOT / "src/core/feature_extractors/ExtractorFactory.cpp").read_text(encoding="utf-8")
        disk_cpp = (ROOT / "src/core/feature_extractors/disk/DiskExtractor.cpp").read_text(encoding="utf-8")
        aliked_cpp = (ROOT / "src/core/feature_extractors/aliked/AlikedExtractor.cpp").read_text(encoding="utf-8")

        self.assertIn("featureMaxImageDim", header)
        self.assertIn("safeDefaultFeatureMaxImageDim", source)
        self.assertIn("disk_extractor_%1_8192.torchscript", source)
        self.assertIn("disk_extractor_%1_8192.pt", source)
        self.assertNotRegex(
            source,
            r'featureAlgorithm\s*==\s*QStringLiteral\("disk"\).*?return\s+1200',
            re.S,
        )
        self.assertIn("if (presets.featureMaxImageDim <= 0)", source)
        self.assertIn("return 0;", source)
        self.assertIn("resolveFeatureMaxImageDim(opts, presets, featureAlgorithm)", source)
        self.assertIn("extractorCfg.maxImageDim   = featureMaxImageDim", source)
        self.assertIn("dcfg.scoreThreshold = cfg.detThreshold", factory)
        self.assertIn("acfg.scoreThreshold = cfg.detThreshold", factory)
        self.assertIn("_config.maxKeypoints", disk_cpp)
        self.assertIn("_config.maxKeypoints", aliked_cpp)
        self.assertNotIn("m_cfg.maxKeypoints", disk_cpp)
        self.assertNotIn("m_cfg.maxKeypoints", aliked_cpp)

    def test_one_click_feature_extraction_filters_pixels_below_five_by_default(self):
        header = (ROOT / "src/core/pipeline/SFMService.h").read_text(encoding="utf-8")
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")

        self.assertIn("featureGrayscaleMin", header)
        self.assertRegex(header, r"featureGrayscaleMin\s*=\s*5\.0f\s*/\s*255\.0f")
        self.assertRegex(header, r"featureGrayscaleMax\s*=\s*1\.0f")
        self.assertRegex(source, r"extractorCfg\.grayscaleMin\s*=\s*opts\.featureGrayscaleMin")
        self.assertRegex(source, r"extractorCfg\.grayscaleMax\s*=\s*opts\.featureGrayscaleMax")

    def test_disk_feature_extraction_adapts_after_cuda_oom(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")
        cli = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        self.assertIn("adaptiveFeatureMaxImageDims", source)
        self.assertIn("isCudaOutOfMemoryError", source)
        self.assertIn("extractFeatureWithAdaptiveRetry", source)
        self.assertIn("CUDA OOM", source)
        self.assertIn("extractorCfg->maxImageDim = retryMaxImageDim", source)
        self.assertIn("0 uses auto/adaptive quality preset", cli)


if __name__ == "__main__":
    unittest.main()
