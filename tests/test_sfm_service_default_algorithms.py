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

    def test_sfm_pipeline_can_generate_sift_traditional_matches(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")
        cli = (ROOT / "src/cli/cli_reconstruct_pipeline.cpp").read_text(encoding="utf-8")

        self.assertIn("TraditionalFeatureMatcher.h", source)
        self.assertIn("isTraditionalSiftMatch", source)
        self.assertIn("TraditionalFeatureMatcher::match", source)
        self.assertIn("sfmFeatureNeedsModel", source)
        self.assertIn("shouldRetrySfmWithSiftFallback", source)
        self.assertIn("fallbackOptions.featureAlgorithm = QStringLiteral(\"sift\")", source)
        self.assertIn("fallbackOptions.matchAlgorithm = QStringLiteral(\"sift_bf_l2\")", source)
        self.assertNotIn("fallbackOptions.matchAlgorithm = QStringLiteral(\"sift_flann\")", source)
        self.assertNotIn("fallbackOptions.device = QStringLiteral(\"cpu\")", source)
        self.assertIn("if (featureNeedsModel && extractorModelPath.isEmpty())", source)
        self.assertNotIn("自动补全匹配当前只支持 DISK/ALIKED/SIFT + LightGlue", source)
        self.assertIn("--sfm-feature-algorithm", cli)
        self.assertIn("--sfm-match-algorithm", cli)
        self.assertIn("--sfm-guided-rematching", cli)
        self.assertIn("sfmOptions.featureAlgorithm", cli)
        self.assertIn("sfmOptions.matchAlgorithm", cli)
        self.assertIn("sfmOptions.enableGuidedRematching = sfmGuidedRematching", cli)

    def test_sfm_two_stage_matching_uses_skeleton_then_guided_fill(self):
        header = (ROOT / "src/core/pipeline/SFMService.h").read_text(encoding="utf-8")
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")

        self.assertIn("enableTwoStageMatching", header)
        self.assertIn("skeletonFeatureMaxKeypoints", header)
        self.assertIn("guidedFillMinPointGainRatio", header)
        self.assertIn("guidedFillMaxRmsRegressionRatio", header)
        self.assertIn("const bool use_skeleton_feature_budget", source)
        self.assertIn("two_stage_skeleton_keypoint_limit", source)
        self.assertIn("guided_min_point_gain", source)
        self.assertIn("opts.guidedFillMinPointGainRatio", source)
        self.assertIn("opts.guidedFillMaxRmsRegressionRatio", source)

    def test_cuda_sift_matching_is_split_into_dedicated_module(self):
        header = ROOT / "src/core/feature_match/tradition/CudaSiftMatcher.h"
        source = ROOT / "src/core/feature_match/tradition/CudaSiftMatcher.cpp"
        cmake = (ROOT / "src/core/feature_match/tradition/CMakeLists.txt").read_text(encoding="utf-8")
        matcher = (ROOT / "src/core/feature_match/tradition/TraditionalFeatureMatcher.cpp").read_text(encoding="utf-8")

        self.assertTrue(header.exists())
        self.assertTrue(source.exists())
        self.assertIn("class CudaSiftMatcher", header.read_text(encoding="utf-8"))
        self.assertIn("CudaSiftMatcher.cpp", cmake)
        self.assertIn("#include \"CudaSiftMatcher.h\"", matcher)
        self.assertNotIn("knnMatchL2Cuda", matcher)

    def test_guided_rematching_can_create_missing_pair_from_registered_cameras(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")

        self.assertIn("fundamentalFromRegisteredCameras", source)
        self.assertIn("pairIndexByKey.insert(guidedPairKey(imageA, imageB), pairs->size())", source)
        self.assertIn("newPair.idA = imageA", source)
        self.assertIn("newPair.idB = imageB", source)
        self.assertIn("estimateFundamentalFromExistingMatches(pair", source)
        self.assertIn("fundamentalFromRegisteredCameras(reconstruction.camera(pair.idA)", source)

    def test_guided_rematching_requires_quality_gain_before_accepting_second_pass(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")

        self.assertIn("guided_point_gain", source)
        self.assertIn("guided_min_point_gain", source)
        self.assertIn("guided_rms_acceptable", source)
        self.assertIn("guided_improved", source)
        self.assertIn("guidedSfmResult.baRmsAfter", source)
        self.assertIn("sfmResult.baRmsAfter", source)
        self.assertIn("insufficient gain or worse RMS", source)
        self.assertIn("guided_improved &&", source)
        self.assertIn("guided_rms_acceptable", source)

    def test_sift_fallback_does_not_replace_usable_disk_result(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")

        self.assertIn("primarySfmResultHasProductionSparseCloud", source)
        helper_start = source.index("bool primarySfmResultHasProductionSparseCloud")
        retry_start = source.index("bool shouldRetrySfmWithSiftFallback", helper_start)
        helper_body = source[helper_start:retry_start]
        self.assertIn("numRegisteredImages", helper_body)
        self.assertIn("numPoints3D", helper_body)
        self.assertIn("minimumUsableSparsePointCountForSiftFallback", helper_body)

        retry_start = source.index("bool shouldRetrySfmWithSiftFallback")
        retry_end = source.index("SFMServiceResult SFMService::run", retry_start)
        retry_body = source[retry_start:retry_end]

        self.assertIn("primarySfmResultHasProductionSparseCloud(opts, result)", retry_body)
        self.assertIn("return false;", retry_body)

    def test_sfm_logs_match_result_catalog_diagnostics_without_using_best_variant_for_input(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")

        self.assertIn('#include "MatchResultCatalog.h"', source)
        self.assertIn("xjw::pipeline::MatchResultCatalog", source)
        self.assertIn("匹配缓存目录诊断", source)
        self.assertIn("SfM 默认仍按当前 feature_algorithm + match_algorithm 选择匹配", source)
        self.assertIn("best variant 只是展示/诊断用途", source)

        diagnostic_start = source.index("void logSfmMatchCacheCatalogDiagnostics")
        diagnostic_end = source.index("SFMServiceResult runSingleSfmAttempt", diagnostic_start)
        diagnostic_body = source[diagnostic_start:diagnostic_end]
        self.assertIn("bestVariantIndex", diagnostic_body)

        selection_start = source.index("auto appendCandidatePair")
        selection_end = source.index("if (allPairs.isEmpty())", selection_start)
        selection_body = source[selection_start:selection_end]
        self.assertNotIn("bestVariantIndex", selection_body)
        self.assertIn("findExistingMatchCache(baseA, baseB)", selection_body)
        self.assertIn("findExistingMatchCache(baseB, baseA)", selection_body)

    def test_disk_and_aliked_lightglue_use_dedicated_torchscript_models(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")
        runner_source = (ROOT / "src/core/pipeline/FeatureMatchRunner.cpp").read_text(encoding="utf-8")
        export_script = (ROOT / "scripts/export_lightglue_torchscript.py").read_text(encoding="utf-8")
        start = source.index("QStringList lightGlueModelCandidates")
        end = source.index("QString findScriptFile", start)
        body = source[start:end]

        disk_start = body.index('featureAlgorithm == QStringLiteral("disk")')
        aliked_start = body.index('featureAlgorithm == QStringLiteral("aliked")')
        sift_start = body.index('featureAlgorithm == QStringLiteral("sift")')
        generic_start = body.index('QStringLiteral("lightglue_matcher_%1.torchscript")')

        disk_branch = body[disk_start:aliked_start]
        aliked_branch = body[aliked_start:sift_start]
        sift_branch = body[sift_start:generic_start]
        self.assertIn("lightglue_disk_%1.torchscript", disk_branch)
        self.assertIn("lightglue_aliked_%1.torchscript", aliked_branch)
        self.assertIn("lightglue_sift_%1.torchscript", sift_branch)
        self.assertNotIn("lightglue_disk_%1.pt", disk_branch)
        self.assertNotIn("lightglue_aliked_%1.pt", aliked_branch)
        self.assertNotIn("lightglue_sift_%1.pt", sift_branch)
        self.assertNotIn("lightglue_matcher", disk_branch)
        self.assertNotIn("lightglue_matcher", aliked_branch)
        self.assertNotIn("lightglue_matcher", sift_branch)

        self.assertIn('featureSuffix == QStringLiteral(".sift")', runner_source)
        self.assertIn('lightglueAlgo = QStringLiteral("sift")', runner_source)
        self.assertIn("lightglue_sift_%1.torchscript", runner_source)
        self.assertNotIn("lightglue_sift_%1.pt", runner_source)
        self.assertIn('"sift": 128', export_script)
        self.assertIn('default=parse_features("disk,aliked,sift")', export_script)
        self.assertIn('output_path = output_dir / f"lightglue_{feature}_{device_name}.torchscript"', export_script)
        self.assertNotIn('output_path = output_dir / f"lightglue_{feature}_{device_name}.pt"', export_script)
        self.assertIn("add_vendored_lightglue_to_path()", export_script)

        self.assertIn("ensureLightGlueTorchScriptModel", source)
        self.assertIn("ensureLightGlueTorchScriptModel", runner_source)
        self.assertIn("export_lightglue_torchscript.py", source)
        self.assertIn("export_lightglue_torchscript.py", runner_source)
        self.assertIn("PLASCAN_ALLOW_PYTHON_LIGHTGLUE_FALLBACK", source)
        self.assertIn("PLASCAN_ALLOW_PYTHON_LIGHTGLUE_FALLBACK", runner_source)
        self.assertNotIn(
            "const bool usePythonLightGlue = lgModelPath.isEmpty() && canUsePythonLightGlue;",
            source,
        )
        self.assertIn("runPythonLightGlue", source)
        self.assertIn("run_lightglue.py", source)
        self.assertNotIn("请先运行 scripts/export_lightglue_torchscript.py 导出 TorchScript", source)

    def test_lightglue_sift_torchscript_export_adds_scale_orientation_channels(self):
        export_script = (ROOT / "scripts/export_lightglue_torchscript.py").read_text(encoding="utf-8")

        self.assertIn("self.model.conf.add_scale_ori", export_script)
        self.assertIn("xy0 = normalize_keypoints(kpts0[..., :2], image_size0).clone()", export_script)
        self.assertIn("torch.cat([xy0, scales0, oris0], dim=-1)", export_script)
        self.assertIn("torch.cat([xy1, scales1, oris1], dim=-1)", export_script)
        self.assertIn('feature == "sift"', export_script)
        self.assertIn("kpt_dim = 4", export_script)

    def test_lightglue_matcher_passes_sift_scale_orientation_to_torchscript(self):
        matcher_source = (ROOT / "src/core/feature_match/lightglue/LightGlueMatcher.cpp").read_text(encoding="utf-8")

        self.assertIn('fd.sourceAlgorithm == "sift"', matcher_source)
        self.assertIn("keypoint.size", matcher_source)
        self.assertIn("keypoint.angle", matcher_source)
        self.assertIn("CV_PI", matcher_source)

    def test_sift_lightglue_uses_traditional_fallback_for_weak_pairs(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")
        runner_source = (ROOT / "src/core/pipeline/FeatureMatchRunner.cpp").read_text(encoding="utf-8")

        self.assertIn("runTraditionalSiftFallback", source)
        self.assertIn('featureAlgorithm == QStringLiteral("sift")', source)
        self.assertIn('traditionalCfg.algorithmName = "sift_bf_l2"', source)
        self.assertIn("traditionalCfg.useCuda = useCuda", source)
        self.assertIn("traditional_sift_fallback", source)
        self.assertIn("fallback_raw_match_count", source)
        self.assertIn('fallback_algorithm")] = QStringLiteral("sift_bf_l2")', source)
        self.assertIn("runTraditionalSiftFallback", runner_source)
        self.assertIn('lightglueAlgo == QStringLiteral("sift")', runner_source)
        self.assertIn('traditionalConfig.algorithmName = "sift_bf_l2"', runner_source)
        self.assertIn("traditionalConfig.useCuda = useCuda", runner_source)
        self.assertIn("traditional_sift_fallback", runner_source)
        self.assertIn("fallback_raw_match_count", runner_source)
        self.assertIn('fallback_algorithm"] = QStringLiteral("sift_bf_l2")', runner_source)

    def test_feature_match_runner_uses_suffix_to_select_lightglue_feature_algorithm(self):
        runner_source = (ROOT / "src/core/pipeline/FeatureMatchRunner.cpp").read_text(encoding="utf-8")

        self.assertIn("featureAlgorithmForSuffix", runner_source)
        self.assertIn('featureSuffix == QStringLiteral(".sift")', runner_source)
        self.assertIn('return QStringLiteral("sift")', runner_source)
        self.assertIn("QString featureAlgo = featureAlgorithmForSuffix(featureSuffix", runner_source)
        self.assertIn("优先由当前特征 suffix 决定算法", runner_source)

    def test_disk_lightglue_has_half_turn_retry_for_opposite_flight_strips(self):
        source = (ROOT / "src/core/pipeline/SFMService.cpp").read_text(encoding="utf-8")
        runner_source = (ROOT / "src/core/pipeline/FeatureMatchRunner.cpp").read_text(encoding="utf-8")

        for text in (source, runner_source):
            self.assertIn("withHalfTurnRotatedKeypoints", text)
            self.assertIn("shouldRunLightGlueHalfTurnRetry", text)
            self.assertIn("lightglue_rotation_retry", text)
            self.assertIn("rotation_retry_degrees", text)
            self.assertIn("half_turn_image1", text)

        self.assertIn("featureAlgorithm == QStringLiteral(\"disk\")", source)
        self.assertIn("featureAlgorithm == QStringLiteral(\"aliked\")", source)
        self.assertIn("lightglueAlgo == QStringLiteral(\"disk\")", runner_source)
        self.assertIn("lightglueAlgo == QStringLiteral(\"aliked\")", runner_source)

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
