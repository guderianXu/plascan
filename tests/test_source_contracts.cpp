#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QRegularExpression>
#include <QString>

#include <initializer_list>

namespace
{

QString repoRoot()
{
    return QStringLiteral(PLASCAN_SOURCE_DIR);
}

QString readSourceFile(const QString &relativePath)
{
    const QString path = QDir(repoRoot()).filePath(relativePath);
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(path);
    if (!file.isOpen())
    {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

bool sourceFileExists(const QString &relativePath)
{
    return QFileInfo::exists(QDir(repoRoot()).filePath(relativePath));
}

QString utf8(const char *text)
{
    return QString::fromUtf8(text);
}

void expectContainsAll(const QString &text, std::initializer_list<const char *> needles)
{
    for (const char *needle : needles)
    {
        EXPECT_TRUE(text.contains(utf8(needle))) << needle;
    }
}

void expectNotContainsAll(const QString &text, std::initializer_list<const char *> needles)
{
    for (const char *needle : needles)
    {
        EXPECT_FALSE(text.contains(utf8(needle))) << needle;
    }
}

int countOccurrences(const QString &text, const char *needle)
{
    return text.count(utf8(needle));
}

int indexOfOrFail(const QString &text, const char *needle, int from = 0)
{
    const int index = text.indexOf(utf8(needle), from);
    EXPECT_GE(index, 0) << needle;
    return index;
}

QString sectionBetween(const QString &text, const char *startNeedle, const char *endNeedle, int from = 0)
{
    const int start = indexOfOrFail(text, startNeedle, from);
    if (start < 0)
    {
        return QString();
    }
    const int end = indexOfOrFail(text, endNeedle, start);
    if (end < 0)
    {
        return QString();
    }
    EXPECT_GT(end, start) << endNeedle;
    return text.mid(start, end - start);
}

QString sectionFrom(const QString &text, const char *startNeedle)
{
    const int start = indexOfOrFail(text, startNeedle);
    if (start < 0)
    {
        return QString();
    }
    return text.mid(start);
}

QString functionBody(const QString &source, const char *signature)
{
    const int start = indexOfOrFail(source, signature);
    if (start < 0)
    {
        return QString();
    }

    const int brace = source.indexOf(QLatin1Char('{'), start);
    EXPECT_GE(brace, 0) << signature;
    if (brace < 0)
    {
        return QString();
    }

    int depth = 0;
    for (int index = brace; index < source.size(); ++index)
    {
        const QChar ch = source.at(index);
        if (ch == QLatin1Char('{'))
        {
            ++depth;
        }
        else if (ch == QLatin1Char('}'))
        {
            --depth;
            if (depth == 0)
            {
                return source.mid(start, index - start + 1);
            }
        }
    }

    ADD_FAILURE() << "Function body not closed: " << signature;
    return source.mid(start);
}

void expectMatches(const QString &text, const char *pattern)
{
    const QRegularExpression re(utf8(pattern), QRegularExpression::DotMatchesEverythingOption);
    EXPECT_TRUE(re.match(text).hasMatch()) << pattern;
}

void expectNotMatches(const QString &text, const char *pattern)
{
    const QRegularExpression re(utf8(pattern), QRegularExpression::DotMatchesEverythingOption);
    EXPECT_FALSE(re.match(text).hasMatch()) << pattern;
}

} // namespace

TEST(FeatureExtractionRunnerContractTest, ReadsImagesThroughCommonUnicodePathIo)
{
    const QString source = readSourceFile(QStringLiteral("src/gui/tasks/FeatureExtractionRunner.cpp"));

    expectContainsAll(source, {
        R"(#include "io/PathIO.h")",
        "xjw::common::io::readImage(imagePath",
    });
    expectNotContainsAll(source, {
        "readImageWithQtPath",
        "cv::imread(imagePath.toStdString()",
    });
}

TEST(CommonPathIoContractTest, FallsBackToGdalWhenOpenCvCannotDecodeImage)
{
    const QString source = readSourceFile(QStringLiteral("src/common/io/PathIO.cpp"));
    const QString cmake = readSourceFile(QStringLiteral("src/common/CMakeLists.txt"));

    expectContainsAll(source, {
        "#include <gdal_priv.h>",
        "cv::Mat readImageWithGdal(const QString &path, int flags)",
        "if (!decoded.empty())",
        "return readImageWithGdal(path, flags)",
    });

    expectContainsAll(cmake, {
        "GDAL_INCLUDE_DIRS",
        "${PLASCAN_GDAL_TARGET}",
    });
}

TEST(SfmSourceContractTest, OneClickSfmDefaultsToDiskLightGlue)
{
    const QString header = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.h"));

    expectMatches(header, R"(featureAlgorithm\s*=\s*QStringLiteral\("disk"\))");
    expectMatches(header, R"(matchAlgorithm\s*=\s*QStringLiteral\("lightglue"\))");
}

TEST(SfmSourceContractTest, UsesAlgorithmAwareFeatureAndMatchPipeline)
{
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));

    expectNotContainsAll(source, {
        "ProjectIO::findSpForImage",
        R"(#include "SuperPoint.h")",
        R"(#include "SuperGlueMatcher.h")",
    });
    expectContainsAll(source, {
        "createExtractor(featureAlgorithm.toStdString(), extractorCfg)",
        "LightGlueMatcher",
        "feature_algorithm",
        "match_algorithm",
    });
}

TEST(SfmSourceContractTest, SupportsSiftTraditionalMatchesAndFallback)
{
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));
    const QString cli = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));

    expectContainsAll(source, {
        "TraditionalFeatureMatcher.h",
        "isTraditionalSiftMatch",
        "TraditionalFeatureMatcher::match",
        "sfmFeatureNeedsModel",
        "shouldRetrySfmWithSiftFallback",
        R"(fallbackOptions.featureAlgorithm = QStringLiteral("sift"))",
        R"(fallbackOptions.matchAlgorithm = QStringLiteral("sift_bf_l2"))",
        "if (featureNeedsModel && extractorModelPath.isEmpty())",
    });
    expectNotContainsAll(source, {
        R"(fallbackOptions.matchAlgorithm = QStringLiteral("sift_flann"))",
        R"(fallbackOptions.device = QStringLiteral("cpu"))",
        "自动补全匹配当前只支持 DISK/ALIKED/SIFT + LightGlue",
    });
    expectContainsAll(cli, {
        "--sfm-feature-algorithm",
        "--sfm-match-algorithm",
        "--sfm-guided-rematching",
        "sfmOptions.featureAlgorithm",
        "sfmOptions.matchAlgorithm",
        "sfmOptions.enableGuidedRematching = sfmGuidedRematching",
    });
}

TEST(SfmSourceContractTest, TwoStageMatchingUsesSkeletonThenGuidedFill)
{
    const QString header = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.h"));
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));

    expectContainsAll(header, {
        "enableTwoStageMatching",
        "skeletonFeatureMaxKeypoints",
        "guidedFillMinPointGainRatio",
        "guidedFillMaxRmsRegressionRatio",
    });
    expectContainsAll(source, {
        "const bool use_skeleton_feature_budget",
        "two_stage_skeleton_keypoint_limit",
        "guided_min_point_gain",
        "opts.guidedFillMinPointGainRatio",
        "opts.guidedFillMaxRmsRegressionRatio",
    });
}

TEST(SfmSourceContractTest, CudaSiftMatchingIsDedicatedModule)
{
    const QString header = readSourceFile(QStringLiteral("src/core/feature_match/tradition/CudaSiftMatcher.h"));
    const QString cmake = readSourceFile(QStringLiteral("src/core/feature_match/tradition/CMakeLists.txt"));
    const QString matcher = readSourceFile(QStringLiteral("src/core/feature_match/tradition/TraditionalFeatureMatcher.cpp"));

    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/feature_match/tradition/CudaSiftMatcher.h")));
    EXPECT_TRUE(sourceFileExists(QStringLiteral("src/core/feature_match/tradition/CudaSiftMatcher.cpp")));
    expectContainsAll(header, {"class CudaSiftMatcher"});
    expectContainsAll(cmake, {"CudaSiftMatcher.cpp"});
    expectContainsAll(matcher, {R"(#include "CudaSiftMatcher.h")"});
    expectNotContainsAll(matcher, {"knnMatchL2Cuda"});
}

TEST(SfmSourceContractTest, GuidedRematchingCreatesMissingPairsAndRequiresQualityGain)
{
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));

    expectContainsAll(source, {
        "fundamentalFromRegisteredCameras",
        "pairIndexByKey.insert(guidedPairKey(imageA, imageB), pairs->size())",
        "newPair.idA = imageA",
        "newPair.idB = imageB",
        "estimateFundamentalFromExistingMatches(pair",
        "fundamentalFromRegisteredCameras(reconstruction.camera(pair.idA)",
        "guided_point_gain",
        "guided_min_point_gain",
        "guided_rms_acceptable",
        "guided_improved",
        "guidedSfmResult.baRmsAfter",
        "sfmResult.baRmsAfter",
        "insufficient gain or worse RMS",
        "guided_improved &&",
    });
}

TEST(SfmSourceContractTest, SiftFallbackDoesNotReplaceUsableDiskResult)
{
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));
    const QString helperBody =
        sectionBetween(source, "bool primarySfmResultHasProductionSparseCloud", "bool shouldRetrySfmWithSiftFallback");
    const QString retryBody = sectionBetween(source,
                                             "bool shouldRetrySfmWithSiftFallback",
                                             "AerialTriangulationServiceResult AerialTriangulationService::run");

    expectContainsAll(helperBody, {
        "numRegisteredImages",
        "numPoints3D",
        "minimumUsableSparsePointCountForSiftFallback",
    });
    expectContainsAll(retryBody, {
        "primarySfmResultHasProductionSparseCloud(opts, result)",
        "return false;",
    });
}

TEST(SfmSourceContractTest, MatchResultCatalogDiagnosticsAreNotUsedForInputSelection)
{
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));

    expectContainsAll(source, {
        R"(#include "MatchResultCatalog.h")",
        "xjw::pipeline::MatchResultCatalog",
        "匹配缓存目录诊断",
        "SfM 默认仍按当前 feature_algorithm + match_algorithm 选择匹配",
        "best variant 只是展示/诊断用途",
    });

    const QString diagnosticBody =
        sectionBetween(source, "void logSfmMatchCacheCatalogDiagnostics", "AerialTriangulationServiceResult runSingleSfmAttempt");
    expectContainsAll(diagnosticBody, {"bestVariantIndex"});

    const QString selectionBody = sectionBetween(source, "auto appendCandidatePair", "if (allPairs.isEmpty())");
    expectNotContainsAll(selectionBody, {"bestVariantIndex"});
    expectContainsAll(selectionBody, {
        "findExistingMatchCache(baseA, baseB)",
        "findExistingMatchCache(baseB, baseA)",
    });
}

TEST(SfmSourceContractTest, DiskAlikedAndSiftLightGlueUseDedicatedTorchScriptModels)
{
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));
    const QString runner = readSourceFile(QStringLiteral("src/core/pipeline/FeatureMatchRunner.cpp"));
    const QString exportScript = readSourceFile(QStringLiteral("scripts/export_lightglue_torchscript.py"));
    const QString modelBody = sectionBetween(source, "QStringList lightGlueModelCandidates", "QString findScriptFile");

    const int diskStart = indexOfOrFail(modelBody, R"(featureAlgorithm == QStringLiteral("disk"))");
    const int alikedStart = indexOfOrFail(modelBody, R"(featureAlgorithm == QStringLiteral("aliked"))");
    const int siftStart = indexOfOrFail(modelBody, R"(featureAlgorithm == QStringLiteral("sift"))");
    const int genericStart = indexOfOrFail(modelBody, R"(QStringLiteral("lightglue_matcher_%1.torchscript"))");
    const QString diskBranch = modelBody.mid(diskStart, alikedStart - diskStart);
    const QString alikedBranch = modelBody.mid(alikedStart, siftStart - alikedStart);
    const QString siftBranch = modelBody.mid(siftStart, genericStart - siftStart);

    expectContainsAll(diskBranch, {"lightglue_disk_%1.torchscript"});
    expectContainsAll(alikedBranch, {"lightglue_aliked_%1.torchscript"});
    expectContainsAll(siftBranch, {"lightglue_sift_%1.torchscript"});
    expectNotContainsAll(diskBranch, {"lightglue_disk_%1.pt", "lightglue_matcher"});
    expectNotContainsAll(alikedBranch, {"lightglue_aliked_%1.pt", "lightglue_matcher"});
    expectNotContainsAll(siftBranch, {"lightglue_sift_%1.pt", "lightglue_matcher"});

    expectContainsAll(runner, {
        R"(featureSuffix == QStringLiteral(".sift"))",
        R"(lightglueAlgo = QStringLiteral("sift"))",
        "lightglue_sift_%1.torchscript",
        "ensureLightGlueTorchScriptModel",
        "export_lightglue_torchscript.py",
        "PLASCAN_ALLOW_PYTHON_LIGHTGLUE_FALLBACK",
    });
    expectNotContainsAll(runner, {"lightglue_sift_%1.pt"});
    expectContainsAll(exportScript, {
        R"("sift": 128)",
        R"(default=parse_features("disk,aliked,sift"))",
        R"(output_path = output_dir / f"lightglue_{feature}_{device_name}.torchscript")",
        "add_vendored_lightglue_to_path()",
    });
    expectNotContainsAll(exportScript, {R"(output_path = output_dir / f"lightglue_{feature}_{device_name}.pt")"});
    expectContainsAll(source, {
        "ensureLightGlueTorchScriptModel",
        "export_lightglue_torchscript.py",
        "PLASCAN_ALLOW_PYTHON_LIGHTGLUE_FALLBACK",
        "runPythonLightGlue",
        "run_lightglue.py",
    });
    expectNotContainsAll(source, {
        "const bool usePythonLightGlue = lgModelPath.isEmpty() && canUsePythonLightGlue;",
        "请先运行 scripts/export_lightglue_torchscript.py 导出 TorchScript",
    });
}

TEST(SfmSourceContractTest, LightGlueSiftCarriesScaleAndOrientation)
{
    const QString exportScript = readSourceFile(QStringLiteral("scripts/export_lightglue_torchscript.py"));
    const QString matcher = readSourceFile(QStringLiteral("src/core/feature_match/lightglue/LightGlueMatcher.cpp"));

    expectContainsAll(exportScript, {
        "self.model.conf.add_scale_ori",
        "xy0 = normalize_keypoints(kpts0[..., :2], image_size0).clone()",
        "torch.cat([xy0, scales0, oris0], dim=-1)",
        "torch.cat([xy1, scales1, oris1], dim=-1)",
        R"(feature == "sift")",
        "kpt_dim = 4",
    });
    expectContainsAll(matcher, {
        R"(fd.sourceAlgorithm == "sift")",
        "keypoint.size",
        "keypoint.angle",
        "CV_PI",
    });
}

TEST(SfmSourceContractTest, SiftLightGlueUsesTraditionalFallbackForWeakPairs)
{
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));
    const QString runner = readSourceFile(QStringLiteral("src/core/pipeline/FeatureMatchRunner.cpp"));

    expectContainsAll(source, {
        "runTraditionalSiftFallback",
        R"(featureAlgorithm == QStringLiteral("sift"))",
        R"(traditionalCfg.algorithmName = "sift_bf_l2")",
        "traditionalCfg.useCuda = useCuda",
        "traditional_sift_fallback",
        "fallback_raw_match_count",
        R"(fallback_algorithm")] = QStringLiteral("sift_bf_l2"))",
    });
    expectContainsAll(runner, {
        "runTraditionalSiftFallback",
        R"(lightglueAlgo == QStringLiteral("sift"))",
        R"(traditionalConfig.algorithmName = "sift_bf_l2")",
        "traditionalConfig.useCuda = useCuda",
        "traditional_sift_fallback",
        "fallback_raw_match_count",
        R"(fallback_algorithm"] = QStringLiteral("sift_bf_l2"))",
    });
}

TEST(SfmSourceContractTest, FeatureMatchRunnerUsesSuffixAndHalfTurnRetry)
{
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));
    const QString runner = readSourceFile(QStringLiteral("src/core/pipeline/FeatureMatchRunner.cpp"));

    expectContainsAll(runner, {
        "featureAlgorithmForSuffix",
        R"(featureSuffix == QStringLiteral(".sift"))",
        R"(return QStringLiteral("sift"))",
        "QString featureAlgo = featureAlgorithmForSuffix(featureSuffix",
        "优先由当前特征 suffix 决定算法",
    });

    for (const QString &text : {source, runner})
    {
        expectContainsAll(text, {
            "withHalfTurnRotatedKeypoints",
            "shouldRunLightGlueHalfTurnRetry",
            "lightglue_rotation_retry",
            "rotation_retry_degrees",
            "half_turn_image1",
        });
    }
    expectContainsAll(source, {
        R"(featureAlgorithm == QStringLiteral("disk"))",
        R"(featureAlgorithm == QStringLiteral("aliked"))",
    });
    expectContainsAll(runner, {
        R"(lightglueAlgo == QStringLiteral("disk"))",
        R"(lightglueAlgo == QStringLiteral("aliked"))",
    });
}

TEST(SfmSourceContractTest, DiskFeatureExtractionQualityControlsAndAdaptiveRetry)
{
    const QString header = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.h"));
    const QString source = readSourceFile(QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));
    const QString factory = readSourceFile(QStringLiteral("src/core/feature_extractors/ExtractorFactory.cpp"));
    const QString disk = readSourceFile(QStringLiteral("src/core/feature_extractors/disk/DiskExtractor.cpp"));
    const QString aliked = readSourceFile(QStringLiteral("src/core/feature_extractors/aliked/AlikedExtractor.cpp"));
    const QString cli = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));

    expectContainsAll(header, {
        "featureMaxImageDim",
        "featureGrayscaleMin",
    });
    expectContainsAll(source, {
        "safeDefaultFeatureMaxImageDim",
        "disk_extractor_%1_8192.torchscript",
        "disk_extractor_%1_8192.pt",
        "if (presets.featureMaxImageDim <= 0)",
        "return 0;",
        "resolveFeatureMaxImageDim(opts, presets, featureAlgorithm)",
        "extractorCfg.maxImageDim   = featureMaxImageDim",
        "adaptiveFeatureMaxImageDims",
        "isCudaOutOfMemoryError",
        "extractFeatureWithAdaptiveRetry",
        "CUDA OOM",
        "extractorCfg->maxImageDim = retryMaxImageDim",
    });
    expectNotMatches(source, R"(featureAlgorithm\s*==\s*QStringLiteral\("disk"\).*?return\s+1200)");
    expectContainsAll(factory, {
        "dcfg.scoreThreshold = cfg.detThreshold",
        "acfg.scoreThreshold = cfg.detThreshold",
    });
    expectContainsAll(disk, {"_config.maxKeypoints"});
    expectContainsAll(aliked, {"_config.maxKeypoints"});
    expectNotContainsAll(disk, {"m_cfg.maxKeypoints"});
    expectNotContainsAll(aliked, {"m_cfg.maxKeypoints"});

    expectMatches(header, R"(featureGrayscaleMin\s*=\s*5\.0f\s*/\s*255\.0f)");
    expectMatches(header, R"(featureGrayscaleMax\s*=\s*1\.0f)");
    expectMatches(source, R"(extractorCfg\.grayscaleMin\s*=\s*opts\.featureGrayscaleMin)");
    expectMatches(source, R"(extractorCfg\.grayscaleMax\s*=\s*opts\.featureGrayscaleMax)");
    expectContainsAll(cli, {"0 uses auto/adaptive quality preset"});
}

TEST(GuiAlgorithmAlignmentContractTest, WorkflowQualityMappingCoversFastStandardQuality)
{
    const QString source = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString helper = functionBody(source, "int sfmQualityLevelFromWorkflowQuality");

    expectContainsAll(helper, {
        R"(quality == QStringLiteral("fast"))",
        R"(quality == QStringLiteral("quality"))",
    });
    expectMatches(helper, R"(return\s+0\s*;)");
    expectMatches(helper, R"(return\s+2\s*;)");
    expectMatches(helper, R"(return\s+1\s*;)");
    EXPECT_GE(countOccurrences(source, "opts.quality = sfmQualityLevelFromWorkflowQuality(quality);"), 1);
}

TEST(GuiAlgorithmAlignmentContractTest, DenseWorkflowAndBundleAdjustSettingsReachCore)
{
    const QString terrainManager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    const QString featureRunner = readSourceFile(QStringLiteral("src/gui/tasks/FeatureExtractionRunner.cpp"));
    const QString projectManager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    const QString denseRunner = readSourceFile(QStringLiteral("src/gui/tasks/DenseMatchRunner.cpp"));
    const QString denseService = readSourceFile(QStringLiteral("src/core/dense_match/DenseMatchService.cpp"));
    const QString cli = readSourceFile(QStringLiteral("src/cli/cli_bundle_adjust.cpp"));

    expectContainsAll(terrainManager, {
        "canonicalFeatureAlgorithmFromMatcher",
        "canonicalMatchAlgorithmFromMatcher",
        R"(featureConfig[QStringLiteral("device")] = QStringLiteral("CUDA");)",
        R"(featureConfig[QStringLiteral("max_num_keypoints")])",
        R"(matchConfig[QStringLiteral("use_cuda")] = true;)",
        R"(matchConfig[QStringLiteral("outlier_method")])",
    });
    expectNotContainsAll(terrainManager, {
        R"(featureConfig[QStringLiteral("max_keypoints")])",
        R"(matchConfig[QStringLiteral("device")])",
        R"(matchConfig[QStringLiteral("outlier_filter")])",
    });
    expectContainsAll(featureRunner, {
        "maxKeypointsFromConfig",
        "deviceString.toLower()",
    });

    expectContainsAll(projectManager, {
        R"(extraSettings.value(QStringLiteral("max_point_iterations")).toInt(12))",
        R"(extraSettings.value(QStringLiteral("max_camera_iterations")).toInt(10))",
        R"(extraSettings.value(QStringLiteral("finite_diff_eps")).toDouble(1e-6))",
        R"(extraSettings.value(QStringLiteral("damping")).toDouble(1e-3))",
        R"(extraSettings.value(QStringLiteral("step_tolerance")).toDouble(1e-8))",
        R"(extraSettings.value(QStringLiteral("filter_max_reproj_error")).toDouble(2.5))",
    });
    expectContainsAll(cli, {
        "int maxPointIterations = 12;",
        "int maxCameraIterations = 10;",
        "double stepTolerance = 1e-8;",
    });

    expectContainsAll(denseRunner, {
        R"(settings.value(QStringLiteral("lr_threshold")))",
        "cfg.lrCheckThreshold",
        "cfg.enableLRCheck",
        R"(settings.value(QStringLiteral("median_filter")))",
        "cfg.medianFilterSize",
    });
    expectContainsAll(denseService, {
        "enableLRCheck",
        "checkLRConsistency",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, MvsDepthMetadataAndDialogContractsAreAligned)
{
    const QString denseManager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString cleanup = readSourceFile(QStringLiteral("src/gui/project/services/ProjectResourceCleanupService.cpp"));
    const QString denseCloud = functionBody(readSourceFile(QStringLiteral("src/gui/dialogs/DenseCloudDialog.cpp")),
                                            "QJsonObject DenseCloudDialog::collectSettings");
    const QString depthFusion = functionBody(readSourceFile(QStringLiteral("src/gui/dialogs/DepthFusionDialog.cpp")),
                                             "QJsonObject DepthFusionDialog::collectSettings");
    const QString triangulation = functionBody(readSourceFile(QStringLiteral("src/gui/dialogs/TriangulationDialog.cpp")),
                                               "QJsonObject TriangulationDialog::collectSettings");
    const QString texture = functionBody(readSourceFile(QStringLiteral("src/gui/dialogs/TextureMappingDialog.cpp")),
                                         "QJsonObject TextureMappingDialog::collectSettings");

    expectContainsAll(denseManager, {
        "existingDepthRecordForPath",
        "validMaskStoragePath",
        R"(depthResult[QStringLiteral("valid_mask_path")])",
    });
    expectContainsAll(functionBody(denseManager, "void removeDepthArtifactsForIndices"), {"validMaskStoragePath"});
    expectContainsAll(cleanup, {R"(record.value(QStringLiteral("valid_mask_path")))"});

    expectNotContainsAll(denseCloud, {
        "num_disparities",
        "block_size",
        "uniqueness_ratio",
        "speckle_window_size",
        "use_full_dp",
        "use_wls_filter",
    });
    expectContainsAll(denseCloud, {
        R"(s["resScale"])",
        R"(s["iterations"])",
        R"(s["patchSize"])",
        R"(s["minViews"])",
    });
    expectNotContainsAll(depthFusion, {R"(o["cuda"])"});
    expectNotContainsAll(triangulation, {
        "depthStability",
        "filterMode",
        "maxReprojError",
        "minAngleFilter",
    });
    expectNotContainsAll(texture, {
        "colorCorrection",
        "ghostFilter",
        "seamsMargin",
        "threads",
    });
}

TEST(GuiAlgorithmAlignmentContractTest, MeshDecimationReachesReconstructionConfig)
{
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));

    expectContainsAll(manager, {
        R"(settings.value(QStringLiteral("decimate")).toBool(false))",
        R"(settings.value(QStringLiteral("decimateRatio"))",
        "cfg.simplifyTargetFaces = std::max(",
        "std::lround(cfg.simplifyTargetFaces * decimateRatio)",
    });
}

TEST(ThreeDReconstructionCliContractTest, TargetExistsAndThreeDOnlyModeSkipsTerrain)
{
    const QString cmake = readSourceFile(QStringLiteral("src/cli/CMakeLists.txt"));
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));

    expectContainsAll(cmake, {
        "three_d_reconstruction_cli",
        "cli_reconstruct_pipeline.cpp",
        "PLASCAN_THREE_D_ONLY",
    });
    expectContainsAll(source, {
        "PLASCAN_THREE_D_ONLY",
        "AerialTriangulationServiceOptions",
        "buildDepthGenConfig",
        "buildMeshAndOptionalTexture",
    });
    expectMatches(source, R"(#ifndef\s+PLASCAN_THREE_D_ONLY(?P<body>.*?)#endif)");
    expectMatches(source, R"(#ifdef\s+PLASCAN_THREE_D_ONLY(?P<body>.*?)three_d_reconstruction_output)");
}

TEST(ThreeDReconstructionCliContractTest, StopsBeforeMvsWhenSfmOutputsAreInsufficient)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));

    expectMatches(source,
                  R"(kMinimumSparsePointsForDenseWorkflow\s*=\s*20.*?if\s*\(\s*sfmResult\.numPoints3D\s*<\s*kMinimumSparsePointsForDenseWorkflow\s*\).*?SFM 稀疏点云点数过少.*?return cli::EXIT_ALGO_ERR)");
    expectContainsAll(source, {"kMinimumRegisteredImagesForDenseWorkflow = 2"});
    expectMatches(source,
                  R"(if\s*\(\s*views\.size\(\)\s*<\s*static_cast<size_t>\(kMinimumRegisteredImagesForDenseWorkflow\)\s*\).*?SFM 后可用于 MVS 的相机不足.*?return cli::EXIT_ALGO_ERR)");
    expectMatches(source,
                  R"(filtered_sparse_points.*?if\s*\(\s*sparse\.points\.size\(\)\s*<\s*static_cast<size_t>\(kMinimumSparsePointsForDenseWorkflow\)\s*\).*?预处理后的 SFM 稀疏点云点数过少.*?return cli::EXIT_ALGO_ERR)");
}

TEST(ThreeDReconstructionCliContractTest, DefaultsMatchGuiWorkflowAndSupportsStageControl)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));

    expectContainsAll(source, {
        R"(std::string device = "auto")",
        "registerCliConsoleLogger",
        "Logger::instance()->registerSink",
        "--feature-max-image-dim",
        "sfmOptions.featureMaxImageDim = featureMaxImageDim",
        "const int denseMinViewCount",
        "denseSettings.minViews = denseMinViewCount",
        "denseSettings.minConsistentViews = denseMinViewCount",
        "--mvs-fusion-confidence",
        "double mvsFusionConfidence = 0.50",
        "denseSettings.fusionMinConfidence = mvsFusionConfidence",
        "denseSettings.depthConsistency = 1.0f",
        "depthConfig.runFusion = false",
        "loadFusionFramesFromDepthMaps",
        "DepthMapFusion fusion",
        "DenseRefineSettings refineSettings",
        "refineDenseCloud",
        "dense_cloud_refined.ply",
        "meshRequest.pointCloudPath = refinedCloudPathForModel",
        "bool exportObj = true",
        "--skip-texture",
        "int meshResolution = 224",
        "meshRequest.reconstruction.poissonDepth = 9",
        "meshRequest.reconstruction.simplifyTargetFaces = 28000",
        "--stop-after-sfm",
        "--skip-mvs",
        "--skip-mesh",
    });

    const QString stopGuard = sectionBetween(source, "if (stopAfterSfm || skipMvs)", "kMinimumRegisteredImagesForDenseWorkflow");
    expectContainsAll(stopGuard, {
        R"(report[QStringLiteral("status")] = QStringLiteral("ok"))",
        "mvs",
        "mesh",
        "return cli::EXIT_OK",
    });
    EXPECT_LT(indexOfOrFail(source, "stopAfterSfm || skipMvs"),
              indexOfOrFail(source, "kMinimumSparsePointsForDenseWorkflow"));
}

TEST(ThreeDReconstructionCliContractTest, StreamsFusionAndUsesRegisteredCameras)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
    const QString gui = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));

    const QString fusionFunction =
        sectionBetween(source, "bool fuseDepthMapsStreamingFromDisk", "QJsonObject depthPostprocessStatsToJson");
    expectContainsAll(fusionFunction, {
        "kStreamingFusionCacheFrameLimit",
        "cachedFrames",
        "loadFusionFramesFromDepthMaps",
        "useCachedFrames",
    });
    expectContainsAll(source, {
        "registeredImagePaths",
        "sfmResult.pendingCamUpdates",
    });
    expectNotContainsAll(source, {
        "cameraByImage.insert(item.imagePath, item.camera);",
    });
    expectContainsAll(gui, {
        "registeredImages",
        "result.pendingCamUpdates.keys()",
        "appendAtResult(result.sparseCloudPath",
    });
}

TEST(MvsSchedulerContractTest, FrameWorkerControlsAndSchedulerBasics)
{
    const QString types = readSourceFile(QStringLiteral("src/core/mvs/MvsTypes.h"));
    const QString configH = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.h"));
    const QString configCpp = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.cpp"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));

    expectContainsAll(types, {
        "gpuFrameWorkerCount",
        "cpuFrameWorkerCount",
        "cudaUseParallelSweep",
    });
    expectContainsAll(configH, {
        "gpuFrameWorkers",
        "cpuFrameWorkers",
    });
    expectContainsAll(configCpp, {
        "gpu_frame_workers",
        "cpu_frame_workers",
        "config.gpuFrameWorkerCount",
        "config.cpuFrameWorkerCount",
        "autoGpuFrameWorkers",
        "settings.gpuFrameWorkers",
        "return std::clamp",
    });
    expectNotContainsAll(configCpp, {
        "return 1;",
        "Q_UNUSED(threads)",
    });
    expectContainsAll(scheduler, {
        "gpuFrameWorkers",
        "workerIndex < gpuFrameWorkers",
        "cpuFrameWorkers",
        "DepthFrameArtifactSaveQueue",
        R"(saveQueue.enqueue(i, res, QStringLiteral("初始")))",
        "saveQueue.waitUntilIdle()",
        "saveQueue.stop()",
        "preloadImagesWorkerCount",
        "std::atomic<int> nextImage",
        "preloadWorkers.emplace_back",
        "preloadImages(): workers=",
    });
    expectNotContainsAll(scheduler, {
        "const int cpuWorkers = cudaAvailable ? 0 : 1;",
        R"(if (!saveDepthFrameArtifacts(i, res, QStringLiteral("初始"))))",
    });
    expectContainsAll(header, {"void releasePixelStorage()"});
}

TEST(MvsSchedulerContractTest, DepthCacheAndArtifactLifecycleContracts)
{
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString helper = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    expectContainsAll(scheduler, {
        "shouldRetainAllDepthFramesInMemory",
        "std::atomic<bool> keepDepthFramesInMemory",
        "DepthFrameResult storedResult = res;",
        "if (!keepDepthFramesInMemory.load())",
        "storedResult.releasePixelStorage();",
        "_depthFrames[i] = storedResult;",
        "if (keepDepthFramesInMemory.load() && NV >= 2)",
        "SystemMemorySnapshot",
        "querySystemMemorySnapshot",
        "estimateDepthFrameCacheBytes",
        "retainedDepthMemoryBudgetBytes",
        "_config.maxDepthCacheRamFraction",
        "_config.minFreeRamBytes",
        "深度图内存策略",
        "refreshViewImageDimensionsFromCache",
        "refreshViewImageDimensionsFromCache();",
        "无有效影像尺寸，采用保守流式模式",
        "memoryPressureRequiresStreaming",
        "keepDepthFramesInMemory.compare_exchange_strong",
        "releaseStoredDepthFramePixelStorage",
        "内存压力升高，切换为流式保存",
        "无法继续本次内存融合",
    });
    expectNotContainsAll(scheduler, {
        "_depthFrames[i] = res;",
        R"(QStringLiteral("无有效影像尺寸");
        }
        return true;)",
    });

    expectContainsAll(manager, {"applyImageSizeToMvsView"});
    EXPECT_GE(countOccurrences(manager, "applyImageSizeToMvsView(imgPath, &view)"), 2);
    expectNotContainsAll(manager, {
        "cv::imread(imagePath.toStdString()",
        "view.imageWidth = 0;",
        "view.imageHeight = 0;",
    });
    expectContainsAll(helper, {
        "const bool shouldLoadConfidence",
        "result.frame.confidence.release();",
    });
    expectNotContainsAll(helper, {"cv::Mat filteredDepth = result.frame.depthMap.clone();"});
}

TEST(MvsSchedulerContractTest, QueueCancelPostprocessAndProgressContracts)
{
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    const QString fusionBody =
        sectionBetween(scheduler, "FusionFrameInput DepthMapGenerator::buildFusionFrame", "void DepthMapGenerator::crossCheckDepthConsistency");
    expectContainsAll(fusionBody, {"frame.confidence.release();"});

    const QString fuseBody = sectionBetween(manager,
                                            "void ProjectDenseReconstructionManager::startFuseDepthMapsAsync",
                                            "void ProjectDenseReconstructionManager::startDenseCloudRefineAsync");
    expectContainsAll(fuseBody, {
        "正在加载深度图 %1/%2",
        "流式深度图融合 %1/%2",
        "mvsProgressChanged",
    });

    const QString queueBlock = sectionBetween(scheduler, "class DepthFrameArtifactSaveQueue", "// =============================================================================");
    expectContainsAll(queueBlock, {
        "maxBufferedTasks",
        "m_capacityCv.wait",
        "m_tasks.size() < m_maxBufferedTasks",
        "m_capacityCv.notify_one()",
        "void cancel()",
        "m_dropPendingTasks",
        "m_tasks.clear()",
    });

    const QString postWorkerBlock = sectionBetween(scheduler, "for (std::thread &worker : workers)", "// 释放图像缓存");
    expectContainsAll(postWorkerBlock, {
        "if (_cancelled.load())",
        "saveQueue.cancel()",
    });
    EXPECT_LT(indexOfOrFail(postWorkerBlock, "if (_cancelled.load())"),
              indexOfOrFail(postWorkerBlock, "saveQueue.waitUntilIdle()"));

    const QString crossBlock = sectionBetween(scheduler,
                                              "void DepthMapGenerator::crossCheckDepthConsistency()",
                                              "bool DepthMapGenerator::saveDepthFrameArtifacts");
    expectContainsAll(crossBlock, {"if (_cancelled.load())"});

    const int filteredSavePos = indexOfOrFail(scheduler, R"(saveQueue.enqueue(i, res, QStringLiteral("过滤后")))");
    const QString filteredBlock = scheduler.mid(qMax(0, filteredSavePos - 400), 600);
    expectContainsAll(filteredBlock, {"if (_cancelled.load())"});
}

TEST(MvsSchedulerContractTest, DenseSparsePreloadAndDepthArtifactSavingAreCancelableAndFast)
{
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    EXPECT_GE(countOccurrences(manager, "QPointer<DepthMapGenerator> genSelf(gen)"), 2);
    EXPECT_GE(countOccurrences(manager, "const QString projectPath = _owner ? _owner->currentProjectPath() : QString()"), 2);
    EXPECT_GE(countOccurrences(manager, "QtConcurrent::run([self, genSelf, sparseXyz, views, request, projectPath]()"), 2);
    EXPECT_GE(countOccurrences(manager, "self->_owner->currentProjectPath() != projectPath"), 2);
    EXPECT_GE(countOccurrences(manager, "if (genSelf->isCancelled())"), 2);
    EXPECT_GE(countOccurrences(manager, "return;"), 2);
    EXPECT_GE(countOccurrences(manager, R"(QMetaObject::invokeMethod(genSelf.data(), "finished")"), 2);
    EXPECT_GE(countOccurrences(manager, "QMetaObject::invokeMethod(genSelf.data(), [self, genSelf, sparseCloud, projectPath]()"), 2);
    expectContainsAll(manager, {"Q_ARG(bool, false)"});
    expectNotContainsAll(manager, {
        "QtConcurrent::run([gen, sparseXyz, views, request]()",
        "QtConcurrent::run([genSelf, sparseXyz, views, request]()",
        "gen->setSparseCloud(sparse)",
        R"(QMetaObject::invokeMethod(gen, "start")",
    });

    expectContainsAll(scheduler, {
        "writeFastDepthMatStorage",
        "saveDepthPreviewPng",
        "maxPreviewDimension",
        "保存%1深度产物耗时",
        "xjw::common::io::writeImage(path, colorVis)",
    });
    expectNotContainsAll(scheduler, {
        "FileStorage storage(path, cv::FileStorage::WRITE)",
        ".yml.gz",
    });
}

TEST(MvsSchedulerContractTest, VisibilityAndSourceViewCachesAvoidRedundantWork)
{
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));

    expectContainsAll(header, {
        "FrameMvsCache",
        "prepareFrameCaches",
        "_visibilityBits",
        "_pairCommonCounts",
        "sourceSharedPointIndices",
    });
    expectContainsAll(scheduler, {
        "prepareFrameCaches();",
        "sourceViewIndicesForFrame",
        "visibleSparsePointIndicesForFrame",
        "sourceSharedPointIndices.reserve",
        "sourceSharedPointIndices.push_back",
        "sourceIndicesMatchCachedPrefix",
    });
    expectNotContainsAll(scheduler, {"selectMvsSourceViewIndices(_views, _sparse, refIdx, numSrc)"});

    const QString visibleBlock =
        sectionBetween(scheduler, "std::vector<size_t> DepthMapGenerator::visibleSparsePointIndicesForFrame", "// =============================================================================");
    expectContainsAll(visibleBlock, {"return cache.sourceSharedPointIndices;"});
    EXPECT_LT(indexOfOrFail(visibleBlock, "return cache.sourceSharedPointIndices;"),
              indexOfOrFail(visibleBlock, "std::vector<size_t> filtered;"));

    const QString frameCacheBlock =
        sectionBetween(scheduler, "void DepthMapGenerator::prepareFrameCaches()", "std::vector<int> DepthMapGenerator::sourceViewIndicesForFrame");
    expectContainsAll(frameCacheBlock, {
        "VisibilityCacheShard",
        "visibilityWorkerCount",
        "#pragma omp parallel",
        "#pragma omp for",
        "shard.visiblePointIndicesByView",
        "shard.pairCommonCounts",
        "mergeVisibilityCacheShards",
        "buildVisibilityBitsFromFrameCaches",
        "rankedSourceCandidates",
        "desiredSourceCount",
        "currentSourceScoreCutoff",
        "remaining candidates are sorted by common count",
        "candidate.commonVisiblePoints <= currentSourceScoreCutoff",
    });
    expectNotContainsAll(frameCacheBlock, {
        "_frameCaches[static_cast<size_t>(viewIdx)].visiblePointIndices.push_back(pointIndex);",
    });
    EXPECT_LT(indexOfOrFail(frameCacheBlock, "candidate.commonVisiblePoints <= currentSourceScoreCutoff"),
              indexOfOrFail(frameCacheBlock, "sampledMedianAngle(refIdx, candidate.viewIndex)"));
}

TEST(MvsSchedulerContractTest, SparseHintsUseProjectedSamplesAndPrescaledPatchMatchInputs)
{
    const QString cameraHeader = readSourceFile(QStringLiteral("src/core/camera/PositiveDepthCameraModel.h"));
    const QString cameraSource = readSourceFile(QStringLiteral("src/core/camera/PositiveDepthCameraModel.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString cuda = readSourceFile(QStringLiteral("src/core/mvs/PatchMatchCUDA.cu"));

    expectContainsAll(scheduler, {
        "estimateDepthRangeFromVisiblePoints",
        "buildHintDepthFromVisiblePoints",
        "buildSparseSupportMaskFromVisiblePoints",
        "const std::vector<size_t> visibleSparsePointIndices",
        "visibleSparsePointIndices)",
        "patchMatchWorkSize",
        "collectProjectedSparseDepthSamples",
        "buildHintDepthFromProjectedSamples",
        "const cv::Size coarseHintSize = patchMatchWorkSize(workRefImg, coarseCfg);",
        "const cv::Size fineHintSize = patchMatchWorkSize(workRefImg, fineCfg);",
        "coarseHintSize.width",
        "coarseHintSize.height",
        "fineHintSize.width",
        "fineHintSize.height",
        "workRefSparseSamples",
        "buildHintDepthFromProjectedSamples(refIdx",
        "buildSparseSupportMaskFromProjectedSamples(refIdx",
    });
    expectContainsAll(header, {
        "ProjectedSparseDepthSample",
        "buildSparseSeedDepthFromProjectedSamples",
    });
    expectNotContainsAll(scheduler, {
        "cv::Mat hintDepth = buildHintDepthFromVisiblePoints(refIdx, W, H, visibleSparsePointIndices);",
        "buildHintDepthForCamera(refIdx,\n                                                 coarseHintCam",
        "buildHintDepthForCamera(refIdx,\n                                                         fineHintCam",
    });

    const QString supportBlock =
        sectionBetween(scheduler, "const std::vector<ProjectedSparseDepthSample> workRefSparseSamples", "timing.hintMs = elapsedMs");
    expectContainsAll(supportBlock, {
        "supportMaskCfg",
        "makeFinePatchMatchConfig(pmCfg, useRectified, 0.0f)",
        "const cv::Size supportMaskSize = patchMatchWorkSize(refImg, supportMaskCfg);",
    });
    expectNotContainsAll(supportBlock, {"patchMatchWorkSize(refImg, pmCfg)"});

    const QString projectedBlock =
        sectionBetween(scheduler,
                       "std::vector<ProjectedSparseDepthSample> DepthMapGenerator::collectProjectedSparseDepthSamples",
                       "cv::Mat DepthMapGenerator::buildHintDepthFromProjectedSamples");
    expectContainsAll(scheduler, {"kMaxProjectedDepthQuantileSamples"});
    expectContainsAll(projectedBlock, {
        "depthQuantileSamples",
        "std::nth_element",
        "projectedCandidates",
        "cam.projectWithDepth",
        "candidate.depth",
    });
    expectNotContainsAll(projectedBlock, {
        "std::sort(allZc",
        "allZc.reserve(visiblePointIndices.size())",
        "float Zc = cam.R_cw[6]*pt[0]",
        "cam.project(pt[0]",
    });
    EXPECT_EQ(countOccurrences(projectedBlock, "for (size_t pointIndex : visiblePointIndices)"), 1);

    expectContainsAll(cameraHeader, {"projectWithDepth"});
    expectContainsAll(cameraSource, {
        "PositiveDepthCameraModel::projectWithDepth",
        "return projectWithDepth",
    });

    const QString fineBlock = sectionBetween(scheduler, "cv::resize(coarseDepth, fineHint", "const int hintValid");
    expectContainsAll(fineBlock, {
        "fineSparseSeedHint",
        "fineSparseSeedHint.copyTo(fineHint",
    });
    expectNotContainsAll(fineBlock, {"buildHintDepthFromProjectedSamples(refIdx"});

    const QString hintBody =
        sectionBetween(scheduler, "cv::Mat DepthMapGenerator::buildHintDepthFromProjectedSamples", "cv::Mat DepthMapGenerator::buildSparseSeedDepthFromProjectedSamples");
    expectContainsAll(hintBody, {
        "cv::distanceTransform",
        "DIST_LABEL_PIXEL",
        "maxHintRadius",
    });
    expectNotContainsAll(hintBody, {
        "cv::Mat distMap",
        "INT_MAX / 2",
    });

    expectContainsAll(cuda, {
        "hintDepth->cols == sW && hintDepth->rows == sH",
        "hintDepth->cols == W && hintDepth->rows == H",
        "hintScaled = *hintDepth",
    });
    const QString gpuBody = sectionBetween(cuda, "bool PatchMatchDepthEstimator::estimateGPU", "bool PatchMatchDepthEstimator::estimateCPU");
    expectContainsAll(gpuBody, {
        "const int sW = std::max(1, refW / ds);",
        "const int sH = std::max(1, refH / ds);",
        "getOrUploadGrayImageGpu(refGray, sW, sH, ds",
    });
    expectNotContainsAll(gpuBody, {"cv::resize(refGray, refScaled"});
}

TEST(MvsSchedulerContractTest, DepthPostprocessAndFusionContracts)
{
    const QString types = readSourceFile(QStringLiteral("src/core/mvs/MvsTypes.h"));
    const QString scheduler = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString cli = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
    const QString fusionH = readSourceFile(QStringLiteral("src/core/mvs/DepthMapFusion.h"));
    const QString fusion = readSourceFile(QStringLiteral("src/core/mvs/DepthMapFusion.cpp"));
    const QString depthUtilsH = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.h"));
    const QString depthUtils = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    expectContainsAll(scheduler, {
        "if (seedHintCnt <= 0)",
        "没有可用 hint seed",
        "FrameTiming",
        "耗时统计",
        "source=",
        "patchmatch=",
        "filter=",
        "applySparseSupportPrior(depthMap, confMap, supportMask, refIdx)",
        "稀疏支撑软约束",
        "removeLocalDepthOutliers",
        "postprocessFusionDepthMap(filteredDepth",
    });
    expectNotContainsAll(scheduler, {
        "depthMap.setTo(0, supportMask == 0)",
        "confMap.setTo(0, supportMask == 0)",
    });
    expectContainsAll(types, {
        "DepthPostProcessStats",
        "enableLocalDepthOutlierFilter",
        "localDepthOutlierRelThresh",
        "maxLocalDepthOutlierRemovalRatio",
    });
    expectContainsAll(cli, {
        "postprocessFusionDepthMap",
        "depth_postprocess",
        "local_depth_outlier_removed",
        "speckle_removed",
        "edge_confidence_removed",
        "geom_consistency_removed",
        "rawDepthStoragePath(depthPngPath)",
        "loadDepthMatStorage",
        "--mvs-res-scale",
        "--mvs-iterations",
        "--mvs-confidence",
        "--mvs-gpu-frame-workers",
        "--mvs-cpu-frame-workers",
        "--mvs-max-frames",
        "denseSettings.resScale = mvsResScale;",
        "denseSettings.iterations = mvsIterations;",
        "denseSettings.patchMatchConfidence = mvsConfidence;",
        "denseSettings.fusionMinConfidence = mvsFusionConfidence;",
        "denseSettings.gpuFrameWorkers = mvsGpuFrameWorkers;",
        "denseSettings.cpuFrameWorkers = mvsCpuFrameWorkers;",
        "limitMvsInputsForRegression",
    });
    expectMatches(scheduler,
                  R"(if\s*\(\s*_config\.runFusion\s*&&\s*keepDepthFramesInMemory\.load\(\)\s*&&\s*\(\s*savePreviewPng\s*\|\|\s*saveRawDepth\s*\)\s*\))");

    expectContainsAll(cli, {
        "QJsonArray depthArtifacts;",
        "&xjw::mvs::DepthMapGenerator::depthMapArtifactSaved",
        "depthArtifacts.append(artifact)",
        R"(denseReport[QStringLiteral("depth_maps")] = depthArtifacts;)",
        R"(denseReport[QStringLiteral("mvs_settings")])",
        R"(denseReport[QStringLiteral("mvs_depth_config")])",
        "bool mvsDepthOnly = false;",
        "--mvs-depth-only",
        R"(denseReport[QStringLiteral("status")] = QStringLiteral("depth_only");)",
        R"(report[QStringLiteral("stop_stage")] = QStringLiteral("mvs_depth");)",
        R"(markSkippedStage(QStringLiteral("mvs_fusion"), depthOnlyReason);)",
        R"(markSkippedStage(QStringLiteral("mesh"), depthOnlyReason);)",
        R"(markSkippedStage(QStringLiteral("terrain"), depthOnlyReason);)",
        "const int depthMapCount = static_cast<int>(depthArtifacts.size());",
        R"(std::fprintf(stdout, "depth_maps=%d\n", depthMapCount);)",
        "streamingFusionWindowIndices",
        "fuseDepthMapsStreamingFromDisk",
        "fusionCfg.fuseOnlyFirstFrame = true",
        "流式深度图融合",
        "voxelDownsampleFusedPointsToTarget",
        "kStreamingFusionCacheFrameLimit",
        "const bool useCachedFrames",
        "if (useCachedFrames)",
        "--mvs-fusion-max-image-dim",
        "mvsFusionMaxImageDim",
        "loadFusionFrameFromDepthMap(",
        "fusionMaxImageDim",
        R"(settings[QStringLiteral("fusion_max_image_dim")])",
    });
    expectContainsAll(depthUtilsH, {"int fusionMaxImageDim"});
    expectContainsAll(depthUtils, {
        "scalePositiveDepthCameraModel",
        "downsampleFusionFrameForMaxDimension",
        "cv::resize(frame->depthMap",
        "cv::resize(frame->confidence",
    });
    expectContainsAll(fusionH, {
        "bool  useColor",
        "int   colorCacheCapacity",
        "bool  fuseOnlyFirstFrame",
        "fuseFirstFrameObservationsFast",
    });
    expectContainsAll(fusion, {
        "class ColorImageCache",
        "ColorImageCache colorCache",
        "colorCache.get",
        "const int fusionStartFrame",
        "const int fusionEndFrame",
        "m_frames[frameIdx].imgW",
        "m_frames[frameIdx].imgH",
        "cv::resize(image, image",
        "fuseFirstFrameObservationsFast",
        "使用已过滤深度图快速反投影",
        "_config.fuseOnlyFirstFrame",
        "resolveFusionWorkerCount",
    });
    expectNotContainsAll(fusion, {
        "std::vector<cv::Mat> colorImages(NF)",
        "colorImages[fi] = cv::imread",
    });
    expectContainsAll(manager, {
        "class DepthFrameLruCache",
        "nearestFusionWindowIndices",
        "streamFusionWindowSize",
        "depthFrameCache.get",
        "fusionCfg.fuseOnlyFirstFrame = true",
        "流式深度图融合",
        "fusionMaxImageDim",
        "_fusionMaxImageDim",
        "request.fusionMaxImageDim",
        "_fusionMaxImageDim)",
    });
    expectNotContainsAll(manager, {
        "frames.reserve(storedFrames.size());\n        for (const auto &stored : storedFrames)",
    });
}

TEST(MvsSchedulerContractTest, DepthConsistencyMetadataAndReuseContracts)
{
    const QString header = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.h"));
    const QString generator = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString tree = readSourceFile(QStringLiteral("src/gui/widgets/DataTreeWidget.cpp"));
    const QString depthUtils = readSourceFile(QStringLiteral("src/core/mvs/DepthFrameUtils.cpp"));
    const QString terrain = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    const QString model = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString heatmapH = readSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.h"));
    const QString heatmap = readSourceFile(QStringLiteral("src/gui/widgets/DisparityHeatmapOverlay.cpp"));

    expectContainsAll(header, {"sourceViewIndices", "void depthMapArtifactSaved(QJsonObject artifact)"});
    expectContainsAll(generator, {
        "std::vector<int> sourceIndices",
        "sourceIndices.push_back(si)",
        "result.sourceViewIndices = sourceIndices",
        "consistencySourceIndicesForFrame",
        "const std::vector<int> consistencySources",
        "saveDepthPreviewPng",
        "xjw::common::io::writeImage(path, colorVis)",
        "saveDepthFrameArtifacts",
        "emit depthMapSaved",
        R"(artifact[QStringLiteral("source_images")])",
        R"(artifact[QStringLiteral("source_indices")])",
        R"(artifact[QStringLiteral("valid_mask_path")])",
        R"(artifact[QStringLiteral("device")])",
        R"(artifact[QStringLiteral("elapsed_ms")])",
        "emit depthMapArtifactSaved(artifact)",
        R"(cancelled("深度范围估计后"))",
        R"(cancelled("极线校正后"))",
        R"(cancelled("构建深度 hint 后"))",
        R"(cancelled("粗层 PatchMatch 后"))",
        R"(cancelled("精细层 PatchMatch 后"))",
        R"(cancelled("深度后处理后"))",
    });
    EXPECT_GT(indexOfOrFail(generator, R"(saveQueue.enqueue(i, res, QStringLiteral("过滤后")))"),
              indexOfOrFail(generator, "crossCheckDepthConsistency();"));
    EXPECT_GT(indexOfOrFail(generator, "emit depthMapSaved"), 0);

    expectContainsAll(manager, {
        "makeProjectDepthRecordFromArtifact",
        "&DepthMapGenerator::depthMapArtifactSaved",
        R"(depthResult[QStringLiteral("mvs_output_dir")])",
        "depthFrameArtifactsExist(pngPath)",
        "cameraForImagePath(camMap, imgPath",
        "cameraForImagePath(_cameraMap, _records[index].refImage",
        "if (request.pipelineMode)",
        "genCfg.runFusion = false",
        "shouldStartFusion = success && (continueMissingMode || pipelineMode)",
        "startFuseDepthMapsAsync(settings)",
    });
    expectNotContainsAll(manager, {
        "camMap.value(imgPath)",
        "camMap.value(stored.refImage)",
    });
    expectContainsAll(tree, {
        R"(obj.value(QStringLiteral("device")).toString())",
        R"(depthResultKind(obj) == QStringLiteral("mvs_depth"))",
        "depth_preview_png",
    });
    expectContainsAll(depthUtils, {
        R"(record.value(QStringLiteral("depth_png")))",
        "depthFrameArtifactsExist(frame)",
        "QFileInfo::exists(frame.depthPng)",
        "QFileInfo::exists(frame.rawDepthPath)",
    });

    expectNotContainsAll(terrain, {
        R"(upsertMetaArrayRecordByPath(&metaUpdated, QStringLiteral("depth_map_results"))",
    });
    expectNotContainsAll(model, {"depth_map_results"});
    expectContainsAll(terrain, {"collectLatestStoredDepthFrames"});

    expectContainsAll(heatmapH, {"QImage heatmapImage() const"});
    expectContainsAll(heatmap, {
        "QImage::Format_RGBA8888",
        "alphaRow[col] = validRow[col] ? 255 : 0",
        "if (_showInvalid)",
    });
}

TEST(MvsSchedulerContractTest, ThreeDWorkflowDenseRefineAndDenseSettingsContracts)
{
    const QString workflow = readSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString depthFusionUi = readSourceFile(QStringLiteral("src/gui/dialogs/DepthFusionDialog.ui"));
    const QString threeDUi = readSourceFile(QStringLiteral("src/gui/dialogs/ThreeDReconstructionDialog.ui"));
    const QString configH = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.h"));
    const QString configCpp = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));

    expectContainsAll(workflow, {
        R"(denseSettings[QStringLiteral("keepNormals")] = true;)",
        "registered_image_count",
        "denseMinViewCount",
        R"(denseSettings[QStringLiteral("minConsistentViews")] = denseMinViewCount;)",
        R"(denseSettings[QStringLiteral("minViews")] = denseMinViewCount;)",
        R"(denseSettings[QStringLiteral("minConfidence")] = 0.50;)",
        R"(denseSettings[QStringLiteral("depthConsistency")] = 1.0;)",
        "startThreeDReconstructionDenseRefineStage(settings)",
        "startDenseCloudRefineAsync(refineSettings)",
        "startThreeDReconstructionMeshStage(settings)",
        R"(refineSettings[QStringLiteral("pipeline_mode")] = true;)",
        R"(refineSettings[QStringLiteral("normalsEnabled")] = true;)",
    });
    expectContainsAll(depthFusionUi, {
        "<string>保留法向量</string>",
        "<number>3</number>",
        "<double>0.500000000000000</double>",
        "<double>1.000000000000000</double>",
    });
    expectContainsAll(threeDUi, {
        R"(<widget class="QCheckBox" name="m_exportObjCheck">)",
        "<bool>true</bool>",
    });

    const QString denseSettings = sectionBetween(configH, "struct DenseGenerationSettings", "DenseGenerationSettings denseGenerationSettingsFromJson");
    const QString denseParse =
        sectionBetween(configCpp, "DenseGenerationSettings denseGenerationSettingsFromJson", "xjw::mvs::DepthGenConfig buildDepthGenConfig");
    expectContainsAll(denseSettings, {"plapoint::ProcessingDevice processingDevice"});
    expectContainsAll(denseParse, {"parsed.processingDevice = processingDeviceFromString"});
    EXPECT_EQ(countOccurrences(manager, "SparseCloudPreprocessor pp(request.processingDevice);"), 2);
    expectNotContainsAll(manager, {"SparseCloudPreprocessor pp;"});

    expectContainsAll(configH, {
        "bool geomConsistency",
        "int speckleMinArea",
        "QString qualityProfile",
        "float fusionRelDepthThreshold = 0.03f",
    });
    expectContainsAll(configCpp, {
        R"(settings.value(QStringLiteral("geomConsistency")).toBool(true))",
        R"(settings.value(QStringLiteral("speckleMinArea")).toInt(16))",
        R"(settings.value(QStringLiteral("qualityProfile")))",
        "applyDenseQualityProfile",
        R"(QStringLiteral("fast_preview"))",
        R"(QStringLiteral("standard"))",
        R"(QStringLiteral("high_quality"))",
        "config.patchMatch.geomConsistency = settings.geomConsistency",
        "config.fusion.minSpeckleComponentArea",
        "config.fusion.enableSpeckleFilter",
        "config.fusion.enableAdaptiveConfidenceFilter",
        "parsed->fusionRelDepthThreshold",
        "config.fusion.relDepthThresh = settings.fusionRelDepthThreshold",
    });
    expectContainsAll(manager, {"fusionCfg.maxDepthError = request.fusionRelDepthThreshold"});
    expectNotContainsAll(manager, {"fusionCfg.maxDepthError = 0.05f;"});
}

TEST(MvsSchedulerContractTest, DenseCloudRefineFilteringAndCancelContracts)
{
    const QString preprocessor = readSourceFile(QStringLiteral("src/core/mvs/SparseCloudPreprocessor.cpp"));
    const QString generator = readSourceFile(QStringLiteral("src/core/mvs/DepthMapGenerator.cpp"));
    const QString header = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.h"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const QString configH = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.h"));
    const QString configCpp = readSourceFile(QStringLiteral("src/gui/project/support/ProjectDenseWorkflowConfig.cpp"));

    expectContainsAll(preprocessor, {
        "kMaxMedianSpacingSamples",
        "sampleCount",
        "#pragma omp parallel for",
    });
    expectNotContainsAll(preprocessor, {"distances.reserve(cloud.size());"});
    expectContainsAll(generator, {
        "kMaxInlineDenseFilterPoints",
        "initialCount <= kMaxInlineDenseFilterPoints",
        "跳过内联稠密点云过滤",
    });
    expectNotContainsAll(generator, {"\"SOR-2\""});

    expectContainsAll(manager, {
        "统计离群点移除 (SOR)",
        "半径离群点移除",
    });
    expectNotContainsAll(manager, {
        "离群点二次清理",
        "strictSorReport",
    });

    expectContainsAll(header, {
        "std::shared_ptr<std::atomic_bool> _activeMvsCancelFlag",
        "createActiveMvsCancelFlag",
        "clearActiveMvsCancelFlag",
    });
    const QString cancelBody = sectionFrom(manager, "void ProjectDenseReconstructionManager::cancelMvs()");
    expectContainsAll(cancelBody, {
        "_activeMvsCancelFlag->store(true",
        "gen->requestCancel()",
        "if (!requestedCancel)",
        "emit mvsProgressFinished(false);",
    });
    EXPECT_LT(indexOfOrFail(cancelBody, "if (!requestedCancel)"),
              indexOfOrFail(cancelBody, R"(qDebug() << "[MVS] 已请求取消")"));

    const QString fuseBody = sectionBetween(manager,
                                            "void ProjectDenseReconstructionManager::startFuseDepthMapsAsync",
                                            "void ProjectDenseReconstructionManager::startDenseCloudRefineAsync");
    const QString refineBody = sectionFrom(manager, "void ProjectDenseReconstructionManager::startDenseCloudRefineAsync");
    expectContainsAll(fuseBody, {
        "const auto cancelFlag = createActiveMvsCancelFlag();",
        "cancelFlag",
        "cancelFlag->load",
        "深度图融合已取消",
        "clearActiveMvsCancelFlag(cancelFlag)",
    });
    expectContainsAll(manager, {
        "kMaxDenseRefineFilterInputPoints",
        "preconditionDenseRefineCloudForFilters",
        "点云过大，先进行预降采样",
        "kStreamingDenseRefineMinPoints",
        "shouldUseStreamingDenseRefine",
        "runStreamingDenseCloudRefineCli",
        "dense_cloud_refine_cli",
        "parseBinaryPlyVertexStreamHeader",
        "QProcess process",
        "--terrain-filter-passes",
    });
    expectContainsAll(refineBody, {
        "const auto cancelFlag = createActiveMvsCancelFlag();",
        "cancelFlag",
        "cancelFlag->load",
        "密集点云后处理已取消",
        "clearActiveMvsCancelFlag(cancelFlag)",
        "!precondition.consumedRequestedVoxel",
        "shouldUseStreamingDenseRefine",
        "runStreamingDenseCloudRefineCli",
    });
    EXPECT_LT(indexOfOrFail(refineBody, "preconditionDenseRefineCloudForFilters"),
              indexOfOrFail(refineBody, "sorFilter(cloud"));
    EXPECT_LT(indexOfOrFail(refineBody, "preconditionDenseRefineCloudForFilters"),
              indexOfOrFail(refineBody, "estimateNormals(cloud"));
    EXPECT_LT(indexOfOrFail(refineBody, "runStreamingDenseCloudRefineCli"),
              indexOfOrFail(refineBody, "readPointCloudPly(inputPly"));

    expectContainsAll(configH, {
        "int terrainSpikeGridResolution = 260;",
        "int terrainSpikeMinCellPoints = 32;",
        "double terrainSpikeMinHeightThreshold = 0.25;",
        "double terrainSpikeMadMultiplier = 3.0;",
        "bool terrainLocalPlaneFilterEnabled = true;",
        "int terrainLocalPlaneMinPoints = 12;",
        "double terrainLocalPlaneMinResidualThreshold = 0.12;",
        "double terrainLocalPlaneMadMultiplier = 4.0;",
        "int terrainFilterPasses = 2;",
    });
    expectContainsAll(configCpp, {
        R"(settings.value(QStringLiteral("terrainSpikeGridResolution")).toInt(260))",
        R"(settings.value(QStringLiteral("terrainSpikeMinCellPoints")).toInt(32))",
        R"(settings.value(QStringLiteral("terrainSpikeMinHeightThreshold")).toDouble(0.25))",
        R"(settings.value(QStringLiteral("terrainSpikeMadMultiplier")).toDouble(3.0))",
        R"(settings.value(QStringLiteral("terrainLocalPlaneFilterEnabled")).toBool(true))",
        R"(settings.value(QStringLiteral("terrainLocalPlaneMinPoints")).toInt(12))",
        R"(settings.value(QStringLiteral("terrainLocalPlaneMinResidualThreshold")).toDouble(0.12))",
        R"(settings.value(QStringLiteral("terrainLocalPlaneMadMultiplier")).toDouble(4.0))",
        R"(settings.value(QStringLiteral("terrainFilterPasses")).toInt(2))",
    });
    expectContainsAll(manager, {
        "options.localPlaneFilterEnabled = request.terrainLocalPlaneFilterEnabled",
        "options.localPlaneMinPoints = request.terrainLocalPlaneMinPoints",
        "options.localPlaneMinResidualThreshold",
        "options.localPlaneMadMultiplier",
        "local_plane_removed_points",
    });
}
