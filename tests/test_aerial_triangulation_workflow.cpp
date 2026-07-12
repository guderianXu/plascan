#include "AerialTriangulationWorkflow.h"
#include "matchphototask/task/MatchPhotosTask.h"

#include <gtest/gtest.h>

namespace
{

xjw::gui::AerialTriangulationWorkflowOptions makeBaseOptions()
{
    xjw::gui::AerialTriangulationWorkflowOptions options;
    options.images = {
        QStringLiteral("E:/data/img_001.jpg"),
        QStringLiteral("E:/data/img_002.jpg"),
        QStringLiteral("E:/data/img_003.jpg")
    };
    options.cameraPaths = {
        QStringLiteral("E:/data/cameras/img_001.tsai"),
        QStringLiteral("E:/data/cameras/img_002.tsai"),
        QStringLiteral("E:/data/cameras/img_003.tsai")
    };
    options.projectPath = QStringLiteral("E:/project/test.plascan");
    options.outputDir = QStringLiteral("E:/project/assets/aerial_triangulation");
    options.projectMeta.insert(QStringLiteral("name"), QStringLiteral("unit"));
    return options;
}

} // namespace

TEST(AerialTriangulationServiceOptionsTest, FullAttemptWritesOutputsByDefault)
{
    const xjw::gui::AerialTriangulationServiceOptions options;

    EXPECT_TRUE(options.writeSfmOutputs);
    EXPECT_EQ(options.sfmExecutionProfile, xjw::SfmExecutionProfile::FullRefinement);
    EXPECT_FALSE(options.useInitialPairHint);
}

TEST(AerialTriangulationServiceOptionsTest, CoarseAttemptCanCarryExplicitInitialPairWithoutOutputs)
{
    xjw::gui::AerialTriangulationServiceOptions options;
    options.writeSfmOutputs = false;
    options.sfmExecutionProfile = xjw::SfmExecutionProfile::CoarseEvaluation;
    options.useInitialPairHint = true;
    options.initialImageId1 = 3;
    options.initialImageId2 = 7;

    EXPECT_FALSE(options.writeSfmOutputs);
    EXPECT_TRUE(options.useInitialPairHint);
    EXPECT_EQ(options.initialImageId1, 3u);
    EXPECT_EQ(options.initialImageId2, 7u);
}

TEST(AerialTriangulationWorkflowCoreTest, HighestQualityMapsToFullResolutionSfmAndTieLimits)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("highest");
    options.keypointLimit = 40000;
    options.tiepointLimit = 4000;
    options.featureAlgorithm = QStringLiteral("sift");
    options.matchAlgorithm = QStringLiteral("lightglue");
    options.guidedImageMatching = true;

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.images, options.images);
    EXPECT_EQ(resolved.serviceOptions.cameraPaths, options.cameraPaths);
    EXPECT_EQ(resolved.serviceOptions.plascanPath, options.projectPath);
    EXPECT_EQ(resolved.serviceOptions.projectMeta, options.projectMeta);
    EXPECT_EQ(resolved.serviceOptions.outputDir,
              QStringLiteral("E:/project/assets/aerial_triangulation/sfm_sparse"));
    EXPECT_EQ(resolved.serviceOptions.quality, 3);
    EXPECT_EQ(resolved.serviceOptions.featureMaxImageDim, -1);
    EXPECT_EQ(resolved.serviceOptions.skeletonFeatureMaxKeypoints, 40000);
    EXPECT_EQ(resolved.serviceOptions.maxTiePointsPerImage, 4000);
    EXPECT_EQ(resolved.serviceOptions.maxTiePointsPerGridCell, 500);
    EXPECT_EQ(resolved.serviceOptions.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(resolved.serviceOptions.matchAlgorithm, QStringLiteral("lightglue"));
    EXPECT_TRUE(resolved.serviceOptions.enableGuidedRematching);
    EXPECT_TRUE(resolved.serviceOptions.enableTwoStageMatching);
    EXPECT_TRUE(resolved.resolvedSettings.value(QStringLiteral("adaptive_known_pose_soft_prior")).toBool());
}

TEST(AerialTriangulationWorkflowCoreTest, TiePointAndSfmStagesUseTheSameCacheDirectories)
{
    auto options = makeBaseOptions();
    options.assetsDir = QStringLiteral("E:/run/assets");
    options.featureDir = QStringLiteral("E:/run/cache/features");
    options.matchDir = QStringLiteral("E:/run/cache/matches");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.tiePointContext.workingDirectory, options.assetsDir);
    EXPECT_EQ(resolved.tiePointContext.featureDirectory, options.featureDir);
    EXPECT_EQ(resolved.tiePointContext.matchDirectory, options.matchDir);
    EXPECT_EQ(resolved.serviceOptions.assetsDir, resolved.tiePointContext.workingDirectory);
    EXPECT_EQ(resolved.serviceOptions.featureDir, resolved.tiePointContext.featureDirectory);
    EXPECT_EQ(resolved.serviceOptions.matchDir, resolved.tiePointContext.matchDirectory);
}

TEST(AerialTriangulationWorkflowCoreTest, LowQualityUsesReducedImageScaleAndConservativeBudgets)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("low");
    options.keypointLimit = 40000;
    options.tiepointLimit = 4000;

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.quality, 0);
    EXPECT_EQ(resolved.serviceOptions.featureMaxImageDim, 2048);
    EXPECT_EQ(resolved.serviceOptions.skeletonFeatureMaxKeypoints, 10000);
    EXPECT_EQ(resolved.serviceOptions.tiePointFeatureMaxKeypoints, 40000);
    EXPECT_TRUE(resolved.serviceOptions.useTiePointDenseSift);
    EXPECT_EQ(resolved.serviceOptions.maxTiePointsPerImage, 1000);
    EXPECT_EQ(resolved.serviceOptions.maxTiePointsPerGridCell, 125);
    EXPECT_FALSE(resolved.serviceOptions.enableGuidedRematching);
}

TEST(AerialTriangulationWorkflowCoreTest, HighestQualityDoesNotOverrideDisabledGuidedMatching)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("highest");
    options.guidedImageMatching = false;

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_FALSE(resolved.serviceOptions.enableGuidedRematching);
    EXPECT_FALSE(resolved.tiePointOptions.enableGuidedMatching);
}

TEST(AerialTriangulationWorkflowCoreTest, LowQualityKeepsRequestedTiePointFeatureBudget)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("low");
    options.keypointLimit = 40000;
    options.tiepointLimit = 4000;
    options.featureAlgorithm = QStringLiteral("sift");
    options.matchAlgorithm = QStringLiteral("lightglue");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.quality, 0);
    EXPECT_EQ(resolved.serviceOptions.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(resolved.serviceOptions.matchAlgorithm, QStringLiteral("lightglue"));
    EXPECT_EQ(resolved.serviceOptions.tiePointFeatureMaxKeypoints, 40000);
    EXPECT_EQ(resolved.serviceOptions.tiePointKeypointLimitPerMegapixel, 0);
    EXPECT_TRUE(resolved.serviceOptions.useTiePointDenseSift);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("resolved_keypoint_budget")).toInt(), 40000);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("skeleton_keypoint_budget")).toInt(), 10000);
}

TEST(AerialTriangulationWorkflowCoreTest, GuidedMatchingUsesPerMegapixelTiePointSignature)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("highest");
    options.keypointLimit = 40000;
    options.guidedImageMatching = true;
    options.featureAlgorithm = QStringLiteral("sift");
    options.matchAlgorithm = QStringLiteral("lightglue");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.tiePointFeatureMaxKeypoints, 0);
    EXPECT_EQ(resolved.serviceOptions.tiePointKeypointLimitPerMegapixel, 40000);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("resolved_keypoint_budget")).toInt(), 0);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("resolved_keypoint_limit_per_megapixel")).toInt(), 40000);
}

TEST(AerialTriangulationWorkflowCoreTest, DefaultsToSiftLightGlueForFormalSfmCacheCompatibility)
{
    auto options = makeBaseOptions();

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(resolved.serviceOptions.matchAlgorithm, QStringLiteral("lightglue"));
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("feature_algorithm")).toString(),
              QStringLiteral("sift"));
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("match_algorithm")).toString(),
              QStringLiteral("lightglue"));
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("match_pipeline")).toString(),
              QStringLiteral("sift-lightglue"));
}

TEST(AerialTriangulationWorkflowCoreTest, AdaptiveCameraModelFittingPropagatesToService)
{
    auto options = makeBaseOptions();
    options.cameraPaths.clear();
    options.adaptiveCameraModelFitting = true;

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.serviceOptions.adaptiveCameraModelFitting);
    EXPECT_TRUE(resolved.resolvedSettings.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
}

TEST(AerialTriangulationWorkflowCoreTest, ResetAlignmentIgnoresProjectCameraMetadata)
{
    auto options = makeBaseOptions();
    options.cameraPaths.clear();
    options.resetAlignment = true;

    const auto resetResolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_FALSE(resetResolved.serviceOptions.useProjectMetaCameras);
    EXPECT_FALSE(resetResolved.resolvedSettings.value(QStringLiteral("use_project_camera_metadata")).toBool());

    options.resetAlignment = false;
    const auto reuseResolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(reuseResolved.serviceOptions.useProjectMetaCameras);
    EXPECT_TRUE(reuseResolved.resolvedSettings.value(QStringLiteral("use_project_camera_metadata")).toBool());
}

TEST(AerialTriangulationWorkflowCoreTest, NoCameraDefaultsToAdaptiveCameraModelFitting)
{
    auto options = makeBaseOptions();
    options.cameraPaths.clear();

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.serviceOptions.adaptiveCameraModelFitting);
    EXPECT_TRUE(resolved.resolvedSettings.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
}

TEST(AerialTriangulationWorkflowCoreTest, ReferenceSequencePreselectionMapsToSequenceWindowPairs)
{
    auto options = makeBaseOptions();
    options.genericPreselection = false;
    options.referencePreselection = true;
    options.referenceMode = QStringLiteral("sequence");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.serviceOptions.autoRestrictKnownCameraPairs);
    EXPECT_FALSE(resolved.serviceOptions.useKnownCameraOverlapPairs);
    EXPECT_EQ(resolved.serviceOptions.knownCameraPairWindow, 6);
    EXPECT_EQ(resolved.serviceOptions.knownCameraSpatialNeighborCount, 0);
    EXPECT_TRUE(resolved.serviceOptions.knownCameraSequenceLoopClosure);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("pair_planning_mode")).toString(),
              QStringLiteral("sequence"));
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("sequence_pair_window")).toInt(), 6);
    EXPECT_TRUE(resolved.resolvedSettings.value(QStringLiteral("sequence_loop_closure")).toBool());
}

TEST(AerialTriangulationWorkflowCoreTest, NoCameraSequencePreselectionForcesSequencePairRestriction)
{
    auto options = makeBaseOptions();
    options.cameraPaths.clear();
    options.genericPreselection = false;
    options.referencePreselection = true;
    options.referenceMode = QStringLiteral("sequence");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.knownCameraAllPairsMaxImages, 0);
    EXPECT_TRUE(resolved.serviceOptions.knownCameraSequenceLoopClosure);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("pair_planning_mode")).toString(),
              QStringLiteral("sequence"));
}

TEST(AerialTriangulationWorkflowCoreTest, HighestQualitySequencePreselectionUsesWiderWindow)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("highest");
    options.genericPreselection = false;
    options.referencePreselection = true;
    options.referenceMode = QStringLiteral("sequence");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.knownCameraPairWindow, 8);
    EXPECT_TRUE(resolved.serviceOptions.knownCameraSequenceLoopClosure);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("sequence_pair_window")).toInt(), 8);
}

TEST(AerialTriangulationWorkflowCoreTest, MatchPipelineOverridesFeatureAndMatcherNames)
{
    auto options = makeBaseOptions();
    options.featureAlgorithm = QStringLiteral("disk");
    options.matchAlgorithm = QStringLiteral("lightglue");
    options.matchPipeline = QStringLiteral("sift-bf-l2");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(resolved.serviceOptions.matchAlgorithm, QStringLiteral("bf_l2"));
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("match_pipeline")).toString(),
              QStringLiteral("sift-bf-l2"));
}

TEST(AerialTriangulationWorkflowCoreTest, ResetAlignmentForcesUnifiedTiePointPreparation)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("highest");
    options.referencePreselection = true;
    options.referenceMode = QStringLiteral("sequence");
    options.resetAlignment = true;
    options.autoGenerateMissingMatches = false;
    options.maskApplyMode = QStringLiteral("keypoints");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.prepareTiePoints);
    EXPECT_TRUE(resolved.forceRebuildTiePoints);
    EXPECT_FALSE(resolved.tiePointOptions.reuseExistingFeatures);
    EXPECT_EQ(resolved.tiePointOptions.pairPolicy.mode,
              xjw::matchphotos::PairSelectionMode::Sequence);
    EXPECT_EQ(resolved.tiePointOptions.pairPolicy.sequenceWindow, 8);
    EXPECT_EQ(resolved.tiePointOptions.maskApplyMode, QStringLiteral("keypoints"));
    EXPECT_FALSE(resolved.serviceOptions.autoGenerateMissingMatches);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("tie_point_preparation")).toString(),
              QStringLiteral("force_rebuild"));
}

TEST(AerialTriangulationWorkflowCoreTest, TiePointPreparationRunsBeforeSfmAndMergesOutputs)
{
    auto options = makeBaseOptions();
    options.resetAlignment = true;
    QStringList events;
    int emittedMatchCount = 0;
    options.pairMatchedFn = [&](const QString &image0,
                                const QString &image1,
                                const QString &matchPath,
                                int matchCount)
    {
        EXPECT_EQ(image0, QStringLiteral("E:/data/img_001.jpg"));
        EXPECT_EQ(image1, QStringLiteral("E:/data/img_002.jpg"));
        EXPECT_TRUE(matchPath.endsWith(QStringLiteral("lightglue.match")));
        EXPECT_EQ(matchCount, 128);
        ++emittedMatchCount;
    };

    const auto result = xjw::gui::AerialTriangulationWorkflow::run(
        options,
        [&](const xjw::gui::AerialTriangulationServiceOptions &serviceOptions)
        {
            events.append(QStringLiteral("sfm"));
            EXPECT_FALSE(serviceOptions.autoGenerateMissingMatches);
            xjw::gui::AerialTriangulationServiceResult serviceResult;
            serviceResult.success = true;
            serviceResult.numRegisteredImages = 3;
            return serviceResult;
        },
        [&](const xjw::matchphotos::MatchPhotosOptions &,
            const xjw::matchphotos::MatchPhotosContext &)
        {
            events.append(QStringLiteral("tie_points"));
            xjw::matchphotos::MatchPhotosResult tieResult;
            tieResult.success = true;
            tieResult.tiePointPath = QStringLiteral("E:/project/assets/tie_points/tie_points.json");
            tieResult.trackCount = 42;
            tieResult.features.push_back({QStringLiteral("E:/data/img_001.jpg"),
                                          QStringLiteral("E:/project/assets/ip/img_001.sift"),
                                          800,
                                          QJsonObject{}});
            xjw::matchphotos::MatchPhotosMatchRecord match;
            match.image0Path = QStringLiteral("E:/data/img_001.jpg");
            match.image1Path = QStringLiteral("E:/data/img_002.jpg");
            match.matchPath = QStringLiteral("E:/project/assets/matches/img_001__img_002_lightglue.match");
            match.sidecarPath = match.matchPath + QStringLiteral(".json");
            match.matchCount = 128;
            match.passedGeometry = true;
            tieResult.matches.push_back(match);
            return tieResult;
        });

    EXPECT_EQ(events, QStringList({QStringLiteral("tie_points"), QStringLiteral("sfm")}));
    EXPECT_EQ(emittedMatchCount, 1);
    ASSERT_TRUE(result.serviceResult.success);
    EXPECT_TRUE(result.tiePointPreparationExecuted);
    EXPECT_EQ(result.tiePointResult.trackCount, 42);
    ASSERT_EQ(result.serviceResult.newFeatureFiles.size(), 1);
    ASSERT_EQ(result.serviceResult.newMatchFiles.size(), 1);
    EXPECT_EQ(result.serviceResult.newMatchFiles.front()
                  .settings.value(QStringLiteral("tie_point_path")).toString(),
              result.tiePointResult.tiePointPath);
    EXPECT_EQ(result.serviceResult.newMatchFiles.front()
                  .settings.value(QStringLiteral("track_count")).toInt(),
              42);
    EXPECT_EQ(result.serviceResult.newMatchFiles.front()
                  .settings.value(QStringLiteral("track_summary")).toObject(),
              result.tiePointResult.trackSummary);
    EXPECT_EQ(result.serviceResult.resultRecordExtra.value(QStringLiteral("tie_point_path")).toString(),
              result.tiePointResult.tiePointPath);
}

TEST(AerialTriangulationWorkflowCoreTest, TiePointPreparationFailureSkipsSfm)
{
    auto options = makeBaseOptions();
    options.resetAlignment = true;
    bool sfmCalled = false;

    const auto result = xjw::gui::AerialTriangulationWorkflow::run(
        options,
        [&](const xjw::gui::AerialTriangulationServiceOptions &)
        {
            sfmCalled = true;
            return xjw::gui::AerialTriangulationServiceResult{};
        },
        [](const xjw::matchphotos::MatchPhotosOptions &,
           const xjw::matchphotos::MatchPhotosContext &)
        {
            xjw::matchphotos::MatchPhotosResult tieResult;
            tieResult.success = false;
            tieResult.errorMessage = QStringLiteral("geometry verification failed");
            return tieResult;
        });

    EXPECT_FALSE(sfmCalled);
    EXPECT_FALSE(result.serviceResult.success);
    EXPECT_TRUE(result.tiePointPreparationExecuted);
    EXPECT_TRUE(result.serviceResult.errorMessage.contains(QStringLiteral("geometry verification failed")));
}
