#include "workflow/AerialTriangulationWorkflow.h"
#include "search/SfmSearchPolicy.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <cmath>

namespace
{

xjw::aerial_triangulation::AerialTriangulationOptions makeBaseOptions(
    const QString &root)
{
    xjw::aerial_triangulation::AerialTriangulationOptions options;
    options.images = {QDir(root).filePath(QStringLiteral("a.png")),
                      QDir(root).filePath(QStringLiteral("b.png")),
                      QDir(root).filePath(QStringLiteral("c.png"))};
    options.projectPath = QDir(root).filePath(QStringLiteral("project.plascan"));
    options.outputDir = QDir(root).filePath(QStringLiteral("assets/aerial_triangulation"));
    options.assetsDir = QDir(root).filePath(QStringLiteral("assets"));
    options.matchingAlgorithmId = QStringLiteral("sift_lightglue");
    return options;
}

xjw::FramePinholeCamera makeInwardRingCamera(double angle)
{
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(1200.0, 1200.0, 320.0, 240.0);
    camera.setPose({{sine, 0.0, -cosine,
                     -cosine, 0.0, -sine,
                     0.0, 1.0, 0.0}},
                   {{5.0 * cosine, 5.0 * sine, 0.0}});
    return camera;
}

} // namespace

TEST(AerialTriangulationWorkflowTest, ResolvesFrontendAndPreparedSfmSettingsSeparately)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    auto options = makeBaseOptions(tempDir.path());
    options.quality = QStringLiteral("highest");
    options.keypointLimit = 40000;
    options.tiepointLimit = 4000;
    options.guidedImageMatching = true;
    options.useInitialPairHint = true;
    options.initialImageId1 = 0;
    options.initialImageId2 = 2;

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.pipelineInput.images, options.images);
    EXPECT_EQ(resolved.pipelineInput.projectPath, options.projectPath);
    EXPECT_EQ(resolved.pipelineInput.quality, 3);
    EXPECT_EQ(resolved.pipelineInput.outputDir,
              QDir(options.outputDir).filePath(QStringLiteral("sfm_sparse")));
    EXPECT_TRUE(resolved.pipelineInput.useInitialPairHint);
    EXPECT_EQ(resolved.pipelineInput.initialImageId1, 0u);
    EXPECT_EQ(resolved.pipelineInput.initialImageId2, 2u);
    EXPECT_TRUE(resolved.tiePointOptions.enableGuidedMatching);
    EXPECT_EQ(resolved.tiePointOptions.maxKeypoints, 40000);
    EXPECT_EQ(resolved.tiePointOptions.keypointLimitPerMegapixel, 0);
    EXPECT_EQ(resolved.tiePointOptions.maxTiePointsPerImage, 4000);
    EXPECT_EQ(resolved.tiePointOptions.maxTiePointsPerGridCell, 62);
    EXPECT_EQ(resolved.tiePointContext.workingDirectory, options.assetsDir);
}

TEST(AerialTriangulationWorkflowTest, ResolvesZeroThreadRequestAutomatically)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.threads = 0;

    const auto automatic =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);
    EXPECT_GE(automatic.pipelineInput.threads, 1);

    options.threads = 13;
    const auto explicitBudget =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);
    EXPECT_EQ(explicitBudget.pipelineInput.threads,
              xjw::aerial_triangulation::resolveSfmThreadBudget(13));
}

TEST(AerialTriangulationWorkflowTest, ExplicitRuntimeLimitsOverrideQualityDefaults)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.quality = QStringLiteral("lowest");
    options.featureMaxImageDim = 6144;
    options.lightGlueTensorRtEnginePath = QStringLiteral("D:/models/lightglue.engine");
    options.lomaRKeypointBudget = 2048;
    options.cudaParallelPairs = 3;
    options.cudaDevice = 1;
    options.featurePrefetchDepth = 4;
    options.matchThreshold = 0.22f;
    options.geometryReprojThreshold = 2.25;
    options.geometryMinInliers = 31;
    options.geometryMaxIterations = 24000;
    options.tiePointGridColumns = 10;
    options.tiePointGridRows = 6;
    options.maxTiePointsPerGridCell = 77;
    options.stationaryTiePointMaxPixelMotion = 1.75f;

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.tiePointOptions.maxImageDim, 6144);
    EXPECT_EQ(resolved.tiePointOptions.lightGlueTensorRtEnginePath,
              QStringLiteral("D:/models/lightglue.engine"));
    EXPECT_EQ(resolved.tiePointOptions.lomaRKeypointBudget, 2048);
    EXPECT_EQ(resolved.tiePointOptions.cudaParallelPairs, 3);
    EXPECT_EQ(resolved.tiePointOptions.cudaDevice, 1);
    EXPECT_EQ(resolved.tiePointOptions.featurePrefetchDepth, 4);
    EXPECT_FLOAT_EQ(resolved.tiePointOptions.matchThreshold, 0.22f);
    EXPECT_DOUBLE_EQ(resolved.tiePointOptions.geometryReprojThreshold, 2.25);
    EXPECT_EQ(resolved.tiePointOptions.geometryMinInliers, 31);
    EXPECT_EQ(resolved.tiePointOptions.geometryMaxIterations, 24000);
    EXPECT_EQ(resolved.tiePointOptions.tiePointGridColumns, 10);
    EXPECT_EQ(resolved.tiePointOptions.tiePointGridRows, 6);
    EXPECT_EQ(resolved.tiePointOptions.maxTiePointsPerGridCell, 77);
    EXPECT_FLOAT_EQ(resolved.tiePointOptions.stationaryTiePointMaxPixelMotion, 1.75f);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("cuda_parallel_pairs_requested")).toInt(), 3);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("cuda_parallel_pairs_effective")).toInt(), 0);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("geometry_max_iterations")).toInt(),
              24000);
    EXPECT_EQ(resolved.pipelineInput.quality, 0);
}

TEST(AerialTriangulationWorkflowTest, RecordsEffectiveCudaPairConcurrencyAfterMatching)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.cudaParallelPairs = 3;

    const auto result = xjw::aerial_triangulation::AerialTriangulationWorkflow::run(
        options,
        [](const xjw::aerial_triangulation::PreparedAerialTriangulationInput &)
        {
            xjw::aerial_triangulation::AerialTriangulationReconstructionResult reconstruction;
            reconstruction.success = true;
            return reconstruction;
        },
        [](const xjw::matchphotos::MatchPhotosOptions &actualOptions,
           const xjw::matchphotos::MatchPhotosContext &)
        {
            EXPECT_EQ(actualOptions.cudaParallelPairs, 3);
            xjw::matchphotos::MatchPhotosResult tiePoints;
            tiePoints.success = true;
            xjw::matchphotos::MatchPhotosMatchRecord match;
            match.settings.insert(
                QStringLiteral("cuda_parallel_pairs_effective"), 2);
            tiePoints.matches.push_back(match);
            return tiePoints;
        });

    EXPECT_EQ(result.config.resolvedSettings.value(
                  QStringLiteral("cuda_parallel_pairs_effective")).toInt(),
              2);
}

TEST(AerialTriangulationWorkflowTest, SequenceModeOnlyChangesPairSelectionPolicy)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.referencePreselection = true;
    options.referenceMode = QStringLiteral("sequence");
    options.quality = QStringLiteral("high");

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.tiePointOptions.pairPolicy.mode,
              xjw::matchphotos::PairSelectionMode::Sequence);
    EXPECT_EQ(resolved.tiePointOptions.pairPolicy.sequenceWindow, 6);
    EXPECT_TRUE(resolved.tiePointOptions.pairPolicy.closeSequenceLoop);
    EXPECT_FALSE(resolved.tiePointOptions.useReferencePreselection);
    EXPECT_TRUE(resolved.pipelineInput.useSequencePoseRecovery);
    EXPECT_FALSE(resolved.pipelineInput.enforceSequencePoseConsistency);
    EXPECT_TRUE(resolved.pipelineInput.sequenceLoopClosure);
}

TEST(AerialTriangulationWorkflowTest, EstimatedPosePreselectionBoundsReferenceAndVocabularyPairs)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.referencePreselection = true;
    options.referenceMode = QStringLiteral("estimated");
    options.genericPreselection = true;
    options.quality = QStringLiteral("highest");
    options.referenceCameras.insert(options.images.front(), xjw::FramePinholeCamera{});

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.tiePointOptions.useReferencePreselection);
    EXPECT_EQ(resolved.tiePointOptions.pairPolicy.cameraOverlapTopKPerImage, 24);
    EXPECT_EQ(resolved.tiePointOptions.pairPolicy.vocabularyTopKPerImage, 4);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("pair_planning_mode")).toString(),
              QStringLiteral("estimated"));
    EXPECT_EQ(resolved.tiePointOptions.referencePreselectionGeometry,
              xjw::matchphotos::ReferencePreselectionGeometry::SparseScene);
    EXPECT_EQ(resolved.tiePointContext.referenceSparsePointsPath,
              QDir(options.outputDir).filePath(QStringLiteral("sfm_sparse/sfm_sparse_points.json")));
}

TEST(AerialTriangulationWorkflowTest, EstimatedInwardRingKeepsClosedSequenceGeometry)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.images.clear();
    options.referencePreselection = true;
    options.referenceMode = QStringLiteral("estimated");
    for (int index = 0; index < 8; ++index)
    {
        const QString image = QDir(tempDir.path())
            .filePath(QStringLiteral("ring_%1.png").arg(index));
        options.images.append(image);
        options.referenceCameras.insert(
            image,
            makeInwardRingCamera(2.0 * 3.14159265358979323846 * index / 8.0));
    }

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.tiePointOptions.useReferencePreselection);
    EXPECT_NE(resolved.tiePointOptions.pairPolicy.mode,
              xjw::matchphotos::PairSelectionMode::Sequence)
        << "已估位姿仍应按稀疏场景规划配对，而不是退化为纯序列窗口";
    EXPECT_TRUE(resolved.pipelineInput.useSequencePoseRecovery);
    EXPECT_TRUE(resolved.pipelineInput.sequenceLoopClosure);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("sequence_geometry_source")).toString(),
              QStringLiteral("estimated_pose_detected"));
    EXPECT_LT(resolved.resolvedSettings
                  .value(QStringLiteral("estimated_sequence_optical_axis_concentration"))
                  .toDouble(),
              0.1);
}

TEST(AerialTriangulationWorkflowTest, MissingTiePointsRunsMatchPhotosBeforePipeline)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.resetAlignment = false;
    options.autoGenerateMissingMatches = false;
    QStringList events;

    const auto result = xjw::aerial_triangulation::AerialTriangulationWorkflow::run(
        options,
        [&](const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
        {
            events.append(QStringLiteral("pipeline"));
            EXPECT_EQ(input.tiePointPath,
                      QDir(options.assetsDir)
                          .filePath(QStringLiteral("tie_points/latest_tie_points.json")));
            xjw::aerial_triangulation::AerialTriangulationReconstructionResult reconstruction;
            reconstruction.success = true;
            reconstruction.numRegisteredImages = 3;
            return reconstruction;
        },
        [&](const xjw::matchphotos::MatchPhotosOptions &,
            const xjw::matchphotos::MatchPhotosContext &)
        {
            events.append(QStringLiteral("matchphotos"));
            xjw::matchphotos::MatchPhotosResult tiePoints;
            tiePoints.success = true;
            tiePoints.tiePointPath = QDir(options.assetsDir)
                .filePath(QStringLiteral("tie_points/latest_tie_points.json"));
            return tiePoints;
        });

    EXPECT_EQ(events, (QStringList{QStringLiteral("matchphotos"), QStringLiteral("pipeline")}));
    EXPECT_TRUE(result.tiePointPreparationExecuted);
    EXPECT_TRUE(result.reconstructionResult.success);
}

TEST(AerialTriangulationWorkflowTest, TiePointFailurePreventsPipelineExecution)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    bool pipelineCalled = false;

    const auto result = xjw::aerial_triangulation::AerialTriangulationWorkflow::run(
        options,
        [&](const xjw::aerial_triangulation::PreparedAerialTriangulationInput &)
        {
            pipelineCalled = true;
            return xjw::aerial_triangulation::AerialTriangulationReconstructionResult{};
        },
        [](const xjw::matchphotos::MatchPhotosOptions &,
           const xjw::matchphotos::MatchPhotosContext &)
        {
            xjw::matchphotos::MatchPhotosResult tiePoints;
            tiePoints.errorMessage = QStringLiteral("matching failed");
            return tiePoints;
        });

    EXPECT_FALSE(pipelineCalled);
    EXPECT_FALSE(result.reconstructionResult.success);
    EXPECT_TRUE(result.reconstructionResult.errorMessage.contains(QStringLiteral("连接点")));
}

TEST(AerialTriangulationWorkflowTest, ExistingTiePointGraphSkipsPreparation)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.resetAlignment = false;
    options.autoGenerateMissingMatches = false;
    const QString tiePointPath = QDir(options.assetsDir)
        .filePath(QStringLiteral("tie_points/latest_tie_points.json"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(tiePointPath).absolutePath()));
    QFile tiePoints(tiePointPath);
    ASSERT_TRUE(tiePoints.open(QIODevice::WriteOnly));
    tiePoints.write("{}");
    tiePoints.close();

    bool tiePointRunnerCalled = false;
    const auto result = xjw::aerial_triangulation::AerialTriangulationWorkflow::run(
        options,
        [](const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
        {
            EXPECT_TRUE(QFileInfo::exists(input.tiePointPath));
            xjw::aerial_triangulation::AerialTriangulationReconstructionResult reconstruction;
            reconstruction.success = true;
            return reconstruction;
        },
        [&](const xjw::matchphotos::MatchPhotosOptions &,
            const xjw::matchphotos::MatchPhotosContext &)
        {
            tiePointRunnerCalled = true;
            return xjw::matchphotos::MatchPhotosResult{};
        });

    EXPECT_FALSE(tiePointRunnerCalled);
    EXPECT_FALSE(result.tiePointPreparationExecuted);
    EXPECT_TRUE(result.reconstructionResult.success);
}

TEST(AerialTriangulationWorkflowTest, ResetAlignmentStillSkipsExistingTiePointGraph)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.resetAlignment = true;
    options.reuseExistingMatches = true;
    options.autoGenerateMissingMatches = false;
    const QString tiePointPath = QDir(options.assetsDir)
        .filePath(QStringLiteral("tie_points/latest_tie_points.json"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(tiePointPath).absolutePath()));
    QFile tiePoints(tiePointPath);
    ASSERT_TRUE(tiePoints.open(QIODevice::WriteOnly));
    tiePoints.write("{}");
    tiePoints.close();

    bool tiePointRunnerCalled = false;
    const auto result = xjw::aerial_triangulation::AerialTriangulationWorkflow::run(
        options,
        [](const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
        {
            EXPECT_FALSE(input.useProjectCameraPoses);
            xjw::aerial_triangulation::AerialTriangulationReconstructionResult reconstruction;
            reconstruction.success = true;
            return reconstruction;
        },
        [&](const xjw::matchphotos::MatchPhotosOptions &,
            const xjw::matchphotos::MatchPhotosContext &)
        {
            tiePointRunnerCalled = true;
            return xjw::matchphotos::MatchPhotosResult{};
        });

    EXPECT_FALSE(tiePointRunnerCalled);
    EXPECT_FALSE(result.tiePointPreparationExecuted);
    EXPECT_TRUE(result.reconstructionResult.success);
}

TEST(AerialTriangulationWorkflowTest, ResetAlignmentReusesExistingMatchesByDefault)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.resetAlignment = true;

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_FALSE(resolved.forceRebuildTiePoints);
    EXPECT_TRUE(resolved.tiePointOptions.reuseExistingMatches);
    EXPECT_TRUE(resolved.pipelineInput.useProjectCameraIntrinsics);
    EXPECT_FALSE(resolved.pipelineInput.useProjectCameraPoses);

    options.resetAlignment = false;
    const auto reuseResolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);
    EXPECT_TRUE(reuseResolved.pipelineInput.useProjectCameraIntrinsics);
    EXPECT_TRUE(reuseResolved.pipelineInput.useProjectCameraPoses);
}

TEST(AerialTriangulationWorkflowTest, KeypointMasksUseFingerprintBasedMatchReuse)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.maskApplyMode = QStringLiteral("keypoints");
    options.reuseExistingMatches = true;

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.tiePointOptions.reuseExistingMatches);
    EXPECT_EQ(resolved.tiePointOptions.maskApplyMode, QStringLiteral("keypoints"));
}

TEST(AerialTriangulationWorkflowTest, DisablingMatchReuseForcesTiePointRebuild)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.resetAlignment = true;
    options.reuseExistingMatches = false;

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.prepareTiePoints);
    EXPECT_TRUE(resolved.forceRebuildTiePoints);
    EXPECT_FALSE(resolved.tiePointOptions.reuseExistingMatches);
    EXPECT_EQ(resolved.resolvedSettings.value(
                  QStringLiteral("tie_point_preparation")).toString(),
              QStringLiteral("force_rebuild"));
}

TEST(AerialTriangulationWorkflowTest, PropagatesExactInputPoseLock)
{
    QTemporaryDir tempDir;
    auto options = makeBaseOptions(tempDir.path());
    options.lockInputCameraPoses = true;

    const auto resolved =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.pipelineInput.lockInputCameraPoses);
    EXPECT_TRUE(resolved.resolvedSettings.value(
        QStringLiteral("lock_input_camera_poses")).toBool());
}
