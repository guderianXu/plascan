#include "GeometryVerifyStage.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchPhotosTask.h"
#include "MatchingStage.h"
#include "MatchPhotosMaskSupport.h"
#include "SparseSceneOverlapAnalyzer.h"
#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

TEST(MatchPhotosTaskTest, PlanOnlyUsesUnifiedAlgorithmWithoutWritingIntermediateFiles)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString image1 = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    xjw::matchphotos::MatchPhotosContext context;
    context.workingDirectory = tempDir.path();
    context.matchDirectory = QDir(tempDir.path()).filePath(QStringLiteral("image_matches"));
    context.pairInput.images = {image0, image1};
    context.pairInput.manualPairKeys = {
        xjw::matchphotos::makePairKey(image0, image1)};

    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = true;
    options.algorithmId = QStringLiteral("sift_lightglue");
    options.pairPolicy.mode = xjw::matchphotos::PairSelectionMode::ManualOnly;

    const auto result = xjw::matchphotos::MatchPhotosTask(options).run(context);

    EXPECT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_TRUE(result.algorithmPlan.valid);
    EXPECT_EQ(result.algorithmPlan.algorithmId, QStringLiteral("sift_lightglue"));
    EXPECT_EQ(result.pairSelection.candidates.size(), 1u);
    EXPECT_TRUE(result.features.empty());
    EXPECT_TRUE(result.matches.empty());
    EXPECT_TRUE(result.imageMatchFiles.empty());
    EXPECT_FALSE(QDir(context.matchDirectory).exists());
}

TEST(MatchPhotosTaskTest, CpuSelectionUsesDefaultAutoSift)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.device = xjw::matchphotos::ComputeDevice::Cpu;
    options.planOnly = true;

    xjw::matchphotos::MatchPhotosContext context;
    context.pairInput.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};

    const auto result = xjw::matchphotos::MatchPhotosTask(options).run(context);

    EXPECT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.algorithmPlan.algorithmId, QStringLiteral("auto_sift"));
    ASSERT_FALSE(result.stages.empty());
    EXPECT_EQ(result.stages.front().stageId, QStringLiteral("algorithm_selection"));
    EXPECT_EQ(result.stages.front().status,
              xjw::matchphotos::MatchPhotosStageStatus::Completed);
}

TEST(MatchPhotosGeometryGateTest, CombinesSupportRatioCoverageAndAdjacencyWithoutInlierCliff)
{
    xjw::matchphotos::MatchPhotosOptions options;
    xjw::matchphotos::GeometryQualityMetrics metrics;
    metrics.rawMatchCount = 100;
    metrics.inlierCount = 20;
    metrics.inlierRatio = 0.20;
    metrics.image0GridCoverage = 0.75;
    metrics.image1GridCoverage = 0.75;
    EXPECT_FALSE(xjw::matchphotos::evaluateGeometryQuality(metrics, options).passed);

    metrics.rawMatchCount = 25;
    metrics.inlierRatio = 0.80;
    metrics.adjacentImages = true;
    EXPECT_TRUE(xjw::matchphotos::evaluateGeometryQuality(metrics, options).passed);

    metrics.rawMatchCount = 200;
    metrics.inlierCount = 63;
    metrics.inlierRatio = 63.0 / 200.0;
    metrics.adjacentImages = false;
    const auto at63 = xjw::matchphotos::evaluateGeometryQuality(metrics, options);
    metrics.inlierCount = 64;
    metrics.inlierRatio = 64.0 / 200.0;
    const auto at64 = xjw::matchphotos::evaluateGeometryQuality(metrics, options);
    EXPECT_TRUE(at63.passed);
    EXPECT_TRUE(at64.passed);
    EXPECT_LT(std::abs(at64.score - at63.score), 0.03);

    metrics.inlierCount = 19;
    metrics.inlierRatio = 19.0 / 200.0;
    EXPECT_FALSE(xjw::matchphotos::evaluateGeometryQuality(metrics, options).passed);

    metrics.inlierCount = 80;
    metrics.inlierRatio = 0.8;
    metrics.image0GridCoverage = 1.0 / 16.0;
    metrics.image1GridCoverage = 1.0 / 16.0;
    EXPECT_FALSE(xjw::matchphotos::evaluateGeometryQuality(metrics, options).passed);
}

TEST(MatchPhotosSoftMaskTest, KeepsUncertainAndBoundaryPointsWithWeights)
{
    xjw::matchphotos::MatchPhotosOptions options;
    cv::Mat exclusion(11, 11, CV_8UC1, cv::Scalar(0));
    exclusion(cv::Rect(2, 2, 7, 7)).setTo(255);
    const cv::Mat softened = xjw::matchphotos::softenedExclusionMask(exclusion, options);

    EXPECT_FLOAT_EQ(xjw::matchphotos::maskPointWeight(
                        softened, cv::Point2f(5.0f, 5.0f), options),
                    0.0f);
    EXPECT_FLOAT_EQ(xjw::matchphotos::maskPointWeight(
                        softened, cv::Point2f(2.0f, 2.0f), options),
                    1.0f);

    cv::Mat uncertain(3, 3, CV_8UC1, cv::Scalar(128));
    options.maskRelaxationRadius = 0;
    const float weight = xjw::matchphotos::maskPointWeight(
        uncertain, cv::Point2f(1.0f, 1.0f), options);
    EXPECT_GT(weight, 0.45f);
    EXPECT_LT(weight, 0.55f);
}

TEST(MatchPhotosSparseSceneOverlapTest, UsesCovisibilityAndFrustumForUnobservedPairs)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::OverlapImageInput> images;
    for (int index = 0; index < 3; ++index)
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(100.0, 100.0, 50.0, 50.0);
        camera.setPose({{1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0}},
                       {{0.5 * (index - 1), 0.0, 0.0}});
        xjw::OverlapImageInput input;
        input.imagePath = QStringLiteral("image_%1.png").arg(index).toStdString();
        input.camera = camera;
        input.width = 100;
        input.height = 100;
        images.push_back(input);
    }

    QJsonArray points;
    for (int index = 0; index < 40; ++index)
    {
        const double x = -0.8 + 1.6 * (index % 8) / 7.0;
        const double y = -0.6 + 1.2 * (index / 8) / 4.0;
        QJsonArray observations;
        observations.append(QJsonObject{{QStringLiteral("image_id"), 0}});
        observations.append(QJsonObject{{QStringLiteral("image_id"), 1}});
        points.append(QJsonObject{
            {QStringLiteral("point_xyz"), QJsonArray{x, y, 5.0}},
            {QStringLiteral("observations"), observations}});
    }
    const QString sidecarPath = QDir(tempDir.path()).filePath(QStringLiteral("sfm_sparse_points.json"));
    QString writeError;
    ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(
        sidecarPath,
        QJsonDocument(QJsonObject{{QStringLiteral("points"), points}})
            .toJson(QJsonDocument::Compact),
        &writeError)) << qPrintable(writeError);

    xjw::OverlapAnalysisResult overlap;
    xjw::matchphotos::SparseSceneOverlapStats stats;
    QString error;
    ASSERT_TRUE(xjw::matchphotos::SparseSceneOverlapAnalyzer::analyzeFile(
        sidecarPath,
        images,
        xjw::matchphotos::SparseSceneOverlapOptions{},
        &overlap,
        &stats,
        &error)) << qPrintable(error);

    EXPECT_EQ(stats.validPointCount, 40);
    EXPECT_GE(stats.covisibilityPairCount, 1);
    EXPECT_GE(stats.frustumPairCount, 3);
    ASSERT_EQ(overlap.pairs.size(), 3u);
    const auto hasPair = [&](int indexA, int indexB)
    {
        return std::any_of(overlap.pairs.cbegin(), overlap.pairs.cend(), [&](const auto &pair)
        {
            return pair.indexA == indexA && pair.indexB == indexB;
        });
    };
    EXPECT_TRUE(hasPair(0, 1));
    EXPECT_TRUE(hasPair(0, 2)) << "没有共同旧轨迹的相机仍应由稀疏场景视锥重叠召回";
    EXPECT_TRUE(hasPair(1, 2));
}

TEST(MatchPhotosSparseSceneOverlapTest, RejectsOpposingFrustaWithoutCovisibility)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::OverlapImageInput> images;
    const std::array<std::array<double, 9>, 2> rotations{{
        {{0.0, 0.0, -1.0,
          -1.0, 0.0, 0.0,
          0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0}}}};
    for (int index = 0; index < 2; ++index)
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(100.0, 100.0, 50.0, 50.0);
        camera.setPose(rotations[static_cast<std::size_t>(index)],
                       {{index == 0 ? 5.0 : -5.0, 0.0, 0.0}});
        double centerPixel[2]{};
        const std::array<double, 3> origin{{0.0, 0.0, 0.0}};
        ASSERT_TRUE(camera.projectWorldPoint(origin.data(), centerPixel));
        xjw::OverlapImageInput input;
        input.imagePath = QStringLiteral("opposite_%1.png").arg(index).toStdString();
        input.camera = camera;
        input.width = 100;
        input.height = 100;
        images.push_back(input);
    }

    QJsonArray points;
    for (int index = 0; index < 40; ++index)
    {
        const double y = -0.5 + (index % 8) / 7.0;
        const double z = -0.5 + (index / 8) / 4.0;
        points.append(QJsonObject{
            {QStringLiteral("point_xyz"), QJsonArray{0.0, y, z}},
            {QStringLiteral("observations"),
             QJsonArray{QJsonObject{{QStringLiteral("image_id"), index % 2}}}}});
    }
    const QString sidecarPath = QDir(tempDir.path()).filePath(QStringLiteral("sfm_sparse_points.json"));
    QString writeError;
    ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(
        sidecarPath,
        QJsonDocument(QJsonObject{{QStringLiteral("points"), points}})
            .toJson(QJsonDocument::Compact),
        &writeError)) << qPrintable(writeError);

    xjw::OverlapAnalysisResult overlap;
    QString error;
    EXPECT_FALSE(xjw::matchphotos::SparseSceneOverlapAnalyzer::analyzeFile(
        sidecarPath,
        images,
        xjw::matchphotos::SparseSceneOverlapOptions{},
        &overlap,
        nullptr,
        &error));
    EXPECT_TRUE(error.isEmpty());
    EXPECT_TRUE(overlap.pairs.empty());
}

TEST(MatchPhotosGeometryCacheTest, FingerprintCoversEveryGeometryOption)
{
    xjw::matchphotos::MatchPhotosOptions base;
    const QByteArray expected = xjw::matchphotos::geometryVerificationFingerprint(base);
    ASSERT_EQ(expected.size(), 32);

    auto expectChanged = [&](const auto &change)
    {
        auto modified = base;
        change(modified);
        EXPECT_NE(xjw::matchphotos::geometryVerificationFingerprint(modified), expected);
    };
    expectChanged([](auto &options) { options.enableGeometryVerification = false; });
    expectChanged([](auto &options) { options.geometryReprojThreshold += 0.25; });
    expectChanged([](auto &options) { ++options.geometryMinInliers; });
    expectChanged([](auto &options) { options.geometryMinInlierRatio += 0.01; });
    expectChanged([](auto &options) { options.geometryMinGridCoverage += 0.01; });
    expectChanged([](auto &options) { ++options.geometryGridColumns; });
    expectChanged([](auto &options) { ++options.geometryGridRows; });
    expectChanged([](auto &options) { ++options.geometryMaxIterations; });
}

TEST(MatchPhotosRawCacheTest, FingerprintCoversAdaptiveSiftAndSoftMaskOptions)
{
    xjw::matchphotos::MatchPhotosOptions base;
    base.maskApplyMode = QStringLiteral("tiepoints");
    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(base);
    ASSERT_TRUE(plan.valid);
    const QByteArray modelFingerprint(32, 'm');
    const QByteArray expected = xjw::matchphotos::rawMatchConfigurationFingerprint(
        base, plan, 0, base.matchThreshold, modelFingerprint);
    const auto expectChanged = [&](const auto& change)
    {
        auto modified = base;
        change(modified);
        EXPECT_NE(xjw::matchphotos::rawMatchConfigurationFingerprint(
                      modified, plan, 0, modified.matchThreshold, modelFingerprint),
                  expected);
    };
    expectChanged([](auto& options) { options.siftMaximumRatio -= 0.01f; });
    expectChanged([](auto& options) { options.siftMinimumAdaptiveRatio += 0.01f; });
    expectChanged([](auto& options) { options.adaptiveSiftRatio = false; });
    expectChanged([](auto& options) { options.maskHardExclusionThreshold -= 0.01f; });
    expectChanged([](auto& options) { options.maskMinimumTiepointWeight += 0.01f; });
    expectChanged([](auto& options) { ++options.maskRelaxationRadius; });
}

TEST(MatchPhotosRawCacheTest, FingerprintCoversResolvedSiftDetectionThreshold)
{
    xjw::matchphotos::MatchPhotosOptions automatic;
    automatic.useExplicitKeypointLimit = true;
    automatic.maxKeypoints = 4096;
    xjw::matchphotos::MatchPhotosOptions fast = automatic;
    fast.profile = xjw::matchphotos::MatchPhotosProfile::Fast;

    const auto automaticPlan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(automatic);
    const auto fastPlan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(fast);
    ASSERT_TRUE(automaticPlan.valid);
    ASSERT_TRUE(fastPlan.valid);
    ASSERT_EQ(automaticPlan.maxKeypoints, fastPlan.maxKeypoints);
    ASSERT_NE(automaticPlan.siftDetectionThreshold, fastPlan.siftDetectionThreshold);

    const QByteArray modelFingerprint(32, 'm');
    EXPECT_NE(xjw::matchphotos::rawMatchConfigurationFingerprint(
                  automatic, automaticPlan, 4096, 0.15f, modelFingerprint),
              xjw::matchphotos::rawMatchConfigurationFingerprint(
                  fast, fastPlan, 4096, 0.15f, modelFingerprint));
}

TEST(MatchPhotosGeometryCacheTest, ReusesCompleteVerifiedPairWithoutRunningUsac)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = false;
    options.geometryMinInliers = 1;

    auto pair = std::make_shared<xjw::image_matching::PairMatchData>();
    pair->rawMatchCount = 2;
    pair->geometryInlierCount = 2;
    pair->geometryPassed = true;
    pair->geometryModel = xjw::image_matching::GeometryModel::Fundamental;
    pair->image0.width = pair->image1.width = 100;
    pair->image0.height = pair->image1.height = 100;
    pair->correspondences.resize(2);
    for (int index = 0; index < 2; ++index)
    {
        auto &correspondence = pair->correspondences[static_cast<std::size_t>(index)];
        correspondence.flags = xjw::image_matching::MatchRecordFlag::GeometryInlier;
        correspondence.observation0.x = correspondence.observation1.x = 10.0f + index * 70.0f;
        correspondence.observation0.y = correspondence.observation1.y = 10.0f + index * 70.0f;
    }

    xjw::matchphotos::MatchPhotosMatchRecord record;
    record.matchCount = 2;
    record.geometricInlierCount = 2;
    record.passedGeometry = true;
    record.pairData = std::move(pair);
    record.settings[QStringLiteral("geometry_cache_reusable")] = true;
    record.settings[QStringLiteral("geometry_config_fingerprint")] =
        QString::fromLatin1(
            xjw::matchphotos::geometryVerificationFingerprint(options).toHex());
    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records;
    records.push_back(std::move(record));

    const xjw::matchphotos::GeometryVerifyStage stage;
    const auto report = stage.run(xjw::matchphotos::MatchPhotosContext{}, options, &records);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Completed);
    EXPECT_EQ(report.itemCount, 1);
    EXPECT_TRUE(report.message.contains(QStringLiteral("缓存全命中")));
    EXPECT_TRUE(report.message.contains(QStringLiteral("未运行 USAC")));
    EXPECT_TRUE(records.front().settings.value(QStringLiteral("geometry_verified")).toBool());
    EXPECT_EQ(records.front().settings.value(QStringLiteral("geometry_raw_matches")).toInt(), 2);
}

TEST(MatchPhotosGeometryCacheTest, DisablingGeometryClearsStaleModelAndResiduals)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = false;
    options.enableGeometryVerification = false;

    auto pair = std::make_shared<xjw::image_matching::PairMatchData>();
    pair->rawMatchCount = 1;
    pair->geometryInlierCount = 1;
    pair->geometryPassed = true;
    pair->geometryModel = xjw::image_matching::GeometryModel::Fundamental;
    pair->geometryMatrix[0] = 3.0;
    pair->correspondences.resize(1);
    pair->correspondences.front().residualPixels = 2.5f;

    xjw::matchphotos::MatchPhotosMatchRecord record;
    record.pairData = pair;
    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records;
    records.push_back(std::move(record));

    const xjw::matchphotos::GeometryVerifyStage stage;
    const auto report = stage.run(xjw::matchphotos::MatchPhotosContext{}, options, &records);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Skipped);
    EXPECT_EQ(pair->geometryModel, xjw::image_matching::GeometryModel::None);
    EXPECT_EQ(pair->geometryMatrix[0], 0.0);
    EXPECT_EQ(pair->correspondences.front().residualPixels, -1.0f);
    EXPECT_FALSE(records.front().settings.value(QStringLiteral("geometry_verified")).toBool(true));
}
