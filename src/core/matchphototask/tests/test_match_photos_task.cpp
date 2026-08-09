#include "GeometryVerifyStage.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchPhotosTask.h"
#include "MatchingStage.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QJsonObject>
#include <QTemporaryDir>

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

TEST(MatchPhotosTaskTest, CpuSelectionFailsBeforeReadingImages)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.device = xjw::matchphotos::ComputeDevice::Cpu;
    options.planOnly = true;

    xjw::matchphotos::MatchPhotosContext context;
    context.pairInput.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};

    const auto result = xjw::matchphotos::MatchPhotosTask(options).run(context);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("CUDA")));
    ASSERT_FALSE(result.stages.empty());
    EXPECT_EQ(result.stages.front().stageId, QStringLiteral("algorithm_selection"));
    EXPECT_EQ(result.stages.front().status,
              xjw::matchphotos::MatchPhotosStageStatus::Failed);
}

TEST(MatchPhotosGeometryGateTest, RequiresStrongSupportForSmallInlierSets)
{
    EXPECT_FALSE(xjw::matchphotos::passesGeometryQualityGate(100, 20, 20));
    EXPECT_TRUE(xjw::matchphotos::passesGeometryQualityGate(25, 20, 20));
    EXPECT_TRUE(xjw::matchphotos::passesGeometryQualityGate(200, 64, 20));
    EXPECT_FALSE(xjw::matchphotos::passesGeometryQualityGate(100, 19, 20));
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
    expectChanged([](auto &options) { ++options.geometryMaxIterations; });
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
    pair->correspondences.resize(2);
    for (auto &correspondence : pair->correspondences)
    {
        correspondence.flags = xjw::image_matching::MatchRecordFlag::GeometryInlier;
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
