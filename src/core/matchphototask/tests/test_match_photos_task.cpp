#include "GeometryVerifyStage.h"
#include "GuidedMatchStage.h"
#include "GuidedMatchPolicy.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosTask.h"
#include "MatchingStage.h"
#include "MatchPhotosMaskSupport.h"
#include "PlaMatchHctPairPreselector.h"
#include "ReferencePoseEpipolarGeometry.h"
#include "plamatch_hct/PlaMatchHctAlgorithm.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QJsonObject>
#include <QTemporaryDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace
{

    xjw::FramePinholeCamera makeGuidedTestCamera(double centerX)
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(1000.0, 1000.0, 500.0, 400.0);
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {centerX, 0.0, 0.0});
        return camera;
    }

} // namespace

TEST(PlaMatchHctPairPreselectorTest, ReusesCoarsePayloadAndKeepsIdenticalPair)
{
    cv::Mat image(256, 256, CV_8UC3);
    cv::RNG random(0x504C414D);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);

    xjw::image_matching::ImageMatchingRuntimeConfig config;
    config.maxKeypoints = 256;
    config.siftBackend = xjw::image_matching::SiftComputeBackend::Cpu;
    xjw::image_matching::PlaMatchHctAlgorithm algorithm(config);
    xjw::image_matching::ImageFeatureInput input;
    input.imagePath = QStringLiteral("synthetic.png");
    input.colorImage = image;
    input.originalWidth = image.cols;
    input.originalHeight = image.rows;
    const auto features = std::make_shared<const xjw::image_matching::FeatureSet>(algorithm.extract(input));

    const QStringList images = {QStringLiteral("left.png"), QStringLiteral("right.png")};
    xjw::matchphotos::MatchPhotosFeatureCache cache;
    cache.insert(images[0], features);
    cache.insert(images[1], features);

    xjw::matchphotos::MatchPhotosOptions options;
    options.useGenericPreselection = true;
    options.useReferencePreselection = false;
    xjw::matchphotos::PairSelectionResult output;
    xjw::matchphotos::PlaMatchHctPairPreselectionStats stats;
    QString error;
    ASSERT_TRUE(xjw::matchphotos::PlaMatchHctPairPreselector::select(
        images, cache, options, {}, xjw::image_matching::SiftComputeBackend::Cpu, 0, &output, &stats, nullptr, &error))
        << qPrintable(error);
    ASSERT_EQ(output.candidates.size(), 1U);
    EXPECT_EQ(output.candidates.front().pair.indexA, 0);
    EXPECT_EQ(output.candidates.front().pair.indexB, 1);
    EXPECT_NE(std::find(output.candidates.front().sources.cbegin(),
                        output.candidates.front().sources.cend(),
                        xjw::matchphotos::PairSource::PlaMatchGeneric),
              output.candidates.front().sources.cend());
    EXPECT_EQ(stats.coarseCandidateCount, 1);
    EXPECT_EQ(stats.genericSelectedCount, 1);
}

TEST(PlaMatchHctPairPreselectorTest, ReusesFloatingPointDescriptorsWithoutExtractingPlaMatchFeatures)
{
    auto features = std::make_shared<xjw::image_matching::FeatureSet>();
    features->sourceAlgorithm = "auto_sift";
    features->computeBackend = "cpu";
    features->imageWidth = 640;
    features->imageHeight = 480;
    features->descriptors = cv::Mat(64, 128, CV_32F);
    cv::RNG random(0x434F4152);
    random.fill(features->descriptors, cv::RNG::UNIFORM, -1.0f, 1.0f);
    for (int index = 0; index < features->descriptors.rows; ++index)
    {
        cv::KeyPoint keypoint;
        keypoint.pt =
            cv::Point2f(20.0f + static_cast<float>(index % 8) * 70.0f, 20.0f + static_cast<float>(index / 8) * 50.0f);
        keypoint.size = 4.0f;
        keypoint.response = 1.0f - static_cast<float>(index) / 100.0f;
        features->keypoints.push_back(keypoint);
        features->scores.push_back(keypoint.response);
    }
    ASSERT_TRUE(features->isConsistent());

    const QStringList images = {QStringLiteral("a.png"), QStringLiteral("b.png"), QStringLiteral("c.png")};
    xjw::matchphotos::MatchPhotosFeatureCache cache;
    for (const QString& image : images)
    {
        cache.insert(image, features);
    }

    xjw::matchphotos::MatchPhotosOptions options;
    options.useGenericPreselection = true;
    options.useReferencePreselection = false;
    xjw::matchphotos::PairSelectionResult output;
    xjw::matchphotos::PlaMatchHctPairPreselectionStats stats;
    QString error;
    ASSERT_TRUE(xjw::matchphotos::PlaMatchHctPairPreselector::select(
        images, cache, options, {}, xjw::image_matching::SiftComputeBackend::Cpu, 0, &output, &stats, nullptr, &error))
        << qPrintable(error);
    EXPECT_TRUE(stats.usedDescriptorAdapter);
    EXPECT_GT(stats.genericSelectedCount, 0);
    EXPECT_FALSE(output.candidates.empty());
    EXPECT_TRUE(stats.detail.contains(QStringLiteral("复用正式算法描述子")));
}

TEST(PlaMatchHctPairPreselectorTest, MatchesReferenceModeFallbackSemantics)
{
    cv::Mat image(128, 128, CV_8UC3);
    cv::RNG random(0x534F5552);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);

    xjw::image_matching::ImageMatchingRuntimeConfig config;
    config.maxKeypoints = 128;
    config.siftBackend = xjw::image_matching::SiftComputeBackend::Cpu;
    xjw::image_matching::PlaMatchHctAlgorithm algorithm(config);
    xjw::image_matching::ImageFeatureInput input;
    input.imagePath = QStringLiteral("reference.png");
    input.colorImage = image;
    input.originalWidth = image.cols;
    input.originalHeight = image.rows;
    const auto features = std::make_shared<const xjw::image_matching::FeatureSet>(algorithm.extract(input));

    const QStringList images = {QStringLiteral("a.png"), QStringLiteral("b.png"), QStringLiteral("c.png")};
    xjw::matchphotos::MatchPhotosFeatureCache cache;
    for (const QString& path : images)
    {
        cache.insert(path, features);
    }

    xjw::matchphotos::MatchPhotosOptions options;
    options.useGenericPreselection = false;
    options.useReferencePreselection = true;
    options.referencePreselectionNeighbors = 1;
    xjw::matchphotos::PairSelectionResult output;
    xjw::matchphotos::PlaMatchHctPairPreselectionStats stats;
    QString error;

    options.referencePreselectionMode = xjw::matchphotos::ReferencePreselectionMode::Source;
    ASSERT_TRUE(xjw::matchphotos::PlaMatchHctPairPreselector::select(
        images, cache, options, {}, xjw::image_matching::SiftComputeBackend::Cpu, 0, &output, &stats, nullptr, &error))
        << qPrintable(error);
    EXPECT_TRUE(stats.usedReferenceIndexFallback);
    EXPECT_FALSE(stats.usedAllPairsFallback);
    EXPECT_EQ(stats.referenceSelectedCount, 2);
    EXPECT_EQ(output.candidates.size(), 2U);

    options.pairPolicy.maxPairs = 1;
    ASSERT_TRUE(xjw::matchphotos::PlaMatchHctPairPreselector::select(
        images, cache, options, {}, xjw::image_matching::SiftComputeBackend::Cpu, 0, &output, &stats, nullptr, &error))
        << qPrintable(error);
    EXPECT_EQ(output.candidates.size(), 1U);
    EXPECT_TRUE(output.restrictPairs);
    options.pairPolicy.maxPairs = 0;

    QMap<QString, xjw::FramePinholeCamera> estimatedCameras;
    estimatedCameras.insert(images[0], makeGuidedTestCamera(0.0));
    estimatedCameras.insert(images[1], makeGuidedTestCamera(10.0));
    estimatedCameras.insert(images[2], makeGuidedTestCamera(11.0));
    options.referencePreselectionMode = xjw::matchphotos::ReferencePreselectionMode::Estimated;
    ASSERT_TRUE(xjw::matchphotos::PlaMatchHctPairPreselector::select(images,
                                                                     cache,
                                                                     options,
                                                                     estimatedCameras,
                                                                     xjw::image_matching::SiftComputeBackend::Cpu,
                                                                     0,
                                                                     &output,
                                                                     &stats,
                                                                     nullptr,
                                                                     &error))
        << qPrintable(error);
    EXPECT_FALSE(stats.usedReferenceIndexFallback);
    EXPECT_EQ(stats.referenceSelectedCount, 2);
    EXPECT_EQ(output.candidates.size(), 2U);
    EXPECT_NE(std::find(output.candidates.front().sources.cbegin(),
                        output.candidates.front().sources.cend(),
                        xjw::matchphotos::PairSource::PlaMatchReferenceEstimated),
              output.candidates.front().sources.cend());

    options.referencePreselectionMode = xjw::matchphotos::ReferencePreselectionMode::Sequential;
    ASSERT_TRUE(xjw::matchphotos::PlaMatchHctPairPreselector::select(
        images, cache, options, {}, xjw::image_matching::SiftComputeBackend::Cpu, 0, &output, &stats, nullptr, &error))
        << qPrintable(error);
    EXPECT_EQ(stats.referenceSelectedCount, 0);
    EXPECT_TRUE(stats.usedAllPairsFallback);
    EXPECT_EQ(output.candidates.size(), 3U);
}

TEST(PlaMatchHctPairPreselectorTest, UsesPositionOnlyReferenceWithoutCoarseFeatures)
{
    const QStringList images = {QStringLiteral("a.png"), QStringLiteral("b.png"), QStringLiteral("c.png")};
    xjw::matchphotos::MatchPhotosFeatureCache emptyCache;
    xjw::matchphotos::MatchPhotosOptions options;
    options.useGenericPreselection = false;
    options.useReferencePreselection = true;
    options.referencePreselectionNeighbors = 1;
    QMap<QString, std::array<double, 3>> positions;
    positions.insert(QStringLiteral("a"), {0.0, 0.0, 0.0});
    positions.insert(QStringLiteral("b.png"), {100.0, 0.0, 0.0});
    positions.insert(QStringLiteral("c.png"), {101.0, 0.0, 0.0});

    xjw::matchphotos::PairSelectionResult output;
    xjw::matchphotos::PlaMatchHctPairPreselectionStats stats;
    QString error;
    ASSERT_TRUE(
        xjw::matchphotos::PlaMatchHctPairPreselector::selectWithPositions(images,
                                                                          emptyCache,
                                                                          options,
                                                                          {},
                                                                          positions,
                                                                          xjw::image_matching::SiftComputeBackend::Cpu,
                                                                          0,
                                                                          &output,
                                                                          &stats,
                                                                          nullptr,
                                                                          &error))
        << qPrintable(error);
    EXPECT_FALSE(stats.usedReferenceIndexFallback);
    EXPECT_EQ(stats.referenceSelectedCount, 2);
    EXPECT_EQ(output.candidates.size(), 2U);
}

TEST(ReferencePoseEpipolarGeometryTest, BuildsHorizontalEpipolarConstraint)
{
    const auto geometry =
        xjw::matchphotos::fundamentalFromReferenceCameras(makeGuidedTestCamera(0.0), makeGuidedTestCamera(1.0));

    ASSERT_TRUE(geometry.valid);
    EXPECT_NEAR(geometry.baseline, 1.0, 1.0e-12);
    EXPECT_NEAR(
        xjw::matchphotos::epipolarSampsonDistance(geometry.fundamental, 100.0, 200.0, 150.0, 200.0), 0.0, 1.0e-9);
    EXPECT_GT(xjw::matchphotos::epipolarSampsonDistance(geometry.fundamental, 100.0, 200.0, 150.0, 210.0), 1.0);
}

TEST(ReferencePoseEpipolarGeometryTest, RejectsDistortedRawPixelGeometry)
{
    xjw::FramePinholeCamera distorted = makeGuidedTestCamera(0.0);
    distorted.setDistortion(0.01, 0.0, 0.0, 0.0, 0.0);

    EXPECT_FALSE(xjw::matchphotos::fundamentalFromReferenceCameras(distorted, makeGuidedTestCamera(1.0)).valid);
}

TEST(MatchPhotosGuidedPolicyTest, UsesTrustedReferencePoseWhenEstimatedModelIsUnavailable)
{
    const QString image0 = QStringLiteral("a.png");
    const QString image1 = QStringLiteral("b.png");
    xjw::matchphotos::MatchPhotosContext context;
    context.referenceCameras.insert(image0, makeGuidedTestCamera(0.0));
    context.referenceCameras.insert(image1, makeGuidedTestCamera(1.0));

    xjw::matchphotos::MatchPhotosOptions options;
    options.guidedMatchingMode = xjw::matchphotos::GuidedMatchingMode::Forced;
    options.guidedUseReferenceCameraPoses = true;

    xjw::matchphotos::MatchPhotosMatchRecord record;
    record.image0Path = image0;
    record.image1Path = image1;
    record.pairData = std::make_shared<xjw::image_matching::PairMatchData>();
    const xjw::image_matching::FeatureSet features0;
    const xjw::image_matching::FeatureSet features1;

    const auto cache = xjw::matchphotos::buildGuidedMatchPolicyCache(context);
    const auto choice =
        xjw::matchphotos::chooseGuidedMatchGeometry(context, options, cache, record, features0, features1);

    EXPECT_TRUE(choice.eligible);
    EXPECT_EQ(choice.geometrySource, QStringLiteral("reference_pose"));
    EXPECT_GE(choice.epipolarThresholdPixels, 2.0);
}

TEST(MatchPhotosGuidedStageTest, ReportsEveryPairWithElapsedTimeAndRemainingCount)
{
    xjw::matchphotos::MatchPhotosContext context;
    context.featureCache = std::make_shared<xjw::matchphotos::MatchPhotosFeatureCache>();
    std::vector<std::tuple<QString, int, int>> updates;
    context.progressCallback = [&updates](const QString& stage_id, const QString& message, int current, int maximum)
    {
        if (stage_id == QStringLiteral("guided_match"))
        {
            updates.emplace_back(message, current, maximum);
        }
    };

    xjw::matchphotos::MatchPhotosOptions options;
    options.guidedMatchingMode = xjw::matchphotos::GuidedMatchingMode::Forced;
    xjw::matchphotos::MatchPhotosAlgorithmPlan plan;
    plan.algorithmId = QStringLiteral("auto_sift");
    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records(2);
    records[0].image0Path = QStringLiteral("a.png");
    records[0].image1Path = QStringLiteral("b.png");
    records[1].image0Path = QStringLiteral("c.png");
    records[1].image1Path = QStringLiteral("d.png");

    const xjw::matchphotos::GuidedMatchStage stage;
    const auto report = stage.run(context, options, plan, &records);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Completed);
    ASSERT_EQ(updates.size(), 2U);
    EXPECT_EQ(std::get<1>(updates[0]), 1);
    EXPECT_EQ(std::get<2>(updates[0]), 2);
    EXPECT_TRUE(std::get<0>(updates[0]).contains(QStringLiteral("耗时")));
    EXPECT_TRUE(std::get<0>(updates[0]).contains(QStringLiteral("剩余 1 对")));
    EXPECT_EQ(std::get<1>(updates[1]), 2);
    EXPECT_TRUE(std::get<0>(updates[1]).contains(QStringLiteral("剩余 0 对")));
}

TEST(MatchPhotosGuidedStageTest, HonorsCancellationBeforeStartingPair)
{
    std::atomic_bool canceled{true};
    xjw::matchphotos::MatchPhotosContext context;
    context.featureCache = std::make_shared<xjw::matchphotos::MatchPhotosFeatureCache>();
    context.cancelFlag = &canceled;
    xjw::matchphotos::MatchPhotosOptions options;
    options.guidedMatchingMode = xjw::matchphotos::GuidedMatchingMode::Forced;
    xjw::matchphotos::MatchPhotosAlgorithmPlan plan;
    plan.algorithmId = QStringLiteral("auto_sift");
    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records(1);

    const xjw::matchphotos::GuidedMatchStage stage;
    const auto report = stage.run(context, options, plan, &records);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Failed);
    EXPECT_TRUE(report.message.contains(QStringLiteral("已取消")));
    EXPECT_TRUE(report.message.contains(QStringLiteral("剩余 1 对")));
}

TEST(MatchPhotosGuidedStageTest, UsesMultipleWorkersForManyPairs)
{
    constexpr int pair_count = 8;
    const int worker_count =
        xjw::matchphotos::resolveGuidedMatchingWorkers(pair_count, std::thread::hardware_concurrency());
    if (worker_count < 2)
    {
        GTEST_SKIP() << "Host exposes fewer than two guided matching workers";
    }

    xjw::matchphotos::MatchPhotosContext context;
    context.featureCache = std::make_shared<xjw::matchphotos::MatchPhotosFeatureCache>();
    std::mutex thread_ids_mutex;
    std::set<std::thread::id> thread_ids;
    context.progressCallback = [&](const QString& stage_id, const QString& message, int, int)
    {
        if (stage_id != QStringLiteral("guided_match") || !message.contains(QStringLiteral("[完成")))
        {
            return;
        }
        {
            std::lock_guard lock(thread_ids_mutex);
            thread_ids.insert(std::this_thread::get_id());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    };

    xjw::matchphotos::MatchPhotosOptions options;
    options.guidedMatchingMode = xjw::matchphotos::GuidedMatchingMode::Forced;
    xjw::matchphotos::MatchPhotosAlgorithmPlan plan;
    plan.algorithmId = QStringLiteral("auto_sift");
    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records(pair_count);
    for (int index = 0; index < pair_count; ++index)
    {
        records[static_cast<std::size_t>(index)].image0Path = QStringLiteral("left_%1.png").arg(index);
        records[static_cast<std::size_t>(index)].image1Path = QStringLiteral("right_%1.png").arg(index);
    }

    const xjw::matchphotos::GuidedMatchStage stage;
    const auto report = stage.run(context, options, plan, &records);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Completed);
    EXPECT_GT(thread_ids.size(), 1U);
    EXPECT_TRUE(report.message.contains(QStringLiteral("CPU worker %1").arg(worker_count)));
}

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
    context.pairInput.manualPairKeys = {xjw::matchphotos::makePairKey(image0, image1)};

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
    EXPECT_EQ(result.matchingFunnel.allPairCount, 1);
    EXPECT_EQ(result.matchingFunnel.selectedPairCount, 1);
    EXPECT_TRUE(result.trackSummary.contains(QStringLiteral("matching_funnel")));
    EXPECT_TRUE(std::any_of(result.stages.begin(),
                            result.stages.end(),
                            [](const auto& stage)
                            {
                                return stage.stageId == QStringLiteral("matching_funnel") &&
                                       stage.status == xjw::matchphotos::MatchPhotosStageStatus::Completed;
                            }));
    EXPECT_FALSE(QDir(context.matchDirectory).exists());
}

TEST(MatchPhotosTaskTest, CpuSelectionUsesDefaultPlaMatchHct)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.device = xjw::matchphotos::ComputeDevice::Cpu;
    options.planOnly = true;

    xjw::matchphotos::MatchPhotosContext context;
    context.pairInput.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};

    const auto result = xjw::matchphotos::MatchPhotosTask(options).run(context);

    EXPECT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.algorithmPlan.algorithmId, QStringLiteral("plamatch_hct"));
    EXPECT_EQ(result.algorithmPlan.executionBackend, xjw::image_matching::SiftComputeBackend::Cpu);
    ASSERT_FALSE(result.stages.empty());
    EXPECT_EQ(result.stages.front().stageId, QStringLiteral("algorithm_selection"));
    EXPECT_EQ(result.stages.front().status, xjw::matchphotos::MatchPhotosStageStatus::Completed);
}

TEST(MatchPhotosTaskTest, DisabledAutomaticPreselectionUsesAllPairsBeyondAutoThreshold)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = true;
    options.useGenericPreselection = false;
    options.useReferencePreselection = false;
    options.pairPolicy.exhaustiveMaxImages = 3;

    xjw::matchphotos::MatchPhotosContext context;
    for (int index = 0; index < 6; ++index)
    {
        context.pairInput.images.append(QStringLiteral("image_%1.png").arg(index));
    }

    const auto result = xjw::matchphotos::MatchPhotosTask(options).run(context);

    EXPECT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.pairSelection.allPairCount, 15);
    EXPECT_EQ(result.pairSelection.candidates.size(), 15U);
    EXPECT_FALSE(result.pairSelection.restrictPairs);
    ASSERT_FALSE(result.pairSelection.candidates.empty());
    EXPECT_NE(std::find(result.pairSelection.candidates.front().sources.cbegin(),
                        result.pairSelection.candidates.front().sources.cend(),
                        xjw::matchphotos::PairSource::Exhaustive),
              result.pairSelection.candidates.front().sources.cend());
}

TEST(MatchPhotosTaskTest, ReusesCompletePlaMatchFeaturesFromDisk)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString matchDirectory = tempDir.filePath(QStringLiteral("matches"));
    const QString image0 = tempDir.filePath(QStringLiteral("a.png"));
    const QString image1 = tempDir.filePath(QStringLiteral("b.png"));
    cv::Mat image(256, 256, CV_8UC3);
    cv::RNG random(0x504C414D);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    ASSERT_TRUE(cv::imwrite(image0.toStdString(), image));
    ASSERT_TRUE(cv::imwrite(image1.toStdString(), image));

    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = false;
    options.device = xjw::matchphotos::ComputeDevice::Cpu;
    options.useExplicitKeypointLimit = true;
    options.maxKeypoints = 128;
    options.useGenericPreselection = false;
    options.useReferencePreselection = false;
    options.guidedMatchingMode = xjw::matchphotos::GuidedMatchingMode::Disabled;

    xjw::matchphotos::MatchPhotosContext context;
    context.workingDirectory = tempDir.path();
    context.matchDirectory = matchDirectory;
    context.pairInput.images = {image0, image1};

    const auto first = xjw::matchphotos::MatchPhotosTask(options).run(context);
    ASSERT_EQ(first.features.size(), 2U);
    EXPECT_FALSE(first.features[0].settings.value(QStringLiteral("feature_cache_reused")).toBool());
    EXPECT_EQ(QDir(matchDirectory).entryList(QStringList{QStringLiteral("*.pihctcache")}, QDir::Files).size(), 2);

    const auto second = xjw::matchphotos::MatchPhotosTask(options).run(context);
    ASSERT_EQ(second.features.size(), 2U);
    EXPECT_TRUE(second.features[0].settings.value(QStringLiteral("feature_cache_reused")).toBool());
    EXPECT_TRUE(second.features[1].settings.value(QStringLiteral("feature_cache_reused")).toBool());
    const auto featureStage = std::find_if(second.stages.cbegin(),
                                           second.stages.cend(),
                                           [](const xjw::matchphotos::MatchPhotosStageReport& stage)
                                           { return stage.stageId == QLatin1String("feature"); });
    ASSERT_NE(featureStage, second.stages.cend());
    EXPECT_TRUE(featureStage->message.contains(QStringLiteral("磁盘缓存命中 2 张")));
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

    EXPECT_FLOAT_EQ(xjw::matchphotos::maskPointWeight(softened, cv::Point2f(5.0f, 5.0f), options), 0.0f);
    EXPECT_FLOAT_EQ(xjw::matchphotos::maskPointWeight(softened, cv::Point2f(2.0f, 2.0f), options), 1.0f);

    cv::Mat uncertain(3, 3, CV_8UC1, cv::Scalar(128));
    options.maskRelaxationRadius = 0;
    const float weight = xjw::matchphotos::maskPointWeight(uncertain, cv::Point2f(1.0f, 1.0f), options);
    EXPECT_GT(weight, 0.45f);
    EXPECT_LT(weight, 0.55f);
}

TEST(MatchPhotosGeometryCacheTest, FingerprintCoversEveryGeometryOption)
{
    xjw::matchphotos::MatchPhotosOptions base;
    const QByteArray expected = xjw::matchphotos::geometryVerificationFingerprint(base);
    ASSERT_EQ(expected.size(), 32);

    auto expectChanged = [&](const auto& change)
    {
        auto modified = base;
        change(modified);
        EXPECT_NE(xjw::matchphotos::geometryVerificationFingerprint(modified), expected);
    };
    expectChanged([](auto& options) { options.enableGeometryVerification = false; });
    expectChanged([](auto& options) { options.geometryReprojThreshold += 0.25; });
    expectChanged([](auto& options) { ++options.geometryMinInliers; });
    expectChanged([](auto& options) { options.geometryMinInlierRatio += 0.01; });
    expectChanged([](auto& options) { options.geometryMinGridCoverage += 0.01; });
    expectChanged([](auto& options) { ++options.geometryGridColumns; });
    expectChanged([](auto& options) { ++options.geometryGridRows; });
    expectChanged([](auto& options) { ++options.geometryMaxIterations; });
}

TEST(MatchPhotosRawCacheTest, FingerprintCoversAdaptiveSiftAndSoftMaskOptions)
{
    xjw::matchphotos::MatchPhotosOptions base;
    base.maskApplyMode = QStringLiteral("tiepoints");
    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(base);
    ASSERT_TRUE(plan.valid);
    const QByteArray modelFingerprint(32, 'm');
    const QByteArray expected =
        xjw::matchphotos::rawMatchConfigurationFingerprint(base, plan, 0, base.matchThreshold, modelFingerprint);
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

    const auto automaticPlan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(automatic);
    const auto fastPlan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(fast);
    ASSERT_TRUE(automaticPlan.valid);
    ASSERT_TRUE(fastPlan.valid);
    ASSERT_EQ(automaticPlan.maxKeypoints, fastPlan.maxKeypoints);
    ASSERT_NE(automaticPlan.siftDetectionThreshold, fastPlan.siftDetectionThreshold);

    const QByteArray modelFingerprint(32, 'm');
    EXPECT_NE(
        xjw::matchphotos::rawMatchConfigurationFingerprint(automatic, automaticPlan, 4096, 0.15f, modelFingerprint),
        xjw::matchphotos::rawMatchConfigurationFingerprint(fast, fastPlan, 4096, 0.15f, modelFingerprint));
}

TEST(MatchPhotosRawCacheTest, FingerprintSeparatesAlignmentAccuracyLevels)
{
    xjw::matchphotos::MatchPhotosOptions high;
    high.accuracy = xjw::matchphotos::AlignmentAccuracy::High;
    xjw::matchphotos::MatchPhotosOptions medium = high;
    medium.accuracy = xjw::matchphotos::AlignmentAccuracy::Medium;
    const auto highPlan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(high);
    const auto mediumPlan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(medium);
    ASSERT_TRUE(highPlan.valid);
    ASSERT_TRUE(mediumPlan.valid);
    const QByteArray modelFingerprint(32, 'm');
    EXPECT_NE(
        xjw::matchphotos::rawMatchConfigurationFingerprint(high, highPlan, 0, high.matchThreshold, modelFingerprint),
        xjw::matchphotos::rawMatchConfigurationFingerprint(
            medium, mediumPlan, 0, medium.matchThreshold, modelFingerprint));
}

TEST(MatchPhotosRawCacheTest, FingerprintCoversLowTextureRecovery)
{
    xjw::matchphotos::MatchPhotosOptions options;
    auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);
    ASSERT_TRUE(plan.valid);
    const QByteArray modelFingerprint(32, 'm');
    const QByteArray baseline =
        xjw::matchphotos::rawMatchConfigurationFingerprint(options, plan, 0, options.matchThreshold, modelFingerprint);

    plan.lowTextureRecovery = !plan.lowTextureRecovery;
    EXPECT_NE(
        xjw::matchphotos::rawMatchConfigurationFingerprint(options, plan, 0, options.matchThreshold, modelFingerprint),
        baseline);
}

TEST(MatchPhotosRawCacheTest, FingerprintCoversResolvedBackendAndDevice)
{
    xjw::matchphotos::MatchPhotosOptions options;
    auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);
    ASSERT_TRUE(plan.valid);
    plan.executionBackend = xjw::image_matching::SiftComputeBackend::Cpu;
    plan.computeDeviceName = QStringLiteral("CPU");
    const QByteArray modelFingerprint(32, 'm');
    const QByteArray baseline =
        xjw::matchphotos::rawMatchConfigurationFingerprint(options, plan, 0, options.matchThreshold, modelFingerprint);

    auto changedBackend = plan;
    changedBackend.executionBackend = xjw::image_matching::SiftComputeBackend::Cuda;
    EXPECT_NE(xjw::matchphotos::rawMatchConfigurationFingerprint(
                  options, changedBackend, 0, options.matchThreshold, modelFingerprint),
              baseline);

    auto changedDevice = plan;
    changedDevice.computeDeviceName = QStringLiteral("another-device");
    EXPECT_NE(xjw::matchphotos::rawMatchConfigurationFingerprint(
                  options, changedDevice, 0, options.matchThreshold, modelFingerprint),
              baseline);
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
        auto& correspondence = pair->correspondences[static_cast<std::size_t>(index)];
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
        QString::fromLatin1(xjw::matchphotos::geometryVerificationFingerprint(options).toHex());
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

TEST(MatchPhotosGeometryReferenceParityTest, NonGuidedPlaMatchUsesLocalConsistencyMinimumEight)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = false;
    options.enableGeometryVerification = true;
    options.guidedMatchingMode = xjw::matchphotos::GuidedMatchingMode::Disabled;

    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records(2);
    for (int index = 0; index < 2; ++index)
    {
        auto pair = std::make_shared<xjw::image_matching::PairMatchData>();
        pair->correspondences.resize(index == 0 ? 8U : 7U);
        records[static_cast<std::size_t>(index)].algorithmId =
            QString::fromLatin1(xjw::image_matching::kPlaMatchHctAlgorithmId);
        records[static_cast<std::size_t>(index)].pairData = std::move(pair);
    }

    const xjw::matchphotos::GeometryVerifyStage stage;
    const auto report = stage.run(xjw::matchphotos::MatchPhotosContext{}, options, &records);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Completed);
    EXPECT_EQ(report.itemCount, 1);
    EXPECT_TRUE(report.message.contains(QStringLiteral("未运行 USAC")));
    EXPECT_TRUE(records[0].passedGeometry);
    EXPECT_FALSE(records[1].passedGeometry);
    for (const auto& record : records)
    {
        ASSERT_TRUE(record.pairData);
        EXPECT_EQ(record.pairData->geometryModel, xjw::image_matching::GeometryModel::None);
        EXPECT_EQ(record.pairData->geometryInlierCount, record.pairData->rawMatchCount);
        EXPECT_EQ(record.settings.value(QStringLiteral("geometry_mode")).toString(),
                  QStringLiteral("plamatch_local_consistency"));
        for (const auto& correspondence : record.pairData->correspondences)
        {
            EXPECT_TRUE(xjw::image_matching::hasFlag(correspondence.flags,
                                                     xjw::image_matching::MatchRecordFlag::GeometryInlier));
        }
    }
}

TEST(MatchPhotosFunnelDiagnosticsTest, SummarizesEveryRetentionBoundary)
{
    xjw::matchphotos::PairSelectionResult selection;
    selection.allPairCount = 10;
    selection.candidates.resize(4);

    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records(3);
    records[0].passedGeometry = true;
    records[0].geometricInlierCount = 80;
    records[1].passedGeometry = true;
    records[1].geometricInlierCount = 35;

    const auto diagnostics = xjw::matchphotos::summarizeMatchPhotosFunnel(selection, records, 3, 200, 2, 100, 15, 40);

    EXPECT_EQ(diagnostics.allPairCount, 10);
    EXPECT_EQ(diagnostics.selectedPairCount, 4);
    EXPECT_EQ(diagnostics.matchedPairCount, 3);
    EXPECT_EQ(diagnostics.rawMatchCount, 200);
    EXPECT_EQ(diagnostics.geometryPassedPairCount, 2);
    EXPECT_EQ(diagnostics.geometryInlierCount, 100);
    EXPECT_EQ(diagnostics.guidedAddedInlierCount, 15);
    EXPECT_EQ(diagnostics.finalPassedPairCount, 2);
    EXPECT_EQ(diagnostics.finalGeometryInlierCount, 115);
    EXPECT_EQ(diagnostics.trackCount, 40);
    EXPECT_DOUBLE_EQ(diagnostics.pairSelectionRatio, 0.4);
    EXPECT_DOUBLE_EQ(diagnostics.matchingYieldRatio, 0.75);
    EXPECT_NEAR(diagnostics.geometryPairRetentionRatio, 2.0 / 3.0, 1.0e-12);
    EXPECT_DOUBLE_EQ(diagnostics.geometryInlierRatio, 0.5);
    EXPECT_DOUBLE_EQ(diagnostics.guidedGainRatio, 0.15);

    const QJsonObject json = diagnostics.toJson();
    EXPECT_EQ(json.value(QStringLiteral("final_geometry_inlier_count")).toInt(), 115);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("guided_gain_ratio")).toDouble(), 0.15);
}
