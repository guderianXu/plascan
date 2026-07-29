#include "MatchPhotosTask.h"
#include "FeatureFileIO.h"
#include "FeatureStage.h"
#include "GeometryVerifyStage.h"
#include "GuidedMatchStage.h"

// 避免 Qt 关键字宏改写 LibTorch 头文件中的 slots()/signals() 成员。
#ifdef slots
#undef slots
#define PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_SLOTS
#endif
#ifdef signals
#undef signals
#define PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_SIGNALS
#endif
#ifdef emit
#undef emit
#define PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_EMIT
#endif

#include "MatchPhotosMaskSupport.h"

#ifdef PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_SLOTS
#define slots Q_SLOTS
#undef PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_SLOTS
#endif
#ifdef PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_SIGNALS
#define signals Q_SIGNALS
#undef PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_SIGNALS
#endif
#ifdef PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_EMIT
#define emit Q_EMIT
#undef PLASCAN_MATCHPHOTOS_MASK_TEST_RESTORE_QT_EMIT
#endif

#include "MatchPhotosAlgorithmSelector.h"
#include "MatchPhotosRuntime.h"
#include "MatchingStage.h"
#include "MatchFileIO.h"
#include "StoredPairGeometryAudit.h"
#include "TrackBuildStage.h"
#include "FeatureOutput.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QTextStream>
#include <QTemporaryDir>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{

QStringList makeImages(int count)
{
    QStringList images;
    const QDir dir(QDir::tempPath());
    for (int i = 0; i < count; ++i)
    {
        images.append(dir.filePath(QStringLiteral("plascan_matchphotos_task_%1.png").arg(i)));
    }
    return images;
}

QString writeSyntheticSiftImage(const QString &dirPath, const QString &name, int shift)
{
    cv::Mat image(160, 160, CV_8UC1, cv::Scalar(20));
    cv::rectangle(image, cv::Rect(25 + shift, 25, 50, 50), cv::Scalar(230), cv::FILLED);
    cv::circle(image, cv::Point(110 + shift, 105), 28, cv::Scalar(180), cv::FILLED);
    cv::line(image, cv::Point(15, 145 - shift), cv::Point(145, 15 + shift), cv::Scalar(255), 3);

    const QString path = QDir(dirPath).filePath(name);
    EXPECT_TRUE(cv::imwrite(path.toStdString(), image));
    return path;
}

QString readProjectSourceFile(const QString &relativePath)
{
    QStringList roots;
#ifdef PLASCAN_SOURCE_DIR
    roots.append(QStringLiteral(PLASCAN_SOURCE_DIR));
#endif
    roots.append(QDir::currentPath());
    roots.append(QCoreApplication::applicationDirPath());

    for (const QString &root : roots)
    {
        QDir dir(root);
        for (int depth = 0; depth < 8; ++depth)
        {
            const QString path = dir.filePath(relativePath);
            if (QFileInfo::exists(path))
            {
                QFile file(path);
                if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
                {
                    return QString();
                }
                QTextStream in(&file);
                return in.readAll();
            }
            if (!dir.cdUp())
            {
                break;
            }
        }
    }
    return QString();
}

} // namespace

TEST(MatchPhotosTaskTest, RunsPairSelectionAndReportsStageSkeleton)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.pairPolicy.exhaustiveMaxImages = 3;

    xjw::matchphotos::MatchPhotosContext context;
    context.pairInput.images = makeImages(5);

    const xjw::matchphotos::MatchPhotosTask task(options);
    const xjw::matchphotos::MatchPhotosResult result = task.run(context);

    EXPECT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.algorithmPlan.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(result.algorithmPlan.matcherAlgorithm, QStringLiteral("lightglue"));
    EXPECT_TRUE(result.algorithmPlan.rotationRobust);
    EXPECT_TRUE(result.pairSelection.restrictPairs);
    ASSERT_EQ(result.stages.size(), 9);
    EXPECT_EQ(result.stages.front().status, xjw::matchphotos::MatchPhotosStageStatus::Completed);
    EXPECT_EQ(result.stages.front().stageId, QStringLiteral("algorithm_selection"));
    EXPECT_EQ(result.stages.at(1).stageId, QStringLiteral("feature"));
    EXPECT_EQ(result.stages.at(2).stageId, QStringLiteral("generic_preselection"));
    EXPECT_EQ(result.stages.at(3).stageId, QStringLiteral("reference_preselection"));
    EXPECT_EQ(result.stages.at(4).stageId, QStringLiteral("pair_selection"));
    EXPECT_EQ(result.stages.at(1).status, xjw::matchphotos::MatchPhotosStageStatus::Skipped);
}

TEST(MatchPhotosTaskTest, ExplicitSequenceModeSurvivesDisabledPreselection)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = true;
    options.useGenericPreselection = false;
    options.useReferencePreselection = false;
    options.pairPolicy.mode = xjw::matchphotos::PairSelectionMode::Sequence;
    options.pairPolicy.sequenceWindow = 1;

    xjw::matchphotos::MatchPhotosContext context;
    context.pairInput.images = makeImages(5);

    const xjw::matchphotos::MatchPhotosTask task(options);
    const xjw::matchphotos::MatchPhotosResult result = task.run(context);

    EXPECT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_TRUE(result.pairSelection.restrictPairs);
    EXPECT_EQ(result.pairSelection.allPairCount, 10);
    ASSERT_EQ(result.pairSelection.candidates.size(), 4);
    for (const xjw::matchphotos::PairCandidate &candidate : result.pairSelection.candidates)
    {
        EXPECT_EQ(candidate.pair.indexB - candidate.pair.indexA, 1);
    }
}

TEST(MatchPhotosTaskTest, ReportsDetailedProgressThroughContextCallback)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.pairPolicy.exhaustiveMaxImages = 3;

    xjw::matchphotos::MatchPhotosContext context;
    context.pairInput.images = makeImages(4);

    std::vector<QString> stageIds;
    std::vector<QString> messages;
    context.progressCallback =
        [&stageIds, &messages](const QString &stageId,
                               const QString &message,
                               int current,
                               int maximum)
    {
        stageIds.push_back(stageId);
        messages.push_back(message);
        EXPECT_GE(current, 0);
        EXPECT_GE(maximum, 1);
    };

    const xjw::matchphotos::MatchPhotosTask task(options);
    const xjw::matchphotos::MatchPhotosResult result = task.run(context);

    EXPECT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_NE(std::find(stageIds.begin(), stageIds.end(), QStringLiteral("algorithm_selection")),
              stageIds.end());
    EXPECT_NE(std::find(stageIds.begin(), stageIds.end(), QStringLiteral("feature")),
              stageIds.end());
    EXPECT_NE(std::find(stageIds.begin(), stageIds.end(), QStringLiteral("pair_selection")),
              stageIds.end());
    EXPECT_TRUE(std::any_of(messages.begin(), messages.end(), [](const QString &message)
    {
        return message.contains(QStringLiteral("特征"));
    }));
    EXPECT_TRUE(std::any_of(messages.begin(), messages.end(), [](const QString &message)
    {
        return message.contains(QStringLiteral("影像对"));
    }));
}

TEST(MatchPhotosTaskTest, FeatureStageWritesSiftFilesForSyntheticImages)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = writeSyntheticSiftImage(tempDir.path(), QStringLiteral("image0.png"), 0);
    const QString image1 = writeSyntheticSiftImage(tempDir.path(), QStringLiteral("image1.png"), 5);

    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = false;
    options.device = xjw::matchphotos::ComputeDevice::Cpu;
    options.maxImageDim = 160;
    options.maxKeypoints = 128;
    options.reuseExistingFeatures = false;

    xjw::matchphotos::MatchPhotosContext context;
    context.featureDirectory = QDir(tempDir.path()).filePath(QStringLiteral("ip"));
    context.pairInput.images = QStringList{image0, image1};

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);
    const xjw::matchphotos::FeatureStage stage;
    std::vector<xjw::matchphotos::MatchPhotosFeatureRecord> records;
    const xjw::matchphotos::MatchPhotosStageReport report =
        stage.run(context, options, plan, &records);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Completed)
        << report.message.toStdString();
    ASSERT_EQ(records.size(), 2u);
    for (const xjw::matchphotos::MatchPhotosFeatureRecord &record : records)
    {
        EXPECT_TRUE(QFile::exists(record.featurePath));
        EXPECT_EQ(FeatureFileIO::peekAlgorithm(record.featurePath), "sift");
        EXPECT_GT(FeatureFileIO::peekCount(record.featurePath), 0);
    }
}

TEST(MatchPhotosTaskTest, FeatureStageUsesTraditionalFeatureConfig)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/FeatureStage.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("TraditionalFeatureConfig config")));
    EXPECT_FALSE(source.contains(QStringLiteral("SuperPointConfig config")));
}

TEST(MatchPhotosTaskTest, FeatureStageUsesDenseSiftThresholdForTiePointExtraction)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/FeatureStage.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("imageConfig.detectionThreshold")));
    EXPECT_TRUE(source.contains(QStringLiteral("0.0005f")))
        << "连接点生成使用 CUDA SIFT 时不能沿用 TraditionalFeatureConfig 默认 0.005，"
           "同时不能低于已验证的 0.0005，否则弱响应点会干扰 LightGlue 的跨视角对应。";
    EXPECT_TRUE(source.contains(QStringLiteral("0.003f")))
        << "快速模式仍应保留较高阈值，避免大项目生成过密候选点。";
}

TEST(MatchPhotosTaskTest, ExplicitCudaSiftDoesNotSilentlyFallBackToCpu)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/FeatureStage.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("config.allowDeviceFallback = options.device != ComputeDevice::Cuda")))
        << "空三默认请求 CUDA SIFT 后，CUDA 后端不可用时必须报错，不能静默改跑 CPU SIFT。";
}

TEST(MatchPhotosTaskTest, GenericPreselectionUsesConnectedNoCameraPairGraphDefaults)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/task/MatchPhotosTask.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int configStart = source.indexOf(QStringLiteral("VocabularyOverlapConfig makeVocabularyConfig"));
    ASSERT_GE(configStart, 0);
    const int configEnd = source.indexOf(QStringLiteral("MatchPhotosStageReport makeVocabularyPreselectionReport"),
                                         configStart);
    ASSERT_GT(configEnd, configStart);
    const QString configBody = source.mid(configStart, configEnd - configStart);

    EXPECT_TRUE(configBody.contains(QStringLiteral("std::max(8, options.pairPolicy.sequenceWindow * 2)")));
    EXPECT_TRUE(configBody.contains(QStringLiteral("config.minPairsPerImage")));
    EXPECT_TRUE(configBody.contains(QStringLiteral("std::max(4, options.pairPolicy.sequenceWindow)")));
    EXPECT_TRUE(configBody.contains(QStringLiteral("config.connectComponents = true")));
    EXPECT_TRUE(configBody.contains(QStringLiteral("config.useSequenceFallback = true")));
    EXPECT_TRUE(configBody.contains(QStringLiteral("config.sequenceWindow")));
    EXPECT_TRUE(configBody.contains(QStringLiteral("config.closeSequenceLoop = true")));
}

TEST(MatchPhotosTaskTest, GenericPreselectionReportsRetrieverProgress)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/task/MatchPhotosTask.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(QStringLiteral("bool buildVocabularyPreselection"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(QStringLiteral("MatchPhotosStageReport makeReferencePreselectionReport"), start);
    ASSERT_GT(end, start);
    const QString body = source.mid(start, end - start);

    EXPECT_TRUE(body.contains(QStringLiteral("Generic 预选")));
    EXPECT_TRUE(body.contains(QStringLiteral("reportMatchPhotosProgress")));
    EXPECT_TRUE(body.contains(QStringLiteral("QString::fromStdString(stage)")));
}

TEST(MatchPhotosMaskSupportTest, KeypointMaskRemovesDescriptorsInMaskedPixels)
{
    FeatureOutput output;
    output.imageWidth = 4;
    output.imageHeight = 4;
    output.keypoints = {
        cv::KeyPoint(cv::Point2f(1.0f, 1.0f), 1.0f),
        cv::KeyPoint(cv::Point2f(3.0f, 1.0f), 1.0f)
    };
    output.scores = {0.9f, 0.8f};
    output.descriptors = torch::tensor({{1.0f, 0.0f}, {0.0f, 1.0f}}, torch::kFloat32);

    cv::Mat mask(4, 4, CV_8UC1, cv::Scalar(0));
    mask.at<uchar>(1, 3) = 255;

    const FeatureOutput filtered = xjw::matchphotos::filterFeatureOutputByMask(output, mask);

    ASSERT_EQ(filtered.keypoints.size(), 1u);
    ASSERT_EQ(filtered.scores.size(), 1u);
    EXPECT_FLOAT_EQ(filtered.keypoints.front().pt.x, 1.0f);
    ASSERT_TRUE(filtered.descriptors.defined());
    ASSERT_EQ(filtered.descriptors.size(0), 1);
    ASSERT_EQ(filtered.descriptors.size(1), 2);
    EXPECT_FLOAT_EQ(filtered.descriptors.index({0, 0}).item<float>(), 1.0f);
}

TEST(MatchPhotosMaskSupportTest, TiepointMaskRemovesMatchesTouchingMaskedPixels)
{
    xjw::feature_extractors::FeatureData feature0;
    feature0.keypoints = {
        cv::KeyPoint(cv::Point2f(1.0f, 1.0f), 1.0f),
        cv::KeyPoint(cv::Point2f(2.0f, 1.0f), 1.0f)
    };
    xjw::feature_extractors::FeatureData feature1;
    feature1.keypoints = {
        cv::KeyPoint(cv::Point2f(1.0f, 1.0f), 1.0f),
        cv::KeyPoint(cv::Point2f(3.0f, 1.0f), 1.0f)
    };

    const std::vector<cv::DMatch> matches = {
        cv::DMatch(0, 0, 0.1f),
        cv::DMatch(1, 1, 0.2f)
    };
    xjw::feature_match::MatchResult result =
        xjw::feature_match::MatchResult::fromCvMatches(matches, 2, 2, "lightglue");

    cv::Mat mask0(4, 4, CV_8UC1, cv::Scalar(0));
    cv::Mat mask1(4, 4, CV_8UC1, cv::Scalar(0));
    mask1.at<uchar>(1, 3) = 255;

    const xjw::feature_match::MatchResult filtered =
        xjw::matchphotos::filterMatchResultByMasks(result, feature0, feature1, mask0, mask1);

    ASSERT_EQ(filtered.cvMatches.size(), 1u);
    EXPECT_EQ(filtered.cvMatches.front().queryIdx, 0);
    EXPECT_EQ(filtered.cvMatches.front().trainIdx, 0);
    EXPECT_EQ(filtered.numMatches, 1);
    ASSERT_EQ(filtered.matches0.size(), 2u);
    EXPECT_EQ(filtered.matches0[0], 0);
    EXPECT_EQ(filtered.matches0[1], -1);
}

TEST(MatchPhotosTaskTest, MatchingStageUsesMemoryAwareLightGlueBudget)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/MatchingStage.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("resolveLightGlueKeypointBudget")));
    EXPECT_TRUE(source.contains(QStringLiteral("budgetFeatureDataForLightGlue")));
    EXPECT_TRUE(source.contains(QStringLiteral("remapLightGlueMatchResultToOriginal")));
    EXPECT_TRUE(source.contains(QStringLiteral("lightglue_keypoint_budget")));
}

TEST(MatchPhotosTaskTest, MatchSidecarCarriesAerialTiePointFrontendSignature)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = QDir(tempDir.path()).filePath(QStringLiteral("left.png"));
    const QString image1 = QDir(tempDir.path()).filePath(QStringLiteral("right.png"));
    const QString feature0Path = QDir(tempDir.path()).filePath(QStringLiteral("left.sift"));
    const QString feature1Path = QDir(tempDir.path()).filePath(QStringLiteral("right.sift"));
    const QString matchPath = QDir(tempDir.path()).filePath(QStringLiteral("left__right_lightglue.match"));
    const QString sidecarPath = matchPath + QStringLiteral(".json");

    xjw::matchphotos::ResolvedImagePair pair;
    pair.image0Path = image0;
    pair.image1Path = image1;
    pair.pairName = QStringLiteral("left__right");
    pair.pairKey = QStringLiteral("left\nright");

    xjw::feature_extractors::FeatureData feature0;
    feature0.keypoints = {cv::KeyPoint(cv::Point2f(10.0f, 11.0f), 1.0f)};
    xjw::feature_extractors::FeatureData feature1;
    feature1.keypoints = {cv::KeyPoint(cv::Point2f(12.0f, 13.0f), 1.0f)};

    xjw::feature_match::MatchResult matchResult =
        xjw::feature_match::MatchResult::fromCvMatches({cv::DMatch(0, 0, 0.1f)}, 1, 1, "lightglue");

    xjw::matchphotos::MatchPhotosOptions options;
    options.featureAlgorithm = QStringLiteral("sift");
    options.matcherAlgorithm = QStringLiteral("lightglue");
    options.maxKeypoints = 40000;
    options.keypointLimitPerMegapixel = 0;
    options.matchThreshold = 0.15f;

    xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    ASSERT_TRUE(xjw::matchphotos::writeMatchPhotosSidecar(sidecarPath,
                                                          pair,
                                                          feature0Path,
                                                          feature1Path,
                                                          matchPath,
                                                          feature0,
                                                          feature1,
                                                          matchResult,
                                                          plan,
                                                          options));

    QFile sidecarFile(sidecarPath);
    ASSERT_TRUE(sidecarFile.open(QIODevice::ReadOnly));
    const QJsonObject sidecar = QJsonDocument::fromJson(sidecarFile.readAll()).object();

    EXPECT_EQ(sidecar.value(QStringLiteral("tie_point_frontend_version")).toInt(), 3);
    EXPECT_EQ(sidecar.value(QStringLiteral("tie_point_feature_max_keypoints")).toInt(), 40000);
    EXPECT_EQ(sidecar.value(QStringLiteral("tie_point_keypoint_limit_per_megapixel")).toInt(), 0);
    EXPECT_NEAR(sidecar.value(QStringLiteral("dense_sift_threshold")).toDouble(), 0.0005, 1e-9);
}

TEST(MatchPhotosTaskTest, StoredPairGeometryAuditRecomputesMissingStatistics)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    constexpr int point_count = 32;
    FeatureOutput feature0;
    FeatureOutput feature1;
    feature0.imageWidth = 640;
    feature0.imageHeight = 480;
    feature1.imageWidth = 640;
    feature1.imageHeight = 480;
    feature0.descriptors = torch::zeros({point_count, 16});
    feature1.descriptors = torch::zeros({point_count, 16});
    std::vector<cv::DMatch> matches;
    for (int index = 0; index < point_count; ++index)
    {
        const int column = index % 8;
        const int row = index / 8;
        const double world_x = (column - 3.5) * 0.25;
        const double world_y = (row - 1.5) * 0.25;
        const double world_z = 4.0 + 0.1 * (index % 3);
        const cv::Point2f point0(
            static_cast<float>(500.0 * world_x / world_z + 320.0),
            static_cast<float>(500.0 * world_y / world_z + 240.0));
        const cv::Point2f point1(
            static_cast<float>(500.0 * (world_x - 0.2) / world_z + 320.0),
            static_cast<float>(500.0 * world_y / world_z + 240.0));
        feature0.keypoints.emplace_back(point0, 1.0f);
        feature1.keypoints.emplace_back(point1, 1.0f);
        feature0.scores.push_back(1.0f);
        feature1.scores.push_back(1.0f);
        matches.emplace_back(index, index, 0.0f);
    }

    const QString feature0_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("left.sift"));
    const QString feature1_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("right.sift"));
    const QString match_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("left__right.match"));
    const QString sidecar_path = match_path + QStringLiteral(".json");
    ASSERT_TRUE(FeatureFileIO::write(
        feature0_path, QStringLiteral("left.png"), feature0, "sift"));
    ASSERT_TRUE(FeatureFileIO::write(
        feature1_path, QStringLiteral("right.png"), feature1, "sift"));
    const xjw::feature_match::MatchResult match_result =
        xjw::feature_match::MatchResult::fromCvMatches(
            matches, point_count, point_count, "lightglue");
    ASSERT_TRUE(xjw::feature_match::writeIndexedMatchFile(
        match_path,
        QStringLiteral("left.png"),
        QStringLiteral("right.png"),
        match_result));

    QFile sidecar_file(sidecar_path);
    ASSERT_TRUE(sidecar_file.open(QIODevice::WriteOnly));
    sidecar_file.write(QJsonDocument(QJsonObject{
        {QStringLiteral("feature0_path"), feature0_path},
        {QStringLiteral("feature1_path"), feature1_path}
    }).toJson());
    sidecar_file.close();

    const xjw::matchphotos::StoredPairGeometryAuditResult audit =
        xjw::matchphotos::auditStoredPairGeometry(
            match_path, sidecar_path, 20, 1.5);

    EXPECT_TRUE(audit.statisticsAvailable);
    EXPECT_TRUE(audit.verified);
    EXPECT_EQ(audit.totalMatches, point_count);
    EXPECT_GE(audit.geometricInliers, 20);
    EXPECT_GT(audit.inlierRatio, 0.60);
    EXPECT_GT(audit.coverageScore, 0.20);
    EXPECT_EQ(audit.reason, QStringLiteral("verified_from_stored_matches"));

    const QString empty_match_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("left__right_empty.match"));
    const xjw::feature_match::MatchResult empty_match_result =
        xjw::feature_match::MatchResult::fromCvMatches(
            {}, point_count, point_count, "lightglue");
    ASSERT_TRUE(xjw::feature_match::writeIndexedMatchFile(
        empty_match_path,
        QStringLiteral("left.png"),
        QStringLiteral("right.png"),
        empty_match_result));
    const xjw::matchphotos::StoredPairGeometryAuditResult empty_audit =
        xjw::matchphotos::auditStoredPairGeometry(
            empty_match_path, sidecar_path, 20, 1.5);
    EXPECT_FALSE(empty_audit.statisticsAvailable);
    EXPECT_FALSE(empty_audit.verified);
    EXPECT_EQ(empty_audit.totalMatches, 0);
    EXPECT_EQ(empty_audit.geometricInliers, 0);
    EXPECT_EQ(
        empty_audit.reason, QStringLiteral("stored_match_evidence_absent"));

    const QString weak_match_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("left__right_weak.match"));
    const std::vector<cv::DMatch> weak_matches(
        matches.cbegin(), matches.cbegin() + 10);
    const xjw::feature_match::MatchResult weak_match_result =
        xjw::feature_match::MatchResult::fromCvMatches(
            weak_matches, point_count, point_count, "lightglue");
    ASSERT_TRUE(xjw::feature_match::writeIndexedMatchFile(
        weak_match_path,
        QStringLiteral("left.png"),
        QStringLiteral("right.png"),
        weak_match_result));
    const xjw::matchphotos::StoredPairGeometryAuditResult weak_audit =
        xjw::matchphotos::auditStoredPairGeometry(
            weak_match_path, sidecar_path, 20, 1.5);
    EXPECT_FALSE(weak_audit.statisticsAvailable);
    EXPECT_FALSE(weak_audit.verified);
    EXPECT_EQ(weak_audit.totalMatches, 10);
    EXPECT_EQ(
        weak_audit.reason,
        QStringLiteral("stored_match_evidence_insufficient"));
}

TEST(MatchPhotosTaskTest, GeometryAndTrackStagesUseExistingCoreImplementations)
{
    const QString geometrySource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/GeometryVerifyStage.cpp"));
    const QString trackSource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/TrackBuildStage.cpp"));
    const QString resultHeader =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/task/MatchPhotosResult.h"));
    const QString tiePointSource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/tie_points/TiePointTrackManager.cpp"));
    ASSERT_FALSE(geometrySource.isEmpty());
    ASSERT_FALSE(trackSource.isEmpty());
    ASSERT_FALSE(resultHeader.isEmpty());
    ASSERT_FALSE(tiePointSource.isEmpty());

    EXPECT_TRUE(geometrySource.contains(QStringLiteral("MatchGeometryFilter::filter")));
    EXPECT_FALSE(geometrySource.contains(QStringLiteral("几何验证阶段尚未接入")));
    EXPECT_TRUE(trackSource.contains(QStringLiteral("TiePointTrackManager")));
    EXPECT_FALSE(trackSource.contains(QStringLiteral("FeatureFileIO::readData")));
    EXPECT_TRUE(tiePointSource.contains(QStringLiteral("MultiViewTrackBuilder")));
    EXPECT_FALSE(resultHeader.contains(QStringLiteral("std::vector<Track> tracks")));
    EXPECT_FALSE(trackSource.contains(QStringLiteral("result->tracks = buildResult.tracks")));
    EXPECT_TRUE(trackSource.contains(QStringLiteral("result->tiePointPath = buildResult.tiePointPath")));
    EXPECT_TRUE(tiePointSource.contains(QStringLiteral("latest_tie_points.json")));
    EXPECT_TRUE(tiePointSource.contains(QStringLiteral("plascan_tie_points")));
    EXPECT_FALSE(trackSource.contains(QStringLiteral("轨迹构建阶段尚未接入")));
}

TEST(MatchPhotosTaskTest, GeometryQualityRejectsWeakAmbiguousPair)
{
    // temple 3-7 像对的 35/59 内点是重复结构产生的伪几何模型，
    // 不能仅因为超过 20 个内点就送入多视轨迹构建。
    EXPECT_FALSE(xjw::matchphotos::passesGeometryQualityGate(59, 35, 20));
}

TEST(MatchPhotosTaskTest, GeometryQualityKeepsCleanLowSupportPair)
{
    // 真实宽基线像对可能内点数不高，但内点率足够高时仍应保留。
    EXPECT_TRUE(xjw::matchphotos::passesGeometryQualityGate(49, 40, 20));
    EXPECT_TRUE(xjw::matchphotos::passesGeometryQualityGate(22, 20, 20));
}

TEST(MatchPhotosTaskTest, GeometryQualityKeepsStrongSupportPair)
{
    // 高内点数本身已经提供足够稳定的两视几何，不应被内点率误杀。
    EXPECT_TRUE(xjw::matchphotos::passesGeometryQualityGate(500, 100, 20));
    EXPECT_FALSE(xjw::matchphotos::passesGeometryQualityGate(100, 19, 20));
}

TEST(MatchPhotosTaskTest, GeometryStageRecoversDenseAndDisconnectedSiftLightGluePairs)
{
    const QString geometrySource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/GeometryVerifyStage.cpp"));
    const QString recoverySource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/SiftLightGlueRecovery.cpp"));
    ASSERT_FALSE(geometrySource.isEmpty());
    ASSERT_FALSE(recoverySource.isEmpty());

    // LightGlue 仍是主匹配器；只有在高精度强重叠像对被预算截断，或实际匹配图
    // 不连通时，才使用全量 SIFT 描述子做按需恢复。
    EXPECT_TRUE(recoverySource.contains(QStringLiteral("TraditionalFeatureMatcher::match")));
    EXPECT_TRUE(geometrySource.contains(QStringLiteral("shouldAugmentDenseSiftPair")));
    EXPECT_TRUE(geometrySource.contains(QStringLiteral("disconnectedRecoveryCandidates")));

    // 恢复结果必须同步落盘，否则后续空三服务会继续读到旧的弱匹配缓存。
    EXPECT_TRUE(recoverySource.contains(QStringLiteral("writeIndexedMatchFile")));
    EXPECT_TRUE(recoverySource.contains(QStringLiteral("writeMatchPhotosSidecar")));
}

TEST(MatchPhotosTaskTest, GuidedKeypointLimitPerMegapixelScalesByImageSize)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.enableGuidedMatching = true;
    options.keypointLimitPerMegapixel = 1000;
    options.useExplicitKeypointLimit = true;
    options.maxKeypoints = 40000;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_EQ(xjw::matchphotos::resolveFeatureKeypointLimit(options, plan, 6000, 4000),
              24000);
    EXPECT_EQ(xjw::matchphotos::resolveFeatureKeypointLimit(options, plan, 160, 160),
              26);
}

TEST(MatchPhotosTaskTest, GuidedMatchStageReportsEnabledDensityMode)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.enableGuidedMatching = true;
    options.keypointLimitPerMegapixel = 1000;

    const xjw::matchphotos::GuidedMatchStage stage;
    const xjw::matchphotos::MatchPhotosContext context;
    const xjw::matchphotos::MatchPhotosStageReport report = stage.run(context, options);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Completed);
    EXPECT_TRUE(report.message.contains(QStringLiteral("每百万像素")));
}

TEST(MatchPhotosTaskTest, TiePointPersistenceStreamsTracksWithoutWholeJsonTree)
{
    const QString tiePointSource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/tie_points/TiePointTrackManager.cpp"));
    ASSERT_FALSE(tiePointSource.isEmpty());

    EXPECT_TRUE(tiePointSource.contains(QStringLiteral("writeTiePointFile(outputPath,")));
    EXPECT_FALSE(tiePointSource.contains(QStringLiteral("QJsonArray tracks;")));
    EXPECT_FALSE(tiePointSource.contains(QStringLiteral("makeTiePointFileObject")));
    EXPECT_FALSE(tiePointSource.contains(QStringLiteral("QJsonDocument(object).toJson(QJsonDocument::Indented)")));
}

TEST(MatchPhotosTaskTest, ResultDropsTransientInlierPairsBeforeReturningToGui)
{
    const QString taskSource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/task/MatchPhotosTask.cpp"));
    ASSERT_FALSE(taskSource.isEmpty());

    EXPECT_TRUE(taskSource.contains(QStringLiteral("clearTransientMatchPayloads")));
    EXPECT_TRUE(taskSource.contains(QStringLiteral("record.inlierIndexPairs.swap")));
    EXPECT_TRUE(taskSource.contains(QStringLiteral("const MatchPhotosStageReport geometryReport")));
    EXPECT_TRUE(taskSource.contains(QStringLiteral("const MatchPhotosStageReport trackReport")));
}
