#include "MvsWorkspaceManifest.h"
#include "DepthFrameUtils.h"
#include "DepthMapGenerator.h"
#include "MvsQualityReport.h"
#include "MvsTypes.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonObject>
#include <QTemporaryDir>

#include <chrono>

using xjw::mvs::MvsDepthFrameRecord;
using xjw::mvs::MvsWorkspaceManifest;
using xjw::mvs::DepthMapGenerator;

namespace
{
MvsDepthFrameRecord makeRecord(int index, const QString &name, const QString &status)
{
    MvsDepthFrameRecord record;
    record.refIndex = index;
    record.refImage = name;
    record.status = status;
    record.device = QStringLiteral("GPU");
    record.depthPng = QStringLiteral("depth_%1.png").arg(index, 3, 10, QLatin1Char('0'));
    record.rawDepthPath = QStringLiteral("depth_%1.bin").arg(index, 3, 10, QLatin1Char('0'));
    record.rawConfidencePath = QStringLiteral("confidence_%1.bin").arg(index, 3, 10, QLatin1Char('0'));
    record.rawGeometrySupportPath = QStringLiteral("geometry_support_%1.bin")
                                        .arg(index, 3, 10, QLatin1Char('0'));
    record.validMaskPath = QStringLiteral("mask_%1.png").arg(index, 3, 10, QLatin1Char('0'));
    record.gridWidth = 6000;
    record.gridHeight = 4000;
    record.elapsedMs = 1000 + index;
    record.configHash = QStringLiteral("cfg-a");
    record.sourceImages = {QStringLiteral("source_a.jpg"), QStringLiteral("source_b.jpg")};
    record.sourceIndices = {7, 9};
    record.rawGeometrySourceMaskPath = QStringLiteral("geometry_source_mask_%1.bin")
                                           .arg(index, 3, 10, QLatin1Char('0'));
    record.rawInverseDepthMeanPath = QStringLiteral("inverse_depth_mean_%1.bin")
                                         .arg(index, 3, 10, QLatin1Char('0'));
    record.rawInverseDepthSpreadPath = QStringLiteral("inverse_depth_spread_%1.bin")
                                           .arg(index, 3, 10, QLatin1Char('0'));
    record.crossViewRepairedMaskPath = QStringLiteral("cross_view_repaired_%1.png")
                                           .arg(index, 3, 10, QLatin1Char('0'));
    QJsonObject source_plan_entry;
    source_plan_entry.insert(QStringLiteral("view_index"), 7);
    source_plan_entry.insert(QStringLiteral("source_image"), QStringLiteral("source_a.jpg"));
    source_plan_entry.insert(QStringLiteral("shared_tracks"), 42);
    source_plan_entry.insert(QStringLiteral("geometric_inliers"), 39);
    source_plan_entry.insert(QStringLiteral("score"), 123.0);
    record.sourcePlan.append(source_plan_entry);
    record.cameraModel = QJsonObject{
        {QStringLiteral("fx"), 1200.0},
        {QStringLiteral("fy"), 1210.0},
        {QStringLiteral("cx"), 640.0},
        {QStringLiteral("cy"), 360.0},
        {QStringLiteral("rotation_world_to_camera"),
         QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}},
        {QStringLiteral("translation_world_to_camera"), QJsonArray{0.0, 0.0, 2.0}},
        {QStringLiteral("camera_center"), QJsonArray{0.0, 0.0, -2.0}}
    };
    return record;
}

void touchFile(const QString &path)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    file.write("x");
}
}

TEST(MvsWorkspaceManifest, SavesAndLoadsFrameRecordsAtomically)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString manifestPath = QDir(tempDir.path()).filePath(QStringLiteral("mvs_manifest.json"));

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    manifest.upsertFrame(makeRecord(2, QStringLiteral("image_002.jpg"), QStringLiteral("completed")));

    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    MvsWorkspaceManifest loaded;
    ASSERT_TRUE(loaded.load(manifestPath, &error)) << error.toStdString();
    ASSERT_EQ(loaded.frames().size(), 1);
    EXPECT_EQ(loaded.frames().front().refImage, QStringLiteral("image_002.jpg"));
    EXPECT_EQ(loaded.frames().front().rawConfidencePath, QStringLiteral("confidence_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawGeometrySupportPath,
              QStringLiteral("geometry_support_002.bin"));
    EXPECT_EQ(loaded.frames().front().sourceIndices, QVector<int>({7, 9}));
    EXPECT_EQ(loaded.frames().front().rawGeometrySourceMaskPath,
              QStringLiteral("geometry_source_mask_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawInverseDepthMeanPath,
              QStringLiteral("inverse_depth_mean_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawInverseDepthSpreadPath,
              QStringLiteral("inverse_depth_spread_002.bin"));
    EXPECT_EQ(loaded.frames().front().crossViewRepairedMaskPath,
              QStringLiteral("cross_view_repaired_002.png"));
    EXPECT_EQ(loaded.frames().front().gridWidth, 6000);
    EXPECT_EQ(loaded.frames().front().gridHeight, 4000);
    EXPECT_EQ(loaded.configHash(), QStringLiteral("cfg-a"));
    EXPECT_DOUBLE_EQ(loaded.frames().front().cameraModel.value(QStringLiteral("fx")).toDouble(), 1200.0);
    EXPECT_EQ(loaded.frames().front().cameraModel
                  .value(QStringLiteral("rotation_world_to_camera"))
                  .toArray()
                  .size(),
              9);
    ASSERT_EQ(loaded.frames().front().sourcePlan.size(), 1);
    EXPECT_EQ(loaded.frames().front().sourcePlan.at(0).toObject().value(QStringLiteral("shared_tracks")).toInt(), 42);
}

TEST(MvsWorkspaceManifest, SortsCompletedFramesByNaturalFileName)
{
    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    manifest.upsertFrame(makeRecord(10, QStringLiteral("depth_010.png"), QStringLiteral("completed")));
    manifest.upsertFrame(makeRecord(1, QStringLiteral("depth_001.png"), QStringLiteral("failed")));
    manifest.upsertFrame(makeRecord(2, QStringLiteral("depth_002.png"), QStringLiteral("completed")));

    const auto sorted = manifest.completedFramesSortedByName();
    ASSERT_EQ(sorted.size(), 2);
    EXPECT_EQ(sorted[0].refImage, QStringLiteral("depth_002.png"));
    EXPECT_EQ(sorted[1].refImage, QStringLiteral("depth_010.png"));
}

TEST(MvsWorkspaceManifest, UpdatesFailedFrameAndInvalidatesConfigMismatch)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    manifest.upsertFrame(makeRecord(3, QStringLiteral("image_003.jpg"), QStringLiteral("running")));
    manifest.markFailed(3, QStringLiteral("cuda out of memory"));

    ASSERT_EQ(manifest.frames().size(), 1);
    EXPECT_EQ(manifest.frames().front().status, QStringLiteral("failed"));
    EXPECT_EQ(manifest.frames().front().error, QStringLiteral("cuda out of memory"));

    EXPECT_TRUE(manifest.hasReusableCompletedFrame(3, QStringLiteral("cfg-a")) == false);

    MvsDepthFrameRecord completed = makeRecord(3, QStringLiteral("image_003.jpg"), QStringLiteral("completed"));
    completed.depthPng = QDir(tempDir.path()).filePath(QStringLiteral("depth_003.png"));
    completed.rawDepthPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_003.bin"));
    touchFile(completed.depthPng);
    touchFile(completed.rawDepthPath);
    manifest.markCompleted(completed);
    EXPECT_TRUE(manifest.hasReusableCompletedFrame(3, QStringLiteral("cfg-a")));
    EXPECT_FALSE(manifest.hasReusableCompletedFrame(3, QStringLiteral("cfg-b")));
}

TEST(MvsWorkspaceManifest, CompletedFrameUpdatePreservesExistingSourcePlan)
{
    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));

    MvsDepthFrameRecord initial = makeRecord(5, QStringLiteral("image_005.jpg"), QStringLiteral("completed"));
    ASSERT_EQ(initial.sourcePlan.size(), 1);
    manifest.markCompleted(initial);

    MvsDepthFrameRecord filtered = initial;
    filtered.depthPng = QStringLiteral("filtered_depth_005.png");
    filtered.sourcePlan = QJsonArray();
    manifest.markCompleted(filtered);

    ASSERT_EQ(manifest.frames().size(), 1);
    EXPECT_EQ(manifest.frames().front().depthPng, QStringLiteral("filtered_depth_005.png"));
    ASSERT_EQ(manifest.frames().front().sourcePlan.size(), 1)
        << "Filtered depth artifact updates must not erase the source plan needed for reproducible MVS fusion";
    EXPECT_EQ(manifest.frames().front().sourcePlan.at(0).toObject().value(QStringLiteral("view_index")).toInt(), 7);
}

TEST(MvsWorkspaceManifest, LoadsLegacyRecordWithoutGeometryEvidencePaths)
{
    const QJsonObject legacy{
        {QStringLiteral("ref_index"), 4},
        {QStringLiteral("ref_image"), QStringLiteral("image_004.jpg")},
        {QStringLiteral("status"), QStringLiteral("completed")},
        {QStringLiteral("raw_depth_path"), QStringLiteral("depth_004.bin")}};

    const MvsDepthFrameRecord record = MvsDepthFrameRecord::fromJson(legacy);

    EXPECT_TRUE(record.sourceIndices.isEmpty());
    EXPECT_TRUE(record.rawGeometrySourceMaskPath.isEmpty());
    EXPECT_TRUE(record.rawInverseDepthMeanPath.isEmpty());
    EXPECT_TRUE(record.rawInverseDepthSpreadPath.isEmpty());
    EXPECT_TRUE(record.crossViewRepairedMaskPath.isEmpty());
}

TEST(MvsWorkspaceManifest, PreservesSourceQualityAndDepthConfidenceSummary)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString manifestPath = QDir(tempDir.path()).filePath(QStringLiteral("mvs_manifest.json"));

    MvsDepthFrameRecord record = makeRecord(6, QStringLiteral("image_006.jpg"), QStringLiteral("completed"));
    record.sourceViewCount = 2;
    record.meanSourceQualityScore = 0.72;
    record.minSourceQualityScore = 0.43;
    record.meanDepthConfidence = 0.81;
    record.validPixelCount = 123456;
    record.validCoverage = 0.625;
    record.supportMaskPath = QStringLiteral("support_006.png");

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    manifest.markCompleted(record);

    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    MvsWorkspaceManifest loaded;
    ASSERT_TRUE(loaded.load(manifestPath, &error)) << error.toStdString();
    ASSERT_EQ(loaded.frames().size(), 1);
    const MvsDepthFrameRecord &loadedRecord = loaded.frames().front();
    EXPECT_EQ(loadedRecord.sourceViewCount, 2);
    EXPECT_DOUBLE_EQ(loadedRecord.meanSourceQualityScore, 0.72);
    EXPECT_DOUBLE_EQ(loadedRecord.minSourceQualityScore, 0.43);
    EXPECT_DOUBLE_EQ(loadedRecord.meanDepthConfidence, 0.81);
    EXPECT_EQ(loadedRecord.validPixelCount, 123456);
    EXPECT_DOUBLE_EQ(loadedRecord.validCoverage, 0.625);
    EXPECT_EQ(loadedRecord.supportMaskPath, QStringLiteral("support_006.png"));

    const QJsonObject json = loadedRecord.toJson();
    EXPECT_EQ(json.value(QStringLiteral("source_view_count")).toInt(), 2);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("source_quality_mean")).toDouble(), 0.72);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("source_quality_min")).toDouble(), 0.43);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("depth_confidence_mean")).toDouble(), 0.81);
    EXPECT_EQ(json.value(QStringLiteral("valid_pixel_count")).toInt(), 123456);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("valid_coverage")).toDouble(), 0.625);
    EXPECT_EQ(json.value(QStringLiteral("support_mask_path")).toString(),
              QStringLiteral("support_006.png"));
}

TEST(MvsWorkspaceManifest, PreservesDepthQualityDiagnostics)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString manifestPath = QDir(tempDir.path()).filePath(QStringLiteral("mvs_manifest.json"));

    cv::Mat depth(8, 10, CV_32F, cv::Scalar(12.0f));
    cv::Mat confidence(8, 10, CV_32F, cv::Scalar(0.55f));
    confidence.at<float>(3, 4) = 0.84f;
    const QJsonObject depthQuality = xjw::mvs::depthMapQualityMetricsToJson(
        xjw::mvs::analyzeDepthMapQuality(depth, confidence, 4));

    MvsDepthFrameRecord record = makeRecord(8, QStringLiteral("image_008.jpg"), QStringLiteral("completed"));
    record.depthQuality = depthQuality;

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    manifest.markCompleted(record);

    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    MvsWorkspaceManifest loaded;
    ASSERT_TRUE(loaded.load(manifestPath, &error)) << error.toStdString();
    ASSERT_EQ(loaded.frames().size(), 1);
    const QJsonObject loadedQuality = loaded.frames().front().depthQuality;
    EXPECT_TRUE(loadedQuality.value(QStringLiteral("low_confidence_full_coverage")).toBool());
    EXPECT_GE(loadedQuality.value(QStringLiteral("recommended_fusion_confidence")).toDouble(), 0.65);
    EXPECT_DOUBLE_EQ(loaded.frames().front().toJson()
                         .value(QStringLiteral("depth_quality"))
                         .toObject()
                         .value(QStringLiteral("valid_coverage"))
                         .toDouble(),
                     1.0);
}

TEST(MvsWorkspaceManifest, PreservesQualityGateAndPyramidDiagnostics)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString manifest_path = QDir(temp_dir.path()).filePath(
        QStringLiteral("mvs_manifest.json"));
    MvsDepthFrameRecord record = makeRecord(
        10,
        QStringLiteral("image_010.jpg"),
        QStringLiteral("completed"));
    record.qualityDecision = QJsonObject{
        {QStringLiteral("acceptance"), QStringLiteral("accepted")},
        {QStringLiteral("calibrated_confidence"), 0.72}
    };
    record.depthCompleteness = QJsonObject{
        {QStringLiteral("available"), true},
        {QStringLiteral("mask_pixel_count"), 1000},
        {QStringLiteral("valid_within_mask_count"), 875},
        {QStringLiteral("valid_within_mask_ratio"), 0.875},
        {QStringLiteral("output_filter_retention_ratio"), 0.91}
    };
    record.sceneProfile = QStringLiteral("orbital_object");
    record.filterMode = QStringLiteral("mild");
    record.acceptance = QStringLiteral("accepted");
    record.maskSource = QStringLiteral("project");
    record.maskCoverage = 0.625;
    record.selectedLevel = 2;
    record.fallbackReason = QStringLiteral("level 1 failed: insufficient support");
    record.pyramidRequestedLevelCount = 3;
    record.pyramidActiveLevelCount = 2;
    record.pyramidMinimumShortSide = 160;
    record.pyramidDegradedReason = QStringLiteral(
        "image short side 480 cannot keep three pyramid levels above 160 pixels");
    record.pyramidLevels = QJsonArray{
        QJsonObject{
            {QStringLiteral("level"), 3},
            {QStringLiteral("downsample_factor"), 4},
            {QStringLiteral("valid_coverage"), 0.58},
            {QStringLiteral("mean_support_views"), 3.25},
            {QStringLiteral("depth_discontinuity_ratio"), 0.04}
        },
        QJsonObject{
            {QStringLiteral("level"), 2},
            {QStringLiteral("downsample_factor"), 2},
            {QStringLiteral("valid_coverage"), 0.61},
            {QStringLiteral("mean_support_views"), 3.75},
            {QStringLiteral("depth_discontinuity_ratio"), 0.07}
        },
        QJsonObject{
            {QStringLiteral("level"), 1},
            {QStringLiteral("downsample_factor"), 1}
        }
    };

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    manifest.markCompleted(record);

    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifest_path, &error)) << error.toStdString();

    MvsWorkspaceManifest loaded;
    ASSERT_TRUE(loaded.load(manifest_path, &error)) << error.toStdString();
    ASSERT_EQ(loaded.frames().size(), 1);
    EXPECT_EQ(loaded.frames().front().qualityDecision
                  .value(QStringLiteral("acceptance"))
                  .toString(),
              QStringLiteral("accepted"));
    EXPECT_DOUBLE_EQ(loaded.frames().front().depthCompleteness
                         .value(QStringLiteral("valid_within_mask_ratio"))
                         .toDouble(),
                     0.875);
    EXPECT_DOUBLE_EQ(loaded.frames().front().toJson()
                         .value(QStringLiteral("depth_completeness"))
                         .toObject()
                         .value(QStringLiteral("output_filter_retention_ratio"))
                         .toDouble(),
                     0.91);
    ASSERT_EQ(loaded.frames().front().pyramidLevels.size(), 3);
    EXPECT_EQ(loaded.frames().front().sceneProfile,
              QStringLiteral("orbital_object"));
    EXPECT_EQ(loaded.frames().front().filterMode, QStringLiteral("mild"));
    EXPECT_EQ(loaded.frames().front().acceptance, QStringLiteral("accepted"));
    EXPECT_EQ(loaded.frames().front().maskSource, QStringLiteral("project"));
    EXPECT_DOUBLE_EQ(loaded.frames().front().maskCoverage, 0.625);
    EXPECT_EQ(loaded.frames().front().selectedLevel, 2);
    EXPECT_EQ(loaded.frames().front().fallbackReason,
              QStringLiteral("level 1 failed: insufficient support"));
    EXPECT_EQ(loaded.frames().front().pyramidRequestedLevelCount, 3);
    EXPECT_EQ(loaded.frames().front().pyramidActiveLevelCount, 2);
    EXPECT_EQ(loaded.frames().front().pyramidMinimumShortSide, 160);
    EXPECT_TRUE(loaded.frames().front().pyramidDegradedReason.contains(
        QStringLiteral("short side 480")));
    EXPECT_EQ(loaded.frames().front().pyramidLevels.at(2)
                  .toObject()
                  .value(QStringLiteral("level"))
                  .toInt(),
              1);
    const QJsonObject level_two = loaded.frames().front().pyramidLevels.at(1).toObject();
    EXPECT_DOUBLE_EQ(level_two.value(QStringLiteral("mean_support_views")).toDouble(), 3.75);
    EXPECT_DOUBLE_EQ(level_two.value(QStringLiteral("depth_discontinuity_ratio")).toDouble(), 0.07);
    const QJsonObject frame_json = loaded.frames().front().toJson();
    EXPECT_EQ(frame_json.value(QStringLiteral("mask_source")).toString(),
              QStringLiteral("project"));
    EXPECT_DOUBLE_EQ(frame_json.value(QStringLiteral("mask_coverage")).toDouble(), 0.625);
    EXPECT_EQ(frame_json.value(QStringLiteral("selected_level")).toInt(), 2);
    EXPECT_EQ(frame_json.value(QStringLiteral("fallback_reason")).toString(),
              QStringLiteral("level 1 failed: insufficient support"));
    EXPECT_EQ(loaded.toJson().value(QStringLiteral("schema")).toString(),
              QStringLiteral("plascan.mvs.workspace.v2"));
}

TEST(MvsWorkspaceManifest, PreservesDepthPostprocessDiagnostics)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString manifestPath = QDir(tempDir.path()).filePath(QStringLiteral("mvs_manifest.json"));

    MvsDepthFrameRecord record = makeRecord(9, QStringLiteral("image_009.jpg"), QStringLiteral("completed"));
    record.depthPostprocess = QJsonObject{
        {QStringLiteral("valid_before"), 1000},
        {QStringLiteral("confidence_removed"), 120},
        {QStringLiteral("local_depth_outlier_removed"), 8},
        {QStringLiteral("speckle_removed"), 24},
        {QStringLiteral("valid_after"), 848},
        {QStringLiteral("effective_confidence_threshold"), 0.65}
    };

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    manifest.markCompleted(record);

    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    MvsWorkspaceManifest loaded;
    ASSERT_TRUE(loaded.load(manifestPath, &error)) << error.toStdString();
    ASSERT_EQ(loaded.frames().size(), 1);
    const QJsonObject postprocess = loaded.frames().front().depthPostprocess;
    EXPECT_EQ(postprocess.value(QStringLiteral("confidence_removed")).toInt(), 120);
    EXPECT_EQ(postprocess.value(QStringLiteral("local_depth_outlier_removed")).toInt(), 8);
    EXPECT_EQ(postprocess.value(QStringLiteral("speckle_removed")).toInt(), 24);
    EXPECT_EQ(loaded.frames().front().toJson()
                  .value(QStringLiteral("depth_postprocess"))
                  .toObject()
                  .value(QStringLiteral("valid_after"))
                  .toInt(),
              848);
}

TEST(MvsDepthFrameLoading, EstimatesWorkingSetAndAdaptsWorkerCountToMemory)
{
    constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
    const std::uint64_t frame_bytes =
        xjw::core::project::estimateFusionFrameWorkingSetBytes(6000, 4000, 0);

    EXPECT_GE(frame_bytes, 6000ULL * 4000ULL * 16ULL);
    EXPECT_EQ(xjw::core::project::recommendedDepthFrameLoadWorkers(16, 64ULL * gib, frame_bytes), 4);
    EXPECT_EQ(xjw::core::project::recommendedDepthFrameLoadWorkers(3, 64ULL * gib, frame_bytes), 3);
    EXPECT_EQ(xjw::core::project::recommendedDepthFrameLoadWorkers(4, gib, frame_bytes), 1);
}

TEST(MvsDepthFrameLoading, AccountsForConfiguredFusionResize)
{
    const std::uint64_t full =
        xjw::core::project::estimateFusionFrameWorkingSetBytes(6000, 4000, 0);
    const std::uint64_t resized =
        xjw::core::project::estimateFusionFrameWorkingSetBytes(6000, 4000, 2048);

    EXPECT_GT(full, resized);
    EXPECT_GE(resized, 2048ULL * 1365ULL * 16ULL);
}

TEST(MvsWorkspaceManifest, CompletedFrameIsNotReusableWhenArtifactsAreMissing)
{
    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    manifest.markCompleted(makeRecord(4, QStringLiteral("image_004.jpg"), QStringLiteral("completed")));

    EXPECT_FALSE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")));
}

TEST(MvsWorkspaceManifest, DepthConfigHashChangesWhenRelevantSettingsChange)
{
    xjw::mvs::DepthGenConfig config;
    config.patchMatch.numIterations = 6;
    config.patchMatch.downsampleFactor = 2;
    config.patchMatch.confidenceThresh = 0.25f;
    config.fusion.minConsistentViews = 2;
    config.numSourceViews = 4;

    const QString hashA = xjw::mvs::makeMvsDepthConfigHash(config, 444);
    const QString hashB = xjw::mvs::makeMvsDepthConfigHash(config, 444);
    EXPECT_FALSE(hashA.isEmpty());
    EXPECT_EQ(hashA, hashB);

    config.patchMatch.downsampleFactor = 4;
    const QString hashC = xjw::mvs::makeMvsDepthConfigHash(config, 444);
    EXPECT_NE(hashA, hashC);

    const auto expect_hash_change = [&config, &hashC](const auto &mutator) {
        xjw::mvs::DepthGenConfig changed = config;
        mutator(changed);
        EXPECT_NE(hashC, xjw::mvs::makeMvsDepthConfigHash(changed, 444));
    };
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.patchMatch.bilateralD += 2;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.patchMatch.bilateralSigmaColor += 1.0f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.patchMatch.bilateralSigmaSpace += 1.0f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.patchMatch.cudaUseParallelSweep = !changed.patchMatch.cudaUseParallelSweep;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.fusion.enableAdaptiveConfidenceFilter =
            !changed.fusion.enableAdaptiveConfidenceFilter;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.fusion.adaptiveFullCoverageThreshold -= 0.01f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.fusion.adaptiveLowMeanConfidenceThreshold -= 0.01f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.fusion.adaptiveStrictConfidenceThreshold -= 0.01f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.depthFilterMode = xjw::mvs::DepthFilterMode::Aggressive;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.fusion.maxLocalDepthOutlierRemovalRatio -= 0.01f;
    });

    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.inputSignature = "at-generation-2";
    });
}

TEST(MvsDepthPostprocess, RemovesIsolatedDepthSpikeAndConfidence)
{
    cv::Mat depth(9, 9, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(9, 9, CV_32F, cv::Scalar(1.0f));
    depth.at<float>(4, 4) = 100.0f;

    const int removed = DepthMapGenerator::removeLocalDepthOutliers(
        depth,
        confidence,
        3,
        0.25f,
        0.20f,
        4);

    EXPECT_EQ(removed, 1);
    EXPECT_EQ(depth.at<float>(4, 4), 0.0f);
    EXPECT_EQ(confidence.at<float>(4, 4), 0.0f);
    EXPECT_EQ(depth.at<float>(4, 3), 10.0f);
}

TEST(MvsDepthPostprocess, RemovesSmallConnectedDepthComponentAndConfidence)
{
    cv::Mat depth(12, 12, CV_32F, cv::Scalar(0.0f));
    cv::Mat confidence(12, 12, CV_32F, cv::Scalar(0.0f));
    depth(cv::Rect(3, 3, 6, 6)).setTo(10.0f);
    confidence(cv::Rect(3, 3, 6, 6)).setTo(1.0f);
    depth(cv::Rect(0, 0, 2, 2)).setTo(9.0f);
    confidence(cv::Rect(0, 0, 2, 2)).setTo(0.8f);

    const int removed = DepthMapGenerator::removeSmallDepthComponents(
        depth,
        confidence,
        8,
        0.20f,
        5);

    EXPECT_EQ(removed, 4);
    EXPECT_EQ(depth.at<float>(0, 0), 0.0f);
    EXPECT_EQ(confidence.at<float>(0, 0), 0.0f);
    EXPECT_EQ(depth.at<float>(5, 5), 10.0f);
    EXPECT_EQ(confidence.at<float>(5, 5), 1.0f);
}

TEST(MvsDepthPostprocess, RemovesManySmallComponentsWithoutRepeatedFullImageScans)
{
    constexpr int kImageSize = 1024;
    cv::Mat depth(kImageSize, kImageSize, CV_32F, cv::Scalar(0.0f));
    cv::Mat confidence(kImageSize, kImageSize, CV_32F, cv::Scalar(0.0f));
    int component_count = 0;
    for (int y = 8; y < kImageSize; y += 16)
    {
        for (int x = 8; x < kImageSize; x += 16)
        {
            depth.at<float>(y, x) = 10.0f;
            confidence.at<float>(y, x) = 1.0f;
            ++component_count;
        }
    }

    const auto start = std::chrono::steady_clock::now();
    const int removed = DepthMapGenerator::removeSmallDepthComponents(
        depth,
        confidence,
        4,
        1.0f,
        6);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    EXPECT_EQ(removed, component_count);
    EXPECT_EQ(cv::countNonZero(depth > 0.0f), 0);
    EXPECT_EQ(cv::countNonZero(confidence > 0.0f), 0);
    EXPECT_LT(elapsed.count(), 5000)
        << "Speckle filtering should remain linear in pixel and component counts.";
}

TEST(MvsDepthPostprocess, PostprocessReportsSmallComponentRemoval)
{
    cv::Mat depth(12, 12, CV_32F, cv::Scalar(0.0f));
    cv::Mat confidence(12, 12, CV_32F, cv::Scalar(1.0f));
    depth(cv::Rect(3, 3, 6, 6)).setTo(10.0f);
    depth(cv::Rect(0, 0, 2, 2)).setTo(9.0f);

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.0f;
    config.enableLocalDepthOutlierFilter = false;
    config.enableSpeckleFilter = true;
    config.minSpeckleComponentArea = 8;

    const auto stats = DepthMapGenerator::postprocessFusionDepthMap(
        depth,
        confidence,
        config,
        6,
        4);

    EXPECT_EQ(stats.smallComponentRemoved, 4);
    EXPECT_EQ(stats.validAfterPostprocess, 36);
}
