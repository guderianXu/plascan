#include "MvsWorkspaceManifest.h"
#include "MvsWorkspaceReplay.h"
#include "DepthFrameUtils.h"
#include "DepthMapGenerator.h"
#include "MvsImagePreprocessor.h"
#include "MvsQualityReport.h"
#include "MvsTypes.h"
#include "io/ImageIO.h"
#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonObject>
#include <QJsonDocument>
#include <QTemporaryDir>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdint>

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
    record.rawAdaptiveGeometrySupportWeightPath =
        QStringLiteral("adaptive_geometry_support_weight_%1.bin")
            .arg(index, 3, 10, QLatin1Char('0'));
    record.rawAdaptiveGeometryEffectiveViewCountPath =
        QStringLiteral("adaptive_geometry_effective_view_count_%1.bin")
            .arg(index, 3, 10, QLatin1Char('0'));
    record.rawAdaptiveGeometryConflictRatioPath =
        QStringLiteral("adaptive_geometry_conflict_ratio_%1.bin")
            .arg(index, 3, 10, QLatin1Char('0'));
    record.validMaskPath = QStringLiteral("mask_%1.png").arg(index, 3, 10, QLatin1Char('0'));
    record.missingReasonPath = QStringLiteral("missing_reason_%1.png")
                                   .arg(index, 3, 10, QLatin1Char('0'));
    record.missingReasonPreviewPath =
        QStringLiteral("missing_reason_preview_%1.png")
            .arg(index, 3, 10, QLatin1Char('0'));
    record.missingReasonSummary = QJsonObject{
        {QStringLiteral("missing_pixel_count"), 123},
        {QStringLiteral("schema_version"), 1}};
    record.gridWidth = 6000;
    record.gridHeight = 4000;
    record.elapsedMs = 1000 + index;
    record.configHash = QStringLiteral("cfg-a");
    record.algorithmRevision = xjw::mvs::kMvsDepthAlgorithmRevision;
    record.sourceImages = {QStringLiteral("source_a.jpg"), QStringLiteral("source_b.jpg")};
    record.sourceIndices = {7, 9};
    record.geometrySourceIndices = {7, 9, 11};
    record.acceptance = QStringLiteral("accepted");
    record.fusionEligible = true;
    record.fusionEligibilityKnown = true;
    record.sceneProfile = QStringLiteral("aerial_terrain");
    record.qualityProfile = QStringLiteral("highest");
    record.configuredSourceViewCount = 8;
    record.sourceViewCount = 2;
    record.requestedSourceViewCount = 4;
    record.sourceViewShortfall = 2;
    record.sourceViewShortfallReason =
        QStringLiteral("missing_pair_verification_statistics");
    record.crossViewRepairDiagnostics = QJsonObject{
        {QStringLiteral("considered_hole_pixel_count"), 20},
        {QStringLiteral("repaired_pixel_count"), 12},
        {QStringLiteral("anchored_interpolation"),
         QJsonObject{{QStringLiteral("interpolated_pixel_count"), 8}}}
    };
    record.geometryEvidenceDiagnostics = QJsonObject{
        {QStringLiteral("valid_inputs"), true},
        {QStringLiteral("native_valid_ratio"), 0.75},
        {QStringLiteral("repaired_valid_ratio"), 0.10}
    };
    record.poseRefinementDiagnostics = QJsonObject{
        {QStringLiteral("enabled"), true},
        {QStringLiteral("candidate_only"), true},
        {QStringLiteral("accepted"), true},
        {QStringLiteral("reason"), QStringLiteral("accepted_candidate")}
    };
    record.derivedCameraModel = QJsonObject{
        {QStringLiteral("fx"), 1200.0},
        {QStringLiteral("camera_center"), QJsonArray{0.01, 0.02, -1.99}}
    };
    record.rawGeometrySourceMaskPath = QStringLiteral("geometry_source_mask_%1.bin")
                                           .arg(index, 3, 10, QLatin1Char('0'));
    record.rawInverseDepthMeanPath = QStringLiteral("inverse_depth_mean_%1.bin")
                                         .arg(index, 3, 10, QLatin1Char('0'));
    record.rawInverseDepthSpreadPath = QStringLiteral("inverse_depth_spread_%1.bin")
                                           .arg(index, 3, 10, QLatin1Char('0'));
    record.crossViewRepairedMaskPath = QStringLiteral("cross_view_repaired_%1.png")
                                           .arg(index, 3, 10, QLatin1Char('0'));
    record.targetedGapRecoveredMaskPath = QStringLiteral(
        "targeted_gap_recovered_%1.png").arg(
            index, 3, 10, QLatin1Char('0'));
    record.targetedGapRecoveryDiagnostics = QJsonObject{
        {QStringLiteral("attempted"), true},
        {QStringLiteral("recovered_pixel_count"), 321}};
    record.depthProvenancePath = QStringLiteral(
        "depth_provenance_%1.png").arg(
            index, 3, 10, QLatin1Char('0'));
    record.depthProvenanceSummary = QJsonObject{
        {QStringLiteral("available"), true},
        {QStringLiteral("anchored_interpolation_pixel_count"), 17}};
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

void writeJson(const QString &path, const QJsonObject &object)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_GT(file.write(QJsonDocument(object).toJson()), 0);
}

QJsonArray doubleArray(const double *values, int count)
{
    QJsonArray result;
    for (int index = 0; index < count; ++index)
    {
        result.append(values[index]);
    }
    return result;
}

QJsonObject cameraJson(const xjw::FramePinholeCamera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto rotation = camera.worldToCameraRotation();
    const auto translation = camera.worldToCameraTranslation();
    const auto center = camera.cameraCenter();
    return QJsonObject{
        {QStringLiteral("fx"), intrinsics.focalX},
        {QStringLiteral("fy"), intrinsics.focalY},
        {QStringLiteral("cx"), intrinsics.principalX},
        {QStringLiteral("cy"), intrinsics.principalY},
        {QStringLiteral("rotation_world_to_camera"),
         doubleArray(rotation.data(), 9)},
        {QStringLiteral("translation_world_to_camera"),
         doubleArray(translation.data(), 3)},
        {QStringLiteral("camera_center"), doubleArray(center.data(), 3)}};
}

xjw::FramePinholeCamera makeBrownCamera()
{
    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(40.0, 42.0, 32.0, 24.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});
    camera.setDistortion(0.35, -0.08, 0.01, 0.006, -0.004);
    camera.setImageSize(xjw::CameraImageSize{64, 48});
    return camera;
}

void attachPreparedArtifactFiles(const QTemporaryDir &temporaryDirectory,
                                 MvsDepthFrameRecord *record)
{
    ASSERT_NE(record, nullptr);
    record->preparedImage = QDir(temporaryDirectory.path()).filePath(
        QStringLiteral("prepared_%1.png").arg(record->refIndex));
    record->preparedValidMaskPath = QDir(temporaryDirectory.path()).filePath(
        QStringLiteral("prepared_%1_valid.png").arg(record->refIndex));
    record->preparedCameraModel = record->cameraModel;
    touchFile(record->preparedImage);
    touchFile(record->preparedValidMaskPath);
}
}

TEST(DepthFrameUtils,
     FastDepthMatStorageHasDeterministicHeaderAndReadsLegacyPadding)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());
    const QDir directory(temporary_directory.path());
    const QString first_path = directory.filePath(QStringLiteral("first.bin"));
    const QString second_path = directory.filePath(QStringLiteral("second.bin"));
    const QString legacy_path = directory.filePath(QStringLiteral("legacy.bin"));

    const cv::Mat matrix(2, 3, CV_32FC1, cv::Scalar(1.25f));
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(first_path, matrix).ok);
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(second_path, matrix).ok);

    QFile first_file(first_path);
    QFile second_file(second_path);
    ASSERT_TRUE(first_file.open(QIODevice::ReadOnly));
    ASSERT_TRUE(second_file.open(QIODevice::ReadOnly));
    const QByteArray first_bytes = first_file.readAll();
    const QByteArray second_bytes = second_file.readAll();
    ASSERT_EQ(first_bytes.size(), second_bytes.size());
    EXPECT_TRUE(first_bytes == second_bytes);
    ASSERT_GE(first_bytes.size(), 40);
    for (qsizetype offset = 28; offset < 32; ++offset)
    {
        EXPECT_EQ(static_cast<unsigned char>(first_bytes.at(offset)), 0u);
    }

    QByteArray legacy_bytes = first_bytes;
    legacy_bytes[28] = static_cast<char>(0x12);
    legacy_bytes[29] = static_cast<char>(0x34);
    legacy_bytes[30] = static_cast<char>(0x56);
    legacy_bytes[31] = static_cast<char>(0x78);
    QFile legacy_file(legacy_path);
    ASSERT_TRUE(legacy_file.open(QIODevice::WriteOnly));
    ASSERT_EQ(legacy_file.write(legacy_bytes), legacy_bytes.size());
    legacy_file.close();

    cv::Mat loaded;
    const auto load_result =
        xjw::core::project::loadDepthMatStorage(legacy_path, &loaded);
    ASSERT_TRUE(load_result.ok) << load_result.errorMessage.toStdString();
    ASSERT_EQ(loaded.type(), matrix.type());
    ASSERT_EQ(loaded.size(), matrix.size());
    EXPECT_EQ(cv::norm(loaded, matrix, cv::NORM_INF), 0.0);
}

TEST(MvsWorkspaceManifest, SavesAndLoadsFrameRecordsAtomically)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString manifestPath = QDir(tempDir.path()).filePath(QStringLiteral("mvs_manifest.json"));

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    MvsDepthFrameRecord record = makeRecord(
        2, QStringLiteral("image_002.jpg"), QStringLiteral("completed"));
    record.preparedImage = QStringLiteral("prepared_images/frame_000002.png");
    record.preparedValidMaskPath = QStringLiteral(
        "prepared_images/frame_000002_valid.png");
    record.preparedCameraModel = record.cameraModel;
    manifest.upsertFrame(record);

    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    MvsWorkspaceManifest loaded;
    ASSERT_TRUE(loaded.load(manifestPath, &error)) << error.toStdString();
    ASSERT_EQ(loaded.frames().size(), 1);
    EXPECT_EQ(loaded.frames().front().refImage, QStringLiteral("image_002.jpg"));
    EXPECT_EQ(loaded.frames().front().preparedImage,
              QStringLiteral("prepared_images/frame_000002.png"));
    EXPECT_EQ(loaded.frames().front().preparedValidMaskPath,
              QStringLiteral("prepared_images/frame_000002_valid.png"));
    EXPECT_DOUBLE_EQ(
        loaded.frames().front().preparedCameraModel
            .value(QStringLiteral("fx"))
            .toDouble(),
        1200.0);
    EXPECT_EQ(loaded.frames().front().rawConfidencePath, QStringLiteral("confidence_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawGeometrySupportPath,
              QStringLiteral("geometry_support_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawAdaptiveGeometrySupportWeightPath,
              QStringLiteral("adaptive_geometry_support_weight_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawAdaptiveGeometryEffectiveViewCountPath,
              QStringLiteral("adaptive_geometry_effective_view_count_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawAdaptiveGeometryConflictRatioPath,
              QStringLiteral("adaptive_geometry_conflict_ratio_002.bin"));
    EXPECT_EQ(loaded.frames().front().sourceIndices, QVector<int>({7, 9}));
    EXPECT_EQ(loaded.frames().front().geometrySourceIndices,
              QVector<int>({7, 9, 11}));
    EXPECT_TRUE(loaded.frames().front().fusionEligibilityKnown);
    EXPECT_TRUE(loaded.frames().front().fusionEligible);
    EXPECT_EQ(loaded.frames().front().role,
              xjw::mvs::DepthFrameRole::Primary);
    EXPECT_EQ(loaded.frames().front().qualityProfile, QStringLiteral("highest"));
    EXPECT_EQ(loaded.frames().front().configuredSourceViewCount, 8);
    EXPECT_EQ(loaded.frames().front().requestedSourceViewCount, 4);
    EXPECT_EQ(loaded.frames().front().sourceViewShortfall, 2);
    EXPECT_EQ(
        loaded.frames().front().sourceViewShortfallReason,
        QStringLiteral("missing_pair_verification_statistics"));
    EXPECT_EQ(
        loaded.frames()
            .front()
            .crossViewRepairDiagnostics
            .value(QStringLiteral("repaired_pixel_count"))
            .toInt(),
        12);
    EXPECT_DOUBLE_EQ(
        loaded.frames()
            .front()
            .geometryEvidenceDiagnostics
            .value(QStringLiteral("native_valid_ratio"))
            .toDouble(),
        0.75);
    EXPECT_TRUE(loaded.frames()
                    .front()
                    .poseRefinementDiagnostics
                    .value(QStringLiteral("candidate_only"))
                    .toBool());
    EXPECT_EQ(loaded.frames()
                  .front()
                  .derivedCameraModel
                  .value(QStringLiteral("camera_center"))
                  .toArray()
                  .size(),
              3);
    EXPECT_EQ(loaded.frames().front().rawGeometrySourceMaskPath,
              QStringLiteral("geometry_source_mask_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawInverseDepthMeanPath,
              QStringLiteral("inverse_depth_mean_002.bin"));
    EXPECT_EQ(loaded.frames().front().rawInverseDepthSpreadPath,
              QStringLiteral("inverse_depth_spread_002.bin"));
    EXPECT_EQ(loaded.frames().front().crossViewRepairedMaskPath,
              QStringLiteral("cross_view_repaired_002.png"));
    EXPECT_EQ(loaded.frames().front().targetedGapRecoveredMaskPath,
              QStringLiteral("targeted_gap_recovered_002.png"));
    EXPECT_EQ(loaded.frames().front().depthProvenancePath,
              QStringLiteral("depth_provenance_002.png"));
    EXPECT_EQ(loaded.frames()
                  .front()
                  .targetedGapRecoveryDiagnostics
                  .value(QStringLiteral("recovered_pixel_count"))
                  .toInt(),
              321);
    EXPECT_EQ(loaded.frames()
                  .front()
                  .depthProvenanceSummary
                  .value(QStringLiteral(
                      "anchored_interpolation_pixel_count"))
                  .toInt(),
              17);
    EXPECT_EQ(loaded.frames().front().missingReasonPath,
              QStringLiteral("missing_reason_002.png"));
    EXPECT_EQ(loaded.frames().front().missingReasonPreviewPath,
              QStringLiteral("missing_reason_preview_002.png"));
    EXPECT_EQ(loaded.frames()
                  .front()
                  .missingReasonSummary
                  .value(QStringLiteral("missing_pixel_count"))
                  .toInt(),
              123);
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

TEST(DepthFrameUtils,
     ManifestArtifactSelectionKeepsOnlyQualifiedSeedsAndReplayMetadata)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    QJsonArray artifacts;
    for (int index = 0; index < 4; ++index)
    {
        MvsDepthFrameRecord record = makeRecord(
            index,
            QDir(temporary_directory.path()).filePath(
                QStringLiteral("image_%1.png").arg(index)),
            QStringLiteral("completed"));
        record.depthPng = QDir(temporary_directory.path()).filePath(
            QStringLiteral("depth_%1.png").arg(index));
        record.rawDepthPath = QDir(temporary_directory.path()).filePath(
            QStringLiteral("depth_%1.bin").arg(index));
        touchFile(record.depthPng);
        touchFile(record.rawDepthPath);
        attachPreparedArtifactFiles(temporary_directory, &record);

        if (index == 1)
        {
            record.acceptance = QStringLiteral("validation_only");
            // A stale/incorrect boolean must never override acceptance.
            record.fusionEligible = true;
        }
        else if (index == 2)
        {
            record.acceptance = QStringLiteral("rejected");
            record.fusionEligible = false;
        }
        if (index == 0)
        {
            record.sourceImages = {
                QDir(temporary_directory.path()).filePath(
                    QStringLiteral("image_3.png")),
                QDir(temporary_directory.path()).filePath(
                    QStringLiteral("image_1.png"))};
            record.qualityDecision = QJsonObject{
                {QStringLiteral("reasons"),
                 QJsonArray{QStringLiteral(
                     "adaptive_geometry_fallback_to_discrete_core")}}};
        }
        artifacts.append(record.toJson());
    }

    const auto discovered =
        xjw::core::project::collectStoredDepthFramesForDirectory(
            artifacts, temporary_directory.path());
    ASSERT_TRUE(discovered.status.ok)
        << discovered.status.errorMessage.toStdString();
    ASSERT_EQ(discovered.frames.size(), 4u);
    for (int index = 0; index < 4; ++index)
    {
        EXPECT_EQ(discovered.frames[static_cast<std::size_t>(index)].refIndex,
                  index);
    }

    const auto selected =
        xjw::core::project::selectFusionEligibleStoredDepthFrames(discovered);
    ASSERT_TRUE(selected.status.ok)
        << selected.status.errorMessage.toStdString();
    ASSERT_EQ(selected.frames.size(), 2u);
    EXPECT_EQ(selected.frames[0].refIndex, 0);
    EXPECT_EQ(selected.frames[1].refIndex, 3);
    EXPECT_TRUE(selected.frames[0].useDiscreteGeometryFallback);
    EXPECT_FALSE(selected.frames[0].preparedImage.isEmpty());
    EXPECT_FALSE(selected.frames[0].preparedCameraModel.isEmpty());
    EXPECT_EQ(
        xjw::core::project::storedFusionSourceIndices(selected.frames, 0),
        (std::vector<int>{1}));
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
    completed.rawGeometrySupportPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_003_geometry_support.bin"));
    completed.rawInverseDepthSpreadPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_003_inverse_depth_spread.bin"));
    completed.rawGeometrySourceMaskPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_003_geometry_source_mask.bin"));
    completed.depthProvenancePath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_003_provenance.png"));
    attachPreparedArtifactFiles(tempDir, &completed);
    touchFile(completed.depthPng);
    touchFile(completed.rawDepthPath);
    touchFile(completed.rawGeometrySupportPath);
    touchFile(completed.rawInverseDepthSpreadPath);
    touchFile(completed.rawGeometrySourceMaskPath);
    touchFile(completed.depthProvenancePath);
    manifest.markCompleted(completed);
    EXPECT_TRUE(manifest.hasReusableCompletedFrame(3, QStringLiteral("cfg-a")));
    EXPECT_FALSE(manifest.hasReusableCompletedFrame(3, QStringLiteral("cfg-b")));
}

TEST(MvsWorkspaceManifest, CompletedFrameUpdatePreservesExistingSourcePlan)
{
    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));

    MvsDepthFrameRecord initial = makeRecord(5, QStringLiteral("image_005.jpg"), QStringLiteral("completed"));
    initial.preparedImage = QStringLiteral("prepared_images/frame_000005.png");
    initial.preparedValidMaskPath = QStringLiteral(
        "prepared_images/frame_000005_valid.png");
    initial.preparedCameraModel = initial.cameraModel;
    ASSERT_EQ(initial.sourcePlan.size(), 1);
    manifest.markCompleted(initial);

    MvsDepthFrameRecord filtered = initial;
    filtered.depthPng = QStringLiteral("filtered_depth_005.png");
    filtered.sourcePlan = QJsonArray();
    filtered.preparedImage.clear();
    filtered.preparedValidMaskPath.clear();
    filtered.preparedCameraModel = QJsonObject();
    filtered.algorithmRevision =
        xjw::mvs::kMvsDepthAlgorithmRevision - 1;
    manifest.markCompleted(filtered);

    ASSERT_EQ(manifest.frames().size(), 1);
    EXPECT_EQ(manifest.frames().front().depthPng, QStringLiteral("filtered_depth_005.png"));
    EXPECT_EQ(manifest.frames().front().algorithmRevision,
              xjw::mvs::kMvsDepthAlgorithmRevision);
    EXPECT_EQ(manifest.frames().front().preparedImage,
              QStringLiteral("prepared_images/frame_000005.png"));
    EXPECT_FALSE(manifest.frames().front().preparedCameraModel.isEmpty());
    ASSERT_EQ(manifest.frames().front().sourcePlan.size(), 1)
        << "Filtered depth artifact updates must not erase the source plan needed for reproducible MVS fusion";
    EXPECT_EQ(manifest.frames().front().sourcePlan.at(0).toObject().value(QStringLiteral("view_index")).toInt(), 7);
}

TEST(MvsWorkspaceManifest, PreservesExplicitFusionIneligibility)
{
    MvsDepthFrameRecord record = makeRecord(
        7, QStringLiteral("image_007.jpg"), QStringLiteral("completed"));
    record.acceptance = QStringLiteral("validation_only");
    record.fusionEligibilityKnown = true;
    record.fusionEligible = false;

    const MvsDepthFrameRecord loaded = MvsDepthFrameRecord::fromJson(
        record.toJson());

    EXPECT_TRUE(loaded.fusionEligibilityKnown);
    EXPECT_FALSE(loaded.fusionEligible);
    EXPECT_EQ(loaded.acceptance, QStringLiteral("validation_only"));
    EXPECT_EQ(loaded.role,
              xjw::mvs::DepthFrameRole::CoverageAuxiliary);
}

TEST(MvsWorkspaceManifest,
     ExplicitAcceptanceDoesNotInheritStalePrimaryEligibility)
{
    MvsWorkspaceManifest manifest;
    MvsDepthFrameRecord primary = makeRecord(
        7, QStringLiteral("image_007.jpg"), QStringLiteral("completed"));
    primary.qualityDecision = QJsonObject{
        {QStringLiteral("acceptance"), QStringLiteral("accepted")}
    };
    manifest.markCompleted(primary);

    MvsDepthFrameRecord validation_update;
    validation_update.refIndex = 7;
    validation_update.refImage = primary.refImage;
    validation_update.acceptance = QStringLiteral("validation_only");
    validation_update.fusionEligibilityKnown = false;
    validation_update.fusionEligible = false;
    manifest.markCompleted(validation_update);

    ASSERT_EQ(manifest.frames().size(), 1);
    const MvsDepthFrameRecord &updated = manifest.frames().front();
    EXPECT_EQ(updated.acceptance, QStringLiteral("validation_only"));
    EXPECT_FALSE(updated.fusionEligibilityKnown);
    EXPECT_FALSE(updated.fusionEligible);
    EXPECT_EQ(updated.role, xjw::mvs::DepthFrameRole::Excluded);
    EXPECT_TRUE(updated.qualityDecision.isEmpty());
    EXPECT_FALSE(updated.toJson().contains(QStringLiteral("fusion_eligible")));
}

TEST(MvsWorkspaceManifest,
     ExplicitEligibilityDoesNotInheritStalePrimaryAcceptance)
{
    MvsWorkspaceManifest manifest;
    MvsDepthFrameRecord primary = makeRecord(
        8, QStringLiteral("image_008.jpg"), QStringLiteral("completed"));
    primary.qualityDecision = QJsonObject{
        {QStringLiteral("acceptance"), QStringLiteral("accepted")}
    };
    manifest.markCompleted(primary);

    MvsDepthFrameRecord partial_update;
    partial_update.refIndex = 8;
    partial_update.refImage = primary.refImage;
    partial_update.fusionEligibilityKnown = true;
    partial_update.fusionEligible = true;
    manifest.markCompleted(partial_update);

    ASSERT_EQ(manifest.frames().size(), 1);
    const MvsDepthFrameRecord &updated = manifest.frames().front();
    EXPECT_TRUE(updated.acceptance.isEmpty());
    EXPECT_TRUE(updated.fusionEligibilityKnown);
    EXPECT_TRUE(updated.fusionEligible);
    EXPECT_EQ(updated.role, xjw::mvs::DepthFrameRole::Excluded);
    EXPECT_TRUE(updated.qualityDecision.isEmpty());
}

TEST(MvsWorkspaceManifest,
     MissingQualificationPairInheritsExistingDecisionAtomically)
{
    MvsWorkspaceManifest manifest;
    MvsDepthFrameRecord primary = makeRecord(
        9, QStringLiteral("image_009.jpg"), QStringLiteral("completed"));
    primary.qualityDecision = QJsonObject{
        {QStringLiteral("acceptance"), QStringLiteral("accepted")},
        {QStringLiteral("reason"), QStringLiteral("original_gate")}
    };
    manifest.markCompleted(primary);

    MvsDepthFrameRecord artifact_update;
    artifact_update.refIndex = 9;
    artifact_update.refImage = primary.refImage;
    artifact_update.depthPng = QStringLiteral("filtered_depth_009.png");
    manifest.markCompleted(artifact_update);

    ASSERT_EQ(manifest.frames().size(), 1);
    const MvsDepthFrameRecord &updated = manifest.frames().front();
    EXPECT_EQ(updated.acceptance, QStringLiteral("accepted"));
    EXPECT_TRUE(updated.fusionEligibilityKnown);
    EXPECT_TRUE(updated.fusionEligible);
    EXPECT_EQ(updated.role, xjw::mvs::DepthFrameRole::Primary);
    EXPECT_EQ(updated.depthPng, QStringLiteral("filtered_depth_009.png"));
    EXPECT_EQ(updated.qualityDecision, primary.qualityDecision);
}

TEST(MvsWorkspaceManifest,
     MalformedFusionEligibilityIsUnknownAndExcluded)
{
    const auto expect_excluded = [](const QJsonValue &malformed_value)
    {
        QJsonObject artifact{
            {QStringLiteral("status"), QStringLiteral("completed")},
            {QStringLiteral("acceptance"), QStringLiteral("validation_only")}
        };
        artifact.insert(QStringLiteral("fusion_eligible"), malformed_value);

        const xjw::mvs::MvsDepthFrameQualification qualification =
            xjw::mvs::qualifyMvsDepthFrameArtifact(artifact);
        EXPECT_FALSE(qualification.fusionEligibilityKnown);
        EXPECT_FALSE(qualification.fusionEligible);
        EXPECT_EQ(qualification.role, xjw::mvs::DepthFrameRole::Excluded);

        const MvsDepthFrameRecord record =
            MvsDepthFrameRecord::fromJson(artifact);
        EXPECT_FALSE(record.fusionEligibilityKnown);
        EXPECT_FALSE(record.fusionEligible);
        EXPECT_EQ(record.role, xjw::mvs::DepthFrameRole::Excluded);
    };

    expect_excluded(QJsonValue(QJsonValue::Null));
    expect_excluded(QJsonValue(QStringLiteral("true")));
    expect_excluded(QJsonValue(QJsonObject{
        {QStringLiteral("value"), true}
    }));

    const QJsonObject explicit_false{
        {QStringLiteral("status"), QStringLiteral("completed")},
        {QStringLiteral("acceptance"), QStringLiteral("accepted")},
        {QStringLiteral("fusion_eligible"), false}
    };
    const xjw::mvs::MvsDepthFrameQualification known_false =
        xjw::mvs::qualifyMvsDepthFrameArtifact(explicit_false);
    EXPECT_TRUE(known_false.fusionEligibilityKnown);
    EXPECT_FALSE(known_false.fusionEligible);
    EXPECT_EQ(known_false.role, xjw::mvs::DepthFrameRole::Excluded);
}

TEST(MvsWorkspaceManifest,
     NestedQualityAcceptanceMustMatchTopLevelQualification)
{
    const auto qualify = [](const QJsonValue &nested_acceptance)
    {
        const QJsonObject artifact{
            {QStringLiteral("status"), QStringLiteral("completed")},
            {QStringLiteral("acceptance"), QStringLiteral("accepted")},
            {QStringLiteral("fusion_eligible"), true},
            {QStringLiteral("quality_decision"), QJsonObject{
                {QStringLiteral("acceptance"), nested_acceptance}
            }}
        };
        const xjw::mvs::MvsDepthFrameQualification qualification =
            xjw::mvs::qualifyMvsDepthFrameArtifact(artifact);
        EXPECT_TRUE(qualification.fusionEligibilityKnown);
        EXPECT_TRUE(qualification.fusionEligible);
        EXPECT_EQ(qualification.role, xjw::mvs::DepthFrameRole::Excluded);
        EXPECT_EQ(MvsDepthFrameRecord::fromJson(artifact).role,
                  xjw::mvs::DepthFrameRole::Excluded);
    };

    qualify(QJsonValue(QStringLiteral("rejected")));
    qualify(QJsonValue(QJsonValue::Null));
    qualify(QJsonValue(QJsonObject{
        {QStringLiteral("value"), QStringLiteral("accepted")}
    }));

    const QJsonObject legacy_without_nested_acceptance{
        {QStringLiteral("status"), QStringLiteral("completed")},
        {QStringLiteral("acceptance"), QStringLiteral("accepted")},
        {QStringLiteral("fusion_eligible"), true},
        {QStringLiteral("quality_decision"), QJsonObject{
            {QStringLiteral("calibrated_confidence"), 0.8}
        }}
    };
    EXPECT_EQ(xjw::mvs::qualifyMvsDepthFrameArtifact(
                  legacy_without_nested_acceptance).role,
              xjw::mvs::DepthFrameRole::Primary);

    QJsonObject normalized_match = legacy_without_nested_acceptance;
    normalized_match[QStringLiteral("quality_decision")] = QJsonObject{
        {QStringLiteral("acceptance"), QStringLiteral(" ACCEPTED ")}
    };
    EXPECT_EQ(xjw::mvs::qualifyMvsDepthFrameArtifact(
                  normalized_match).role,
              xjw::mvs::DepthFrameRole::Primary);
}

TEST(DepthFrameQualificationPolicy, AssignsRolesFailClosed)
{
    using xjw::mvs::DepthFrameRole;
    using xjw::mvs::qualifyDepthFrameRole;

    EXPECT_EQ(xjw::mvs::canonicalDepthSceneProfile(
                  QStringLiteral(" Orbital_Object ")),
              QStringLiteral("orbital_object"));
    EXPECT_TRUE(xjw::mvs::isOrbitalDepthSceneProfile(
        QStringLiteral(" Orbital_Object ")));
    EXPECT_EQ(xjw::mvs::canonicalDepthSceneProfile(
                  QStringLiteral(" GENERAL ")),
              QStringLiteral("custom"));
    EXPECT_FALSE(xjw::mvs::isKnownDepthSceneProfile(
        QStringLiteral("mystery_profile")));

    QString canonical_batch_profile;
    EXPECT_TRUE(xjw::mvs::extendCanonicalDepthSceneProfileBatch(
        QStringLiteral(" Orbital_Object "), &canonical_batch_profile));
    EXPECT_EQ(canonical_batch_profile, QStringLiteral("orbital_object"));
    EXPECT_TRUE(xjw::mvs::extendCanonicalDepthSceneProfileBatch(
        QStringLiteral("orbital_object"), &canonical_batch_profile));
    EXPECT_FALSE(xjw::mvs::extendCanonicalDepthSceneProfileBatch(
        QStringLiteral("aerial_terrain"), &canonical_batch_profile));
    EXPECT_FALSE(xjw::mvs::extendCanonicalDepthSceneProfileBatch(
        QStringLiteral("mystery_profile"), &canonical_batch_profile));

    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("accepted"), true, true,
                  QStringLiteral("completed")),
              DepthFrameRole::Primary);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral(" ACCEPTED "), true, true,
                  QStringLiteral(" Completed ")),
              DepthFrameRole::Primary);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("validation_only"), true, false,
                  QStringLiteral("completed")),
              DepthFrameRole::CoverageAuxiliary);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("validation_only"), true, true,
                  QStringLiteral("completed")),
              DepthFrameRole::CoverageAuxiliary);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("accepted"), false, true,
                  QStringLiteral("completed")),
              DepthFrameRole::Excluded);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("validation_only"), false, false,
                  QStringLiteral("completed")),
              DepthFrameRole::Excluded);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("accepted"), true, false,
                  QStringLiteral("completed")),
              DepthFrameRole::Excluded);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("rejected"), true, true,
                  QStringLiteral("completed")),
              DepthFrameRole::Excluded);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("accepted"), true, true,
                  QStringLiteral("failed")),
              DepthFrameRole::Excluded);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("accepted"), true, true,
                  QStringLiteral("running")),
              DepthFrameRole::Excluded);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("unknown"), true, true,
                  QStringLiteral("completed")),
              DepthFrameRole::Excluded);
    EXPECT_EQ(qualifyDepthFrameRole(
                  QStringLiteral("accepted"), true, true, QString()),
              DepthFrameRole::Excluded);
}

TEST(MvsWorkspaceManifest, PersistsEffectiveNativeGridAndPixelDomainAudit)
{
    MvsDepthFrameRecord record = makeRecord(
        7, QStringLiteral("image_007.jpg"), QStringLiteral("completed"));
    record.effectiveNativeFinalDepthGrid = true;
    record.gridWidth = 1555;
    record.gridHeight = 1036;
    record.pixelDomainDiagnostics = QJsonObject{
        {QStringLiteral("configured_pixel_domain"),
         QStringLiteral("prepared_full_raster")},
        {QStringLiteral("effective_pixel_domain"),
         QStringLiteral("depth_grid")},
        {QStringLiteral("requested_native_final_depth_grid"), true},
        {QStringLiteral("effective_native_final_depth_grid"), true},
        {QStringLiteral("raster_width"), 6221},
        {QStringLiteral("raster_height"), 4146},
        {QStringLiteral("grid_width"), 1555},
        {QStringLiteral("grid_height"), 1036},
        {QStringLiteral("scale_x"), 1555.0 / 6221.0},
        {QStringLiteral("scale_y"), 1036.0 / 4146.0},
        {QStringLiteral("linear_scale"), 0.25},
        {QStringLiteral("area_scale"), 0.0625},
        {QStringLiteral("grid_matches_raster"), false},
        {QStringLiteral("parameters"),
         QJsonObject{
             {QStringLiteral("boundary_edge_radius_pixels"),
              QJsonObject{
                  {QStringLiteral("configured_full_raster"), 1},
                  {QStringLiteral("quantized_grid"), 0},
                  {QStringLiteral("effective_grid"), 0},
                  {QStringLiteral("active"), false},
                  {QStringLiteral("disabled_reason"),
                   QStringLiteral("edge_radius_subpixel_on_depth_grid")}}},
             {QStringLiteral("fusion_reprojection_base_error_pixels"),
              QJsonObject{
                  {QStringLiteral("configured_full_raster"), 1.5},
                  {QStringLiteral("effective_grid"), 0.375},
                  {QStringLiteral("scope"),
                   QStringLiteral(
                       "base_before_view_count_or_streaming_runtime_override")},
                  {QStringLiteral("runtime_scaled_per_target_frame"), true}}}}}};

    const QJsonObject json = record.toJson();
    EXPECT_TRUE(json.value(
        QStringLiteral("effective_native_final_depth_grid")).toBool());
    EXPECT_EQ(json.value(QStringLiteral("grid_width")).toInt(), 1555);
    EXPECT_EQ(json.value(QStringLiteral("grid_height")).toInt(), 1036);
    ASSERT_TRUE(json.value(
        QStringLiteral("pixel_domain_diagnostics")).isObject());

    const MvsDepthFrameRecord loaded = MvsDepthFrameRecord::fromJson(json);
    EXPECT_TRUE(loaded.effectiveNativeFinalDepthGrid);
    EXPECT_EQ(loaded.gridWidth, 1555);
    EXPECT_EQ(loaded.gridHeight, 1036);
    EXPECT_EQ(loaded.pixelDomainDiagnostics.value(
        QStringLiteral("raster_width")).toInt(), 6221);
    EXPECT_DOUBLE_EQ(loaded.pixelDomainDiagnostics.value(
        QStringLiteral("scale_x")).toDouble(), 1555.0 / 6221.0);
    const QJsonObject parameters = loaded.pixelDomainDiagnostics.value(
        QStringLiteral("parameters")).toObject();
    const QJsonObject boundary_edge = parameters.value(
        QStringLiteral("boundary_edge_radius_pixels")).toObject();
    EXPECT_FALSE(boundary_edge.value(QStringLiteral("active")).toBool(true));
    EXPECT_EQ(boundary_edge.value(
        QStringLiteral("disabled_reason")).toString(),
        QStringLiteral("edge_radius_subpixel_on_depth_grid"));
    const QJsonObject fusion_base = parameters.value(
        QStringLiteral("fusion_reprojection_base_error_pixels")).toObject();
    EXPECT_TRUE(fusion_base.value(
        QStringLiteral("runtime_scaled_per_target_frame")).toBool());
    EXPECT_EQ(fusion_base.value(QStringLiteral("scope")).toString(),
              QStringLiteral(
                  "base_before_view_count_or_streaming_runtime_override"));

    const MvsDepthFrameRecord legacy = MvsDepthFrameRecord::fromJson(
        makeRecord(8, QStringLiteral("image_008.jpg"),
                   QStringLiteral("completed")).toJson());
    EXPECT_FALSE(legacy.effectiveNativeFinalDepthGrid);
    EXPECT_TRUE(legacy.pixelDomainDiagnostics.isEmpty());
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
    EXPECT_EQ(loadedRecord.algorithmRevision,
              xjw::mvs::kMvsDepthAlgorithmRevision);

    const QJsonObject json = loadedRecord.toJson();
    EXPECT_EQ(json.value(QStringLiteral("source_view_count")).toInt(), 2);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("source_quality_mean")).toDouble(), 0.72);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("source_quality_min")).toDouble(), 0.43);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("depth_confidence_mean")).toDouble(), 0.81);
    EXPECT_EQ(json.value(QStringLiteral("valid_pixel_count")).toInt(), 123456);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("valid_coverage")).toDouble(), 0.625);
    EXPECT_EQ(json.value(QStringLiteral("support_mask_path")).toString(),
              QStringLiteral("support_006.png"));
    EXPECT_EQ(json.value(QStringLiteral("algorithm_revision")).toInt(),
              xjw::mvs::kMvsDepthAlgorithmRevision);
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

TEST(MvsWorkspaceReplay, RestoresOrderedViewsAndProjectMasks)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString maskDir = QDir(tempDir.path()).filePath(QStringLiteral("masks"));
    ASSERT_TRUE(QDir().mkpath(maskDir));

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("legacy"));
    for (int index = 0; index < 2; ++index)
    {
        const QString imagePath =
            QDir(tempDir.path()).filePath(QStringLiteral("image_%1.png").arg(index));
        ASSERT_TRUE(cv::imwrite(
            imagePath.toStdString(),
            cv::Mat(12, 18, CV_8U, cv::Scalar(80 + index))));
        const QString maskPath =
            QDir(maskDir).filePath(QStringLiteral("image_%1_mask.png").arg(index));
        ASSERT_TRUE(cv::imwrite(
            maskPath.toStdString(),
            cv::Mat(12, 18, CV_8U, cv::Scalar(0))));

        MvsDepthFrameRecord record =
            makeRecord(index, imagePath, QStringLiteral("completed"));
        manifest.markCompleted(record);
    }

    const QString manifestPath =
        QDir(tempDir.path()).filePath(QStringLiteral("mvs_manifest.json"));
    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    std::vector<xjw::mvs::CameraView> views;
    ASSERT_TRUE(xjw::mvs::loadMvsReplayViews(
        manifestPath, maskDir, &views, &error)) << error.toStdString();
    ASSERT_EQ(views.size(), 2);
    EXPECT_EQ(views[0].imageWidth, 18);
    EXPECT_EQ(views[0].imageHeight, 12);
    EXPECT_TRUE(views[0].camera.isValid());
    EXPECT_FALSE(views[0].validRegionMaskPath.empty());
}

TEST(MvsWorkspaceReplay,
     UsesPreparedRasterAndFullResolutionCameraForBrownWorkspace)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("brown-prepared-raster"));
    const xjw::FramePinholeCamera source_camera = makeBrownCamera();
    for (int index = 0; index < 2; ++index)
    {
        cv::Mat source_color(48, 64, CV_8UC3);
        for (int row = 0; row < source_color.rows; ++row)
        {
            for (int column = 0; column < source_color.cols; ++column)
            {
                source_color.at<cv::Vec3b>(row, column) = cv::Vec3b(
                    static_cast<std::uint8_t>((row * 5 + index * 11) % 251),
                    static_cast<std::uint8_t>((column * 7 + index * 13) % 251),
                    static_cast<std::uint8_t>((row + column * 3) % 251));
            }
        }
        const QString source_path = QDir(temporary_directory.path()).filePath(
            QStringLiteral("source_%1.png").arg(index));
        ASSERT_TRUE(xjw::common::io::writeImage(source_path, source_color));

        cv::Mat source_gray;
        cv::cvtColor(source_color, source_gray, cv::COLOR_BGR2GRAY);
        cv::Mat source_valid_mask(
            source_gray.size(), CV_8UC1, cv::Scalar(255));
        cv::rectangle(
            source_valid_mask,
            cv::Rect(20, 14, 18, 16),
            cv::Scalar(0),
            cv::FILLED);
        cv::Mat prepared_gray;
        cv::Mat prepared_valid_mask;
        xjw::FramePinholeCamera prepared_camera;
        std::string preparation_error;
        ASSERT_TRUE(xjw::mvs::prepareMvsImageAndMask(
            source_gray,
            source_valid_mask,
            source_camera,
            &prepared_gray,
            &prepared_valid_mask,
            &prepared_camera,
            &preparation_error)) << preparation_error;

        xjw::mvs::MvsPreparedRasterArtifact prepared_artifact;
        ASSERT_TRUE(xjw::mvs::saveMvsPreparedRasterArtifact(
            xjw::common::io::toUtf8Path(source_path),
            source_camera,
            prepared_valid_mask,
            xjw::common::io::toUtf8Path(temporary_directory.path()),
            index,
            &prepared_artifact,
            &preparation_error)) << preparation_error;

        MvsDepthFrameRecord record = makeRecord(
            index, source_path, QStringLiteral("completed"));
        record.preparedImage = xjw::common::io::fromUtf8Path(
            prepared_artifact.imagePath);
        record.preparedValidMaskPath = xjw::common::io::fromUtf8Path(
            prepared_artifact.validMaskPath);
        record.maskSource = index == 0
            ? QStringLiteral("content")
            : QStringLiteral("project");
        record.preparedCameraModel = cameraJson(prepared_artifact.camera);
        record.cameraModel = cameraJson(
            prepared_artifact.camera.scaledIntrinsics(0.5, 0.5));
        record.gridWidth = 32;
        record.gridHeight = 24;
        manifest.markCompleted(record);
    }

    const QString manifest_path = QDir(temporary_directory.path()).filePath(
        QStringLiteral("mvs_manifest.json"));
    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifest_path, &error))
        << error.toStdString();

    std::vector<xjw::mvs::CameraView> views;
    ASSERT_TRUE(xjw::mvs::loadMvsReplayViews(
        manifest_path, QString(), &views, &error)) << error.toStdString();
    ASSERT_EQ(views.size(), 2);
    EXPECT_EQ(views[0].imageWidth, 64);
    EXPECT_EQ(views[0].imageHeight, 48);
    EXPECT_DOUBLE_EQ(views[0].camera.focalX(), 40.0);
    EXPECT_DOUBLE_EQ(views[0].camera.focalY(), 42.0);
    EXPECT_FALSE(views[0].preparedImagePath.empty());
    EXPECT_FALSE(views[0].preparedValidMaskPath.empty());
    EXPECT_EQ(views[0].preparedValidMaskSource, "content");
    EXPECT_EQ(views[1].preparedValidMaskSource, "project");
    EXPECT_EQ(
        QFileInfo(xjw::common::io::fromUtf8Path(views[0].imagePath))
            .canonicalFilePath(),
        QFileInfo(QDir(temporary_directory.path()).filePath(
                      QStringLiteral("source_0.png")))
            .canonicalFilePath());

    const cv::Mat source_color = xjw::common::io::readImage(
        views[0].imagePath, cv::IMREAD_COLOR);
    const cv::Mat prepared_color = xjw::common::io::readImage(
        views[0].preparedImagePath, cv::IMREAD_COLOR);
    ASSERT_FALSE(source_color.empty());
    ASSERT_FALSE(prepared_color.empty());
    EXPECT_GT(cv::norm(source_color, prepared_color, cv::NORM_INF), 0.0);
}

TEST(DepthFrameUtils,
     StoredNativeGridRestoresPreparedRasterDomainBeforeFusionDownsample)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());
    const QDir directory(temporary_directory.path());

    const QString raw_depth_path = directory.filePath(
        QStringLiteral("depth_0.bin"));
    const QString geometry_support_path = directory.filePath(
        QStringLiteral("depth_0_geometry_support.bin"));
    const QString inverse_depth_spread_path = directory.filePath(
        QStringLiteral("depth_0_inverse_depth_spread.bin"));
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(
        raw_depth_path,
        cv::Mat(12, 16, CV_32FC1, cv::Scalar(8.0f))).ok);
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(
        geometry_support_path,
        cv::Mat(12, 16, CV_16UC1, cv::Scalar(3))).ok);
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(
        inverse_depth_spread_path,
        cv::Mat(12, 16, CV_32FC1, cv::Scalar(0.01f))).ok);

    xjw::FramePinholeCamera prepared_camera = makeBrownCamera();
    prepared_camera.setDistortion(xjw::FramePinholeCamera::Distortion{});
    const xjw::FramePinholeCamera grid_camera =
        prepared_camera.scaledIntrinsics(0.25, 0.25);

    xjw::core::project::StoredDepthFrameRecord stored;
    stored.sceneProfile = QStringLiteral("aerial_terrain");
    stored.refIndex = 0;
    stored.refImage = directory.filePath(QStringLiteral("source.png"));
    stored.preparedImage = directory.filePath(
        QStringLiteral("prepared.png"));
    touchFile(stored.preparedImage);
    stored.preparedCameraModel = cameraJson(prepared_camera);
    stored.cameraModel = cameraJson(grid_camera);
    stored.rawDepthPath = raw_depth_path;
    stored.rawGeometrySupportPath = geometry_support_path;
    stored.rawInverseDepthSpreadPath = inverse_depth_spread_path;
    stored.algorithmRevision = xjw::mvs::kMvsDepthAlgorithmRevision;
    stored.effectiveNativeFinalDepthGrid = true;
    stored.gridWidth = 16;
    stored.gridHeight = 12;
    stored.pixelDomainDiagnostics = QJsonObject{
        {QStringLiteral("effective_native_final_depth_grid"), true},
        {QStringLiteral("raster_width"), 64},
        {QStringLiteral("raster_height"), 48},
        {QStringLiteral("grid_width"), 16},
        {QStringLiteral("grid_height"), 12}};

    xjw::mvs::FusionConfig fusion_config;
    fusion_config.confidenceThresh = 0.0f;
    fusion_config.enableAdaptiveConfidenceFilter = false;
    fusion_config.enableLocalDepthOutlierFilter = false;
    fusion_config.enableSpeckleFilter = false;

    const auto result = xjw::core::project::buildStoredFusionFrame(
        stored,
        makeBrownCamera(),
        fusion_config,
        3,
        8);
    ASSERT_TRUE(result.status.ok) << result.status.errorMessage.toStdString();
    ASSERT_TRUE(result.frame.sourceCamera.imageSize().has_value());
    EXPECT_EQ(result.frame.sourceCamera.imageSize()->samples, 64);
    EXPECT_EQ(result.frame.sourceCamera.imageSize()->lines, 48);
    EXPECT_EQ(result.frame.depthMap.size(), cv::Size(8, 6));
    ASSERT_TRUE(result.frame.cameraModel.imageSize().has_value());
    EXPECT_EQ(result.frame.cameraModel.imageSize()->samples, 8);
    EXPECT_EQ(result.frame.cameraModel.imageSize()->lines, 6);

    xjw::core::project::StoredDepthFrameRecord missing_diagnostics = stored;
    missing_diagnostics.pixelDomainDiagnostics = QJsonObject{};
    const auto missing_result = xjw::core::project::buildStoredFusionFrame(
        missing_diagnostics,
        makeBrownCamera(),
        fusion_config,
        3,
        8);
    EXPECT_FALSE(missing_result.status.ok);
    EXPECT_TRUE(missing_result.status.errorMessage.contains(
        QStringLiteral("pixel_domain_diagnostics")));

    xjw::core::project::StoredDepthFrameRecord contradictory_grid = stored;
    contradictory_grid.pixelDomainDiagnostics.insert(
        QStringLiteral("grid_width"), 15);
    const auto contradictory_result =
        xjw::core::project::buildStoredFusionFrame(
            contradictory_grid,
            makeBrownCamera(),
            fusion_config,
            3,
            8);
    EXPECT_FALSE(contradictory_result.status.ok);
    EXPECT_TRUE(contradictory_result.status.errorMessage.contains(
        QStringLiteral("互相矛盾")));
}

TEST(MvsWorkspaceReplay,
     RejectsIncompleteOrMismatchedPreparedRasterTriplet)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("prepared-triplet-validation"));
    MvsDepthFrameRecord prepared_record;
    for (int index = 0; index < 2; ++index)
    {
        const QString source_path = QDir(temporary_directory.path()).filePath(
            QStringLiteral("source_%1.png").arg(index));
        ASSERT_TRUE(xjw::common::io::writeImage(
            source_path,
            cv::Mat(48, 64, CV_8UC3, cv::Scalar(20 + index, 40, 80))));
        MvsDepthFrameRecord record = makeRecord(
            index, source_path, QStringLiteral("completed"));
        if (index == 0)
        {
            const QString prepared_path = QDir(
                temporary_directory.path()).filePath(
                    QStringLiteral("prepared.png"));
            ASSERT_TRUE(xjw::common::io::writeImage(
                prepared_path,
                cv::Mat(48, 64, CV_8UC3, cv::Scalar(30, 50, 90))));
            record.preparedImage = prepared_path;
            record.preparedCameraModel = cameraJson(makeBrownCamera());
            prepared_record = record;
        }
        manifest.markCompleted(record);
    }

    const QString manifest_path = QDir(temporary_directory.path()).filePath(
        QStringLiteral("mvs_manifest.json"));
    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifest_path, &error))
        << error.toStdString();

    std::vector<xjw::mvs::CameraView> views;
    EXPECT_FALSE(xjw::mvs::loadMvsReplayViews(
        manifest_path, QString(), &views, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("不完整")))
        << error.toStdString();

    prepared_record.preparedValidMaskPath = QDir(
        temporary_directory.path()).filePath(
            QStringLiteral("prepared_valid.png"));
    ASSERT_TRUE(xjw::common::io::writeImage(
        prepared_record.preparedValidMaskPath,
        cv::Mat(24, 32, CV_8UC1, cv::Scalar(255))));
    manifest.markCompleted(prepared_record);
    ASSERT_TRUE(manifest.saveAtomic(manifest_path, &error))
        << error.toStdString();
    EXPECT_FALSE(xjw::mvs::loadMvsReplayViews(
        manifest_path, QString(), &views, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("尺寸不一致")))
        << error.toStdString();
}

TEST(MvsWorkspaceManifest, UpdatesPoseCandidateWithoutChangingFrameStatus)
{
    MvsWorkspaceManifest manifest;
    manifest.upsertFrame(makeRecord(
        4, QStringLiteral("image_004.jpg"), QStringLiteral("completed")));
    const QJsonObject diagnostics{
        {QStringLiteral("candidate_only"), true},
        {QStringLiteral("accepted"), false},
        {QStringLiteral("reason"), QStringLiteral("projection_coverage_regressed")}
    };
    const QJsonObject derived{
        {QStringLiteral("camera_center"), QJsonArray{0.0, 0.0, 1.0}}
    };

    manifest.updatePoseRefinement(4, diagnostics, derived);

    ASSERT_EQ(manifest.frames().size(), 1);
    EXPECT_EQ(manifest.frames().front().status, QStringLiteral("completed"));
    EXPECT_EQ(manifest.frames()
                  .front()
                  .poseRefinementDiagnostics
                  .value(QStringLiteral("reason"))
                  .toString(),
              QStringLiteral("projection_coverage_regressed"));
    EXPECT_EQ(manifest.frames().front().derivedCameraModel, derived);
}

TEST(MvsWorkspaceReplay, ResolvesRelativeImagePathsAgainstManifestDirectory)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imageDir =
        QDir(tempDir.path()).filePath(QStringLiteral("影像 目录"));
    ASSERT_TRUE(QDir().mkpath(imageDir));

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("relative-paths"));
    for (int index = 0; index < 2; ++index)
    {
        const QString imagePath = QDir(imageDir).filePath(
            QStringLiteral("影像_%1.png").arg(index));
        ASSERT_TRUE(xjw::common::io::writeImage(
            imagePath, cv::Mat(12, 18, CV_8U, cv::Scalar(80 + index))));
        MvsDepthFrameRecord record = makeRecord(
            index,
            QDir(tempDir.path()).relativeFilePath(imagePath),
            QStringLiteral("completed"));
        manifest.markCompleted(record);
    }

    const QString manifestPath =
        QDir(tempDir.path()).filePath(QStringLiteral("mvs_manifest.json"));
    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    std::vector<xjw::mvs::CameraView> views;
    ASSERT_TRUE(xjw::mvs::loadMvsReplayViews(
        manifestPath, QString(), &views, &error)) << error.toStdString();
    ASSERT_EQ(views.size(), 2);
    EXPECT_EQ(QFileInfo(xjw::common::io::fromUtf8Path(views[0].imagePath))
                  .canonicalFilePath(),
              QFileInfo(QDir(imageDir).filePath(QStringLiteral("影像_0.png")))
                  .canonicalFilePath());
}

TEST(MvsWorkspaceReplay, RejectsDuplicateImagePathsAcrossFrameIndices)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imagePath =
        QDir(tempDir.path()).filePath(QStringLiteral("same_image.png"));
    ASSERT_TRUE(cv::imwrite(
        imagePath.toStdString(), cv::Mat(12, 18, CV_8U, cv::Scalar(80))));

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("duplicate-images"));
    manifest.markCompleted(makeRecord(0, imagePath, QStringLiteral("completed")));
    manifest.markCompleted(makeRecord(1, imagePath, QStringLiteral("completed")));
    const QString manifestPath =
        QDir(tempDir.path()).filePath(QStringLiteral("mvs_manifest.json"));
    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    std::vector<xjw::mvs::CameraView> views;
    EXPECT_FALSE(xjw::mvs::loadMvsReplayViews(
        manifestPath, QString(), &views, &error));
    EXPECT_TRUE(views.empty());
    EXPECT_TRUE(error.contains(QStringLiteral("重复 ref_image")))
        << error.toStdString();
}

TEST(MvsWorkspaceReplay, LoadsVerifiedFailedAndMissingPairAuditStates)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString reportPath =
        QDir(tempDir.path()).filePath(QStringLiteral("pair_audit.json"));
    writeJson(
        reportPath,
        QJsonObject{
            {QStringLiteral("pairs"),
             QJsonArray{
                 QJsonObject{
                     {QStringLiteral("image_a"), QStringLiteral("a.png")},
                     {QStringLiteral("image_b"), QStringLiteral("b.png")},
                     {QStringLiteral("status"), QStringLiteral("verified")},
                     {QStringLiteral("total_matches"), 100},
                     {QStringLiteral("geometric_inliers"), 90},
                     {QStringLiteral("coverage_score"), 0.5}},
                 QJsonObject{
                     {QStringLiteral("image_a"), QStringLiteral("b.png")},
                     {QStringLiteral("image_b"), QStringLiteral("c.png")},
                     {QStringLiteral("status"), QStringLiteral("failed")}},
                 QJsonObject{
                     {QStringLiteral("image_a"), QStringLiteral("c.png")},
                     {QStringLiteral("image_b"), QStringLiteral("d.png")},
                     {QStringLiteral("status"), QStringLiteral("missing_statistics")}}
             }}
        });

    std::vector<xjw::mvs::MvsSourcePairQuality> qualities;
    xjw::mvs::MvsPairAuditSummary summary;
    QString error;
    ASSERT_TRUE(xjw::mvs::loadMvsPairAuditReport(
        reportPath, &qualities, &summary, &error)) << error.toStdString();
    ASSERT_EQ(qualities.size(), 3);
    EXPECT_EQ(summary.auditedPairCount, 3);
    EXPECT_EQ(summary.verifiedPairCount, 1);
    EXPECT_EQ(summary.failedPairCount, 1);
    EXPECT_EQ(summary.missingStatisticsPairCount, 1);
    EXPECT_TRUE(qualities[0].verified);
    EXPECT_EQ(QDir::cleanPath(xjw::common::io::fromUtf8Path(qualities[0].imageA)),
              QDir(tempDir.path()).filePath(QStringLiteral("a.png")));
    EXPECT_EQ(QDir::cleanPath(xjw::common::io::fromUtf8Path(qualities[0].imageB)),
              QDir(tempDir.path()).filePath(QStringLiteral("b.png")));
    EXPECT_TRUE(qualities[1].hasVerificationStatistics);
    EXPECT_FALSE(qualities[1].verified);
    EXPECT_FALSE(qualities[2].hasVerificationStatistics);
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
    record.sceneProfile = QStringLiteral(" Orbital_Object ");
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

TEST(MvsWorkspaceManifest, CurrentFrameRequiresGeometrySupportAndInverseDepthSpread)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    MvsDepthFrameRecord record = makeRecord(
        4,
        QStringLiteral("image_004.jpg"),
        QStringLiteral("completed"));
    record.sceneProfile = QStringLiteral("aerial_terrain");
    record.depthPng = QDir(tempDir.path()).filePath(QStringLiteral("depth_004.png"));
    record.rawDepthPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_004.bin"));
    record.rawGeometrySupportPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_geometry_support.bin"));
    record.rawInverseDepthSpreadPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_inverse_depth_spread.bin"));
    record.rawGeometrySourceMaskPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_geometry_source_mask.bin"));
    record.depthProvenancePath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_provenance.png"));
    attachPreparedArtifactFiles(tempDir, &record);
    touchFile(record.depthPng);
    touchFile(record.rawDepthPath);
    touchFile(record.depthProvenancePath);
    touchFile(record.rawGeometrySourceMaskPath);
    manifest.upsertFrame(record);

    EXPECT_FALSE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")))
        << "A current frame without geometry support must not be reused";

    touchFile(record.rawGeometrySupportPath);
    EXPECT_FALSE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")))
        << "A current frame without inverse-depth spread must not be reused";

    touchFile(record.rawInverseDepthSpreadPath);
    EXPECT_TRUE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")));

    record.geometrySourceIndices.clear();
    manifest.upsertFrame(record);
    EXPECT_FALSE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")))
        << "A source mask path without its exact ordinal table must not be reused";

    const QString geometry_source_mask_path =
        record.rawGeometrySourceMaskPath;
    record.rawGeometrySourceMaskPath.clear();
    manifest.upsertFrame(record);
    EXPECT_TRUE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")))
        << "A current frame may explicitly omit both the all-zero source mask "
           "and its empty ordinal table";
    const QString zero_source_manifest_path = QDir(tempDir.path()).filePath(
        QStringLiteral("zero_source_manifest.json"));
    QString manifest_error;
    ASSERT_TRUE(manifest.saveAtomic(
        zero_source_manifest_path, &manifest_error))
        << manifest_error.toStdString();
    MvsWorkspaceManifest reloaded_zero_source_manifest;
    ASSERT_TRUE(reloaded_zero_source_manifest.load(
        zero_source_manifest_path, &manifest_error))
        << manifest_error.toStdString();
    EXPECT_TRUE(reloaded_zero_source_manifest.hasReusableCompletedFrame(
        4, QStringLiteral("cfg-a")));

    record.rawGeometrySourceMaskPath = geometry_source_mask_path;
    record.geometrySourceIndices = {7, 7};
    manifest.upsertFrame(record);
    EXPECT_FALSE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")))
        << "A duplicate source ordinal must not be reused";

    record.geometrySourceIndices = {7, 9, 11};
    manifest.upsertFrame(record);
    EXPECT_TRUE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")));
}

TEST(MvsWorkspaceManifest, PreviousRevisionFrameIsNotReusableEvenWhenArtifactsMatch)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    MvsDepthFrameRecord record = makeRecord(
        4,
        QStringLiteral("image_004.jpg"),
        QStringLiteral("completed"));
    record.algorithmRevision = xjw::mvs::kMvsAdaptiveGeometryEvidenceRevision;
    record.sceneProfile = QStringLiteral("orbital_object");
    record.depthPng = QDir(tempDir.path()).filePath(QStringLiteral("depth_004.png"));
    record.rawDepthPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_004.bin"));
    record.rawGeometrySupportPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_geometry_support.bin"));
    record.rawInverseDepthSpreadPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_inverse_depth_spread.bin"));
    record.rawGeometrySourceMaskPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_geometry_source_mask.bin"));
    record.rawAdaptiveGeometrySupportWeightPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_adaptive_geometry_support_weight.bin"));
    record.rawAdaptiveGeometryEffectiveViewCountPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_adaptive_geometry_effective_view_count.bin"));
    record.depthProvenancePath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_provenance.png"));
    attachPreparedArtifactFiles(tempDir, &record);
    touchFile(record.depthPng);
    touchFile(record.rawDepthPath);
    touchFile(record.rawGeometrySupportPath);
    touchFile(record.rawInverseDepthSpreadPath);
    touchFile(record.rawGeometrySourceMaskPath);
    touchFile(record.rawAdaptiveGeometrySupportWeightPath);
    touchFile(record.rawAdaptiveGeometryEffectiveViewCountPath);
    touchFile(record.depthProvenancePath);
    manifest.upsertFrame(record);

    EXPECT_FALSE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")));
}

TEST(MvsWorkspaceManifest, CurrentOrbitalFrameRequiresConflictRatio)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("cfg-a"));
    MvsDepthFrameRecord record = makeRecord(
        4,
        QStringLiteral("image_004.jpg"),
        QStringLiteral("completed"));
    record.algorithmRevision = xjw::mvs::kMvsDepthAlgorithmRevision;
    record.sceneProfile = QStringLiteral("orbital_object");
    record.depthPng = QDir(tempDir.path()).filePath(QStringLiteral("depth_004.png"));
    record.rawDepthPath = QDir(tempDir.path()).filePath(QStringLiteral("depth_004.bin"));
    record.rawGeometrySupportPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_geometry_support.bin"));
    record.rawInverseDepthSpreadPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_inverse_depth_spread.bin"));
    record.rawGeometrySourceMaskPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_geometry_source_mask.bin"));
    record.rawAdaptiveGeometrySupportWeightPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_adaptive_geometry_support_weight.bin"));
    record.rawAdaptiveGeometryEffectiveViewCountPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_adaptive_geometry_effective_view_count.bin"));
    record.rawAdaptiveGeometryConflictRatioPath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_adaptive_geometry_conflict_ratio.bin"));
    record.depthProvenancePath = QDir(tempDir.path()).filePath(
        QStringLiteral("depth_004_provenance.png"));
    attachPreparedArtifactFiles(tempDir, &record);
    touchFile(record.depthPng);
    touchFile(record.rawDepthPath);
    touchFile(record.rawGeometrySupportPath);
    touchFile(record.rawInverseDepthSpreadPath);
    touchFile(record.rawGeometrySourceMaskPath);
    touchFile(record.rawAdaptiveGeometrySupportWeightPath);
    touchFile(record.rawAdaptiveGeometryEffectiveViewCountPath);
    touchFile(record.depthProvenancePath);
    manifest.upsertFrame(record);

    EXPECT_FALSE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")));

    touchFile(record.rawAdaptiveGeometryConflictRatioPath);
    EXPECT_TRUE(manifest.hasReusableCompletedFrame(4, QStringLiteral("cfg-a")));
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
        changed.patchMatch.backend = xjw::mvs::PatchMatchBackend::OpenCl;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.patchMatch.openClDeviceIndex = 1;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.preserveNativeFinalDepthGrid =
            !changed.preserveNativeFinalDepthGrid;
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
        changed.fusion.enableGeometrySupportedLowConfidenceRetention =
            !changed.fusion.enableGeometrySupportedLowConfidenceRetention;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.fusion.geometrySupportedMaximumInverseDepthSpread += 0.001f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.depthFilterMode = xjw::mvs::DepthFilterMode::Aggressive;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.enableAdaptiveGeometryEvidence =
            !changed.enableAdaptiveGeometryEvidence;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.enableTargetedGapRecovery = !changed.enableTargetedGapRecovery;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.targetedGapRecoveryConfidence += 0.01f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.targetedGapRecoveryHypothesisCount += 1;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.targetedGapRecoveryConsensusInverseDepthSpread += 0.001f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.enableTargetedGapSurfacePrior =
            !changed.enableTargetedGapSurfacePrior;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.targetedGapSurfacePriorMaximumFitResidual += 0.001f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.targetedGapRecoveryMaximumPriorDistancePixels += 1;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.enablePostConsistencyResidualReestimation =
            !changed.enablePostConsistencyResidualReestimation;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.postConsistencyResidualMaximumLayerSpread += 0.001f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.depthPoseRefinement.enabled = true;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.fusion.maxLocalDepthOutlierRemovalRatio -= 0.01f;
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.resolvedImageCacheStrategy = "bounded";
    });
    expect_hash_change([](xjw::mvs::DepthGenConfig &changed) {
        changed.resolvedImageCacheCapacity += 1;
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
