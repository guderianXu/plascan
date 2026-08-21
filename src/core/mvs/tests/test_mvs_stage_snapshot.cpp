#include "DepthFrameUtils.h"
#include "DepthMapGenerator.h"
#include "MvsStageSnapshot.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace
{

QJsonObject loadObject(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}

xjw::mvs::DepthFrameResult makeFrame(int reference_index,
                                     int width,
                                     int height)
{
    xjw::mvs::DepthFrameResult result;
    result.refViewIdx = reference_index;
    result.success = true;
    result.preparedRasterSize = cv::Size(width, height);
    result.depthMap = QSharedPointer<cv::Mat>::create(
        height, width, CV_32FC1, cv::Scalar(2.0f));
    result.confidence = QSharedPointer<cv::Mat>::create(
        height, width, CV_32FC1, cv::Scalar(0.75f));
    result.validMask = QSharedPointer<cv::Mat>::create(
        height, width, CV_8UC1, cv::Scalar(255));
    result.depthMap->at<float>(0, 0) = 0.0f;

    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(800.0, 810.0, 3.5, 1.5);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});
    camera.setImageSize(xjw::CameraImageSize{width, height});
    result.cameraModel = camera;
    result.qualityMetrics.width = width;
    result.qualityMetrics.height = height;
    result.qualityMetrics.validPixelCount = width * height - 1;
    result.qualityMetrics.validCoverage =
        static_cast<float>(width * height - 1) /
        static_cast<float>(width * height);
    return result;
}

const QJsonObject *recordForStage(const QJsonArray &records,
                                  const QString &stage,
                                  QJsonObject *storage)
{
    for (const QJsonValue &value : records)
    {
        const QJsonObject object = value.toObject();
        if (object.value(QStringLiteral("stage")).toString() == stage)
        {
            *storage = object;
            return storage;
        }
    }
    return nullptr;
}

} // namespace

TEST(MvsStageSnapshotTest, DefaultConfigurationProducesNoArtifacts)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    xjw::mvs::DepthGenConfig config;
    config.stageSnapshotDirectory = temporary.filePath("snapshots").toStdString();

    xjw::mvs::MvsStageSnapshotRecorder recorder(config, 3);
    EXPECT_FALSE(recorder.enabled());
    EXPECT_FALSE(QFileInfo::exists(temporary.filePath("snapshots/manifest.json")));
}

TEST(MvsStageSnapshotTest, CapturesBoundedTripletAndRecordsMissingStages)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString snapshot_directory = temporary.filePath("snapshots");
    xjw::mvs::DepthGenConfig config;
    config.stageSnapshotReferenceIndices = {2};
    config.stageSnapshotMaximumLongEdge = 4;
    config.stageSnapshotBudgetBytes = 1024 * 1024;
    config.stageSnapshotDirectory = snapshot_directory.toStdString();
    xjw::mvs::DepthFrameResult frame = makeFrame(2, 8, 4);

    xjw::mvs::MvsStageSnapshotRecorder recorder(config, 3);
    ASSERT_TRUE(recorder.enabled()) << recorder.initializationError().toStdString();
    recorder.capture(
        2,
        xjw::mvs::MvsStageSnapshotStage::PatchMatchOutput,
        QStringLiteral("test_boundary"),
        frame,
        *frame.depthMap,
        *frame.confidence,
        *frame.validMask);
    recorder.finalize();

    const QJsonObject manifest = loadObject(recorder.manifestPath());
    EXPECT_EQ(manifest.value(QStringLiteral("status")).toString(),
              QStringLiteral("complete"));
    EXPECT_FALSE(manifest.value(QStringLiteral("authoritative")).toBool(true));
    const QJsonArray records = manifest.value(QStringLiteral("records")).toArray();
    ASSERT_EQ(records.size(), 4);
    QJsonObject captured;
    ASSERT_NE(recordForStage(records, QStringLiteral("patchmatch_output"), &captured),
              nullptr);
    EXPECT_EQ(captured.value(QStringLiteral("status")).toString(),
              QStringLiteral("captured"));
    EXPECT_EQ(captured.value(QStringLiteral("snapshot_width")).toInt(), 4);
    EXPECT_EQ(captured.value(QStringLiteral("snapshot_height")).toInt(), 2);

    const QString depth_path = captured.value(QStringLiteral("depth"))
                                   .toObject()
                                   .value(QStringLiteral("path"))
                                   .toString();
    cv::Mat restored_depth;
    const auto load_result = xjw::core::project::loadDepthMatStorage(
        depth_path, &restored_depth);
    ASSERT_TRUE(load_result.ok) << load_result.errorMessage.toStdString();
    EXPECT_EQ(restored_depth.size(), cv::Size(4, 2));
    EXPECT_EQ(restored_depth.type(), CV_32FC1);

    QJsonObject missing;
    ASSERT_NE(recordForStage(
                  records, QStringLiteral("cross_view_consistency"), &missing),
              nullptr);
    EXPECT_EQ(missing.value(QStringLiteral("status")).toString(),
              QStringLiteral("missing"));
    EXPECT_EQ(missing.value(QStringLiteral("reason")).toString(),
              QStringLiteral("stage_not_reached"));
}

TEST(MvsStageSnapshotTest, BudgetSkipsWholeStageWithoutPartialPayload)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    xjw::mvs::DepthGenConfig config;
    config.stageSnapshotReferenceIndices = {0};
    config.stageSnapshotMaximumLongEdge = 8;
    config.stageSnapshotBudgetBytes = 1;
    config.stageSnapshotDirectory = temporary.filePath("snapshots").toStdString();
    xjw::mvs::DepthFrameResult frame = makeFrame(0, 8, 4);

    xjw::mvs::MvsStageSnapshotRecorder recorder(config, 1);
    ASSERT_TRUE(recorder.enabled());
    recorder.capture(
        0,
        xjw::mvs::MvsStageSnapshotStage::PatchMatchOutput,
        QString(),
        frame,
        *frame.depthMap,
        *frame.confidence,
        *frame.validMask);
    recorder.finalize();

    const QJsonArray records = loadObject(recorder.manifestPath())
                                   .value(QStringLiteral("records"))
                                   .toArray();
    QJsonObject skipped;
    ASSERT_NE(recordForStage(records, QStringLiteral("patchmatch_output"), &skipped),
              nullptr);
    EXPECT_EQ(skipped.value(QStringLiteral("status")).toString(),
              QStringLiteral("skipped"));
    EXPECT_EQ(skipped.value(QStringLiteral("reason")).toString(),
              QStringLiteral("snapshot_budget_exhausted"));
    EXPECT_FALSE(QDir(temporary.filePath("snapshots/ref_0000")).exists());
}
