#include "DepthArtifactIO.h"
#include "DepthMatStorage.h"
#include "DenseCloudArtifactValidation.h"
#include "MvsTypes.h"
#include "MvsWorkspaceManifest.h"
#include "MvsWorkspaceReplay.h"
#include "PointCloudArtifactIO.h"
#include "io/PathIO.h"

#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QTemporaryDir>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace
{

    QJsonObject cameraJson()
    {
        return {{QStringLiteral("fx"), 120.0},
                {QStringLiteral("fy"), 120.0},
                {QStringLiteral("cx"), 9.0},
                {QStringLiteral("cy"), 6.0},
                {QStringLiteral("rotation_world_to_camera"), QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}},
                {QStringLiteral("camera_center"), QJsonArray{0.0, 0.0, 0.0}}};
    }

} // namespace

class MvsStorageContract : public testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(QDir().mkpath(QStringLiteral(PLASCAN_MVS_STORAGE_TEST_TMP)));
    }
};

TEST_F(MvsStorageContract, FastDepthMatStorageHasDeterministicHeaderAndReadsLegacyPadding)
{
    QTemporaryDir temporary_directory(QStringLiteral(PLASCAN_MVS_STORAGE_TEST_TMP "/matrix-XXXXXX"));
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
    const auto load_result = xjw::core::project::loadDepthMatStorage(legacy_path, &loaded);
    ASSERT_TRUE(load_result.ok) << load_result.errorMessage.toStdString();
    ASSERT_EQ(loaded.type(), matrix.type());
    ASSERT_EQ(loaded.size(), matrix.size());
    EXPECT_EQ(cv::norm(loaded, matrix, cv::NORM_INF), 0.0);
}

TEST_F(MvsStorageContract, RoundTripsNoncontiguousEvidenceAndRejectsTruncatedPayload)
{
    QTemporaryDir temporary_directory(QStringLiteral(PLASCAN_MVS_STORAGE_TEST_TMP "/evidence-XXXXXX"));
    ASSERT_TRUE(temporary_directory.isValid());
    const QString path = QDir(temporary_directory.path()).filePath(QStringLiteral("evidence.bin"));
    cv::Mat parent(4, 6, CV_32SC1, cv::Scalar(7));
    const cv::Mat evidence = parent(cv::Rect(1, 1, 3, 2));
    ASSERT_FALSE(evidence.isContinuous());
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(path, evidence).ok);
    cv::Mat loaded;
    ASSERT_TRUE(xjw::core::project::loadDepthMatStorage(path, &loaded).ok);
    EXPECT_EQ(loaded.type(), CV_32SC1);
    EXPECT_EQ(loaded.size(), evidence.size());
    EXPECT_EQ(cv::norm(loaded, evidence, cv::NORM_INF), 0.0);

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::ReadWrite));
    ASSERT_TRUE(file.resize(file.size() - 1));
    file.close();
    const auto result = xjw::core::project::loadDepthMatStorage(path, &loaded);
    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("数据不完整"))) << qUtf8Printable(result.errorMessage);
}

TEST_F(MvsStorageContract, ReportsInvalidInputAndWriteFailure)
{
    QTemporaryDir temporary_directory(QStringLiteral(PLASCAN_MVS_STORAGE_TEST_TMP "/errors-XXXXXX"));
    ASSERT_TRUE(temporary_directory.isValid());
    const QDir directory(temporary_directory.path());
    cv::Mat loaded;
    EXPECT_FALSE(xjw::core::project::loadDepthMatStorage(directory.filePath(QStringLiteral("absent.bin")), &loaded).ok);
    EXPECT_FALSE(xjw::core::project::loadDepthMatStorage(directory.filePath(QStringLiteral("depth.png")), &loaded).ok);
    EXPECT_FALSE(
        xjw::core::project::writeDepthMatStorage(directory.filePath(QStringLiteral("empty.bin")), cv::Mat()).ok);
    const auto failure =
        xjw::core::project::writeDepthMatStorage(temporary_directory.path(), cv::Mat::ones(2, 3, CV_32F));
    EXPECT_FALSE(failure.ok);
    EXPECT_TRUE(failure.errorMessage.contains(temporary_directory.path()));
}

TEST_F(MvsStorageContract, PreviewPreservesInvalidPixelsAndMaximumDimension)
{
    QTemporaryDir temporary_directory(QStringLiteral(PLASCAN_MVS_STORAGE_TEST_TMP "/preview-XXXXXX"));
    ASSERT_TRUE(temporary_directory.isValid());
    const QString path = QDir(temporary_directory.path()).filePath(QStringLiteral("preview.png"));
    cv::Mat depth(8, 4096, CV_32F, cv::Scalar(2.0f));
    depth.colRange(0, 2048).setTo(0.0f);
    std::string error;
    ASSERT_TRUE(xjw::mvs::saveDepthPreviewPng(xjw::common::io::toUtf8Path(path), depth, &error)) << error;
    const cv::Mat preview = xjw::common::io::readImage(path, cv::IMREAD_COLOR);
    ASSERT_EQ(preview.size(), cv::Size(2048, 4));
    EXPECT_EQ(cv::norm(preview.colRange(0, 1024), cv::NORM_INF), 0.0);
    EXPECT_GT(cv::norm(preview.colRange(1024, 2048), cv::NORM_INF), 0.0);
}

TEST_F(MvsStorageContract, ManifestRoundTripAndReplayKeepOrderedCameraAndMaskIdentity)
{
    QTemporaryDir temporary_directory(QStringLiteral(PLASCAN_MVS_STORAGE_TEST_TMP "/replay-XXXXXX"));
    ASSERT_TRUE(temporary_directory.isValid());
    const QDir directory(temporary_directory.path());
    const QString mask_directory = directory.filePath(QStringLiteral("masks"));
    ASSERT_TRUE(QDir().mkpath(mask_directory));
    xjw::mvs::MvsWorkspaceManifest manifest;
    manifest.setConfigHash(QStringLiteral("storage-contract"));
    for (int index : {1, 0})
    {
        const QString image = directory.filePath(QStringLiteral("image_%1.png").arg(index));
        const QString mask = QDir(mask_directory).filePath(QStringLiteral("image_%1_mask.png").arg(index));
        ASSERT_TRUE(xjw::common::io::writeImage(image, cv::Mat(12, 18, CV_8UC1, cv::Scalar(80 + index))));
        ASSERT_TRUE(xjw::common::io::writeImage(mask, cv::Mat::zeros(12, 18, CV_8UC1)));
        xjw::mvs::MvsDepthFrameRecord record;
        record.refIndex = index;
        record.refImage = image;
        record.cameraModel = cameraJson();
        record.cameraModel.insert(QStringLiteral("camera_center"), QJsonArray{static_cast<double>(index), 0.0, 0.0});
        record.cameraModel.insert(QStringLiteral("instance_id"), QStringLiteral("mvs-instance-%1").arg(index));
        record.cameraModel.insert(QStringLiteral("image_id"), QStringLiteral("mvs-image-%1").arg(index));
        record.cameraModel.insert(QStringLiteral("world_frame"), QStringLiteral("project-world"));
        record.status = QStringLiteral("completed");
        manifest.upsertFrame(record);
    }
    const QString path = directory.filePath(QStringLiteral("nested/mvs_manifest.json"));
    QString error;
    ASSERT_TRUE(manifest.saveAtomic(path, &error)) << qUtf8Printable(error);
    xjw::mvs::MvsWorkspaceManifest loaded;
    ASSERT_TRUE(loaded.load(path, &error)) << qUtf8Printable(error);
    EXPECT_EQ(loaded.toJson(), manifest.toJson());
    EXPECT_EQ(loaded.configHash(), QStringLiteral("storage-contract"));
    std::vector<xjw::mvs::CameraView> views;
    ASSERT_TRUE(xjw::mvs::loadMvsReplayViews(path, mask_directory, &views, &error)) << qUtf8Printable(error);
    ASSERT_EQ(views.size(), 2u);
    for (int index = 0; index < 2; ++index)
    {
        EXPECT_EQ(views[index].imageWidth, 18);
        EXPECT_EQ(views[index].imageHeight, 12);
        EXPECT_DOUBLE_EQ(views[index].camera.focalX(), 120.0);
        EXPECT_DOUBLE_EQ(views[index].camera.principalX(), 9.0);
        EXPECT_DOUBLE_EQ(views[index].camera.cameraCenter()[0], static_cast<double>(index));
        EXPECT_TRUE(views[index].camera.hasBoundIdentity());
        EXPECT_EQ(views[index].camera.instanceId().value(), "mvs-instance-" + std::to_string(index));
        EXPECT_EQ(views[index].camera.imageId().value(), "mvs-image-" + std::to_string(index));
        EXPECT_EQ(views[index].camera.worldFrame().value(), "project-world");
        EXPECT_EQ(views[index].imagePath,
                  xjw::common::io::toUtf8Path(directory.filePath(QStringLiteral("image_%1.png").arg(index))));
        EXPECT_EQ(
            views[index].validRegionMaskPath,
            xjw::common::io::toUtf8Path(QDir(mask_directory).filePath(QStringLiteral("image_%1_mask.png").arg(index))));
    }
    ASSERT_TRUE(QFile::remove(QDir(mask_directory).filePath(QStringLiteral("image_1_mask.png"))));
    EXPECT_FALSE(xjw::mvs::loadMvsReplayViews(path, mask_directory, &views, &error));
    EXPECT_TRUE(views.empty());
    EXPECT_TRUE(error.contains(QStringLiteral("蒙版不存在"))) << qUtf8Printable(error);
}

TEST_F(MvsStorageContract, PointCloudWriterAndValidatorRejectTruncatedPublication)
{
    QTemporaryDir temporary_directory(QStringLiteral(PLASCAN_MVS_STORAGE_TEST_TMP "/ply-XXXXXX"));
    ASSERT_TRUE(temporary_directory.isValid());
    const QString path = QDir(temporary_directory.path()).filePath(QStringLiteral("nested/cloud.ply"));
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(2, 3);
    for (int index = 0; index < 2; ++index)
    {
        points(index, 0) = static_cast<float>(index);
        points(index, 1) = 2.0f;
        points(index, 2) = 3.0f;
    }
    xjw::mvs::DensePointCloud cloud(std::move(points));
    QString error;
    ASSERT_TRUE(xjw::mvs::writeDensePointCloudPly(path, cloud, true, &error)) << qUtf8Printable(error);
    std::string validation_error;
    ASSERT_TRUE(
        xjw::mvs::detail::validateDenseCloudPlyArtifact(xjw::common::io::toFilesystemPath(path), 2, &validation_error))
        << validation_error;
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::ReadWrite));
    ASSERT_TRUE(file.resize(file.size() - 1));
    file.close();
    EXPECT_FALSE(
        xjw::mvs::detail::validateDenseCloudPlyArtifact(xjw::common::io::toFilesystemPath(path), 2, &validation_error));
    EXPECT_NE(validation_error.find("byte length mismatch"), std::string::npos) << validation_error;
}
